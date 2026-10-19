"""
Library wrapper around the existing NAFRec pipeline in main.py.

Starts the model (inference) workers and the CPU pre/postprocessing pool once,
then accepts image processing requests until closed. All the actual work is done
by the existing functions in main.py (start_inference_workers, init_cpu_worker,
predict_single_image), so predictions are identical to the CLI. Nothing is
written to disk: each request returns the internal (preds, xml_input) pair that
xml_output.get_xml (and utils.save_json_output) take.

Example:
    from nafrec import NAFRec

    if __name__ == "__main__":      # required: workers use the "spawn" start method
        with NAFRec(detection_model_path="det.pth",
                    recognition_model_path="trocr", processor_path="proc",
                    device="cuda", gpu_ids=[0]) as rec:
            result = rec.process("page.jpg")                     # blocking
            if result:                                           # None = nothing detected
                preds, xml_input = result
                get_xml(preds, xml_input)                        # from nafrec.xml_output
            fut = rec.submit("page2.jpg")                        # non-blocking
            results = rec.process_many(["a.jpg", "b.jpg"])       # parallel, input order

For callers with their own CPU processes (e.g. SfRD), NAFRec(cpu_pool=False, ...) starts only the
model workers. Pass the NAFRec object itself to those processes (it pickles as a lightweight handle
to the model workers) and call the building blocks of process() there:
    lines = handle.detect("page.jpg")                  # 7-tuple of raw line / region polygons
    preds, model_name = handle.recognize("page.jpg", ordered_lines)   # or None
    # handle.recognize(..., with_model_type=True) also returns the nafrec.model_types.ModelType
"""
import os
from multiprocessing import get_context

import psutil

from .main import (make_args, validate_args, resolve_devices,
                   start_inference_workers, stop_inference_workers,
                   init_cpu_worker, predict_single_image,
                   detect_polygons, recognize_lines, _CPU_STATE)


def _predict(image_path):
    preds, xml_input, _msg = predict_single_image(image_path)
    return None if preds is None else (preds, xml_input)


class NAFRec:
    """
    Accepts the same options as the CLI (names without leading dashes), e.g.
    NAFRec(detection_model_path=..., device="cpu", cpu_processes=4).
    `input_folder` is not needed. page_xml / alto_xml / xml_folder
    are carried into the returned xml_input.

    cpu_pool=False starts only the model workers (no CPU pre/postprocessing pool), for callers
    that bring their own CPU processes; see detect() / recognize().
    """

    def __init__(self, *, cpu_pool=True, **options):
        options.setdefault("input_folder", None)  # unused in library mode
        self.args = make_args(**options)
        validate_args(self.args)
        self.devices = resolve_devices(self.args)

        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        if self.args.cpu_processes:
            cpu_processes = self.args.cpu_processes
        elif self.devices == ["cpu"]:
            cpu_processes = max(1, physical // 2)
        else:
            cpu_processes = max(1, min(physical, 6 * len(self.devices)))
        self.cpu_processes = cpu_processes  # also the sensible default size for a caller's own CPU pool

        ctx = get_context("spawn")  # required for CUDA
        self._manager = ctx.Manager()
        self._queues = {role: self._manager.JoinableQueue(maxsize=self.args.inference_in_flight_limit)
                        for role in ("det", "rec")}
        self._results = self._manager.dict()
        self._slots = self._manager.BoundedSemaphore(
            self.args.inference_in_flight_limit)

        self._workers = start_inference_workers(
            ctx, self.args, self.devices, self._queues, self._results)
        self._pool = ctx.Pool(
            processes=cpu_processes,
            # init_cpu_worker will set the correct queues in _CPU_STATE
            initializer=init_cpu_worker,
            initargs=(self.args, self._queues, self._results, self._slots),
        ) if cpu_pool else None
        self._closed = False
        self._owner = True

    def __getstate__(self):
        # Pickles as a handle to the running model workers (queues, results, slots), e.g. to
        # hand to the initializer of a caller's own pool. A handle can't submit() or close().
        return {k: getattr(self, k) for k in (
            "args", "devices", "cpu_processes", "_queues", "_results", "_slots")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._manager = self._pool = None
        self._workers = []
        self._closed = False
        self._owner = False

    def submit(self, image_path):
        """Queue an image; returns a multiprocessing AsyncResult whose value is
        (preds, xml_input), or None if nothing was detected/transcribed.
        Errors are raised on .get()."""
        if self._closed:
            raise RuntimeError("NAFRec is closed")
        if self._pool is None:
            raise RuntimeError(
                "NAFRec has no CPU pool (cpu_pool=False, or a handle in another process)")
        return self._pool.apply_async(_predict, (str(image_path),))

    def _wait(self, future):
        while True:
            try:
                return future.get(timeout=1)
            except Exception as e:
                if type(e).__name__ != "TimeoutError":
                    raise
                if not all(w.is_alive() for w in self._workers):
                    raise RuntimeError(
                        "A model worker process died (see its traceback above)")

    def process(self, image_path):
        """Process one image and block until done. Returns (preds, xml_input) or None."""
        return self._wait(self.submit(image_path))

    def _run(self, fn, *fn_args):
        """Run a CPU-side step in the CPU pool if there is one, else in the calling process
        (the model workers do the model part either way)."""
        if self._closed:
            raise RuntimeError("NAFRec is closed")
        if self._pool is not None:
            return self._wait(self._pool.apply_async(fn, fn_args))
        if not _CPU_STATE:
            init_cpu_worker(self.args, self._queues, self._results, self._slots)
        return fn(*fn_args)

    def detect(self, image_path):
        """Segment one image; blocks. Returns (line_polygons, line_confs, line_max_mins,
        region_polygons, region_confs, region_max_mins, image_shape) after the usual
        thresholding / merging / sliver filtering."""
        return self._run(detect_polygons, str(image_path))

    def recognize(self, image_path, ordered_lines, with_model_type=False):
        """Classify (when both recognizers are configured) and recognize caller-supplied lines;
        blocks. ordered_lines is a list of regions as made by nafrec.utils.order_regions_lines
        (each with 'lines', 'line_confs' and 'img_shape'). Returns (preds, model_name), with preds
        a list with one dict per region (its 'text_lines' holding polygon, text, text_conf, ...),
        or None if there were no lines. With with_model_type=True returns
        (preds, model_name, model_type) instead, model_type being the nafrec.model_types.ModelType
        of the recognizer(s) used (e.g. LATIN_PPOCRV6_AND_CYRILLIC_TROCR), for building an
        XmlInput / the XML outputs."""
        preds, model_name, model_type = self._run(
            recognize_lines, str(image_path), ordered_lines, True)
        if preds is None:
            return None
        return (preds, model_name, model_type) if with_model_type else (preds, model_name)

    def process_many(self, image_paths):
        """Process several images in parallel; returns results in input order (each (preds, xml_input) or None)."""
        futures = [self.submit(p) for p in image_paths]
        return [self._wait(f) for f in futures]

    def close(self):
        if self._closed or not self._owner:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        stop_inference_workers(self.args, self.devices,
                               self._workers, self._queues)
        self._manager.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
