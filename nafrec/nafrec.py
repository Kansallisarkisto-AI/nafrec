"""
Library wrapper around the existing NAFRec pipeline in main.py.

Starts the model (inference) workers and the CPU pre/postprocessing pool once,
then accepts image processing requests until closed. All the actual work is done
by the existing functions in main.py (inference_worker_loop, init_cpu_worker,
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
"""
import os
from multiprocessing import get_context

import psutil

from .main import (make_args, validate_args, resolve_devices,
                   inference_worker_loop, init_cpu_worker, predict_single_image)


def _predict(image_path):
    preds, xml_input, _msg = predict_single_image(image_path)
    return None if preds is None else (preds, xml_input)


class NAFRec:
    """
    Accepts the same options as the CLI (names without leading dashes), e.g.
    NAFRec(detection_model_path=..., device="cpu", cpu_processes=4).
    `input_folder` is not needed. page_xml / alto_xml / xml_folder
    are carried into the returned xml_input.
    """

    def __init__(self, **options):
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

        ctx = get_context("spawn")  # required for CUDA
        self._manager = ctx.Manager()
        self._queue = self._manager.JoinableQueue(maxsize=self.args.inference_in_flight_limit)
        self._results = self._manager.dict()
        self._slots = self._manager.BoundedSemaphore(self.args.inference_in_flight_limit)

        self._workers = [
            ctx.Process(target=inference_worker_loop,
                        args=(self.args, device, self._queue, self._results))
            for device in self.devices
        ]
        for w in self._workers:
            w.start()
        self._pool = ctx.Pool(
            processes=cpu_processes,
            initializer=init_cpu_worker,  # init_cpu_worker will set the correct queues in _CPU_STATE
            initargs=(self.args, self._queue, self._results, self._slots),
        )
        self._closed = False

    def submit(self, image_path):
        """Queue an image; returns a multiprocessing AsyncResult whose value is
        (preds, xml_input), or None if nothing was detected/transcribed.
        Errors are raised on .get()."""
        if self._closed:
            raise RuntimeError("NAFRec is closed")
        return self._pool.apply_async(_predict, (str(image_path),))

    def _wait(self, future):
        while True:
            try:
                return future.get(timeout=1)
            except Exception as e:
                if type(e).__name__ != "TimeoutError":
                    raise
                if not all(w.is_alive() for w in self._workers):
                    raise RuntimeError("A model worker process died (see its traceback above)")

    def process(self, image_path):
        """Process one image and block until done. Returns (preds, xml_input) or None."""
        return self._wait(self.submit(image_path))

    def process_many(self, image_paths):
        """Process several images in parallel; returns results in input order (each (preds, xml_input) or None)."""
        futures = [self.submit(p) for p in image_paths]
        return [self._wait(f) for f in futures]

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.close()
        self._pool.join()
        for _ in self._workers:
            self._queue.put(None)
        self._queue.join()
        for w in self._workers:
            w.join()
        self._manager.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()