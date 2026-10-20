# Using nafrec as a Python library

Besides the `nafrec` command, the pipeline can be used from Python through the `NAFRec` class. It starts the model workers and the CPU pre/postprocessing pool **once**, then accepts image processing requests until it is closed. This avoids reloading the models for every batch of images, and lets another application (such as [SfRD](https://github.com/Kansallisarkisto-AI/sfrd)) use nafrec's detection and recognition as building blocks.

The predictions should be identical to the CLI, because `NAFRec` runs the same code. The differences are that `NAFRec` itself does not write anything to disk, and that results are returned to the caller.

This library documentation (USAGE.md) is generated with Claude and not thoroughly checked. For reference, see [SfRD](https://github.com/Kansallisarkisto-AI/sfrd) for an example of a currently maintained application that uses the library. *Features that are not used by SfRD are not tested*.

- [Using nafrec as a Python library](#using-nafrec-as-a-python-library)
  - [Quick start](#quick-start)
  - [Constructing NAFRec](#constructing-nafrec)
    - [Attributes](#attributes)
    - [`cpu_pool=False`](#cpu_poolfalse)
  - [Processing images](#processing-images)
  - [Return values](#return-values)
    - [`preds`](#preds)
    - [`xml_input`](#xml_input)
  - [Writing XML and JSON](#writing-xml-and-json)
  - [Using your own CPU processes](#using-your-own-cpu-processes)
    - [`detect(image_path)`](#detectimage_path)
    - [`recognize(image_path, ordered_lines, with_model_type=False)`](#recognizeimage_path-ordered_lines-with_model_typefalse)
    - [Where the work runs](#where-the-work-runs)
    - [Handles](#handles)
  - [Accepting nafrec options on your own command line](#accepting-nafrec-options-on-your-own-command-line)
  - [Other functions](#other-functions)
  - [Lifecycle](#lifecycle)

## Quick start

```python
from nafrec import NAFRec
from nafrec.xml_output import save_xml
from nafrec.utils import save_json_output

if __name__ == "__main__":  # required: workers use the "spawn" start method
    with NAFRec(
        detection_model_path="/path/to/rfdetr/model.pth",
        recognition_model_path="/path/to/trocr/model/folder/",
        processor_path="/path/to/trocr/processor/folder/",
        device="cuda",
        gpu_ids=[0],
        alto_xml=1,
        xml_folder="/output/path",
    ) as rec:
        result = rec.process("page.jpg")      # blocks until the page is done
        if result is not None:                # None = nothing detected / transcribed
            preds, xml_input = result
            save_xml(preds, xml_input)        # writes /output/path/alto/page.xml
            save_json_output(preds, "page.jpg", rec.args)
```

The `if __name__ == "__main__":` guard is required in scripts. Model and CPU workers are started with the `spawn` method (needed for CUDA), which re-imports your main module in every worker.

## Constructing NAFRec

```python
NAFRec(*, cpu_pool=True, **options)
```

`options` accepts **every CLI option, using the same names without the leading dashes**. Defaults are the CLI defaults. See the [Arguments section of the README](README.md#arguments) for the full list and their meanings. For example:

```python
NAFRec(
    detection_model_path="det.pth",
    recognition_model_path="trocr",
    processor_path="proc",
    cyrillic_recognition_model_path="trocr_cyr",
    cyrillic_processor_path="proc_cyr",
    script_classification_model_path="script.pt",
    device="cuda",
    gpu_ids=[0, 1],
    workers_per_gpu=1,
    cpu_processes=12,
    fp16_rec=True,
    tile_size=768,
)
```

Flag options (`use_ppocr`, `split_workers`, `fp16_rec`, ...) take `True`/`False`. Multi-value options (`gpu_ids`, `ppocr_width_buckets`, `paddle_layout_mean`, ...) take lists. `page_xml`, `alto_xml` and `output_json` are integers (`0`/`1`).

Notes on the options in library mode:

- `input_folder` is not needed and is ignored.
- Unknown option names raise `TypeError`.
- The model options are validated immediately (`ValueError` if, for example, no recognition model is configured, or a required processor path is missing). The rules are the same as for the CLI, see [Choosing models](README.md#choosing-models).
- `page_xml`, `alto_xml` and `xml_folder` do not make `NAFRec` write anything. They are carried into the returned `xml_input` so that [`save_xml`](#writing-xml-and-json) knows what to write and where. If `xml_folder` is not set, `xml_input.xml_path` is the directory of the image.
- With `detector="none"` no detection model is loaded, and `detect()` / `process()` cannot be used (they raise an error). Use this when you always supply your own lines to `recognize()`.

### Attributes

| Attribute | Description |
|-----------|-------------|
| `rec.args` | The resolved options (an `argparse.Namespace`). Can be passed as `args` to `save_json_output`. |
| `rec.devices` | List of device strings of the model workers, e.g. `["cuda:0", "cuda:1"]` or `["cpu"]`. |
| `rec.cpu_processes` | Size of the CPU pool: `cpu_processes` if you set it, otherwise half of the physical cores on CPU, or min(physical cores, 6 × model workers) with GPUs. This is also a sensible default size for your own pool when `cpu_pool=False`. |

### `cpu_pool=False`

By default `NAFRec` starts a pool of CPU workers that do image loading, cropping, polygon post-processing and the orchestration of each page. With `cpu_pool=False` only the model workers are started, for callers that bring their own CPU processes. In that case `submit()`, `process()` and `process_many()` are not available (they raise `RuntimeError`); use [`detect()` and `recognize()`](#using-your-own-cpu-processes) instead.

## Processing images

All methods take an image path (`str` or `Path`).

| Method | Blocks | Description |
|--------|--------|-------------|
| `process(image_path)` | yes | Process one image and return `(preds, xml_input)`, or `None`. |
| `submit(image_path)` | no | Queue an image and return a `multiprocessing` `AsyncResult`. `.get()` gives `(preds, xml_input)` or `None`; errors are raised on `.get()`. |
| `process_many(image_paths)` | yes | Process several images in parallel. Returns the results in input order, each `(preds, xml_input)` or `None`. |

```python
with NAFRec(**options) as rec:
    # one at a time
    result = rec.process("a.jpg")

    # queue now, collect later
    futures = [rec.submit(p) for p in paths]
    results = [f.get() for f in futures]

    # or let nafrec do it
    results = rec.process_many(paths)
```

`process()` and `process_many()` also notice if a model worker process has died and raise `RuntimeError` rather than waiting forever (the worker's traceback is printed). When using `submit()` and `.get()` directly, pass a `timeout` to `.get()` if you want similar protection.

The pool processes images in parallel, so to use the hardware well, submit many images at once (`submit` / `process_many`) rather than calling `process()` in a loop.

## Return values

Each result is `(preds, xml_input)`, or `None` if no text lines were detected or none could be transcribed (the CLI reports these cases as `[info]` lines).

### `preds`

A list with one dictionary per text region, in reading order. Page-level values are repeated in every region:

| Key | Description |
|-----|-------------|
| `img_name` | Image file name |
| `height`, `width` | Image size in pixels |
| `page_conf_mean`, `page_conf_median`, `page_conf_25`, `page_conf_75` | Page-level statistics of the text recognition confidences |
| `n_long_rowtext` | Number of lines where the model generated over 100 characters |
| `language` | Predicted (majority) language of the recognized text (`"unknown"` if detection fails) |
| `region_conf` | Region detection confidence |
| `region_coords` | Region polygon, `[[x, y], ...]` |
| `region_name` | Region name |
| `text_lines` | List of line dictionaries, in reading order (below) |

Each entry of `text_lines`:

| Key | Description |
|-----|-------------|
| `polygon` | Line polygon, `[[x, y], ...]` |
| `text` | Recognized text |
| `conf` | Line detection confidence |
| `text_conf` | Text recognition confidence |
| `row_length` | Number of generated characters |
| `predicted_script` | `"latin"` or `"cyrillic"`. Only present when script classification was used (both a main and a cyrillic recognition model configured) |
| `script_pred_conf` | Confidence of the script prediction (same condition) |

These are the same values as in the `.json` files written by the CLI.

### `xml_input`

A `nafrec.main.XmlInput` (a pydantic model) that describes the page and the models used. It is what [`save_xml`](#writing-xml-and-json) takes:

| Field | Description |
|-------|-------------|
| `image_path` | Path of the processed image |
| `page_xml`, `alto_xml` | Which XML formats to write (0/1) |
| `xml_path` | Base folder for the XML output (`xml_folder`, or the image directory) |
| `region_segment_model_name`, `line_segment_model_name` | Names from `region_model_name` / `line_model_name` |
| `region_segment_model_type`, `line_segment_model_type` | `nafrec.model_types.ModelType` of the detector, inferred from `detector` and the Paddle model paths |
| `classification_model_name` | Name of the script classification model, or `None` if no classification was used |
| `text_recognition_model_name` | Name(s) of the recognition model(s) actually used on this page. If a page mixed both scripts, both names are joined with `" and "` |
| `text_recognition_model_type` | `ModelType` of the recognizer(s) actually used, e.g. `ModelType.TROCR`, `ModelType.PPOCRV6_MEDIUM_REC`, `ModelType.LATIN_PPOCRV6_AND_CYRILLIC_TROCR` |

## Writing XML and JSON

`NAFRec` never writes files. To get the same output as the CLI, use the output functions on each result:

```python
from nafrec.xml_output import save_xml
from nafrec.utils import save_json_output

preds, xml_input = result
save_xml(preds, xml_input)                      # ALTO and/or PAGE XML, as set in xml_input
save_json_output(preds, image_path, rec.args)   # <json_folder>/<image stem>.json
```

- `save_xml` writes `<xml_path>/alto/<image stem>.xml` and/or `<xml_path>/page/<image stem>.xml`, depending on `page_xml` and `alto_xml`.
- `save_json_output` writes to `json_folder` if set, otherwise to `<xml_folder>/json`, otherwise to `json/` next to the image. It uses the options object only for these two folder settings, and does not check `output_json`: whether to call it is up to you.

You can of course skip both and use `preds` directly (for example to store the text in a database).

## Using your own CPU processes

Applications that have their own multiprocessing (and their own idea of what happens between detection and recognition) can use nafrec for just the model work. This is how SfRD uses nafrec: it detects/aligns text lines from annotations, clips and filters them, and only then asks nafrec to recognize the resulting lines.

The pattern:

1. Create `NAFRec(cpu_pool=False, ...)` in the main process. It starts only the model workers.
2. Give the `NAFRec` object to your worker processes, typically as an argument to the pool initializer. When pickled, the object becomes a lightweight **handle** to the running model workers (their request queues, result store and in-flight limiter).
3. In the workers, call `handle.detect(...)` and `handle.recognize(...)`.
4. In the main process, call `rec.close()` when done (use `try`/`finally`).

```python
from multiprocessing import get_context

from nafrec import NAFRec, get_default_region, get_line_regions, order_regions_lines

HANDLE = None

def init_worker(handle):
    global HANDLE
    HANDLE = handle                    # a handle: can detect() and recognize(), not close()

def work(image_path):
    # 1. detection (blocks; the polygon post-processing runs in this process)
    (line_polygons, line_confs, line_max_mins,
     region_polygons, region_confs, region_max_mins, image_shape) = HANDLE.detect(image_path)

    # 2. assemble regions and order the lines (your own filtering can go here)
    line_preds = {"coords": line_polygons, "max_min": line_max_mins, "confs": line_confs}
    if region_polygons:
        region_preds = [
            {"coords": poly, "id": str(i), "max_min": mm, "name": "paragraph",
             "img_shape": image_shape, "conf": conf}
            for i, (poly, conf, mm) in enumerate(
                zip(region_polygons, region_confs, region_max_mins))
        ]
    else:
        region_preds = get_default_region(image_shape=image_shape)  # whole page as one region

    lines_connected = get_line_regions(lines=line_preds, regions=region_preds)
    ordered_lines = order_regions_lines(lines=lines_connected, regions=region_preds)

    # 3. recognition (script classification is done inside if both recognizers are configured)
    recognized = HANDLE.recognize(image_path, ordered_lines)
    if recognized is None:             # no lines
        return image_path, []
    preds, model_name = recognized
    return image_path, preds

if __name__ == "__main__":
    rec = NAFRec(cpu_pool=False, detection_model_path="det.pth",
                 recognition_model_path="trocr", processor_path="proc")
    ctx = get_context("spawn")
    try:
        with ctx.Pool(rec.cpu_processes, initializer=init_worker, initargs=(rec,)) as pool:
            for image_path, preds in pool.imap_unordered(work, image_paths):
                ...
    finally:
        rec.close()
```

### `detect(image_path)`

Runs segmentation on one image and blocks. The text lines and regions have already been through the usual thresholding, merging and sliver filtering (`confidence_threshold`, the `line_*` / `region_*` options, `sliver_threshold`). Returns a 7-tuple:

| Position | Name | Description |
|----------|------|-------------|
| 0 | `line_polygons` | List of line polygons |
| 1 | `line_confs` | Detection confidence of each line |
| 2 | `line_max_mins` | Bounding box `[x_min, y_min, x_max, y_max]` of each line |
| 3 | `region_polygons` | List of region polygons (may be empty, e.g. with `detector="paddle"` and no layout model) |
| 4 | `region_confs` | Detection confidence of each region |
| 5 | `region_max_mins` | Bounding box of each region |
| 6 | `image_shape` | `(height, width)` of the image |

The `line_preds` and `region_preds` dictionaries built in the example above are the input format of `get_line_regions` and `order_regions_lines`, which assign lines to regions and put both in reading order. If you do not want regions at all, `get_default_region(image_shape)` gives a single region covering the whole page.

### `recognize(image_path, ordered_lines, with_model_type=False)`

Classifies (when both a main and a cyrillic recognizer are configured) and recognizes the lines you supply, and blocks. `ordered_lines` is a list of regions in the format that `nafrec.utils.order_regions_lines` returns. Each region needs:

| Key | Description |
|-----|-------------|
| `lines` | List of line polygons (`[[x, y], ...]`) in the order you want them in the output. They are cropped from the image at these coordinates, so they may come from any source: nafrec's detector, your own annotations, etc. |
| `line_confs` | One confidence value per line (copied to `conf` in the output) |
| `img_shape` | `(height, width)` of the image |
| `region_coords`, `region_name`, `region_conf` | Copied to the output. Without them the output has `None` for these |

Other keys in the region dictionaries (such as the `line_labels` that SfRD adds) are ignored by `recognize()` and are not copied to the result. The result is in the same order as `ordered_lines`, so you can zip the two to get your own data back, as SfRD does.

Returns `(preds, model_name)`, or `None` if there were no lines at all. `preds` is in the format [described above](#preds): one entry per input region, **including regions without lines** (empty `text_lines`). `model_name` is the name of the recognition model(s) used.

With `with_model_type=True` the return value is `(preds, model_name, model_type)`, where `model_type` is the `nafrec.model_types.ModelType` of the recognizer(s) actually used on the page. This is what you need, together with the other information you know, to build an `XmlInput` and write XML for lines that you supplied yourself:

```python
from nafrec.main import XmlInput
from nafrec.model_types import ModelType
from nafrec.xml_output import save_xml

recognized = HANDLE.recognize(image_path, ordered_lines, with_model_type=True)
if recognized:
    preds, model_name, model_type = recognized
    xml_input = XmlInput(
        image_path=image_path,
        page_xml=0, alto_xml=1,
        xml_path="/output/path",
        region_segment_model_name="my-regions",
        region_segment_model_type=ModelType.CALLER_SUPPLIED_LINES,
        line_segment_model_name="my-lines",
        line_segment_model_type=ModelType.CALLER_SUPPLIED_LINES,
        classification_model_name=None,         # or the classifier's name if one was used
        text_recognition_model_name=model_name,
        text_recognition_model_type=model_type,
    )
    save_xml(preds, xml_input)
```

`ModelType.CALLER_SUPPLIED_LINES` makes the XML metadata state that the regions or lines were supplied by the calling application, not detected by nafrec.

### Where the work runs

For an object created with `cpu_pool=True`, `detect()` and `recognize()` run their CPU-side work (image loading, cropping, polygon post-processing) in the CPU pool. For an object created with `cpu_pool=False`, and for handles in other processes, that work runs in the calling process. The model work always runs in the model workers.

### Handles

A pickled copy of a `NAFRec` object (a handle) can call `detect()` and `recognize()`, and can read `args`, `devices` and `cpu_processes`. It cannot `submit()`, `process()` or `close()`, and, unlike the owning object, it cannot notice that a model worker has died (a call then waits forever). Only the process that created `NAFRec` owns the workers and should close them.

Because the model workers limit the number of requests in flight (`inference_in_flight_limit`), many handles can safely share one set of model workers.

## Accepting nafrec options on your own command line

If you write a command line tool that wraps nafrec (as SfRD does), you can let your own parser handle your options and pass everything it does not recognize on to nafrec, so that all nafrec options are available without redefining them:

```python
import argparse
from nafrec import NAFRec
from nafrec.main import parse_args as parse_nafrec_args, make_args, validate_args

parser = argparse.ArgumentParser(allow_abbrev=False)   # an abbreviation must not swallow a nafrec option
parser.add_argument("--my_option")
args, nafrec_argv = parser.parse_known_args()

# library=True: --input_folder is not required
nafrec_options = vars(parse_nafrec_args(nafrec_argv, library=True))

# optional: report mistakes early, before any long-running step of your own
try:
    validate_args(make_args(**nafrec_options))
except ValueError as e:
    parser.error(str(e))

rec = NAFRec(cpu_pool=False, **nafrec_options)
```

SfRD additionally sets `nafrec_options["detector"] = "none"` when it will not use nafrec's segmentation at all (its `--disregard_regions` and `--disregard_lines` options together), so that no detection model is needed or loaded.

`make_args(**options)` builds the same options object as the CLI parser with the defaults filled in, and raises `TypeError` for unknown names. `validate_args(args)` raises `ValueError` for invalid model combinations. `NAFRec` calls both itself on construction, so validating beforehand only matters for failing early.

## Other functions

Importable from `nafrec`:

| Function | Description |
|----------|-------------|
| `run(input_folder, **options)` | The CLI as a function. Processes all images in a folder and **writes** the XML/JSON files. Returns `(n_ok, n_failed)`. Needs the `if __name__ == "__main__":` guard. |
| `make_args(**options)` | Options object with the CLI defaults, see above. |
| `get_default_region(image_shape)` | A single region covering the whole page, for when there are no region predictions. `image_shape` is `(height, width)`. |
| `get_line_regions(lines, regions)` | Assign each line to the region it overlaps most (or to the closest one). |
| `order_regions_lines(lines, regions)` | Order the lines inside each region and the regions on the page. Drops regions without lines. |
| `get_iou(poly1, poly2)` | Intersection over union of two polygons. |
| `load_rfdetr_model`, `predict_polygons` | Load an RF-DETR model and run segmentation in the calling process, without any workers. |
| `load_paddle_layout_model`, `PaddleLayoutDetector` | The corresponding Paddle detector. |
| `load_trocr_model`, `get_text_predictions`, `classify_and_recognize` | Lower-level recognition functions used by the workers. |

Most applications only need `NAFRec`. The lower-level functions load and call a single model in the current process, and leave device placement, batching and parallelism to you. Some of the lower-level functions are left over from an older implementation and their direct usage may not be supported in the latest nafrec version.

## Lifecycle

- **Always close.** Use `with NAFRec(...) as rec:` or call `rec.close()` in a `finally` block. Closing waits for the CPU pool, stops the model workers and shuts down the manager process. If you do not, the worker processes (and the VRAM they hold) can keep your script from exiting.
- **Startup time.** The constructor returns as soon as the worker processes have been started; the models are loaded inside the workers. Requests wait in the queue until a worker is ready, so the first request can take noticeably longer, especially with `enable_tensorrt` (engines are built on the first run and cached in `./trt_enginecache`).
- **`spawn` start method.** Everything passed to the workers must be picklable, and scripts need the `__main__` guard. Functions that you give to your own pool must be defined at module level.
- **Memory.** Each model worker holds a full copy of all configured models. `workers_per_gpu > 1` multiplies this; `split_workers` does not.
- **Back-pressure.** At most `inference_in_flight_limit` requests are queued at once; further requests wait for a free slot. Requests carry cropped line images, so a large limit uses a lot of memory.
- **`None` results.** A `None` from `process()` is not an error: it means that nothing was detected or nothing could be transcribed in that image. Actual errors (unreadable image, failed model call, etc.) are raised as exceptions from `process()`, `.get()` or `process_many()`. In a batch, use `submit()` and catch the exception from each `.get()` if you want the rest of the batch to continue.
- **Using a closed object** raises `RuntimeError("NAFRec is closed")`.