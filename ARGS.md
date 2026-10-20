# Arguments

The options below are grouped by purpose. Paths marked "conditionally required" depend on which models you configure (see [Choosing models](README.md#choosing-models)).

#### Input, models and output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_folder` | str | **required** | Path to folder containing input images |
| `--detection_model_path` | str | `/path/to/detection_model.pth` | Path to the RF-DETR detection model file (used with `--detector rfdetr`) |
| `--recognition_model_path` | str | None | Path to the (latin-script / main) TrOCR recognition model folder. Optional if only the cyrillic model is used, or if `--use_ppocr` is set |
| `--cyrillic_recognition_model_path` | str | None | Path to the cyrillic TrOCR recognition model folder. Optional |
| `--processor_path` | str | None | Path to the TrOCR processor folder. Required with `--recognition_model_path` (not needed with `--use_ppocr`) |
| `--cyrillic_processor_path` | str | None | Path to the cyrillic TrOCR processor folder. Required with `--cyrillic_recognition_model_path` |
| `--script_classification_model_path` | str | None | Path to the script classification model file (`.pt`/`.pth`). Required when both the main and the cyrillic recognition models are configured |
| `--region_model_name` | str | `Default` | Region detection model name (recorded in the XML metadata) |
| `--line_model_name` | str | `Default` | Line detection model name (recorded in the XML metadata) |
| `--script_classification_model_name` | str | `naf_script_classifier_202610` | Script classification model name (recorded in the XML metadata) |
| `--text_rec_model_name` | str | `Default` | Latin text recognition model name (recorded in the XML metadata) |
| `--cyrillic_text_rec_model_name` | str | `Default` | Cyrillic text recognition model name (recorded in the XML metadata) |
| `--trocr_batch_size` | int | 8 | Batch size for TrOCR text recognition |
| `--classifier_batch_size` | int | 32 | Batch size for script type classification |
| `--classifier_default_label` | str | latin | Default label used by script type classifier in cases where all line classifications fail |
| `--page_xml` | int | 0 | Whether to save output as PAGE XML (0 = no, 1 = yes) |
| `--alto_xml` | int | 1 | Whether to save output as ALTO XML (0 = no, 1 = yes) |
| `--output_json` | int | 1 | Whether to save output as .json file (0 = no, 1 = yes) |
| `--xml_folder` | str | None | Custom path for XML output. If None, saves to `<image folder>/alto` or `<image folder>/page` |
| `--json_folder` | str | None | Custom path for .json output. If None, saves to `<xml_folder>/json` (or `<image folder>/json` if `--xml_folder` is not set) |

#### Detection thresholds and post-processing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--confidence_threshold` | float | 0.15 | Detection confidence threshold for filtering detections (RF-DETR only; the Paddle detector uses the `--paddle_*` thresholds) |
| `--line_percentage_threshold` | float | 7e-05 | Threshold value for filtering out small line polygons, as a fraction of the image area (applies to both `rfdetr` and `paddle` detectors) |
| `--region_percentage_threshold` | float | 7e-05 | Threshold value for filtering out small region polygons, as a fraction of the image area (applies to both detectors) |
| `--line_iou` | float | 0.3 | Threshold value for merging lines based on intersection over union (IoU) (applies to both detectors) |
| `--region_iou` | float | 0.3 | Threshold value for merging regions based on intersection over union (IoU) (applies to both detectors) |
| `--line_overlap_threshold` | float | 0.5 | Threshold value for merging lines based on overlapping area (applies to both detectors) |
| `--region_overlap_threshold` | float | 0.5 | Threshold value for merging regions based on overlapping area (applies to both detectors) |
| `--sliver_threshold` | float | 0.01 | Given as a proportion of the smaller dimension of the image, so 0.01 would be 1% of the shorter side. Text lines thinner than this threshold will be discarded. |

#### Tiled detection (SAHI)

Slicing Aided Hyper Inference runs the RF-DETR detector on overlapping tiles of the page instead of on the whole downscaled image, which helps with large images and small text. It is disabled by default and only applies to the RF-DETR detector.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tile_size` | int | 0 | Tile size in pixels. 768 is usually a good value for RF-DETR segmentation models. 0 disables tiling |
| `--tile_overlap` | int | 128 | Tile overlap in pixels |
| `--tiles_across` | int | 2 | The number of tiles across the shorter dimension of the image. The size the image is resized to before tiling is `tile_size * tiles_across - tile_overlap` (768 without tiling) |
| `--tile_iou_threshold` | float | 0.85 | IoU threshold for suppressing overlapping detections when tiling is used. 0.85 for general material, 0.5 is sometimes better for tables |
| `--tile_batch_size` | int | 3 | Batch size for tiled detection. 6 is a reasonable value |

#### Devices, parallelism and precision

nafrec runs *model workers* (processes that hold the models and run inference on a CPU or GPU) and *CPU workers* (processes that do image loading, cropping, polygon post-processing and writing the output). Pages are processed in parallel by the CPU workers, which send detection and recognition requests to the model workers.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `cuda` | Device to use for inference: `cuda` or `cpu`. If `cuda` is requested but no CUDA device is available, nafrec warns and falls back to `cpu` |
| `--gpu_ids` | int (one or more) | None | GPU indices to use, as seen after `CUDA_VISIBLE_DEVICES` (e.g. `--gpu_ids 0 2`). Default: all visible GPUs. Ignored with `--device cpu` |
| `--workers_per_gpu` | int | 1 | Number of model worker processes per GPU. Each holds a full copy of all models in VRAM, so 2 takes twice the VRAM for a possible small speedup. `--split_workers` is generally a better option than 2 here |
| `--split_workers` | flag | off | Start two model worker processes per device, one for detection and one for recognition/classification, so that detection and recognition run in parallel on the same device. By default each worker does both. Takes about as much VRAM as `--workers_per_gpu 1` (each model is only loaded once per device) |
| `--cpu_processes` | int | 0 (auto) | Number of CPU pre/postprocessing workers. Default: half of the physical cores when running on CPU, otherwise min(physical cores, 6 × number of model workers) |
| `--inference_in_flight_limit` | int | 16 | Maximum number of model requests queued or in flight at once. Requests carry cropped line images, so keep this modest |
| `--fp16_rec` | flag | off | Run the recognition models (TrOCR, PP-OCR) in fp16. PP-OCR ONNX models are converted in memory at load time. Only applies on GPUs (CUDA/ROCm); ignored on CPU |
| `--fp16_det` | flag | off | Run the detection/segmentation models (RF-DETR, Paddle layout and text detection) in fp16. Paddle ONNX models are converted in memory at load time. Only applies on GPUs (CUDA/ROCm); ignored on CPU |

#### PP-OCRv6 recognition

PP-OCRv6 (ONNX) can be used instead of TrOCR for latin-script lines. The cyrillic model is always TrOCR.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_ppocr` | flag | off | Use a PP-OCRv6 ONNX recognition model instead of TrOCR for latin-script lines. Requires `--ppocr_model_path` (and then `--recognition_model_path` / `--processor_path` are not used) |
| `--ppocr_model_path` | str | None | Path to the PP-OCRv6 recognition ONNX model |
| `--ppocr_char_dict_path` | str | None | Path to the character dictionary the PP-OCRv6 model was trained with. If None, the default dictionary is used |
| `--ppocr_img_height` | int | 96 | PP-OCR input height. Must match training (e.g. 96 for a custom NAF model) |
| `--ppocr_img_width_min` | int | 320 | PP-OCR minimum input width |
| `--ppocr_img_width_max` | int | 3200 | PP-OCR maximum input width. Text lines that would be wider are squeezed to this width. Must be >= `--ppocr_img_width_min`. With TensorRT this is also the maximum width of the shape profile |
| `--ppocr_width_buckets` | int (one or more) | None | Pad each batch's input width up to the smallest of these bucket widths (e.g. `320 640 1280 2560`), so that the model sees only a few distinct input shapes. Batches wider than the largest bucket keep their natural width. Default: no bucketing |
| `--ppocr_batch_size` | int | 32 | Batch size for PP-OCR text recognition |
| `--ppocr_pad_batch` | str | `auto` | Pad every PP-OCR batch (including the last partial one and single-line retries) to `--ppocr_batch_size`, so that the model always sees the same batch shape. Choices: `auto`, `on`, `off`. MIGraphX compiles for one fixed shape and fails on others; `auto` turns padding on only when MIGraphX is the active execution provider |

#### TensorRT

TensorRT applies to the PP-OCR recognition model and to the Paddle layout and text detection models. TrOCR does not use TensorRT even in optimum-onnx mode. See also the [notes](README.md#notes) on TensorRT.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_tensorrt` | flag | off | Use the TensorRT execution provider if available. By default GPUs use CUDA/ROCm/MIGraphX. Engines are cached per model in `./trt_enginecache` (relative to the working directory), so the first run builds them and is slow |
| `--no_trt_dynamic_shapes` | flag | off | Don't pass min/opt/max shape profiles to the TensorRT provider for PP-OCR / Paddle models. TensorRT then rebuilds engines whenever a new input shape appears, which for the Paddle text detector means for nearly every page size |
| `--ppocr_trt_opt_width` | int | 1536 | PP-OCR with TensorRT: optimal input width for the shape profile |
| `--paddle_det_trt_opt_shape` | int int (H W) | None | Paddle text detection with TensorRT: optimal input size (height and width) for the shape profile, e.g. `1536 1152`. Default: a 3:4 portrait page at the `--paddle_det_limit_side_len` resize limit. The minimum and maximum sizes of the profile are derived from the detection resize settings automatically |

#### Detector selection and Paddle detection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detector` | str | `rfdetr` | Segmentation backend. `rfdetr`: RF-DETR model for both regions and lines (`--detection_model_path`). `paddle`: ONNX DB text detection model for lines (`--paddle_det_model_path`) plus an optional PP-DocLayout layout model for regions (`--paddle_layout_model_path`). `none`: no detection model is loaded (library use with caller-supplied lines only, see [USAGE.md](USAGE.md)) |
| `--paddle_det_model_path` | str | None | Paddle DB text detection model (ONNX, e.g. PP-OCRv6 det) giving the text lines. Required with `--detector paddle` |
| `--paddle_layout_model_path` | str | None | Optional Paddle layout model (ONNX, e.g. PP-DocLayout) giving the text regions. Without it, each page is treated as one full-page region |
| `--paddle_layout_threshold` | float | 0.3 | Minimum score for layout regions |
| `--paddle_layout_input_size` | int | 0 | Layout model input size for models with a dynamic input shape (0 = 800). Ignored for fixed-size models |
| `--paddle_layout_mean` | float float float | `0 0 0` | Layout model input normalization mean (RGB, after scaling to 0-1). PP-DocLayout-L/plus-L: `0 0 0` |
| `--paddle_layout_std` | float float float | `1 1 1` | Layout model input normalization standard deviation (RGB). PP-DocLayout-L/plus-L: `1 1 1`; PicoDet-based S/M models typically use ImageNet values `0.229 0.224 0.225` |
| `--paddle_layout_boxes_in_input_space` | flag | off | Set if the layout model returns boxes in resized-input pixels instead of original-image pixels |
| `--paddle_layout_labels_path` | str | None | Text file with one layout class name per line (class id = line number). Default: the built-in PP-DocLayout_plus-L class names |
| `--paddle_layout_ignore_classes` | str (zero or more) | none | Layout classes (ids, or names if `--paddle_layout_labels_path` is given) to drop as regions. Text lines inside them are dropped too (e.g. `image chart seal`) |
| `--paddle_det_limit_side_len` | int | 1536 | Text detection input resize limit in pixels (see `--paddle_det_limit_type`) |
| `--paddle_det_limit_type` | str | `max` | `max`: downscale so that the longer side is at most the limit. `min`: upscale so that the shorter side is at least the limit |
| `--paddle_det_max_side_limit` | int | 4000 | Hard cap for the longer side of the text detection input |
| `--paddle_det_thresh` | float | 0.3 | DB binarization threshold |
| `--paddle_det_box_thresh` | float | 0.6 | DB minimum box score (also used as the line confidence) |
| `--paddle_det_unclip_ratio` | float | 1.5 | DB box expansion ratio |