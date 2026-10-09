import os
import time

import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import itertools
import traceback
from queue import Empty
import cv2
import psutil
from multiprocessing import get_context

# Import PyTorch and prevent useless warning
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch.jit.script.*is deprecated.*",
    category=FutureWarning,
)
import torch

from .xml_output import save_xml
from .trocr import get_text_preds, get_text_lines, get_line_dicts, load_trocr_model
from .ppocr import load_ppocr_model, get_ppocr_preds, PPOCRRecognizer
from .seg_inference import load_rfdetr_model, predict_masks, masks_to_polygons
from .paddle_layout import load_paddle_layout_model
from .image_processing import load_with_torchvision, crop_lines
from .script_classifier import load_classification_model, classify_lines
from .utils import load_image_paths, get_default_region, get_line_regions, order_regions_lines, flatten_lines, process_text_predictions, get_page_stats, save_json_output

class XmlInput(BaseModel):
    image_path: str
    page_xml: int
    alto_xml: int
    xml_path: str
    region_segment_model_name: str
    line_segment_model_name: str
    classification_model_name: Optional[str] = None
    text_recognition_model_name: str

class ClassifierInput(BaseModel):
    line_images: list
    batch_size: int
    default_label: str

class OCRInput(BaseModel):
    line_images: list
    line_polygons: list
    line_confs: list
    batch_size: int

def build_parser():
    parser = argparse.ArgumentParser(description="Load and run inference model")
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use for inference"
    )
    parser.add_argument(
        "--detection_model_path",
        type=str,
        default='/path/to/detection_model.pth',
        help="Path to the detection model file"
    )
    parser.add_argument(
        "--recognition_model_path",
        type=str,
        default=None,
        help="Path to the (latin-script / main) recognition model folder. Optional if only the cyrillic model is used."
    )
    parser.add_argument(
        "--cyrillic_recognition_model_path",
        type=str,
        default=None,
        help="Path to the cyrillic recognition model folder. Optional. Script classification only runs when both the main and the cyrillic recognition models are configured; with only one of them, all lines go to it."
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Path to the processor folder (required with recognition_model_path, unless --use_ppocr)"
    )
    parser.add_argument(
        "--cyrillic_processor_path",
        type=str,
        default=None,
        help="Path to the cyrillic processor folder (required if cyrillic_recognition_model_path is given)"
    )
    parser.add_argument(
        "--script_classification_model_path",
        type=str,
        default=None,
        help="Path to the script classification model file (required when both the main and the cyrillic recognition models are configured)"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Image input folder"
    )
    parser.add_argument(
        "--region_model_name",
        type=str,
        default = 'rfdetr_text_seg_model_202510',
        help="region model name"
    )
    parser.add_argument(
        "--line_model_name",
        type=str,
        default='rfdetr_text_seg_model_202510',
        help="line  model name"
    )
    parser.add_argument(
        "--text_rec_model_name",
        type=str,
        default='202509_tf32',
        help="Text rec model name"
    )
    parser.add_argument(
        "--cyrillic_text_rec_model_name",
        type=str,
        default='cyrillic_large_202603',
        help="Text rec model name"
    )
    parser.add_argument(
        "--script_classification_model_name",
        type=str,
        default='script_classifier_202610',
        help="Text rec model name"
    )
    parser.add_argument(
        "--trocr_batch_size",
        type=int,
        default=8,
        help="Batch size for text_rec"
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=32,
        help="Batch size for script type classifier"
    )
    parser.add_argument(
        "--classifier_default_label",
        type=str,
        default="latin",
        help="Default label used by script type classifier in cases where all line classifications fail"
    )
    parser.add_argument(
        "--page_xml",
        type=int,
        default=0,
        help="Whether to save as page xml (0 = no, 1 = yes)"
    )
    parser.add_argument(
        "--alto_xml",
        type=int,
        default=1,
        help="Whether to save as alto xml (0 = no, 1 = yes)"
    )
    parser.add_argument(
        "--output_json",
        type=int,
        default=1,
        help="Whether to save output as .json file (0 = no, 1 = yes)"
    )
    parser.add_argument(
        "--xml_folder",
        type=str,
        default=None,
        help="Where to save xmls. If None, saves to img_folder under alto or page folders"
    )
    parser.add_argument(
        "--json_folder",
        type=str,
        default=None,
        help="Where to save json files. If None, saves to xml_folder."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--line_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small line polygons"
    )
    parser.add_argument(
        "--region_percentage_threshold",
        type=float,
        default=7e-05,
        help="Threshold value for filtering out small region polygons"
    )
    parser.add_argument(
        "--line_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_iou",
        type=float,
        default=0.3,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--line_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping lines"
    )
    parser.add_argument(
        "--region_overlap_threshold",
        type=float,
        default=0.5,
        help="Threshold value for merging overlapping regions"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="Tile size for Slicing Aided Hyper Inference (SAHI), in pixels. 768 is usually a good value for RF-DETR segmentation models. 0 to disable."
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=128,
        help="Tile overlap for SAHI, in pixels."
    )
    parser.add_argument(
        "--tiles_across",
        type=int,
        default=2,
        help="The number of SAHI tiles across the shorter dimension of the image."
    )
    parser.add_argument(
        "--tile_iou_threshold",
        type=float,
        default=0.85,
        help="IoU threshold for suppressing overlapping detections _when SAHI is used_. 0.85 for general material, 0.5 is sometimes better for tables."
    )
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=6,
        help="Batch size for SAHI. 6 is a reasonable value."
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU indices to use (as seen after CUDA_VISIBLE_DEVICES). Default: all visible GPUs. Ignored with --device cpu."
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of model worker processes per GPU (each holds a full copy of all models in VRAM)"
    )
    parser.add_argument(
        "--cpu_processes",
        type=int,
        default=0,
        help="Number of CPU pre/postprocessing workers. Default: min(physical cores, 6 * number of model workers)."
    )
    parser.add_argument(
        "--inference_in_flight_limit",
        type=int,
        default=16,
        help="Maximum number of GPU requests queued/in flight at once. Requests carry cropped line images, so keep this modest."
    )

    parser.add_argument(
        "--recognition_max_wait_ms",
        type=float,
        default=200.0,
        help="Work stealing for text recognition: lines from queued pages are pooled so that complete batches can be formed. A partial batch is only run when its oldest line has waited this long (ms). 0 = never wait."
    )

    # PP-OCRv6 args
    parser.add_argument(
        "--use_ppocr",
        action="store_true",
        help="Use a PP-OCRv6 ONNX recognition model instead of TrOCR for latin-script lines"
    )
    parser.add_argument(
        "--ppocr_model_path",
        type=str,
        default=None,
        help="Path to the PP-OCRv6 recognition ONNX model"
    )
    parser.add_argument(
        "--ppocr_char_dict_path",
        type=str,
        default=None,
        help="Path to the character dictionary used to train the PP-OCRv6 model, or None to use the default dictionary."
    )
    parser.add_argument(
        "--ppocr_img_height",
        type=int,
        default=96,
        help="PP-OCR input height, must match training (e.g. 96 for a custom NAF model)"
    )
    parser.add_argument(
        "--ppocr_img_width_min",
        type=int,
        default=1536,
        help="PP-OCR minimum input width (scalable)"
    )
    parser.add_argument(
        "--ppocr_batch_size",
        type=int,
        default=32,
        help="Batch size for PP-OCR text recognition"
    )

    # Detector selection / Paddle layout + text detection args
    parser.add_argument(
        "--detector",
        type=str,
        choices=["rfdetr", "paddle"],
        default="rfdetr",
        help="Segmentation backend: 'rfdetr' (--detection_model_path) or 'paddle' (ONNX DB text detection model for lines, plus an optional PP-DocLayout layout model for regions)"
    )
    parser.add_argument("--paddle_layout_model_path", type=str, default=None,
                        help="Optional Paddle layout model (ONNX, e.g. PP-DocLayout) giving the text regions. Without it each page is one full-page region.")
    parser.add_argument("--paddle_det_model_path", type=str, default=None,
                        help="Paddle DB text detection model (ONNX, e.g. PP-OCRv6 det) giving the text lines")
    parser.add_argument("--paddle_layout_threshold", type=float, default=0.3,
                        help="Minimum score for layout regions")
    parser.add_argument("--paddle_layout_input_size", type=int, default=0,
                        help="Layout model input size for models with dynamic input shape (0 = 800). Ignored for fixed-size models.")
    parser.add_argument("--paddle_layout_mean", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        help="Layout model input normalization mean (RGB, after scaling to 0-1). PP-DocLayout-L/plus-L: 0 0 0")
    parser.add_argument("--paddle_layout_std", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        help="Layout model input normalization std (RGB). PP-DocLayout-L/plus-L: 1 1 1; PicoDet-based S/M models typically use ImageNet 0.229 0.224 0.225")
    parser.add_argument("--paddle_layout_boxes_in_input_space", action="store_true",
                        help="Set if the layout model returns boxes in resized-input pixels instead of original-image pixels")
    parser.add_argument("--paddle_layout_labels_path", type=str, default=None,
                        help="Text file with one layout class name per line (class id = line number). Default: the built-in PP-DocLayout_plus-L class names")
    parser.add_argument("--paddle_layout_ignore_classes", type=str, nargs="*", default=[],
                        help="Layout classes (ids, or names if --paddle_layout_labels_path is given) to drop as regions; text lines inside them are dropped too (e.g. image chart seal)")
    parser.add_argument("--paddle_det_limit_side_len", type=int, default=1536,
                        help="Text detection input resize limit in pixels (see --paddle_det_limit_type)")
    parser.add_argument("--paddle_det_limit_type", type=str, choices=["max", "min"], default="max",
                        help="'max': downscale so the longer side is at most the limit; 'min': upscale so the shorter side is at least the limit")
    parser.add_argument("--paddle_det_max_side_limit", type=int, default=4000,
                        help="Hard cap for the longer side of the text detection input")
    parser.add_argument("--paddle_det_thresh", type=float, default=0.3,
                        help="DB binarization threshold")
    parser.add_argument("--paddle_det_box_thresh", type=float, default=0.6,
                        help="DB minimum box score (also used as the line confidence)")
    parser.add_argument("--paddle_det_unclip_ratio", type=float, default=1.5,
                        help="DB box expansion ratio")

    return parser

def parse_args(argv=None):
    return build_parser().parse_args(argv)

def make_args(**overrides):
    """
    Build an argument namespace with all CLI defaults, overridden by keyword
    arguments (names are the CLI option names without leading dashes), e.g.
    make_args(input_folder="imgs", gpu_ids=[0, 2], cpu_processes=12).
    """
    defaults = {a.dest: a.default for a in build_parser()._actions if a.dest != "help"}
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown option(s): {sorted(unknown)}")
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

def run(input_folder, **options):
    """
    Library entry point, equivalent to the CLI. Returns (n_ok, n_failed).

    Parallelism options: device ("cuda"/"cpu"), gpu_ids (list[int] or None = all
    visible), workers_per_gpu, cpu_processes (0 = auto), inference_in_flight_limit.
    All other CLI options (detection_model_path, tile_size, ...) work the same way.

    Uses the "spawn" start method, so call this from under an
    `if __name__ == "__main__":` guard in scripts.
    """
    return main(make_args(input_folder=input_folder, **options))

def split_by_label(classification_results):
    """
    Single pass over classification_results, splitting into index lists
    per predicted label.
    """
    indices_by_label = {}
    for i, result in enumerate(classification_results):
        indices_by_label.setdefault(result["predicted_label"], []).append(i)
    return indices_by_label

def get_text_predictions(
    classification_results,
    cropped_lines,
    line_polygons,
    img_line_confs,
    model_config,
    args
):
    """
    Routes each classified line to its matching HTR model and returns
    results back in original line order.

    model_config: {predicted_label: (recognition_model, processor)}

    Returns (all_text_predictions, model_name):
      all_text_predictions -- list in the same order as cropped_lines,
        one entry per input line, always (see the missing-results check
        below).
      model_name -- the HTR model name(s) actually used on this page,
        for recording in the output XML/json (e.g. both names joined if
        a page mixed scripts).

    """
    n_lines = len(classification_results)
    indices_by_label = split_by_label(classification_results)
 
    all_text_predictions = [None] * n_lines
    model_names = {
        "cyrillic": args.cyrillic_text_rec_model_name, 
        "latin": args.text_rec_model_name
    }
    used_models = []
 
    for label, indices in indices_by_label.items():
        if not indices:
            continue
        if label not in model_config:
            raise ValueError(
                f"No HTR model configured for predicted_label={label} "
                f"({len(indices)} line(s) affected)"
            )
 
        model, processor = model_config[label]
 
        subset_lines = [cropped_lines[i] for i in indices]
        subset_polygons = [line_polygons[i] for i in indices]
        subset_confs = [img_line_confs[i] for i in indices]

        # PP-OCRv6 or not
        use_ppocr = isinstance(model, PPOCRRecognizer)

        payload = OCRInput(
            line_images=subset_lines,
            line_polygons=subset_polygons,
            line_confs=subset_confs,
            batch_size=args.ppocr_batch_size if use_ppocr else args.trocr_batch_size,
        )

        if use_ppocr:
            subset_predictions = get_ppocr_preds(payload, model)
        else:
            subset_predictions = get_text_preds(payload, model, processor)

        used_models.append(model_names[label])

        if len(subset_predictions) != len(indices):
            raise ValueError(
                f"{label} HTR model returned {len(subset_predictions)} result(s) for "
                f"{len(indices)} input line(s) -- results can't be safely matched back to "
                f"their original positions."
            )
 
        # Place each result back to its original line index.
        for orig_idx, prediction in zip(indices, subset_predictions):
            classification_pred = classification_results[orig_idx]
            if classification_pred.get("classified", True):
                prediction["predicted_script"] = classification_pred["predicted_label"]
                prediction["script_pred_conf"] = classification_pred["confidence"]
            all_text_predictions[orig_idx] = prediction
 
    missing = [i for i, pred in enumerate(all_text_predictions) if pred is None]
    if missing:
        raise ValueError(
            f"{len(missing)} line(s) never received a transcription result "
            f"(indices: {missing[:10]}{'...' if len(missing) > 10 else ''}) -- a "
            f"predicted_label wasn't covered by model_config, or a result came back None."
        )

    # In case two models were used for text recognition, both names are combined
    model_name = " and ".join(used_models)

    return all_text_predictions, model_name

def cyrillic_enabled(args):
    """Is the cyrillic recognition model configured?"""
    return bool(args.cyrillic_recognition_model_path)

def latin_enabled(args):
    """Is the main (latin-script) recognition model configured? (TrOCR, or PP-OCR with --use_ppocr)"""
    return bool(args.ppocr_model_path if args.use_ppocr else args.recognition_model_path)

def classification_enabled(args):
    """Script classification is only meaningful (and only used) when there are two models to choose between."""
    return latin_enabled(args) and cyrillic_enabled(args)

def validate_args(args):
    if args.detector == "paddle":
        if not args.paddle_det_model_path:
            raise ValueError("detector='paddle' requires paddle_det_model_path (text line detection ONNX model)")
        if not args.paddle_layout_model_path:
            print("[info] no paddle layout model: each page is treated as a single full-page region")
    latin, cyr = latin_enabled(args), cyrillic_enabled(args)
    if not (latin or cyr):
        raise ValueError("No recognition model configured: set recognition_model_path "
                         "(or ppocr_model_path with use_ppocr) and/or cyrillic_recognition_model_path")
    required = []
    if latin and not args.use_ppocr:
        required.append("processor_path")
    if cyr:
        required.append("cyrillic_processor_path")
    if latin and cyr:
        required.append("script_classification_model_path")
    missing = [n for n in required if not getattr(args, n)]
    if missing:
        raise ValueError(f"Missing required option(s) for the configured models: {missing}")
    if not (latin and cyr):
        print(f"[info] only the {'main' if latin else 'cyrillic'} recognition model is configured: "
              "script classification disabled, all lines go to it")

def classify_or_skip(cropped_lines, classification_model, model_config, args):
    """
    Script classification, or -- when classification_model is None -- trivial results
    routing every line to the single configured recognizer (model_config must then have
    exactly one entry). Such results carry classified=False so no
    predicted_script/script_pred_conf fields are written.
    """
    if classification_model is None:
        if len(model_config) != 1:
            raise ValueError("classification_model is None but model_config has "
                             f"{len(model_config)} recognizers; need exactly one")
        label = next(iter(model_config))
        return [{"predicted_label": label, "confidence": None, "fallback": False,
                 "error": None, "classified": False} for _ in cropped_lines]
    classifier_payload = ClassifierInput(
        line_images=cropped_lines,
        batch_size=args.classifier_batch_size,
        default_label=args.classifier_default_label,
    )
    return classify_lines(classifier_payload, classification_model)

def build_model_config(recognition_model=None, processor=None, cyrillic_recognition_model=None, cyrillic_processor=None):
    """{label: (model, processor)} for the models that are not None (at least one required)."""
    config = {}
    if recognition_model is not None:
        config["latin"] = (recognition_model, processor)
    if cyrillic_recognition_model is not None:
        config["cyrillic"] = (cyrillic_recognition_model, cyrillic_processor)
    if not config:
        raise ValueError("At least one recognition model is required")
    return config

def classify_and_recognize(
    image_path, 
    ordered_lines, 
    classification_model, 
    recognition_model, 
    cyrillic_recognition_model, 
    processor, 
    cyrillic_processor,
    args
):
    """
    Classify detected text lines based on their script type (latin / cyrillic)
    and forward each line to correct text recognition model.
 
    Any of recognition_model / cyrillic_recognition_model / classification_model may be None
    (see build_model_config / classify_or_skip).

    Returns (preds, trocr_model_name), or (None, None) if there was
    nothing to transcribe.
    """
    # Load image file
    image = load_with_torchvision(image_path)
    # Merge text lines from regions into flat lists
    line_polygons, img_line_confs, n_lines = flatten_lines(ordered_lines)
    # Get cropped line images
    cropped_lines = crop_lines(line_polygons, image)
    # Get classification results
    # Recognizers that are None are simply unused. With exactly one recognizer,
    # pass classification_model=None: all lines go to it without classification.
    model_config = build_model_config(recognition_model, processor, cyrillic_recognition_model, cyrillic_processor)
    classification_results = classify_or_skip(cropped_lines, classification_model, model_config, args)
    # Get text predictions
    text_predictions, trocr_model_name = get_text_predictions(
        classification_results,
        cropped_lines,
        line_polygons,
        img_line_confs,
        model_config, 
        args
    )
    if text_predictions:
        # Add page level stats to line level prediction results
        height, width = ordered_lines[0]['img_shape']
        image_name = Path(image_path).name
        lines_dict = get_page_stats(text_predictions, image_name, height, width)
        # Combine text region predictions with output data
        preds = process_text_predictions(lines_dict, ordered_lines, n_lines)
        return preds, trocr_model_name
    else:
        # No lines to transcribe (e.g. detection found regions but no
        # actual lines within them)
        return None, None

def load_latin_recognizer(args, device):
    """Returns (recognition_model, processor). processor is None for PP-OCR."""
    if args.use_ppocr:
        model = load_ppocr_model(
            args.ppocr_model_path,
            args.ppocr_char_dict_path,
            device=device,
            img_height=args.ppocr_img_height,
            img_width_min=args.ppocr_img_width_min,
        )
        return model, None
    return load_trocr_model(args.recognition_model_path, args.processor_path, device)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_devices(args):
    """Return the list of device strings for model workers, one per worker process."""
    if args.device != "cuda" or not torch.cuda.is_available():
        if args.device == "cuda":
            print("[warn] requested cuda but no CUDA device is available -- falling back to cpu")
        return ["cpu"]

    n = torch.cuda.device_count()
    ids = args.gpu_ids if args.gpu_ids else list(range(n))
    bad = [i for i in ids if i < 0 or i >= n]
    if bad:
        raise ValueError(f"Invalid --gpu_ids {bad}: {n} GPU(s) visible")

    return [f"cuda:{i}" for i in ids for _ in range(args.workers_per_gpu)]


def segmentation_max_size(args):
    return args.tile_size * args.tiles_across - args.tile_overlap if args.tile_size else 768


# ---------------------------------------------------------------------------
# GPU side
# ---------------------------------------------------------------------------

def load_detection_model(args, device):
    """Load the configured segmentation backend onto `device`."""
    if args.detector == "paddle":
        return load_paddle_layout_model(
            args.paddle_layout_model_path,
            args.paddle_det_model_path,
            device,
            layout_threshold=args.paddle_layout_threshold,
            layout_input_size=args.paddle_layout_input_size,
            layout_mean=tuple(args.paddle_layout_mean),
            layout_std=tuple(args.paddle_layout_std),
            layout_boxes_in_input_space=args.paddle_layout_boxes_in_input_space,
            layout_labels_path=args.paddle_layout_labels_path,
            layout_ignore_classes=args.paddle_layout_ignore_classes,
            det_limit_side_len=args.paddle_det_limit_side_len,
            det_limit_type=args.paddle_det_limit_type,
            det_max_side_limit=args.paddle_det_max_side_limit,
            det_thresh=args.paddle_det_thresh,
            det_box_thresh=args.paddle_det_box_thresh,
            det_unclip_ratio=args.paddle_det_unclip_ratio,
        )
    return load_rfdetr_model(
        args.detection_model_path,
        device=device,
        batch_size=args.tile_batch_size if args.tile_size else 1,
    )

def run_detection(detection_model, image_path, args):
    """
    Model-worker part of segmentation.

    rfdetr: returns the raw masks (see seg_inference.predict_masks); the polygon
    post-processing happens in the CPU worker (see postprocess_detection).
    paddle: the backend already returns the final 7-tuple (see
    seg_inference.masks_to_polygons), so there is nothing left to post-process.
    """
    if args.detector == "paddle":
        return detection_model.predict_polygons(image_path)
    return predict_masks(
        detection_model,
        image_path,
        max_size=segmentation_max_size(args),
        confidence_threshold=args.confidence_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_iou_threshold=args.tile_iou_threshold,
        tile_batch_size=args.tile_batch_size,
    )

def postprocess_detection(detection_result, args):
    """CPU-worker part of segmentation: raw masks -> merged line/region polygons (the 7-tuple)."""
    if args.detector == "paddle":
        return detection_result
    line_mask, line_confs, region_mask, region_confs, image_shape = detection_result
    return masks_to_polygons(
        line_mask, line_confs, region_mask, region_confs, image_shape,
        line_percentage_threshold=args.line_percentage_threshold,
        region_percentage_threshold=args.region_percentage_threshold,
        line_iou=args.line_iou,
        region_iou=args.region_iou,
        line_overlap_threshold=args.line_overlap_threshold,
        region_overlap_threshold=args.region_overlap_threshold,
    )

def run_classify_and_recognize(cropped_lines, line_polygons, line_confs,
                               classification_model, model_config, args):
    """Pure model work: script classification + routing to the right HTR model."""
    classification_results = classify_or_skip(cropped_lines, classification_model, model_config, args)

    # Exception objects in fallback results aren't needed downstream; keep payloads picklable
    for r in classification_results:
        r["error"] = None if r["error"] is None else str(r["error"])

    return get_text_predictions(
        classification_results, cropped_lines, line_polygons, line_confs,
        model_config, args,
    )


def _aspect(img):
    try:
        return img.shape[1] / max(1, img.shape[0])
    except Exception:
        return 0.0


class RecognitionPool:
    """
    Worker-local pool of cropped lines (from any number of queued pages) awaiting recognition.
    Complete batches are run as soon as they can be formed; a partial batch is run only once its
    oldest line has waited max_wait seconds. A page's result is published once all its lines are done.
    """

    def __init__(self, model_config, args, inference_results):
        self.model_config = model_config
        self.args = args
        self.results = inference_results
        self.max_wait = args.recognition_max_wait_ms / 1000.0
        self.pending = {label: [] for label in model_config}  # (arrival, request_id, line_idx, image, aspect)
        self.requests = {}

    def _batch_size(self, label):
        use_ppocr = isinstance(self.model_config[label][0], PPOCRRecognizer)
        return self.args.ppocr_batch_size if use_ppocr else self.args.trocr_batch_size

    def add(self, request_id, classification_results, cropped_lines, line_polygons, line_confs):
        indices_by_label = split_by_label(classification_results)
        for label, indices in indices_by_label.items():
            if label not in self.model_config:
                raise ValueError(
                    f"No HTR model configured for predicted_label={label} "
                    f"({len(indices)} line(s) affected)"
                )
        n = len(cropped_lines)
        self.requests[request_id] = {
            "classification": classification_results, "polygons": line_polygons, "confs": line_confs,
            "texts": [None] * n, "scores": [None] * n, "remaining": n,
            "labels": list(indices_by_label),
        }
        if n == 0:
            self._finish(request_id)
            return
        now = time.monotonic()
        for label, indices in indices_by_label.items():
            self.pending[label].extend(
                (now, request_id, i, cropped_lines[i], _aspect(cropped_lines[i])) for i in indices)

    def _recognize(self, label, images):
        """One batch (len(images) <= batch size) -> (scores, texts)."""
        model, processor = self.model_config[label]
        if isinstance(model, PPOCRRecognizer):
            return model.recognize(images, len(images))
        return get_text_lines(images, len(images), model, processor)

    def _run_batch(self, label, entries):
        try:
            scores, texts = self._recognize(label, [e[3] for e in entries])
            if len(texts) != len(entries) or len(scores) != len(entries):
                raise ValueError(f"{label} HTR model returned {len(texts)} result(s) for {len(entries)} input line(s)")
        except Exception:
            error = traceback.format_exc()
            for request_id in {e[1] for e in entries}:
                self._fail(request_id, error)
            return
        for (_, request_id, i, _, _), text, score in zip(entries, texts, scores):
            req = self.requests.get(request_id)
            if req is None:  # request already failed
                continue
            req["texts"][i], req["scores"][i] = text, score
            req["remaining"] -= 1
            if req["remaining"] == 0:
                self._finish(request_id)

    def _fail(self, request_id, error):
        self.requests.pop(request_id, None)
        self.results[request_id] = {"ok": False, "error": error}

    def _finish(self, request_id):
        req = self.requests.pop(request_id)
        model_names = {"cyrillic": self.args.cyrillic_text_rec_model_name,
                       "latin": self.args.text_rec_model_name}
        predictions = get_line_dicts(req["polygons"], req["texts"], req["confs"], req["scores"])
        for prediction, c in zip(predictions, req["classification"]):
            if c.get("classified", True):
                prediction["predicted_script"] = c["predicted_label"]
                prediction["script_pred_conf"] = c["confidence"]
        model_name = " and ".join(model_names[label] for label in req["labels"])
        self.results[request_id] = {"ok": True, "result": (predictions, model_name)}

    def _drain(self, label, only_full):
        entries = self.pending[label]
        bs = self._batch_size(label)
        n_run = len(entries) // bs * bs if only_full else len(entries)
        if n_run == 0:
            return
        entries.sort(key=lambda e: e[4])  # similar widths together -> less padding
        to_run, self.pending[label] = entries[:n_run], entries[n_run:]
        for start in range(0, n_run, bs):
            self._run_batch(label, to_run[start:start + bs])

    def run_full(self):
        """Run all complete batches."""
        for label in self.pending:
            self._drain(label, only_full=True)

    def run_expired(self):
        """Run everything pending for a label whose oldest line has waited at least max_wait."""
        for label in self.pending:
            entries = self.pending[label]
            if entries and time.monotonic() - min(e[0] for e in entries) >= self.max_wait:
                self._drain(label, only_full=False)

    def run_all(self):
        for label in self.pending:
            self._drain(label, only_full=False)

    def seconds_until_expiry(self):
        """None if nothing is pending, else seconds until the oldest pending line hits max_wait."""
        oldest = [min(e[0] for e in entries) for entries in self.pending.values() if entries]
        if not oldest:
            return None
        return max(0.0, min(oldest) + self.max_wait - time.monotonic())


def inference_worker_loop(args, device_string, inference_task_queue, inference_results):
    """
    Model worker: pins itself to one device, loads all models there, then serves
    tasks from the shared queue until it receives None.
    """
    if device_string.startswith("cuda"):
        torch.cuda.set_device(torch.device(device_string))

    print(f"[{device_string}] loading models...")
    detection_model = load_detection_model(args, device_string)
    recognition_model = processor = None
    if latin_enabled(args):
        recognition_model, processor = load_latin_recognizer(args, device_string)

    cyrillic_recognition_model = cyrillic_processor = None
    if cyrillic_enabled(args):
        cyrillic_recognition_model, cyrillic_processor = load_trocr_model(
            args.cyrillic_recognition_model_path,
            args.cyrillic_processor_path,
            device_string,
        )

    classification_model = None
    if classification_enabled(args):
        classification_model = load_classification_model(
            args.script_classification_model_path, device_string)

    model_config = build_model_config(
        recognition_model, processor, cyrillic_recognition_model, cyrillic_processor)
    pool = RecognitionPool(model_config, args, inference_results)
    print(f"[{device_string}] ready")

    while True:
        pool.run_full()
        try:
            # While lines are waiting, always take whatever job is available (detection jobs
            # produce more lines to fill the waiting batches). timeout=0 once the wait is up.
            task = inference_task_queue.get(timeout=pool.seconds_until_expiry())
        except Empty:
            # Nothing left to take and the oldest pending line has waited long enough
            pool.run_expired()
            continue
        try:
            if task is None:
                pool.run_all()
                return

            kind, request_id, payload = task

            if kind == "predict_polygons":
                result = run_detection(detection_model, payload["image_path"], args)
            elif kind == "classify_and_recognize":
                # Classify now, recognize later: the lines join the pool and the result is
                # published by the pool once all of this page's lines are done.
                classification_results = classify_or_skip(
                    payload["cropped_lines"], classification_model, model_config, args)
                pool.add(request_id, classification_results, payload["cropped_lines"],
                         payload["line_polygons"], payload["line_confs"])
                continue
            else:
                raise ValueError(f"Unknown GPU task kind: {kind}")

            inference_results[request_id] = {"ok": True, "result": result}

        except Exception:
            inference_results[request_id] = {"ok": False, "error": traceback.format_exc()}

        finally:
            inference_task_queue.task_done()


_INFERENCE_REQUEST_COUNTER = itertools.count()


def run_inference_task(inference_task_queue, inference_results, device_slots, kind, payload):
    """Submit a task to whichever model worker is free and block until its result arrives."""
    request_id = f"{os.getpid()}-{next(_INFERENCE_REQUEST_COUNTER)}"

    device_slots.acquire()
    try:
        inference_task_queue.put((kind, request_id, payload))
        while True:
            result = inference_results.pop(request_id, None)
            if result is not None:
                if result["ok"]:
                    return result["result"]
                raise RuntimeError(result["error"])
            time.sleep(0.01)
    finally:
        device_slots.release()


# ---------------------------------------------------------------------------
# CPU side
# ---------------------------------------------------------------------------

_CPU_STATE = {}


def init_cpu_worker(args, inference_task_queue, inference_results, device_slots):
    # We are parallelising with processes; avoid thread oversubscription.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _CPU_STATE.update(args=args, queue=inference_task_queue,
                      results=inference_results, slots=device_slots)


def predict_single_image(image_path):
    """
    CPU worker: segmentation request -> polygon post-processing -> ordering ->
    cropping -> recognition request -> page stats.
    Returns (preds, xml_input, msg). preds/xml_input are exactly what get_xml(preds, xml_input)
    (and save_json_output(preds, ...)) take; both are None, with an explanation in msg,
    when there was nothing to output.
    """
    args = _CPU_STATE["args"]
    q, res, slots = _CPU_STATE["queue"], _CPU_STATE["results"], _CPU_STATE["slots"]

    detection_result = run_inference_task(
        q, res, slots, "predict_polygons", {"image_path": image_path})
    # Mask -> polygon post-processing runs here in the CPU worker, not in the model worker
    (line_polygons, line_confs, line_max_mins, region_polygons,
     region_confs, region_max_mins, image_shape) = postprocess_detection(detection_result, args)

    line_preds = {'coords': line_polygons, 'max_min': line_max_mins, 'confs': line_confs}

    if len(region_polygons) > 0:
        region_preds = [
            {'coords': poly, 'id': str(num), 'max_min': mm, 'name': 'paragraph',
             'img_shape': image_shape, 'conf': conf}
            for num, (poly, conf, mm) in enumerate(
                zip(region_polygons, region_confs, region_max_mins))
        ]
    else:
        region_preds = get_default_region(image_shape=image_shape)

    lines_connected = get_line_regions(lines=line_preds, regions=region_preds)
    ordered_lines = order_regions_lines(lines=lines_connected, regions=region_preds)

    if not ordered_lines:
        return None, None, "no lines/regions detected, skipping output"

    flat_polygons, flat_confs, n_lines = flatten_lines(ordered_lines)
    if not flat_polygons:
        return None, None, "no transcribable lines found, skipping output"

    image = load_with_torchvision(image_path)
    cropped_lines = crop_lines(flat_polygons, image)
    del image

    text_predictions, htr_model_name = run_inference_task(
        q, res, slots, "classify_and_recognize",
        {"cropped_lines": cropped_lines,
         "line_polygons": flat_polygons,
         "line_confs": flat_confs})

    if not text_predictions:
        return None, None, "no transcribable lines found, skipping output"

    height, width = ordered_lines[0]['img_shape']
    lines_dict = get_page_stats(text_predictions, Path(image_path).name, height, width)
    preds = process_text_predictions(lines_dict, ordered_lines, n_lines)

    xml_input = XmlInput(
        image_path=image_path,
        page_xml=args.page_xml,
        alto_xml=args.alto_xml,
        xml_path=os.path.dirname(image_path) if not args.xml_folder else args.xml_folder,
        region_segment_model_name=args.region_model_name,
        line_segment_model_name=args.line_model_name,
        classification_model_name=args.script_classification_model_name if classification_enabled(args) else None,
        text_recognition_model_name=htr_model_name,
    )
    return preds, xml_input, ""


def process_single_image(image_path):
    """
    CPU worker: predict_single_image + write XML/JSON.
    Returns (image_path, status, message) with status in {"ok", "info", "error"}.
    """
    args = _CPU_STATE["args"]
    try:
        preds, xml_input, msg = predict_single_image(image_path)
        if preds is None:
            return image_path, "info", msg
        save_xml(preds, xml_input)
        if args.output_json:
            save_json_output(preds, image_path, args)
        return image_path, "ok", ""
    except Exception:
        return image_path, "error", traceback.format_exc()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(args):
    validate_args(args)
    devices = resolve_devices(args)
    print(f"Model workers: {devices}")

    print('Find images in folder', str(args.input_folder))
    images = load_image_paths(args.input_folder)
    print('Found ', str(len(images)))
    if not images:
        return 0, 0

    n_model_workers = len(devices)
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    if args.cpu_processes:
        cpu_processes = args.cpu_processes
    elif devices == ["cpu"]:  # also using CPU for model inference
        cpu_processes = max(1, physical // 2)   # leave cores for the CPU-side inference worker
    else:
        cpu_processes = max(1, min(physical, 6 * n_model_workers))
    print(f"Starting {cpu_processes} CPU workers")

    ctx = get_context("spawn")  # required for CUDA
    manager = ctx.Manager()
    inference_task_queue = manager.JoinableQueue(maxsize=args.inference_in_flight_limit)
    inference_results = manager.dict()
    device_slots = manager.BoundedSemaphore(args.inference_in_flight_limit)

    inference_workers = [
        ctx.Process(target=inference_worker_loop,
                    args=(args, device, inference_task_queue, inference_results))
        for device in devices
    ]
    for w in inference_workers:
        w.start()

    n_ok = n_err = 0
    try:
        with ctx.Pool(
            processes=cpu_processes,
            initializer=init_cpu_worker,
            initargs=(args, inference_task_queue, inference_results, device_slots),
        ) as pool:
            for image_path, status, msg in tqdm(
                pool.imap_unordered(process_single_image, images, chunksize=1),
                total=len(images), desc="Processing images", dynamic_ncols=True,
            ):
                if status == "ok":
                    n_ok += 1
                elif status == "info":
                    tqdm.write(f"[info] {image_path}: {msg}")
                else:
                    n_err += 1
                    tqdm.write(f"[error] failed to process {image_path}:\n{msg}")
    finally:
        for _ in inference_workers:
            inference_task_queue.put(None)
        inference_task_queue.join()
        for w in inference_workers:
            w.join()

    print(f'Processing finished: {n_ok} ok, {n_err} failed')
    return n_ok, n_err


def entrypoint():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()