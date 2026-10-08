import os
import time

import argparse
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel
import torch.multiprocessing as mp

# Import PyTorch and prevent useless warning
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch.jit.script.*is deprecated.*",
    category=FutureWarning,
)
import torch

from .xml_koodit import get_xml
from .trocr import get_text_preds, load_trocr_model
from .ppocr import load_ppocr_model, get_ppocr_preds, PPOCRRecognizer
from .seg_inference import load_rfdetr_model, predict_polygons
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
    classification_model_name: str
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

def parse_args():
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
        default='/path/to/recognition_model/',
        help="Path to the recognition model folder"
    )
    parser.add_argument(
        "--cyrillic_recognition_model_path",
        type=str,
        default='/path/to/cyrillic_recognition_model/',
        help="Path to the cyrillic recognition model folder"
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default='/path/to/processor/',
        help="Path to the processor folder"
    )
    parser.add_argument(
        "--cyrillic_processor_path",
        type=str,
        default='/path/to/cyrillic_processor/',
        help="Path to the cyrillic processor folder"
    )
    parser.add_argument(
        "--script_classification_model_path",
        type=str,
        default="/path/to/classification_model.pt", 
        help="Path to the script classification model file"
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
        "--multi_gpu",
        type=bool,
        default=False,
        help="Whether to use all GPUs on system instead of just the first one. Requires a patched version of the RF-DETR library that accepts a device argument with the rank specified."
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
        default="/path/to/ppocr_rec/inference.onnx",
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
        
    args = parser.parse_args()
    return args

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
    classifier_payload = ClassifierInput(
        line_images = cropped_lines,
        batch_size = args.classifier_batch_size,
        default_label = args.classifier_default_label
    )
    classification_results = classify_lines(classifier_payload, classification_model)
    # Get text predictions
    model_config={
        "cyrillic": (cyrillic_recognition_model, cyrillic_processor),
        "latin": (recognition_model, processor),
    }
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
        return None

def process_all_images(
    images, 
    detection_model, 
    classification_model, 
    recognition_model, 
    cyrillic_recognition_model, 
    processor, 
    cyrillic_processor, 
    args, 
    rank=0
):
    """
    Process a collection of images through detection, recognition, and XML output pipeline.

    This function performs end-to-end OCR processing on a batch of images by:
    1. Detecting text lines and regions using a detection model
    2. Organizing detected lines into regions and ordering them
    3. Classifying line images based on their script type (latin / cyrillic)
    4. Passing text lines into cyrillic or non-cyrillic recognition model based on their text type
    5. Recognizing text content using a the selected recognition model
    6. Generating XML output (PAGE or ALTO format) with the recognized text

    Args:
        images: Iterable of image file paths to process
        detection_model: Model for detecting text lines and regions in images
        classification_model: Model for classifying line images based on their script type (latin / cyrillic)
        recognition_model: Model for recognizing text content (using lating script) from detected lines
        cyrillic_recognition_model: Model for recognizing text content (using cyrillic script) from detected lines
        processor: Processor for preparing data for the recognition model
        cyrillic_processor: Processor for preparing data for the cyrillic recognition model
        args: Argument object containing configuration parameters including:
            - batch_size: Threshold for line detection
            - page_xml: Flag for PAGE XML output
            - alto_xml: Flag for ALTO XML output
            - region_model_name: Name of the region segmentation model
            - line_model_name: Name of the line segmentation model
            - text_rec_model_name: Name of the text recognition model

    Returns:
        None. Outputs are written as XML files to the same directory as input images.

    Note:
        Progress is displayed via tqdm progress bar during processing.
    """
    bar = tqdm(
        images,
        desc=f"GPU {rank}",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for image_path in bar:
        try:
            start_time = time.time()
            line_polygons, line_confs, line_max_mins, region_polygons, region_confs, region_max_mins, image_shape = predict_polygons(
                                        detection_model, 
                                        image_path, 
                                        max_size = args.tile_size * args.tiles_across - args.tile_overlap if args.tile_size else 768, 
                                        confidence_threshold = args.confidence_threshold,
                                        line_percentage_threshold = args.line_percentage_threshold,
                                        region_percentage_threshold = args.region_percentage_threshold,
                                        line_iou = args.line_iou,
                                        region_iou = args.region_iou,
                                        line_overlap_threshold = args.line_overlap_threshold,
                                        region_overlap_threshold = args.region_overlap_threshold,
                                        tile_size = args.tile_size,
                                        tile_overlap = args.tile_overlap,
                                        tile_iou_threshold = args.tile_iou_threshold,
                                        tile_batch_size=args.tile_batch_size)
            
            predict_polygons_time = time.time() - start_time

            start_time = time.time()
            line_preds = {'coords':line_polygons,
                        'max_min': line_max_mins,
                        'confs':line_confs
                        }

            if len (region_polygons) > 0:
                region_preds = []
                for num, (region_polygon, region_conf, region_max_min) in enumerate(zip(region_polygons, region_confs, region_max_mins)):
                    region_preds.append({'coords': region_polygon,
                                        'id': str(num),
                                        'max_min': region_max_min,
                                        'name': 'paragraph',
                                        'img_shape': image_shape,
                                        'conf': region_conf})
            else:
                region_preds = get_default_region(image_shape=image_shape)

            lines_connected_to_regions = get_line_regions(lines=line_preds, regions=region_preds)
            ordered_lines = order_regions_lines(lines=lines_connected_to_regions, regions=region_preds)

            if ordered_lines:
                text_predictions, trocr_model_name = classify_and_recognize(
                    image_path, 
                    ordered_lines, 
                    classification_model, 
                    recognition_model, 
                    cyrillic_recognition_model, 
                    processor, 
                    cyrillic_processor,
                    args
                )
                if text_predictions:
                    xml_input = XmlInput(image_path = image_path,
                                        page_xml = args.page_xml,
                                        alto_xml = args.alto_xml,
                                        xml_path = os.path.dirname(image_path) if not args.xml_folder else args.xml_folder,
                                        region_segment_model_name=args.region_model_name,
                                        line_segment_model_name=args.line_model_name,
                                        classification_model_name=args.script_classification_model_name,
                                        text_recognition_model_name=trocr_model_name)
                    get_xml(text_predictions, xml_input)
                    if args.output_json:
                        save_json_output(text_predictions, image_path, args)
                else:
                    print(f"[info] no transcribable lines found for {image_path}, skipping output")

            else:
                print(f"[info] no lines/regions detected for {image_path}, skipping output")

            get_text_predictions_time = time.time() - start_time
            bar.set_postfix_str(f"  predict_polygons: {predict_polygons_time:.2f} s, get_text_predictions: {get_text_predictions_time:.2f} s")
        
        except Exception as e:
            print(f"[error] failed to process {image_path}: {e}")
            continue

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

def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] requested cuda but no CUDA device is available -- falling back to cpu")
        args.device = "cpu"

    print("Loading rfdetr model")
    detection_model = load_rfdetr_model(args.detection_model_path, device=args.device, batch_size=args.tile_batch_size if args.tile_size else 1)
    print("Loading script type classification model")
    classification_model = load_classification_model(args.script_classification_model_path, args.device)

    print('Loading PP-OCRv6 model for latin script' if args.use_ppocr else 'Loading TrOCR model for latin script')
    recognition_model, processor = load_latin_recognizer(args, args.device)
    
    print('Loading TrOCR model for cyrillic script')
    cyrillic_recognition_model, cyrillic_processor = load_trocr_model(args.cyrillic_recognition_model_path, args.cyrillic_processor_path, args.device)

    print('Find images in folder', str(args.input_folder))
    images = load_image_paths(args.input_folder)
    print('Found ', str(len(images)))

    print('Starting HTR')
    process_all_images(images, detection_model, classification_model, recognition_model, cyrillic_recognition_model, processor, cyrillic_processor, args)
    print('Processing Finished')

def worker(rank, world_size, args):
    # Pin this process to one GPU
    torch.cuda.set_device(rank)
    device_string = f"cuda:{rank}"

    print(f"[GPU {rank}] loading models...")
    detection_model = load_rfdetr_model(args.detection_model_path, device=device_string, batch_size=args.tile_batch_size if args.tile_size else 1)
    classification_model = load_classification_model(args.script_classification_model_path, device=device_string)
    recognition_model, processor = load_latin_recognizer(args, device_string)
    cyrillic_recognition_model, cyrillic_processor = load_trocr_model(args.cyrillic_recognition_model_path, args.cyrillic_processor_path, device=device_string)

    images = load_image_paths(args.input_folder)

    # select the range of images for this worker, images are split across world_size workers
    my_images = images[rank::world_size]
    print(f"[GPU {rank}] got {len(my_images)} images")

    process_all_images(my_images, detection_model, classification_model, recognition_model, cyrillic_recognition_model, processor, cyrillic_processor, args, rank=rank)

def entrypoint():
    args = parse_args()

    if args.multi_gpu:
        ngpu = torch.cuda.device_count()
        print(f"{ngpu} GPUs detected")
    else:
        ngpu = 1
        print("Using only first GPU.")
    
    if ngpu <= 1:
        main(args)
    else:
        mp.set_start_method("spawn", force=True)  # important for CUDA
        mp.spawn(worker, args=(ngpu, args), nprocs=ngpu, join=True)

if __name__ == "__main__":
    entrypoint()