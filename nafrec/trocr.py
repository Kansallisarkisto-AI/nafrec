import numpy as np
import cv2
import math
import torch
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 192
IMG_WIDTH = 1024

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor
        self.chat_template = None

def load_trocr_model(model_path, processor_path, device='cuda', revision="main"):
    """
    Load a TrOCR model with custom image size support and positional encoding interpolation.

    This function applies patches to ViTPatchEmbeddings and ViTEmbeddings to enable
    models to handle custom image sizes by interpolating positional encodings.

    Args:
        model_path: Path to the pretrained TrOCR model directory or checkpoint.
        processor_path: Path to the TrOCR processor directory or checkpoint.

    Returns:
        tuple: A 2-element tuple containing:
            - processor (TrOCRProcessor): Configured image processor with custom dimensions.
            - model (VisionEncoderDecoderModel): Loaded TrOCR model on the specified device.
    """
    resolved_device = device or DEVICE
    use_fp16 = torch.device(resolved_device).type == "cuda"
    model_dtype = torch.float16 if use_fp16 else torch.float32
    # Store original
    original_embeddings_forward = ViTEmbeddings.forward
    
    # Always apply patches for models saved with custom image sizes
    def universal_patch_forward(self, *args, **kwargs):
        pixel_values = args[0] if args else kwargs['pixel_values']
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
    def universal_embeddings_forward(self, *args, **kwargs):
        kwargs['interpolate_pos_encoding'] = True
        return original_embeddings_forward(self, *args, **kwargs)
    
    # Apply patches
    ViTPatchEmbeddings.forward = universal_patch_forward
    ViTEmbeddings.forward = universal_embeddings_forward
    
    # Load model and processor
    model = VisionEncoderDecoderModel.from_pretrained(
                                                    model_path,
                                                    torch_dtype=model_dtype,
                                                    revision=revision
                                                ).to(resolved_device)

    if model.config.encoder.model_type == "dinov2":
        processor = TrOCRProcessorCustom.from_pretrained(processor_path, revision=revision)
    elif "microsoft" in processor_path.lower():
        processor = TrOCRProcessor.from_pretrained(processor_path,
                                        backend="torchvision",
                                        revision=revision)
        
    else:
        processor = TrOCRProcessor.from_pretrained(processor_path,
                                                backend="torchvision",
                                                do_resize=True, 
                                                size={'height': IMG_HEIGHT,'width': IMG_WIDTH},
                                                revision=revision)
    
    return model, processor


def predict_text(cropped_lines, recognition_model, processor):
    """
    Predict text content from cropped line images using a recognition model.

    Args:
        cropped_lines: List of cropped text line images.
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        tuple: A 2-element tuple containing:
            - scores (list): Confidence scores for each prediction.
            - generated_text (list): Predicted text strings for each line.
    """
    # Device and dtype are read from the model's OWN parameters, not a module-level global
    device = next(recognition_model.parameters()).device
    dtype = next(recognition_model.parameters()).dtype

    pixel_values = processor(cropped_lines, return_tensors="pt").pixel_values
    generated_dict = recognition_model.generate(pixel_values.to(device, dtype=dtype), max_new_tokens=128, max_length=None, return_dict_in_generate=True, output_scores=True)
    generated_ids = generated_dict['sequences']
    generated_scores = generated_dict['scores']

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Works for both:
    # - greedy decoding: num_beams=1
    # - beam search: num_beams>1
    token_log_probs = recognition_model.compute_transition_scores(
        sequences=generated_ids,
        scores=generated_scores,
        beam_indices=getattr(generated_dict, "beam_indices", None),
        normalize_logits=True,
    )

    pad_token_id = processor.tokenizer.pad_token_id

    # For encoder-decoder models, generated_ids normally begins with a decoder
    # start token, while transition scores correspond to subsequent tokens.
    generated_token_ids = generated_ids[:, -token_log_probs.shape[1]:]

    if pad_token_id is None:
        token_mask = torch.ones_like(
            generated_token_ids,
            dtype=torch.bool,
        )
    else:
        token_mask = generated_token_ids.ne(pad_token_id)

    # Exclude special tokens from confidence aggregation where possible.
    special_token_ids = set(processor.tokenizer.all_special_ids)
    for special_token_id in special_token_ids:
        token_mask &= generated_token_ids.ne(special_token_id)

    token_counts = token_mask.sum(dim=1).clamp(min=1)

    # Geometric mean of generated-token probabilities.
    mean_log_probs = (
        token_log_probs * token_mask
    ).sum(dim=1) / token_counts

    scores = mean_log_probs.exp().float().cpu().tolist()

    return scores, generated_text


def get_text_lines(cropped_lines, batch_size, recognition_model, processor):
    """
    Process text lines in batches and predict text content with confidence scores.

    Args:
        cropped_lines: List of cropped text line images.
        batch_size: Maximum number of lines to process in a single batch.
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        tuple: A 2-element tuple containing:
            - scores (list): Confidence scores for all predictions.
            - generated_text (list): Predicted text strings for all lines.
    """
    scores, generated_text = [], []
    if not cropped_lines:
        return scores, generated_text

    n = math.ceil(len(cropped_lines) / batch_size)
    for i in range(n):
        start = i * batch_size
        end = min(start + batch_size, len(cropped_lines))
        batch = cropped_lines[start:end]

        try:
            sc, gt = predict_text(batch, recognition_model, processor)
        except Exception as e:
            print(
                f"[warn] batch [{start}:{end}] failed ({e}) -- retrying its lines "
                f"individually to isolate which one is the problem"
            )
            sc, gt = [], []
            for line in batch:
                try:
                    line_sc, line_gt = predict_text([line], recognition_model, processor)
                    sc.extend(line_sc)
                    gt.extend(line_gt)
                except Exception as line_e:
                    print(f"[warn] line also failed individually, inserting empty placeholder: {line_e}")
                    sc.append(0.0)
                    gt.append("")
 
        scores += sc
        generated_text += gt
 
    return scores, generated_text

def get_line_dicts(polygons, generated_text, line_confs, scores):
    """
    Combine OCR results into a structured dictionary with metadata and statistics.

    Args:
        polygons: List of polygon coordinates for each text line.
        generated_text: List of predicted text strings.
        line_confs: List of confidence scores for line detection.
        scores: List of confidence scores for text recognition.

    Returns:
        line_dicts: List of text line dictionaries, each containing:
            - polygon (list): Polygon coordinates for the text line.
            - text (str): Predicted text content.
            - conf (float): Confidence score for line detection.
            - text_conf (float): Confidence score for text recognition.
            - row_length (int): Character count of the text.
    """
    line_dicts = []
    for i in range(len(generated_text)):
        row_length = len(generated_text[i])
        line_dict = {
            'polygon': polygons[i], 
            'text': generated_text[i], 
            'conf': line_confs[i], 
            'text_conf':scores[i], 
            'row_length': row_length
        }
        line_dicts.append(line_dict)
    return line_dicts

def get_text_preds(data, recognition_model, processor):
    """
    Process an image to extract and recognize text from detected text lines.

    Args:
        data: Object containing input parameters including:
            - line_images: List of cropped line images.
            - line_polygons: List of polygon coordinates for text lines.
            - line_confs: Confidence scores for line detection.
            - batch_size: batch size for number of line images processed
        recognition_model: Pre-trained text recognition model.
        processor: Image processor for the recognition model.

    Returns:
        line_dicts: List of text line dictionaries, each containing:
            - polygon (list): Polygon coordinates for the text line.
            - text (str): Predicted text content.
            - conf (float): Confidence score for line detection.
            - text_conf (float): Confidence score for text recognition.
            - row_length (int): Character count of the text.
    """
    # Get text predictions
    scores, generated_text = get_text_lines(data.line_images, data.batch_size, recognition_model, processor)
    # Get results in dictionary form
    line_dicts = get_line_dicts(data.line_polygons, generated_text, data.line_confs, scores)
    return line_dicts