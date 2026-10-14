import os
import glob
import numpy as np
import cv2
import math
import torch
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .ppocr import _get_providers_and_options, format_trt_profile, onnx_input_specs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 128      # generation limit; also sizes the TensorRT decoder profiles
TRT_OPT_PAST_LEN = 32     # typical number of already generated tokens a line has (TensorRT "opt" shape)

class TrOCRProcessorCustom(TrOCRProcessor):
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.current_processor = self.image_processor
        self.chat_template = None


# ---------------------------------------------------------------------------
# TensorRT shape profile for the ONNX TrOCR parts (encoder, decoder, decoder_with_past)
# ---------------------------------------------------------------------------
# optimum hands ONE provider-option list to all parts, so there is one combined profile. That works because the
# TensorRT EP looks profile entries up by the graph's own input names and ignores names a graph doesn't have.
# The only requirement is that an input name shared by two files (input_ids, encoder_hidden_states) gets the
# same range in both.

_TROCR_ONNX_NAMES = {"encoder": "encoder_model.onnx",
                     "decoder": "decoder_model.onnx",
                     "decoder_with_past": "decoder_with_past_model.onnx"}


def _trt_dim(name, axis, batch, enc_seq, ids_seq, past_seq, image_hw):
    """(min, opt, max) for one dynamic dimension, decided by the input's name (optimum's export naming)."""
    if axis == 0:
        return batch
    if name == "pixel_values" and axis in (2, 3) and image_hw:
        v = int(image_hw[axis - 2])
        return (v, v, v)
    if name == "input_ids" and axis == 1:
        return ids_seq
    if name in ("encoder_hidden_states", "encoder_attention_mask") and axis == 1 and enc_seq:
        return (enc_seq, enc_seq, enc_seq)
    if name.startswith("past_key_values") and axis == 2:
        if ".encoder." in name:          # cross-attention cache: always the encoder sequence length
            if not enc_seq:
                raise ValueError("encoder sequence length unknown")
            return (enc_seq, enc_seq, enc_seq)
        return past_seq                   # self-attention cache: grows by one token per step
    raise ValueError(f"don't know how to size dynamic dim {axis} of input '{name}'")


def trocr_trt_shapes(specs, batch, enc_seq=None, ids_seq=(1, 1, 1),
                     past_seq=(1, TRT_OPT_PAST_LEN, MAX_NEW_TOKENS + 1), image_hw=None):
    """{input_name: (min_dims, opt_dims, max_dims)} for the dynamic inputs of ONE TrOCR ONNX file.

    specs: onnx_input_specs() of that file ([(name, dims)], dim = int or None for dynamic); static dims keep
    their value. batch / ids_seq / past_seq are (min, opt, max); enc_seq is the encoder output length.
    """
    shapes = {}
    for name, dims in specs:
        if all(d is not None for d in dims):
            continue
        per_axis = [(d, d, d) if d is not None
                    else _trt_dim(name, axis, batch, enc_seq, ids_seq, past_seq, image_hw)
                    for axis, d in enumerate(dims)]
        shapes[name] = tuple(tuple(t[i] for t in per_axis) for i in range(3))
    return shapes


def _find_trocr_onnx(model_path, revision, filename):
    """Path of one exported ONNX part (local folder, any subfolder), or downloaded from the hub; None if absent."""
    if os.path.isdir(model_path):
        hits = sorted(glob.glob(os.path.join(model_path, "**", filename), recursive=True))
        return hits[0] if hits else None
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_path, filename, revision=revision)
    except Exception:
        return None


def _encoder_seq_len(encoder_path, image_hw, model_path, revision):
    """Encoder output length: run the encoder once on zeros (CPU); formula from the config as a fallback."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dtype = np.float16 if "float16" in inp.type else np.float32
        return int(sess.run(None, {inp.name: np.zeros((1, 3, *image_hw), dtype=dtype)})[0].shape[1])
    except Exception as e:
        print(f"[info] TrOCR encoder length probe failed ({type(e).__name__}: {e}), using the config formula")
        from transformers import AutoConfig
        enc = AutoConfig.from_pretrained(model_path, revision=revision).encoder
        patch = enc.patch_size if isinstance(enc.patch_size, (list, tuple)) else (enc.patch_size,) * 2
        return (image_hw[0] // patch[0]) * (image_hw[1] // patch[1]) + 1   # + CLS token (ViT / BEiT / DINOv2)


def _processor_image_hw(processor):
    """(H, W) of the images the processor hands to the model, read from the loaded TrOCRProcessor.
    Works for dict sizes and for the torchvision backend's SizeDict. None if it can't tell."""
    ip = getattr(processor, "image_processor", processor)

    def get(size, key):
        return size.get(key) if isinstance(size, dict) else getattr(size, key, None)

    # a centre crop, when enabled, is the final size; otherwise the resize target
    crop = getattr(ip, "crop_size", None)
    size = crop if getattr(ip, "do_center_crop", False) and crop else getattr(ip, "size", None)
    if isinstance(size, int):
        return size, size
    if size is None:
        return None
    if get(size, "height") and get(size, "width"):
        return int(get(size, "height")), int(get(size, "width"))
    for k in ("shortest_edge", "longest_edge"):
        if get(size, k):
            return int(get(size, k)), int(get(size, k))
    return None


def predetermine_trocr_trt_profile(model_path, revision, batch_size, use_cache=True, image_hw=None):
    """Work out, before loading anything on the GPU, the TensorRT profile for an optimum TrOCR ONNX export.

    Batch 1..batch_size (encoder) and 1..batch_size*num_beams (decoders). image_hw: (H, W) of pixel_values,
    normally what the image processor produces (see _processor_image_hw). When None: a static size in the
    encoder graph, else encoder.image_size in the model config.
    Returns (input_names, (min_shapes, opt_shapes, max_shapes)) for _get_providers_and_options(trt_profile=...).
    """
    paths = {k: _find_trocr_onnx(model_path, revision, f) for k, f in _TROCR_ONNX_NAMES.items()}
    needed = ["encoder", "decoder"] + (["decoder_with_past"] if use_cache else [])
    missing = [k for k in needed if not paths[k]]
    if missing:
        raise FileNotFoundError(f"missing {[_TROCR_ONNX_NAMES[k] for k in missing]} in {model_path}")

    batch_size = max(1, int(batch_size))
    try:
        from transformers import GenerationConfig
        beams = max(1, int(GenerationConfig.from_pretrained(model_path, revision=revision).num_beams or 1))
    except Exception:
        beams = 1
    enc_batch = (1, batch_size, batch_size)
    dec_batch = (1, batch_size * beams, batch_size * beams)

    enc_specs = onnx_input_specs(paths["encoder"])
    onnx_hw = None
    for name, dims in enc_specs:
        if name == "pixel_values" and len(dims) == 4 and None not in dims[2:]:
            onnx_hw = (dims[2], dims[3])
    if image_hw is None:
        image_hw = onnx_hw
    if image_hw is None:
        from transformers import AutoConfig
        size = AutoConfig.from_pretrained(model_path, revision=revision).encoder.image_size
        image_hw = (size, size) if isinstance(size, int) else tuple(size)
    if onnx_hw is not None and tuple(onnx_hw) != tuple(image_hw):
        print(f"[warn] TrOCR encoder ONNX has a fixed input size {onnx_hw[0]}x{onnx_hw[1]} but the image "
              f"processor produces {image_hw[0]}x{image_hw[1]}; the model will not accept the processor's images")
    enc_seq = _encoder_seq_len(paths["encoder"], image_hw, model_path, revision)

    # with a cache both decoders only ever see one new token; without it the decoder sees the whole sequence
    ids_seq = (1, 1, 1) if use_cache else (1, 16, MAX_NEW_TOKENS + 1)
    merged = {}
    for key in needed:
        batch = enc_batch if key == "encoder" else dec_batch
        shapes = trocr_trt_shapes(onnx_input_specs(paths[key]), batch, enc_seq=enc_seq,
                                  ids_seq=ids_seq, image_hw=image_hw)
        for name, triple in shapes.items():
            if merged.get(name, triple) != triple:
                raise ValueError(f"input '{name}' needs different ranges in two ONNX files")
            merged[name] = triple
    print(f"[info] TrOCR TensorRT profile: batch 1..{batch_size} (decoders x{beams} beams), "
          f"image {image_hw[0]}x{image_hw[1]}, encoder length {enc_seq}")
    return format_trt_profile(merged)


def _load_trocr_processor(processor_path, encoder_type, revision):
    if encoder_type == "dinov2":
        return TrOCRProcessorCustom.from_pretrained(processor_path, revision=revision)
    if "microsoft" in processor_path.lower():
        return TrOCRProcessor.from_pretrained(processor_path,
                                              backend="torchvision",
                                              revision=revision)
    return TrOCRProcessor.from_pretrained(processor_path,
                                          backend="torchvision",
                                          do_resize=True,
                                          revision=revision)


def load_trocr_model(model_path, processor_path, device='cuda', revision="main", fp16=False,
                     enable_tensorrt=False, batch_size=8, trt_cache_subdir="trocr",
                     trt_image_hw=None):
    """
    Load a TrOCR model with custom image size support and positional encoding interpolation.

    This function applies patches to ViTPatchEmbeddings and ViTEmbeddings to enable
    models to handle custom image sizes by interpolating positional encodings.

    Args:
        model_path: Path to the pretrained TrOCR model directory or checkpoint.
        processor_path: Path to the TrOCR processor directory or checkpoint.
        enable_tensorrt: run the ONNX encoder and decoders on TensorRT (ONNX models only). The shape profile is
            worked out from the exported ONNX files before the sessions are created.
        trt_image_hw: (H, W) override for the TensorRT pixel_values profile (default: the size the TrOCRProcessor
            produces, then the ONNX, then the model config).
        batch_size: largest recognition batch (--trocr_batch_size); sets the TensorRT batch range 1..batch_size.
        trt_cache_subdir: prefix of the engine cache subfolders under ./trt_enginecache.

    Returns:
        tuple: A 2-element tuple containing:
            - processor (TrOCRProcessor): Configured image processor with custom dimensions.
            - model (VisionEncoderDecoderModel): Loaded TrOCR model on the specified device.
    """
    resolved_device = device or DEVICE
    use_fp16 = fp16 and torch.device(resolved_device).type == "cuda"
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
    
    # The processor is built first when possible: its image size sets the TensorRT profile. Which processor class
    # to use depends on the encoder type, read from the config (built after the model if that fails).
    processor = None
    try:
        from transformers import AutoConfig
        encoder_type = AutoConfig.from_pretrained(model_path, revision=revision).encoder.model_type
        processor = _load_trocr_processor(processor_path, encoder_type, revision)
    except Exception:
        pass

    # Load model
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        # Same provider / session-option selection as PP-OCR (CUDA device_id, OpenVINO, CPU threads, ...).
        # optimum wants provider names and their options as two separate lists.
        use_cache = True
        trt_profile = None
        if enable_tensorrt and str(resolved_device).startswith("cuda"):
            try:
                image_hw = trt_image_hw or (_processor_image_hw(processor) if processor is not None else None)
                trt_profile = predetermine_trocr_trt_profile(model_path, revision, batch_size,
                                                             use_cache=use_cache, image_hw=image_hw)
            except Exception as e:
                print(f"[warn] no TensorRT profile for TrOCR ({type(e).__name__}: {e}) -- using CUDA")
                enable_tensorrt = False
        providers, session_options = _get_providers_and_options(
            resolved_device, trt_profile=trt_profile, enable_tensorrt=enable_tensorrt,
            trt_cache_subdir=trt_cache_subdir, label="trocr")
        model = ORTModelForVision2Seq.from_pretrained(
            model_path,
            revision=revision,
            providers=[p[0] if isinstance(p, tuple) else p for p in providers],
            provider_options=[p[1] if isinstance(p, tuple) else {} for p in providers],
            session_options=session_options,
            use_cache=use_cache,
            use_merged=False,  # the merged decoder's `If` node can't be built by TensorRT
            use_io_binding=True
        )

        def nope():
            return False
        model.is_remote_code = nope

        # something is causing issues with io binding and the kv cache branch in the decoder
        #model.decoder.use_io_binding = False
        print(f"[info] onnx trocr model loaded on {model.device}")
    except Exception as e:
        #print(f"[info] onnx trocr model not used ({type(e).__name__}: {e})")
        model = VisionEncoderDecoderModel.from_pretrained(
                                                        model_path,
                                                        torch_dtype=model_dtype,
                                                        revision=revision
                                                    ).to(resolved_device)
        print("[info] regular trocr model loaded")

    if processor is None:
        processor = _load_trocr_processor(processor_path, model.config.encoder.model_type, revision)

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
    # Device and dtype are read from the model itself, not a module-level global.
    # (.device / .dtype exist on both the PyTorch model and the ONNX Runtime model)
    device = recognition_model.device
    dtype = recognition_model.dtype

    pixel_values = processor(cropped_lines, return_tensors="pt").pixel_values
    generated_dict = recognition_model.generate(pixel_values.to(device, dtype=dtype), max_new_tokens=MAX_NEW_TOKENS, max_length=None, return_dict_in_generate=True, output_scores=True)
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