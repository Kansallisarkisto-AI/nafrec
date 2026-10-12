"""PP-OCRv6 recognition-only inference (ONNX Runtime) for nafrec.

Takes already-cropped text line images (as produced by image_processing.crop_lines)
and returns results in the same line-dict format as trocr.get_text_preds, so the
rest of the pipeline (XML / JSON output) doesn't need to know which model was used.
"""
import numpy as np
import cv2
import os


GPU_PROVIDERS = ("TensorrtExecutionProvider",    # NVIDIA TensorRT
                 "CUDAExecutionProvider",        # NVIDIA regular CUDA
                 "MIGraphXExecutionProvider",    # AMD, ORT >= 1.23
                 "ROCMExecutionProvider")        # AMD, ORT <= 1.22 (legacy)


def _is_gpu_session(providers):
    """True if the primary provider (str or (name, options) tuple) is a GPU one."""
    first = providers[0]
    return (first[0] if isinstance(first, tuple) else first) in GPU_PROVIDERS


MAX_REC_WIDTH = 3200  # RecPreProcess._max_w: wider lines are squeezed to this width


def build_trt_profile(input_name, img_height, img_width_min, width_buckets, batch_size,
                      max_width=MAX_REC_WIDTH, opt_width=None, channels=3):
    """Build TensorRT optimization-profile shape strings for the rec model input (N, C, H, W).

    The widths RecPreProcess can produce follow from the bucketing logic:
      * no buckets: any width in [img_width_min, max_width]
      * buckets:    batch width = smallest bucket >= natural width (natural >= img_width_min),
                    else the natural width when it exceeds the largest bucket (capped at max_width)
    so   min W = smallest bucket >= img_width_min (or img_width_min without buckets)
         max W = max_width (covers batches beyond the largest bucket)
         opt W = the bucket at the median position (or a user supplied value)
    Batch: min 1 (single-line retry path), opt = max = batch_size.
    Returns (input_name, (min_shapes, opt_shapes, max_shapes)) -- see apply_trt_profile().
    """
    buckets = sorted({int(b) for b in width_buckets}) if width_buckets else []
    usable = [b for b in buckets if img_width_min <= b <= max_width]
    w_min = usable[0] if usable else img_width_min
    w_max = max(max_width, usable[-1] if usable else 0)
    if opt_width is None:
        opt_width = usable[len(usable) // 2] if usable else min(max(img_width_min, 640), w_max)
    opt_width = min(max(int(opt_width), w_min), w_max)
    mk = lambda n, w: f"{input_name}:{n}x{channels}x{img_height}x{w}"
    return input_name, (mk(1, w_min), mk(batch_size, opt_width), mk(batch_size, w_max))


def apply_trt_profile(trt_options, input_name, profile):
    """Write a build_trt_profile() result into the TensorRT EP option dict."""
    mn, opt, mx = profile
    trt_options.update({"trt_profile_min_shapes": mn,
                        "trt_profile_opt_shapes": opt,
                        "trt_profile_max_shapes": mx})
    return trt_options


def _get_providers_and_options(device, trt_profile=None):
    """trt_profile: optional (input_name, (min, opt, max)) from build_trt_profile; TensorRT only."""
    """Map a torch-style device string ('cuda', 'cuda:1', 'cpu') to ORT providers.
    Note: torch uses 'cuda' for ROCm builds too, so 'cuda' here means "the GPU"."""
    import onnxruntime as ort
    ort.preload_dlls()   # loads CUDA/cuDNN from pip wheels (or system paths)

    device = str(device)
    # get available providers
    available = ort.get_available_providers()
    option = ort.SessionOptions()

    # If CPU-only
    if not device.startswith("cuda"):
        # Special settings for CPU providers

        option.log_severity_level = 3

        option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        option.intra_op_num_threads = max(1, os.cpu_count())
        option.inter_op_num_threads = 1

        option.enable_cpu_mem_arena = True
        option.enable_mem_pattern = True
        option.enable_mem_reuse = True

        for name in ("OpenVINOExecutionProvider",):  # special CPU providers in priority order
            if name in available:
                print(f"[info] using accelerated {name}")
                return [name, "CPUExecutionProvider"], option

        print(f"[info] using generic CPUExecutionProvider")
        return ["CPUExecutionProvider"], option  # generic CPU fallback

    device_id = int(device.split(":")[1]) if ":" in device else 0

    for name in GPU_PROVIDERS:
        if name in available:
            if "Tensorrt" in name:
                print(f"[info] using TensorRT")
                trt_ep_options = {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_timing_cache_enable": True,
                    "trt_dump_ep_context_model": True,
                    "trt_force_timing_cache": True,
                    "trt_timing_cache_path": "./trt_timingcache/",
                    "trt_engine_cache_path": "./trt_enginecache/",
                    "trt_ep_context_file_path": "./trt_epcontext/",
                    "trt_builder_optimization_level": 2,  # level 2 enables build heuristics
                    "device_id": device_id
                }
                if trt_profile is not None:
                    apply_trt_profile(trt_ep_options, *trt_profile)
                return [(name, trt_ep_options), "CPUExecutionProvider"], option
            
            return [(name, {"device_id": device_id}), "CPUExecutionProvider"], option

    print("[warn] no GPU execution provider found in onnxruntime "
          f"(available: {available}) -- PP-OCR will run on CPU")

    # fall back to generic CPU provider
    return ["CPUExecutionProvider"], option


def _create_session(model_path, providers, options=None, fp16=False):
    """InferenceSession; with fp16 and a GPU provider (CUDA / ROCm / MIGraphX) the
    model is converted to fp16 in memory (the .onnx file is untouched). Inputs and
    outputs stay fp32, so callers don't change. On CPU, fp16 is ignored."""
    import onnxruntime as ort

    if fp16 and _is_gpu_session(providers):
        import onnx
        from onnxconverter_common import float16

        model = float16.convert_float_to_float16(onnx.load(model_path), keep_io_types=True)
        return ort.InferenceSession(model.SerializeToString(), providers=providers, sess_options=options)
    if fp16:
        print(f"[info] fp16 ignored for {os.path.basename(model_path)}: not running on a GPU")
    return ort.InferenceSession(model_path, providers=providers, sess_options=options)


def _onnx_input_name(model_path):
    import onnx
    m = onnx.load(model_path, load_external_data=False)
    return m.graph.input[0].name


class PPOCRRecognizer:
    """Wraps an ONNX PP-OCRv6 recognition model + preprocessing + CTC decoding."""

    def __init__(self, model_path, char_dict_path, device="cpu", img_height=48, img_width_min=320, fp16=False,
                 width_buckets=None, batch_size=32, trt_dynamic_shapes=True, trt_opt_width=None):
        # Lazy imports so nafrec still works without onnxruntime / ppocrv6_onnx
        # when the PP-OCR path isn't used.
        from ppocrv6_onnx import RecPreProcess, CTCLabelDecode

        trt_profile = None
        if trt_dynamic_shapes and str(device).startswith("cuda"):
            trt_profile = build_trt_profile(_onnx_input_name(model_path), img_height, img_width_min,
                                            width_buckets, batch_size, opt_width=trt_opt_width)
            print(f"[info] TensorRT profile: min={trt_profile[1][0]} opt={trt_profile[1][1]} max={trt_profile[1][2]}")
        providers, options = _get_providers_and_options(device, trt_profile)
        self.session = _create_session(model_path, providers, options, fp16)
        self.input_name = self.session.get_inputs()[0].name

        self.pre = RecPreProcess(rec_image_shape=(3, img_height, img_width_min), width_buckets=width_buckets)
        self.post = CTCLabelDecode(char_dict_path)

        # PaddleOCR appends a trailing space class when Global.use_space_char=true
        # (the dict file itself doesn't contain it). Detect this from the model's
        # output size so no extra CLI flag is needed.
        n_classes = None
        for out in self.session.get_outputs():
            if out.shape and len(out.shape) == 3 and isinstance(out.shape[-1], int):
                n_classes = out.shape[-1]
                break
        if n_classes is None or n_classes == self.post.vocab_size + 1:
            self.post._chars = self.post._chars + (" ",)
        if n_classes is not None and n_classes != self.post.vocab_size:
            raise ValueError(
                f"PP-OCR model outputs {n_classes} classes but the character dictionary "
                f"gives {self.post.vocab_size} (blank + dict [+ space]). "
                f"Check --ppocr_char_dict_path."
            )

    def _recognize_batch(self, imgs_bgr):
        x = self.pre(imgs_bgr)
        out = self.session.run(None, {self.input_name: x})[0]
        texts, scores = self.post(out)
        return texts, scores

    def recognize(self, line_images_rgb, batch_size=32):
        """line_images_rgb: list of RGB uint8 arrays (H, W, 3).
        Returns (scores, texts) in the input order."""
        n = len(line_images_rgb)
        texts, scores = [""] * n, [0.0] * n

        valid = []  # (original index, BGR image)
        for i, img in enumerate(line_images_rgb):
            if img is None or img.ndim != 3 or img.shape[0] == 0 or img.shape[1] == 0:
                continue  # degenerate crop -> empty text
            valid.append((i, cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)))

        # Sort by aspect ratio so batches have similar widths (less padding)
        valid.sort(key=lambda t: t[1].shape[1] / float(t[1].shape[0]))

        for start in range(0, len(valid), batch_size):
            chunk = valid[start:start + batch_size]
            idxs = [i for i, _ in chunk]
            imgs = [im for _, im in chunk]
            try:
                t, s = self._recognize_batch(imgs)
            except Exception as e:
                print(f"[warn] PP-OCR batch failed ({e}) -- retrying its lines individually")
                t, s = [], []
                for im in imgs:
                    try:
                        ti, si = self._recognize_batch([im])
                        t.extend(ti); s.extend(si)
                    except Exception as line_e:
                        print(f"[warn] line also failed individually, inserting empty placeholder: {line_e}")
                        t.append(""); s.append(0.0)
            for i, ti, si in zip(idxs, t, s):
                texts[i], scores[i] = ti, float(si)
        return scores, texts


def load_ppocr_model(model_path, char_dict_path, device="cpu", img_height=48, img_width_min=320, fp16=False,
                     width_buckets=None, batch_size=32, trt_dynamic_shapes=True, trt_opt_width=None):
    return PPOCRRecognizer(model_path, char_dict_path, device, img_height, img_width_min, fp16, width_buckets,
                           batch_size, trt_dynamic_shapes, trt_opt_width)


def get_ppocr_preds(data, recognizer):
    """Drop-in counterpart of trocr.get_text_preds. `data` is an OCRInput."""
    from .trocr import get_line_dicts

    scores, texts = recognizer.recognize(data.line_images, data.batch_size)
    return get_line_dicts(data.line_polygons, texts, data.line_confs, scores)