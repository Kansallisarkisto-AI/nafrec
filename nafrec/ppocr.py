"""PP-OCRv6 recognition-only inference (ONNX Runtime) for nafrec.

Takes already-cropped text line images (as produced by image_processing.crop_lines)
and returns results in the same line-dict format as trocr.get_text_preds, so the
rest of the pipeline (XML / JSON output) doesn't need to know which model was used.
"""
import numpy as np
import cv2
import os


def _get_providers_and_options(device):
    """Map a torch-style device string ('cuda', 'cuda:1', 'cpu') to ORT providers.
    Note: torch uses 'cuda' for ROCm builds too, so 'cuda' here means "the GPU"."""
    import onnxruntime as ort

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

    for name in ("CUDAExecutionProvider",        # NVIDIA
                 "MIGraphXExecutionProvider",    # AMD, ORT >= 1.23
                 "ROCMExecutionProvider"):       # AMD, ORT <= 1.22 (legacy)
        if name in available:
            return [(name, {"device_id": device_id}), "CPUExecutionProvider"], option

    print("[warn] no GPU execution provider found in onnxruntime "
          f"(available: {available}) -- PP-OCR will run on CPU")

    # fall back to generic CPU provider
    return ["CPUExecutionProvider"], option


class PPOCRSession:
    """Model-worker side: just the ONNX session. Takes an already preprocessed batch and
    returns the raw model output; pre/post-processing live in PPOCRRecognizer (CPU side)."""

    def __init__(self, model_path, device="cpu"):
        # Lazy import so nafrec still works without onnxruntime when the PP-OCR path isn't used.
        import onnxruntime as ort

        providers, options = _get_providers_and_options(device)
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=options)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        return self.session.run(None, {self.input_name: x})[0]


class PPOCRConfigError(ValueError):
    """Model and character dictionary don't match; never swallowed by the retry logic."""


class PPOCRRecognizer:
    """PP-OCRv6 preprocessing + CTC decoding around `run`, a callable that maps a
    preprocessed batch to the raw model output: a PPOCRSession in-process, or a request
    to a model worker when this runs in a CPU worker."""

    def __init__(self, char_dict_path, run, img_height=48, img_width_min=320):
        # Lazy import so nafrec still works without ppocrv6_onnx when the PP-OCR path isn't used.
        from ppocrv6_onnx import RecPreProcess, CTCLabelDecode

        self.run = run
        self.pre = RecPreProcess(rec_image_shape=(3, img_height, img_width_min))
        self.post = CTCLabelDecode(char_dict_path)
        self._classes_checked = False

    def _check_classes(self, n_classes):
        # PaddleOCR appends a trailing space class when Global.use_space_char=true
        # (the dict file itself doesn't contain it). Detect this from the model's
        # output size so no extra CLI flag is needed. Needs a model output, so it
        # runs on the first batch instead of at load time.
        if n_classes == self.post.vocab_size + 1:
            self.post._chars = self.post._chars + (" ",)
        if n_classes != self.post.vocab_size:
            raise PPOCRConfigError(
                f"PP-OCR model outputs {n_classes} classes but the character dictionary "
                f"gives {self.post.vocab_size} (blank + dict [+ space]). "
                f"Check --ppocr_char_dict_path."
            )
        self._classes_checked = True

    def _recognize_batch(self, imgs_bgr):
        x = self.pre(imgs_bgr)
        out = self.run(x)
        if not self._classes_checked:
            self._check_classes(out.shape[-1])
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
            except PPOCRConfigError:
                raise
            except Exception as e:
                print(f"[warn] PP-OCR batch failed ({e}) -- retrying its lines individually")
                t, s = [], []
                for im in imgs:
                    try:
                        ti, si = self._recognize_batch([im])
                        t.extend(ti); s.extend(si)
                    except PPOCRConfigError:
                        raise
                    except Exception as line_e:
                        print(f"[warn] line also failed individually, inserting empty placeholder: {line_e}")
                        t.append(""); s.append(0.0)
            for i, ti, si in zip(idxs, t, s):
                texts[i], scores[i] = ti, float(si)
        return scores, texts


def load_ppocr_model(model_path, char_dict_path, device="cpu", img_height=48, img_width_min=320):
    """Self-contained (in-process) recognizer."""
    return PPOCRRecognizer(char_dict_path, PPOCRSession(model_path, device), img_height, img_width_min)


def load_ppocr_session(model_path, device="cpu"):
    """Model worker: the ONNX session only."""
    return PPOCRSession(model_path, device)


def get_ppocr_preds(data, recognizer):
    """Drop-in counterpart of trocr.get_text_preds. `data` is an OCRInput."""
    from .trocr import get_line_dicts

    scores, texts = recognizer.recognize(data.line_images, data.batch_size)
    return get_line_dicts(data.line_polygons, texts, data.line_confs, scores)