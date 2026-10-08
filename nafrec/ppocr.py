"""PP-OCRv6 recognition-only inference (ONNX Runtime) for nafrec.

Takes already-cropped text line images (as produced by image_processing.crop_lines)
and returns results in the same line-dict format as trocr.get_text_preds, so the
rest of the pipeline (XML / JSON output) doesn't need to know which model was used.
"""
import numpy as np
import cv2


def _get_providers(device):
    """Map a torch-style device string ('cuda', 'cuda:1', 'cpu') to ORT providers.
    Note: torch uses 'cuda' for ROCm builds too, so 'cuda' here means "the GPU"."""
    import onnxruntime as ort

    device = str(device)
    if not device.startswith("cuda"):
        return ["CPUExecutionProvider"]

    device_id = int(device.split(":")[1]) if ":" in device else 0
    available = ort.get_available_providers()

    for name in ("CUDAExecutionProvider",        # NVIDIA
                 "MIGraphXExecutionProvider",    # AMD, ORT >= 1.23
                 "ROCMExecutionProvider"):       # AMD, ORT <= 1.22 (legacy)
        if name in available:
            return [(name, {"device_id": device_id}), "CPUExecutionProvider"]

    print("[warn] no GPU execution provider found in onnxruntime "
          f"(available: {available}) -- PP-OCR will run on CPU")
    return ["CPUExecutionProvider"]


class PPOCRRecognizer:
    """Wraps an ONNX PP-OCRv6 recognition model + preprocessing + CTC decoding."""

    def __init__(self, model_path, char_dict_path, device="cpu", img_height=48, img_width_min=320):
        # Lazy imports so nafrec still works without onnxruntime / ppocrv6_onnx
        # when the PP-OCR path isn't used.
        import onnxruntime as ort
        from ppocrv6_onnx import RecPreProcess, CTCLabelDecode

        self.session = ort.InferenceSession(model_path, providers=_get_providers(device))
        self.input_name = self.session.get_inputs()[0].name

        self.pre = RecPreProcess(rec_image_shape=(3, img_height, img_width_min))
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


def load_ppocr_model(model_path, char_dict_path, device="cpu", img_height=48, img_width_min=320):
    return PPOCRRecognizer(model_path, char_dict_path, device, img_height, img_width_min)


def get_ppocr_preds(data, recognizer):
    """Drop-in counterpart of trocr.get_text_preds. `data` is an OCRInput."""
    from .trocr import get_line_dicts

    scores, texts = recognizer.recognize(data.line_images, data.batch_size)
    return get_line_dicts(data.line_polygons, texts, data.line_confs, scores)