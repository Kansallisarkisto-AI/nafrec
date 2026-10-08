"""Paddle (ONNX) layout + text line detection, as a drop-in alternative to the RF-DETR model.

PP-DocLayout-style layout models only find *regions* (boxes with a class), not text
lines. nafrec needs both, so this combines two ONNX models:

  * a layout model (e.g. PP-DocLayout-L/M/S exported to ONNX)  -> regions
  * a DB text detection model (e.g. PP-OCRv6 det, via ppocrv6_onnx) -> text lines

`PaddleLayoutDetector.predict_polygons(image_path)` returns exactly what
`seg_inference.predict_polygons` returns:

    (line_polygons, line_confs, line_max_mins,
     region_polygons, region_confs, region_max_mins, image_shape)

i.e. lists of int (N, 2) numpy polygons, float confidences, (xmin, ymin, xmax, ymax)
tuples, and image_shape = (height, width), all in original-image pixel coordinates.
"""
import numpy as np
import cv2

from .ppocr import _get_providers_and_options

DEFAULT_LAYOUT_SIZE = 800  # PP-DocLayout-L / plus-L input size

# With some help for PP-DocLayout_plus-L implementation from
# https://huggingface.co/cimo001/paddle/tree/main/PP-DocLayout_plus-L/src

# Class id -> name for PP-DocLayout_plus-L (used unless layout_labels_path is given;
# other PP-DocLayout variants have different class lists, so pass a labels file for them)
PP_DOCLAYOUT_PLUS_L_LABELS = [
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "reference", "doc_title", "footnote",
    "header", "algorithm", "footer", "seal", "chart", "formula_number",
    "aside_text", "reference_content",
]


def _static_hw(shape):
    """(H, W) from an ONNX input shape like [1, 3, 800, 800], or None if dynamic."""
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return shape[2], shape[3]
    return None


class PaddleLayoutDetector:
    def __init__(
        self,
        layout_model_path,
        det_model_path,
        device="cpu",
        *,
        layout_threshold=0.3,
        layout_input_size=0,
        layout_mean=(0.0, 0.0, 0.0),
        layout_std=(1.0, 1.0, 1.0),
        layout_boxes_in_input_space=False,
        layout_labels_path=None,
        layout_ignore_classes=(),
        det_limit_side_len=1536,
        det_limit_type="max",
        det_max_side_limit=4000,
        det_thresh=0.3,
        det_box_thresh=0.6,
        det_unclip_ratio=1.5,
    ):
        # Lazy imports so nafrec works without onnxruntime / ppocrv6_onnx
        # when this detector isn't used.
        import onnxruntime as ort
        from ppocrv6_onnx import DetPreProcess, DBPostProcess

        providers, options = _get_providers_and_options(device)

        # ---- layout model (optional: without it there are no regions and nafrec
        # falls back to one full-page region) ----
        self.layout = None
        if layout_model_path:
            self.layout = ort.InferenceSession(layout_model_path, providers=providers, sess_options=options)
            inputs = {i.name: i for i in self.layout.get_inputs()}
            self._img_name = "image" if "image" in inputs else next(
                i.name for i in self.layout.get_inputs() if len(i.shape) == 4)
            self._has_im_shape = "im_shape" in inputs
            self._has_scale_factor = "scale_factor" in inputs
            hw = _static_hw(inputs[self._img_name].shape)
            if hw is None:
                size = layout_input_size or DEFAULT_LAYOUT_SIZE
                hw = (size, size)
            elif layout_input_size and (layout_input_size, layout_input_size) != hw:
                print(f"[warn] layout_input_size={layout_input_size} ignored: "
                      f"model has a fixed input size {hw}")
            self._in_h, self._in_w = hw
        self._mean = np.asarray(layout_mean, dtype=np.float32)
        self._std = np.asarray(layout_std, dtype=np.float32)
        self.layout_threshold = layout_threshold
        self.boxes_in_input_space = layout_boxes_in_input_space

        # ---- class names / ignored classes ----
        labels = list(PP_DOCLAYOUT_PLUS_L_LABELS)
        if layout_labels_path:
            with open(layout_labels_path, encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        self.ignore_ids = set()
        for spec in layout_ignore_classes or ():
            spec = str(spec)
            if spec.lstrip("-").isdigit():
                self.ignore_ids.add(int(spec))
            elif spec in labels:
                self.ignore_ids.add(labels.index(spec))
            else:
                raise ValueError(
                    f"layout_ignore_classes: '{spec}' is not an integer id and not in the "
                    f"label list ({layout_labels_path or 'built-in PP-DocLayout_plus-L'}): {labels}")

        # ---- DB text line detection model ----
        self.det = ort.InferenceSession(det_model_path, providers=providers)
        self._det_input = self.det.get_inputs()[0].name
        self._det_pre = DetPreProcess(
            limit_side_len=det_limit_side_len,
            limit_type=det_limit_type,
            max_side_limit=det_max_side_limit,
        )
        self._det_post = DBPostProcess(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
        )

    # ------------------------------------------------------------------
    def _detect_regions(self, img_bgr):
        """Returns list of (class_id, score, x1, y1, x2, y2) in original image pixels."""
        if self.layout is None:
            return []
        h, w = img_bgr.shape[:2]
        ih, iw = self._in_h, self._in_w

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Paddle models expect RGB
        resized = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_CUBIC)
        x = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

        feed = {self._img_name: x}
        if self._has_im_shape:
            feed["im_shape"] = np.array([[ih, iw]], dtype=np.float32)
        if self._has_scale_factor:
            feed["scale_factor"] = np.array([[ih / h, iw / w]], dtype=np.float32)
        outs = self.layout.run(None, feed)

        # detections: (n, 6) = [class_id, score, x1, y1, x2, y2]; an optional second
        # output holds the number of valid rows.
        dets = next((np.asarray(o) for o in outs if np.ndim(o) == 2 and np.shape(o)[1] == 6), None)
        if dets is None:
            raise RuntimeError(
                "Unrecognized layout model output: expected an (N, 6) array "
                f"[class, score, x1, y1, x2, y2], got shapes {[np.shape(o) for o in outs]}")
        counts = [o for o in outs if np.ndim(o) == 1 and np.size(o) == 1]
        if counts:
            dets = dets[: int(counts[0][0])]

        result = []
        for cls, score, x1, y1, x2, y2 in dets.tolist():
            if cls < 0 or score < self.layout_threshold:
                continue
            if self.boxes_in_input_space:
                x1, x2 = x1 * w / iw, x2 * w / iw
                y1, y2 = y1 * h / ih, y2 * h / ih
            x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
            y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            result.append((int(cls), float(score), x1, y1, x2, y2))
        return result

    def _detect_lines(self, img_bgr):
        x, shape = self._det_pre(img_bgr)
        pred = self.det.run(None, {self._det_input: x})[0]
        boxes, scores = self._det_post(pred, shape)  # (N, 4, 2) int16 in original pixels
        return boxes, scores

    # ------------------------------------------------------------------
    def predict_polygons(self, image_path):
        """Same return value as seg_inference.predict_polygons."""
        from .image_processing import load_with_torchvision

        # Load the same way the rest of the pipeline does (torchvision, RGB, no EXIF
        # rotation) so coordinates line up with the later line cropping.
        img_rgb = load_with_torchvision(image_path)
        img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
        h, w = img_bgr.shape[:2]
        image_shape = (h, w)

        regions = self._detect_regions(img_bgr)
        kept = [r for r in regions if r[0] not in self.ignore_ids]
        ignored = [r for r in regions if r[0] in self.ignore_ids]

        region_polygons, region_confs, region_max_mins = [], [], []
        for _, score, x1, y1, x2, y2 in kept:
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            region_polygons.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int))
            region_confs.append(score)
            region_max_mins.append((x1, y1, x2, y2))

        def inside(cx, cy, boxes):
            return any(b[2] <= cx <= b[4] and b[3] <= cy <= b[5] for b in boxes)

        boxes, scores = self._detect_lines(img_bgr)
        line_polygons, line_confs, line_max_mins = [], [], []
        for quad, score in zip(boxes, scores):
            poly = np.asarray(quad).astype(int)
            xmin, ymin = poly.min(axis=0)
            xmax, ymax = poly.max(axis=0)
            if ignored:
                # drop lines (e.g. text inside figures / seals) that sit in an ignored
                # region and are not also covered by a kept region
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                if inside(cx, cy, ignored) and not inside(cx, cy, kept):
                    continue
            line_polygons.append(poly)
            line_confs.append(float(score))
            line_max_mins.append((xmin, ymin, xmax, ymax))

        return (line_polygons, line_confs, line_max_mins,
                region_polygons, region_confs, region_max_mins, image_shape)


def load_paddle_layout_model(layout_model_path, det_model_path, device="cpu", **kwargs):
    """Counterpart of seg_inference.load_rfdetr_model."""
    return PaddleLayoutDetector(layout_model_path, det_model_path, device, **kwargs)