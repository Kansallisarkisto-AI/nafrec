"""Paddle (ONNX) layout + text line detection, as a drop-in alternative to the RF-DETR model.

PP-DocLayout-style layout models only find *regions* (boxes with a class), not text
lines. nafrec needs both, so this combines two ONNX models:

  * a layout model (e.g. PP-DocLayout-L/M/S exported to ONNX)  -> regions
  * a DB text detection model (e.g. PP-OCRv6 det, via ppocrv6_onnx) -> text lines

`PaddleLayoutDetector.predict_polygons(image_path)` returns exactly what
`seg_inference.predict_polygons` returns, after the same polygon post-processing
(small polygon filtering, merging of overlapping polygons, sliver removal; see
`seg_inference.raw_polygons_to_polygons`):

    (line_polygons, line_confs, line_max_mins,
     region_polygons, region_confs, region_max_mins, image_shape)

i.e. lists of int (N, 2) numpy polygons, float confidences, (xmin, ymin, xmax, ymax)
tuples, and image_shape = (height, width), all in original-image pixel coordinates.
"""
import numpy as np
import cv2

from .ppocr import (_get_providers_and_options, _create_session, apply_trt_profile,
                    format_trt_profile, onnx_input_specs)

DEFAULT_LAYOUT_SIZE = 800  # PP-DocLayout-L / plus-L input size
DET_STRIDE = 32            # DetPreProcess rounds both sides to a multiple of this

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


def _pick_image_input(inputs):
    """Name of the image tensor among [(name, shape)] model inputs."""
    names = [n for n, _ in inputs]
    return "image" if "image" in names else next(n for n, shape in inputs if len(shape) == 4)


def _resolve_layout_hw(shape, layout_input_size, warn=True):
    """(H, W) the layout model is fed: its static size, else layout_input_size / the default."""
    hw = _static_hw(shape)
    if hw is None:
        size = layout_input_size or DEFAULT_LAYOUT_SIZE
        return size, size
    if warn and layout_input_size and (layout_input_size, layout_input_size) != hw:
        print(f"[warn] layout_input_size={layout_input_size} ignored: "
              f"model has a fixed input size {hw}")
    return hw


# ---------------------------------------------------------------------------
# TensorRT optimization profiles (dynamic shapes)
# ---------------------------------------------------------------------------
def _round_side(v):
    """Same rounding DetPreProcess applies to each side: nearest multiple of 32, at least 32."""
    return max(int(round(v / DET_STRIDE)) * DET_STRIDE, DET_STRIDE)


def det_side_bounds(limit_side_len, limit_type, max_side_limit):
    """(min_side, max_side) covering both H and W of every tensor DetPreProcess can produce.

    DetPreProcess scales the page so its longest ('max') / shortest ('min') side matches
    limit_side_len, clamps the longest side to max_side_limit, then rounds each side to a
    multiple of 32. Hence:
      * 'max': both sides <= min(limit_side_len, max_side_limit)
      * 'min': the long side is only bounded by max_side_limit
      * either side can be as small as 32 (tiny / extreme-aspect-ratio images)
    The rounding is monotonic, so rounding the bound gives the exact upper limit.
    """
    cap = min(limit_side_len, max_side_limit) if limit_type == "max" else max_side_limit
    return DET_STRIDE, _round_side(cap)


def default_det_opt_shape(limit_side_len, limit_type, max_side_limit):
    """Typical (H, W) of a portrait page: the shape TensorRT tunes its kernels for."""
    _, hi = det_side_bounds(limit_side_len, limit_type, max_side_limit)
    if limit_type == "max":
        long_side = hi
        short_side = _round_side(0.75 * hi)
    else:
        short_side = min(_round_side(limit_side_len), hi)
        long_side = min(_round_side(limit_side_len * 4 / 3), hi)
    return long_side, short_side


def build_trt_det_profile(input_spec, limit_side_len, limit_type, max_side_limit, opt_shape=None):
    """TensorRT profile for the DB text detection model, input (1, C, H, W) with H, W dynamic.

    input_spec: (name, dims) from onnx_input_specs(); dims are ints (static) or None (dynamic).
    opt_shape: optional (H, W) to tune for; default = a portrait page at the resize limit.
    Batch is always 1 (one page per call). Returns None when the model has no dynamic dims
    (nothing to declare), else (input_names, (min, opt, max)) for apply_trt_profile().
    """
    name, dims = input_spec
    if len(dims) != 4:
        raise ValueError(f"det input '{name}' has rank {len(dims)}, expected 4 (N, C, H, W)")
    if all(d is not None for d in dims):
        return None

    lo, hi = det_side_bounds(limit_side_len, limit_type, max_side_limit)
    opt_h, opt_w = opt_shape if opt_shape else default_det_opt_shape(limit_side_len, limit_type, max_side_limit)
    opt_h, opt_w = (min(max(_round_side(v), lo), hi) for v in (opt_h, opt_w))

    def shape(h, w):
        # static dims stay as in the model; dynamic ones: batch 1, 3 channels, H, W
        return tuple(d if d is not None else (1, 3, h, w)[axis] for axis, d in enumerate(dims))

    triple = (shape(lo, lo), shape(opt_h, opt_w), shape(hi, hi))
    return format_trt_profile({name: triple})


def build_trt_layout_profile(input_specs, in_h, in_w):
    """TensorRT profile for the layout model: every dynamic input pinned to what nafrec feeds it.

    nafrec always runs the layout model with batch 1 at a single resolution (in_h x in_w), so
    min = opt = max. The TensorRT EP needs *every* dynamic input in the profile, and Paddle
    exports typically have a symbolic batch dim on `image`, `im_shape` and `scale_factor`.
    input_specs: [(name, dims)] from onnx_input_specs(). Returns None when nothing is dynamic.
    """
    shapes = {}
    for name, dims in input_specs:
        if all(d is not None for d in dims):
            continue
        fixed = []
        for axis, d in enumerate(dims):
            if d is not None:
                fixed.append(d)
            elif axis == 0:
                fixed.append(1)
            elif len(dims) == 4 and axis in (1, 2, 3):
                fixed.append((3, in_h, in_w)[axis - 1])
            else:
                raise ValueError(f"cannot infer dynamic dimension {axis} of layout input '{name}' {dims}")
        shapes[name] = (tuple(fixed),) * 3
    return format_trt_profile(shapes) if shapes else None


class PaddleLayoutDetector:
    def __init__(
        self,
        layout_model_path,
        det_model_path,
        device="cpu",
        *,
        fp16=False,
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
        enable_tensorrt=False,
        trt_dynamic_shapes=True,
        det_trt_opt_shape=None,
    ):
        # Lazy imports so nafrec works without onnxruntime / ppocrv6_onnx
        # when this detector isn't used.
        from ppocrv6_onnx import DetPreProcess, DBPostProcess

        # TensorRT: per-model providers because each model gets its own shape profile and
        # cache folder. Without a profile the TensorRT EP rebuilds an engine for every new
        # input shape, so dynamic shapes matter (see build_trt_*_profile).
        use_trt_profiles = enable_tensorrt and trt_dynamic_shapes and str(device).startswith("cuda")

        # ---- layout model (optional: without it there are no regions and nafrec
        # falls back to one full-page region) ----
        self.layout = None
        if layout_model_path:
            layout_profile = None
            if use_trt_profiles:
                layout_profile = self._layout_trt_profile(layout_model_path, layout_input_size)
            providers, options = _get_providers_and_options(
                device, layout_profile, enable_tensorrt, trt_cache_subdir="layout", label="layout model")
            self.layout = _create_session(layout_model_path, providers, options)  # ignore fp16 here due to incompatibility
            inputs = {i.name: i for i in self.layout.get_inputs()}
            self._img_name = _pick_image_input([(n, i.shape) for n, i in inputs.items()])
            self._has_im_shape = "im_shape" in inputs
            self._has_scale_factor = "scale_factor" in inputs
            self._in_h, self._in_w = _resolve_layout_hw(inputs[self._img_name].shape, layout_input_size)
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
        det_profile = None
        if use_trt_profiles:
            det_profile = self._det_trt_profile(det_model_path, det_limit_side_len, det_limit_type,
                                                det_max_side_limit, det_trt_opt_shape)
        det_providers, _ = _get_providers_and_options(
            device, det_profile, enable_tensorrt, trt_cache_subdir="det", label="text detection model")
        self.det = _create_session(det_model_path, det_providers)  # ignore fp16 here due to incompatibility
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
    @staticmethod
    def _layout_trt_profile(model_path, layout_input_size):
        """Shape profile for the layout model, or None (with a warning) if it can't be built."""
        try:
            specs = onnx_input_specs(model_path)
            img_shape = dict(specs)[_pick_image_input(specs)]
            in_h, in_w = _resolve_layout_hw(img_shape, layout_input_size, warn=False)
            profile = build_trt_layout_profile(specs, in_h, in_w)
        except Exception as e:
            print(f"[warn] could not build a TensorRT shape profile for the layout model ({e}) -- "
                  "TensorRT will rebuild its engine if the input shape changes")
            return None
        if profile is not None:
            print(f"[info] TensorRT layout profile: {profile[1][1]}")
        return profile

    @staticmethod
    def _det_trt_profile(model_path, limit_side_len, limit_type, max_side_limit, opt_shape):
        """Shape profile for the text detection model, or None (with a warning) if it can't be built."""
        try:
            profile = build_trt_det_profile(onnx_input_specs(model_path)[0], limit_side_len,
                                            limit_type, max_side_limit, opt_shape)
        except Exception as e:
            print(f"[warn] could not build a TensorRT shape profile for the text detection model ({e}) -- "
                  "TensorRT will rebuild its engine for every new page size")
            return None
        if profile is not None:
            mn, opt, mx = profile[1]
            print(f"[info] TensorRT det profile: min={mn} opt={opt} max={mx}")
        return profile

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
    def predict_raw(self, image_path):
        """
        Model (GPU) part of polygon prediction: the raw, not yet post-processed detections.
        Counterpart of seg_inference.predict_masks; the CPU post-processing (merging, filtering)
        is done by seg_inference.raw_polygons_to_polygons.

        Returns:
            (line_polygons, line_confs, region_polygons, region_confs, image_shape): lists of int
            (N, 2) numpy polygons and float confidences, image_shape = (height, width), all in
            original-image pixel coordinates.
        """
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

        region_polygons, region_confs = [], []
        for _, score, x1, y1, x2, y2 in kept:
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            region_polygons.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int))
            region_confs.append(score)

        def inside(cx, cy, boxes):
            return any(b[2] <= cx <= b[4] and b[3] <= cy <= b[5] for b in boxes)

        boxes, scores = self._detect_lines(img_bgr)
        line_polygons, line_confs = [], []
        for quad, score in zip(boxes, scores):
            poly = np.asarray(quad).astype(int)
            if ignored:
                # drop lines (e.g. text inside figures / seals) that sit in an ignored
                # region and are not also covered by a kept region
                xmin, ymin = poly.min(axis=0)
                xmax, ymax = poly.max(axis=0)
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                if inside(cx, cy, ignored) and not inside(cx, cy, kept):
                    continue
            line_polygons.append(poly)
            line_confs.append(float(score))

        return line_polygons, line_confs, region_polygons, region_confs, image_shape

    def predict_polygons(self, image_path,
                         line_percentage_threshold=7e-05,
                         region_percentage_threshold=7e-05,
                         line_iou=0.3,
                         region_iou=0.3,
                         line_overlap_threshold=0.5,
                         region_overlap_threshold=0.5,
                         sliver_threshold=0.01):
        """
        Convenience wrapper doing both steps in the calling process: `predict_raw` followed by
        seg_inference.raw_polygons_to_polygons. Same arguments and return value as
        seg_inference.predict_polygons. The pipeline in main.py calls the two steps separately
        so that post-processing runs in the CPU workers instead of the model workers.
        """
        from .seg_inference import raw_polygons_to_polygons

        line_polygons, line_confs, region_polygons, region_confs, image_shape = self.predict_raw(image_path)
        return raw_polygons_to_polygons(
            line_polygons, line_confs, region_polygons, region_confs, image_shape,
            line_percentage_threshold=line_percentage_threshold,
            region_percentage_threshold=region_percentage_threshold,
            line_iou=line_iou, region_iou=region_iou,
            line_overlap_threshold=line_overlap_threshold,
            region_overlap_threshold=region_overlap_threshold,
            sliver_threshold=sliver_threshold)


def load_paddle_layout_model(layout_model_path, det_model_path, device="cpu", **kwargs):
    """Counterpart of seg_inference.load_rfdetr_model."""
    return PaddleLayoutDetector(layout_model_path, det_model_path, device, **kwargs)