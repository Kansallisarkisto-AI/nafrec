from pathlib import Path
 
import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, read_image
 
# Keep OpenCV from fighting with DataLoader workers; harmless otherwise.
# For single-process use you can remove this or call cv2.setNumThreads(n).
cv2.setNumThreads(0)
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _target_size(h, w, max_size, mode="smaller"):
    """
    Return (new_h, new_w), or None if no resize is needed.
 
    mode="smaller": the smaller edge is capped at max_size (larger edge free).
    mode="longer":  the longer edge is capped at max_size.
    """
    ref = min(h, w) if mode == "smaller" else max(h, w)
    if ref <= max_size:
        return None
    s = max_size / ref
    return max(1, round(h * s)), max(1, round(w * s))
 
 
def _to_chw_tensor(image):
    """numpy (H, W, C) / torch (C, H, W) -> torch (C, H, W), no extra copies if avoidable."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return image
 
 
def worker_init_fn(_worker_id):
    """Pass as DataLoader(worker_init_fn=...) to avoid thread oversubscription."""
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
 
 
# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_with_torchvision(img_path):
    """Original-style loader: returns contiguous (H, W, C) uint8 RGB numpy array."""
    img_tensor = read_image(str(img_path), mode=ImageReadMode.RGB)
    # ascontiguousarray avoids hidden copies in later cv2 / torch calls
    return np.ascontiguousarray(img_tensor.permute(1, 2, 0).numpy())
 
 
def load_resized_fast(img_path, max_size=1024, mode="smaller", interpolation=cv2.INTER_AREA):
    """
    Load an image and downsize it as cheaply as possible.
 
    For JPEGs, PIL's draft() makes the decoder downscale during decode (by 1/2, 1/4, 1/8),
    which is far cheaper than decoding full-res and resizing. draft() never goes below the
    requested size, so the final cv2 resize still has enough resolution.
 
    Returns: uint8 numpy array (H, W, 3), RGB.
    """
    with Image.open(str(img_path)) as im:
        w, h = im.size
        target = _target_size(h, w, max_size, mode)
        if target is not None:
            th, tw = target
            im.draft("RGB", (tw, th))  # no-op for non-JPEG formats
        arr = np.asarray(im.convert("RGB"))  # contiguous HWC uint8
 
    # Final exact resize (draft only gets within a power-of-two factor)
    h, w = arr.shape[:2]
    target = _target_size(h, w, max_size, mode)
    if target is not None:
        th, tw = target
        arr = cv2.resize(arr, (tw, th), interpolation=interpolation)
    return arr
 
 
# --------------------------------------------------------------------------
# Preprocessing (drop-in replacements for the originals)
# --------------------------------------------------------------------------
def _preprocess(image, max_size, normalize, mode, backend, interpolation):
    # --- cv2 path: numpy in, stay in numpy until the very end ---
    if backend == "cv2" and isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = image[:, :, None]
        h, w = image.shape[:2]
        target = _target_size(h, w, max_size, mode)
        if target is not None:
            th, tw = target
            image = cv2.resize(image, (tw, th), interpolation=interpolation)
            if image.ndim == 2:  # cv2 drops the channel dim for 1-channel input
                image = image[:, :, None]
        out = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        if normalize:
            out = out.float().div_(255)  # in-place divide on the small image
        return out
 
    # --- torchvision path (tensor or numpy input) ---
    if isinstance(image, torch.Tensor):
        _, h, w = image.shape
    elif isinstance(image, np.ndarray):
        image = _to_chw_tensor(image)
        _, h, w = image.shape
    else:  # PIL
        w, h = image.size
 
    target = _target_size(h, w, max_size, mode)
    if target is not None:
        image = F.resize(image, list(target), antialias=True)  # uint8 resize
    if normalize:
        image = F.to_dtype(image, torch.float32, scale=True)   # float conversion last
    return image
 
 
def preprocess_resize_smallerof_wh_torch_transform(
    image, max_size=1024, normalize=True, backend="cv2", interpolation=cv2.INTER_AREA
):
    """
    Resize so the *smaller* of (H, W) is at most max_size (larger edge unconstrained).
 
    backend="cv2":         fastest on CPU; numpy input stays numpy until the end.
    backend="torchvision": antialiased bilinear, matches the original numerics.
    Returns a torch.Tensor (C, H, W) (or PIL for PIL input on the torchvision backend).
    """
    return _preprocess(image, max_size, normalize, "smaller", backend, interpolation)
 
 
def preprocess_resize_torch_transform(
    image, max_size=1024, normalize=True, backend="cv2", interpolation=cv2.INTER_AREA
):
    """Resize so the *longer* of (H, W) is at most max_size."""
    return _preprocess(image, max_size, normalize, "longer", backend, interpolation)
 
 
def load_and_preprocess(path, max_size=1024, normalize=True, mode="smaller",
                        interpolation=cv2.INTER_AREA):
    """One-call fast path: draft-decode + resize + (optional) float conversion."""
    arr = load_resized_fast(path, max_size=max_size, mode=mode, interpolation=interpolation)
    out = torch.from_numpy(arr).permute(2, 0, 1)
    if normalize:
        out = out.float().div_(255)
    return out

def upscale_mask_opencv(mask, bbox, upscaled_bbox_shape):
    """Upscale using OpenCV resize with nearest neighbor."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_mask = mask[y1:y2, x1:x2]
    mask_uint8 = cropped_mask.astype(np.uint8)
    upscaled = cv2.resize(mask_uint8, 
                         upscaled_bbox_shape, 
                         interpolation=cv2.INTER_NEAREST)

    return upscaled * 255

def upscale_bbox(bbox, original_shape, mask_shape):
    """
    Upscale bounding box coordinates from mask resolution to original image resolution.

    Parameters:
    -----------
    bbox : np.ndarray or list
        Bounding box coordinates in format [x_min, y_min, x_max, y_max]
        in the mask's coordinate system
    original_shape : tuple
        Original image shape (H, W) or (H, W, C) - e.g., (4545, 5527, 3)
    mask_shape : tuple
        Mask shape (H, W) - e.g., (631, 768)

    Returns:
    --------
    np.ndarray
        Upscaled bounding box as integer coordinates [x_min, y_min, x_max, y_max]
    """

    # Ensure bbox is a numpy array
    bbox = np.array(bbox)

    # Extract height and width from shapes
    original_h, original_w = original_shape[0], original_shape[1]
    mask_h, mask_w = mask_shape[0], mask_shape[1]

    # Calculate scale factors
    scale_x = original_w / mask_w  # Width scaling
    scale_y = original_h / mask_h  # Height scaling

    # Unpack bbox coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates
    x_min_scaled = x_min * scale_x
    y_min_scaled = y_min * scale_y
    x_max_scaled = x_max * scale_x
    y_max_scaled = y_max * scale_y

    # limit to range 0 to original width/height
    if x_min_scaled < 0:
        x_min_scaled = 0
    if y_min_scaled < 0:
        y_min_scaled = 0
    if x_max_scaled > original_w:
        x_max_scaled = original_w
    if y_max_scaled > original_h:
        y_max_scaled = original_h

    # Convert to integers (rounding to nearest)
    bbox_scaled = np.array([
        x_min_scaled,
        y_min_scaled,
        x_max_scaled,
        y_max_scaled
    ]).astype(np.int32)

    return bbox_scaled

def crop_line(image, polygon):
    """Crops predicted text line based on the polygon coordinates
    and returns pasted text line image with white background.

    Args:
        image: Input image array.
        polygon: List of coordinate pairs defining the text line polygon.

    Returns:
        numpy.ndarray: Cropped text line image with white background around polygon.
    """
    polygon = np.array([[int(lst[0]), int(lst[1])] for lst in polygon], dtype=np.int32)
    rect = cv2.boundingRect(polygon)
    cropped_image = image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    mask = np.zeros([cropped_image.shape[0], cropped_image.shape[1]], dtype=np.uint8)
    cv2.drawContours(mask, [polygon- np.array([[rect[0],rect[1]]])], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(cropped_image, cropped_image, mask = mask)
    wbg = np.ones_like(cropped_image, np.uint8)*255
    cv2.bitwise_not(wbg,wbg, mask=mask)
    # Overlap the resulted cropped image on the white background
    dst = wbg+res
    return dst

def crop_lines(polygons, image):
    """Returns a list of line images cropped following the 
    detected polygon coordinates.
    
    Crop multiple text lines from an image based on polygon coordinates.

    Args:
        polygons: List of polygons, each containing coordinate pairs.
        image: Input image array.

    Returns:
        list: List of cropped text line images.
    """
    cropped_lines = []
    for polygon in polygons:
        cropped_line = crop_line(image, polygon)
        cropped_lines.append(cropped_line)
    return cropped_lines