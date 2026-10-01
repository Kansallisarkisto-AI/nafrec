import os 
import time

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from collections import Counter
import torchvision.models as models
import torchvision.transforms.v2.functional as TF
from pydantic import BaseModel

from .image_processing import load_with_torchvision

# Stabe parameters used by the cassification model
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 96
NUM_CLASSES = 2
IDX2LABEL = {0: "cyrillic", 1: "latin"}

class ModelBackend:
    def __init__(self, checkpoint_path: str, device: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] requested cuda but no CUDA device is available -- falling back to cpu")
            device = "cpu"
        self.device = torch.device(device)

        model = self.build_mobilenet_v3_small()
        model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to(self.device).eval()
    
    def build_mobilenet_v3_small(self):
        """
        Builds a MobileNetV3-Small model using trained model weights.
        """
        # Load model weights
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        return model
    
    def predict_batch(self, batch_tensor: torch.Tensor):
        """
        batch_tensor: (N, C, H, W). 
        Returns (pred_indices, confidences), each length N.
        """
        with torch.no_grad():
            logits = self.model(batch_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        if self.device.type == "cuda":
            # Forward pass on CUDA is launched asynchronously -- without this,
            # timing around this call would under-report actual GPU time.
            torch.cuda.synchronize()
        return pred.cpu().tolist(), conf.cpu().tolist()

def load_classification_model(model_path: str, device: str):
    """Loads trained PyTorch model using ModelBackend class."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in (".pt", ".pth"):
        raise ValueError(f"Unrecognized model extension: {ext} (expected .pt/.pth)")
    return ModelBackend(model_path, device)

def preprocess_image(image) -> torch.Tensor:
    """
    Converts an image directly into a normalized (C, H, W) float tensor.
    Accepts a raw (H, W, C) uint8 numpy array or a PIL.Image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
 
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) uint8 -> (C, H, W) uint8
 
    tensor = TF.resize(tensor, [IMAGE_HEIGHT, IMAGE_WIDTH], antialias=True)
    tensor = tensor.float() / 255.0
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor

def classify_lines(data, model):
    """
    Returns a list the same length as line_images, in the same order.
 
    A line that fails to preprocess is never dropped: it's assigned a
    fallback result using this call's majority predicted label among
    the lines that did classify successfully. Its
    result["fallback"] is True and result["confidence"] is None.
 
    If every line in the call fails, default_label is used if given.
    """
    results = [None] * len(data.line_images)
    failed_indices = []

    for i in range(0, len(data.line_images), data.batch_size):
        batch_images = data.line_images[i : i + data.batch_size]
        tensors = []
        tensor_indices = []
        for j, image in enumerate(batch_images):
            idx = i + j
            try:
                tensors.append(preprocess_image(image))
                tensor_indices.append(idx)
            except Exception as e:
                print(f"[warn] failed to preprocess line {idx}: {e}")
                failed_indices.append({"idx": idx, "error": e})
        if not tensors:
            continue

        batch_tensor = torch.stack(tensors)
        pred_indices, confidences = model.predict_batch(batch_tensor)

        for idx, pred_idx, confidence in zip(tensor_indices, pred_indices, confidences):
            results[idx] = {
                "predicted_label": IDX2LABEL[pred_idx],
                "confidence": confidence,
                "fallback": False,
                "error": None
            }
        
    if failed_indices:
        successful_labels = [r["predicted_label"] for r in results if r is not None]
        if successful_labels:
            majority_label = Counter(successful_labels).most_common(1)[0][0]
        elif data.default_label is not None:
            majority_label = data.default_label
        else:
            raise ValueError(
                f"All {len(failed_indices)} line(s) in this call failed to preprocess."
            )
        print(
            f"[warn] {len(failed_indices)} line(s) failed preprocessing -- assigned fallback "
            f"label {majority_label} (this call's majority among successful predictions)"
        )
        for item in failed_indices:
            results[item["idx"]] = {"predicted_label": majority_label, "confidence": None, "fallback": True, "error": item["error"]}

    return results
