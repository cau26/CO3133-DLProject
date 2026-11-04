"""One bounded image-to-logits path, shared by Python, export checks and API."""
from __future__ import annotations

import io
import math
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_image(data: bytes) -> Image.Image:
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("Image must contain 1 to 5 MiB of bytes.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Only PNG, JPEG and WEBP are supported.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel input limit.")
                source.load()
                return ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Image cannot be decoded safely.") from exc


def preprocess_image(data: bytes, manifest: dict, crop=None):
    """Crop coordinates refer to the EXIF-oriented original; resize by letterbox."""
    image = decode_image(data)
    original = {"width": image.width, "height": image.height, "mode": image.mode, "crop": None}
    prep = manifest["preprocessing"]
    if manifest["domain"] == "fashion_mnist":
        if crop is not None or image.size != (28, 28) or image.mode != "L":
            raise ValueError("Reference accepts only original 28x28 grayscale images, without cropping.")
        if prep != {"version": "fashion_mnist_totensor_v1", "height": 28, "width": 28,
                    "channels": 1, "mode": "L", "resize": "none", "normalization": "zero_one"}:
            raise ValueError("Unsupported reference preprocessing contract.")
        array = np.array(image, dtype=np.float32)[None] / 255.0
    else:
        if prep.get("version") != "garment_rgb_letterbox_v1":
            raise ValueError("Unsupported garment preprocessing version.")
        if crop is None:
            raise ValueError("Select a crop containing one supported garment.")
        if (not isinstance(crop, (list, tuple)) or len(crop) != 4
                or any(type(v) is not int for v in crop)):
            raise ValueError("Crop must be four integer pixel coordinates.")
        left, top, right, bottom = crop
        if not 0 <= left < right <= image.width or not 0 <= top < bottom <= image.height:
            raise ValueError("Crop must lie inside the EXIF-oriented image.")
        if right - left < 8 or bottom - top < 8:
            raise ValueError("Crop must be at least 8x8 pixels.")
        original["crop"] = list(crop)
        if (prep.get("resize") != "letterbox_bilinear" or prep.get("mode") != "RGB"
                or prep.get("channels") != 3 or prep.get("fill") != [0, 0, 0]
                or prep.get("normalization") != "imagenet_v1"):
            raise ValueError("Unsupported garment preprocessing contract.")
        height, width = prep["height"], prep["width"]
        if any(type(v) is not int or not 32 <= v <= 512 for v in [height, width]):
            raise ValueError("Input resolution must be within 32 to 512 pixels.")
        image = image.convert("RGB").crop(crop)
        scaled = ImageOps.contain(image, (width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        array = np.array(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        array = (array - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    if not torch.isfinite(tensor).all():
        raise ValueError("Preprocessed image is not finite.")
    return tensor, original


class Predictor:
    """CPU-only draft. Uncalibrated reference always requests human review."""

    def __init__(self, release_dir: str | Path):
        from .release import validate_release
        checked = validate_release(release_dir, require_ready=False)
        manifest = checked["manifest"]
        if manifest["domain"] == "garment_crop_rgb":
            checked = validate_release(release_dir, require_ready=True)
        self.manifest = manifest
        if manifest["architecture"] == "cnn_c5":
            from a1.cnn import CustomCNN
            self.model = CustomCNN(dropout=0.3)
        else:
            # No model download or ImageNet fallback during serving.
            from torchvision.models import mobilenet_v3_small
            self.model = mobilenet_v3_small(weights=None, num_classes=len(manifest["labels"]))
            from a1.cnn import GlobalAveragePool2d
            self.model.avgpool = GlobalAveragePool2d()
        weights = torch.load(checked["weights_path"], map_location="cpu", weights_only=True)
        state = weights.get("model_state_dict", weights)
        self.model.load_state_dict(state, strict=True)
        for value in self.model.state_dict().values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Model contains non-finite weights.")
        self.model.eval()
        self.model.requires_grad_(False)
        self._lock = threading.Lock()
        calibration = manifest.get("calibration", {})
        self.calibrated = manifest["status"] == "ready" and calibration.get("status") == "fitted"
        self.temperature = calibration.get("temperature", 1.0) if self.calibrated else 1.0
        self.threshold = calibration.get("review_threshold") if self.calibrated else None
        self.review_mode = calibration.get("policy", {}).get("mode", "confidence_threshold")
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (float, int))
                or not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("Temperature must be finite and positive.")
        if self.calibrated and (isinstance(self.threshold, bool) or
                not isinstance(self.threshold, (float, int)) or not 0 <= self.threshold <= 1):
            raise ValueError("Calibrated model requires a review threshold in [0,1].")
        self.metadata = {
            "model_version": manifest["release_id"], "domain": manifest["domain"],
            "release_status": manifest["status"], "architecture": manifest["architecture"],
            "labels": manifest["labels"], "preprocessing": manifest["preprocessing"],
            "calibrated": self.calibrated, "weights_sha256": manifest["weights"]["sha256"],
            "review_policy": self.review_mode if self.calibrated else "uncalibrated_reference",
            "scope": "reference benchmark only" if manifest["domain"] == "fashion_mnist" else "selected garment crops",
        }

    def predict_bytes(self, data: bytes, crop=None) -> dict:
        started = time.perf_counter()
        tensor, input_info = preprocess_image(data, self.manifest, crop)
        decoded = time.perf_counter()
        with self._lock, torch.inference_mode():
            logits = self.model(tensor)
        forwarded = time.perf_counter()
        if logits.shape != (1, len(self.manifest["labels"])) or not torch.isfinite(logits).all():
            raise RuntimeError("Model output failed shape or finite-value checks.")
        # Use exactly the same arithmetic as threshold selection, including ties.
        from .probabilities import log_probabilities
        try:
            probs = np.exp(log_probabilities(logits.detach().cpu().numpy(), self.temperature))[0].tolist()
        except ValueError as error:
            raise RuntimeError("Model probabilities are numerically unusable.") from error
        rows = [{"index": i, "label": name, "probability": prob}
                for i, (name, prob) in enumerate(zip(self.manifest["labels"], probs))]
        best = max(rows, key=lambda row: row["probability"])
        review_mode = getattr(self, "review_mode", "confidence_threshold")
        review_required = (not self.calibrated or review_mode == "review_all"
                           or best["probability"] < self.threshold)
        return {
            "model_version": self.manifest["release_id"], "domain": self.manifest["domain"],
            "release_status": self.manifest["status"], "calibrated": self.calibrated,
            "review_required": review_required,
            "review_policy": review_mode if self.calibrated else "uncalibrated_reference",
            "review_reason": ("insufficient_risk_control_evidence" if review_mode == "review_all"
                              else "below_threshold_or_uncalibrated" if review_required else None),
            "predicted_class": best, "probabilities": rows, "input": input_info,
            "preprocessing": self.manifest["preprocessing"], "scope": self.metadata["scope"],
            "timing_ms": {"decode_preprocess": (decoded - started) * 1000,
                          "forward": (forwarded - decoded) * 1000,
                          "total": (time.perf_counter() - started) * 1000},
        }
