"""Input and inference contracts using synthetic images and untrained local weights.

These tests check software behavior, not classifier accuracy or calibration quality.
"""
import concurrent.futures
import hashlib
import io
import json
import math
import threading

import numpy as np
import pytest
import torch
from PIL import Image

from a1.cnn import CustomCNN
from garmentlab import inference
from garmentlab.inference import Predictor, decode_image, preprocess_image


def image_bytes(image, format="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=format, **kwargs)
    return stream.getvalue()


def manifest(domain="fashion_mnist", status=None):
    reference = domain == "fashion_mnist"
    return {
        "format_version": 1, "release_id": "synthetic-inference-test", "domain": domain,
        "architecture": "cnn_c5" if reference else "mobilenet_v3_small",
        "status": status or ("reference" if reference else "candidate"),
        "labels": [f"class_{index}" for index in range(10)] if reference else ["shirt", "trousers"],
        "preprocessing": ({"version": "fashion_mnist_totensor_v1", "height": 28, "width": 28,
                           "channels": 1, "mode": "L", "resize": "none", "normalization": "zero_one"}
                          if reference else
                          {"version": "garment_rgb_letterbox_v1", "height": 64, "width": 64,
                           "channels": 3, "mode": "RGB", "resize": "letterbox_bilinear",
                           "fill": [0, 0, 0], "normalization": "imagenet_v1"}),
    }


def write_release(directory, description, weights):
    directory.mkdir()
    torch.save(weights, directory / "weights.pt")
    description["weights"] = {"file": "weights.pt",
                              "sha256": hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest()}
    (directory / "release.json").write_text(json.dumps(description), encoding="utf-8")
    return directory


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reference_release(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = CustomCNN(dropout=0.3)
    return write_release(tmp_path / "reference", manifest(), {"model_state_dict": model.state_dict()})


@pytest.mark.parametrize("value", [b"", b"not an image", "text is not bytes", bytearray(b"PNG")])
def test_decode_rejects_empty_malformed_and_wrong_input_types(value):
    with pytest.raises(ValueError):
        decode_image(value)


def test_decode_enforces_encoded_byte_limit():
    with pytest.raises(ValueError, match="MiB"):
        decode_image(b"x" * (inference.MAX_BYTES + 1))


def test_decode_enforces_pixel_limit_before_loading_large_image(monkeypatch):
    encoded = image_bytes(Image.new("RGB", (20, 20), "red"))
    assert inference.MAX_PIXELS == 16_000_000
    monkeypatch.setattr(inference, "MAX_PIXELS", 399)
    with pytest.raises(ValueError, match="megapixel"):
        decode_image(encoded)


def test_pillow_decompression_bomb_is_converted_to_invalid_input(monkeypatch):
    encoded = image_bytes(Image.new("RGB", (20, 20), "red"))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ValueError, match="decoded safely"):
        decode_image(encoded)


def test_unsupported_format_and_animated_png_are_rejected():
    with pytest.raises(ValueError, match="Only PNG"):
        decode_image(image_bytes(Image.new("RGB", (12, 12), "red"), format="BMP"))
    animated = image_bytes(Image.new("RGB", (12, 12), "red"), save_all=True,
                           append_images=[Image.new("RGB", (12, 12), "blue")], duration=100, loop=0)
    with pytest.raises(ValueError, match="Animated"):
        decode_image(animated)


def test_reference_preprocessing_preserves_every_original_pixel():
    pixels = (np.arange(28 * 28, dtype=np.uint16).reshape(28, 28) % 256).astype(np.uint8)
    tensor, info = preprocess_image(image_bytes(Image.fromarray(pixels)), manifest())
    expected = pixels.astype(np.float32)[None, None] / np.float32(255)
    assert tensor.shape == (1, 1, 28, 28) and tensor.dtype == torch.float32
    np.testing.assert_array_equal(tensor.numpy(), expected)
    assert info == {"width": 28, "height": 28, "mode": "L", "crop": None}


@pytest.mark.parametrize("image,crop", [(Image.new("RGB", (28, 28)), None),
                                       (Image.new("L", (29, 28)), None),
                                       (Image.new("L", (28, 28)), [0, 0, 28, 28])])
def test_reference_refuses_rgb_wrong_shape_and_cropping(image, crop):
    with pytest.raises(ValueError, match="original 28x28 grayscale"):
        preprocess_image(image_bytes(image), manifest(), crop=crop)


def test_reference_rejects_a_changed_preprocessing_contract():
    description = manifest()
    description["preprocessing"]["normalization"] = "imagenet_v1"
    with pytest.raises(ValueError, match="reference preprocessing"):
        preprocess_image(image_bytes(Image.new("L", (28, 28))), description)


@pytest.mark.parametrize("crop", [None, [0, 0, 20], [False, 0, 20, 20], [0.0, 0, 20, 20],
                                  [-1, 0, 20, 20], [0, 0, 21, 20], [0, 0, 0, 20], [0, 0, 7, 20]])
def test_real_domain_requires_a_valid_nontrivial_crop(crop):
    with pytest.raises(ValueError):
        preprocess_image(image_bytes(Image.new("RGB", (20, 20), "red")),
                         manifest("garment_crop_rgb"), crop=crop)


@pytest.mark.parametrize("size,content_slice,fill_slice", [((40, 20), (slice(16, 48), slice(None)), (0, 0)),
                                                         ((20, 40), (slice(None), slice(16, 48)), (0, 0))])
def test_real_letterbox_preserves_aspect_ratio_and_adds_padding(size, content_slice, fill_slice):
    tensor, info = preprocess_image(image_bytes(Image.new("RGB", size, "red")),
                                    manifest("garment_crop_rgb"), crop=[0, 0, *size])
    normalized = tensor.numpy()[0]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    reconstructed = normalized * std + mean
    np.testing.assert_allclose(reconstructed[:, fill_slice[0], fill_slice[1]], [0, 0, 0], atol=1e-6)
    content = reconstructed[:, content_slice[0], content_slice[1]]
    np.testing.assert_allclose(content[0], 1, atol=1e-6)
    np.testing.assert_allclose(content[1:], 0, atol=1e-6)
    assert int(np.count_nonzero(reconstructed[0] > 0.9)) == 32 * 64
    assert info["crop"] == [0, 0, *size]


def test_crop_coordinates_use_exif_oriented_dimensions():
    source = Image.new("RGB", (20, 10), "red")
    exif = source.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise; displayed size becomes 10x20.
    encoded = image_bytes(source, exif=exif)
    assert decode_image(encoded).size == (10, 20)
    tensor, info = preprocess_image(encoded, manifest("garment_crop_rgb"), crop=[0, 12, 10, 20])
    assert info["width"] == 10 and info["height"] == 20
    assert tensor.shape == (1, 3, 64, 64)


def test_reference_model_is_frozen_eval_and_threaded_predictions_are_repeatable(reference_release):
    predictor = Predictor(reference_release)
    assert not predictor.model.training
    assert all(not layer.training for layer in predictor.model.modules())
    assert not any(parameter.requires_grad for parameter in predictor.model.parameters())
    encoded = image_bytes(Image.new("L", (28, 28), 120))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: predictor.predict_bytes(encoded), range(6)))
    assert all(result["probabilities"] == results[0]["probabilities"] for result in results)
    assert all(result["review_required"] is True and result["calibrated"] is False for result in results)
    assert math.isclose(sum(row["probability"] for row in results[0]["probabilities"]), 1, abs_tol=1e-6)
    assert all(value >= 0 for value in results[0]["timing_ms"].values())
    assert results[0]["scope"] == "reference benchmark only"


def test_predictor_uses_weights_only_loading(reference_release, monkeypatch):
    actual_load = torch.load
    calls = []
    def record_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(inference.torch, "load", record_load)
    Predictor(reference_release)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_predictor_rejects_invalid_checkpoint_shapes_and_nonfinite_weights(tmp_path):
    for index, kind in enumerate(("shape", "nonfinite")):
        state = CustomCNN().state_dict()
        if kind == "shape":
            state["classifier.2.weight"] = torch.zeros(9, 128)
        else:
            state["classifier.2.weight"][0, 0] = float("nan")
        release = write_release(tmp_path / f"bad-{index}", manifest(), state)
        with pytest.raises((RuntimeError, ValueError)):
            Predictor(release)


def test_real_candidate_and_unbound_ready_flag_are_blocked_before_model_loading(tmp_path):
    for index, status in enumerate(("candidate", "ready")):
        release = write_release(tmp_path / f"real-{index}", manifest("garment_crop_rgb", status), {})
        with pytest.raises(ValueError):
            Predictor(release)


class FixedLogits(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor([values], dtype=torch.float32))

    def forward(self, images):
        return self.values


def test_temperature_and_review_threshold_apply_to_the_same_probability_distribution():
    # Isolate output policy with synthetic logits; this does not claim a fitted model.
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = manifest("garment_crop_rgb", "ready")
    predictor.calibrated = True
    predictor.temperature = 2.0
    predictor.threshold = 0.8
    predictor.model = FixedLogits([2.0, 0.0]).eval()
    predictor._lock = threading.Lock()
    predictor.metadata = {"scope": "synthetic output-policy test"}
    result = predictor.predict_bytes(image_bytes(Image.new("RGB", (20, 20), "red")), crop=[0, 0, 20, 20])
    expected = torch.softmax(torch.tensor([2.0, 0.0], dtype=torch.float64) / 2.0, dim=0).tolist()
    assert [row["probability"] for row in result["probabilities"]] == pytest.approx(expected, abs=1e-15)
    assert result["predicted_class"]["label"] == "shirt"
    assert result["review_required"] is True


@pytest.mark.parametrize("logits", [[float("nan"), 0.0], [0.0, 0.0, 0.0]])
def test_nonfinite_or_wrong_class_count_outputs_fail_closed(logits):
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = manifest("garment_crop_rgb", "ready")
    predictor.calibrated, predictor.temperature, predictor.threshold = True, 1.0, 0.8
    predictor.model = FixedLogits(logits).eval()
    predictor._lock = threading.Lock()
    predictor.metadata = {"scope": "synthetic output-policy test"}
    with pytest.raises(RuntimeError, match="shape or finite"):
        predictor.predict_bytes(image_bytes(Image.new("RGB", (20, 20), "red")), crop=[0, 0, 20, 20])


def test_review_threshold_one_does_not_accept_confidence_rounded_to_one_in_float32():
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = manifest("garment_crop_rgb", "ready")
    predictor.calibrated, predictor.temperature, predictor.threshold = True, 1.0, 1.0
    predictor.model = FixedLogits([20.0, 0.0]).eval()
    predictor._lock = threading.Lock()
    predictor.metadata = {"scope": "synthetic threshold precision boundary"}
    result = predictor.predict_bytes(image_bytes(Image.new("RGB", (20, 20), "red")), crop=[0, 0, 20, 20])
    assert result["probabilities"][0]["probability"] < 1.0
    assert result["review_required"] is True


def test_finite_logits_with_numerically_unusable_temperature_fail_closed():
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = manifest("garment_crop_rgb", "ready")
    predictor.calibrated, predictor.temperature, predictor.threshold = True, 1e-320, 0.8
    predictor.model = FixedLogits([2.0, 0.0]).eval()
    predictor._lock = threading.Lock()
    predictor.metadata = {"scope": "synthetic numeric-boundary test"}
    with pytest.raises(RuntimeError, match="finite|probabilit"):
        predictor.predict_bytes(image_bytes(Image.new("RGB", (20, 20), "red")), crop=[0, 0, 20, 20])


def test_explicit_review_all_rejects_even_float64_confidence_equal_to_one():
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = manifest("garment_crop_rgb", "ready")
    predictor.calibrated, predictor.temperature, predictor.threshold = True, 1.0, 1.0
    predictor.review_mode = "review_all"
    predictor.model = FixedLogits([1000.0, 0.0]).eval()
    predictor._lock = threading.Lock()
    predictor.metadata = {"scope": "synthetic review-all boundary"}
    result = predictor.predict_bytes(image_bytes(Image.new("RGB", (20, 20), "red")), crop=[0, 0, 20, 20])
    assert result["predicted_class"]["probability"] == 1.0
    assert result["review_required"] is True
    assert result["review_reason"] == "insufficient_risk_control_evidence"
