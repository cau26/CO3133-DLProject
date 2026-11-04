"""HTTP contracts with fake predictors; not checkpoint or model-quality tests."""
from __future__ import annotations

import asyncio
import io
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from garmentlab.api import MAX_FILE_BYTES, MAX_REQUEST_BYTES, create_app


def png_bytes():
    buffer = io.BytesIO()
    Image.new("L", (28, 28), color=128).save(buffer, format="PNG")
    return buffer.getvalue()


class FakePredictor:
    def __init__(self, error=None, domain="fashion_mnist"):
        self.calls = []
        self.error = error
        self.metadata = {"model_version": "fake-unittest-only", "domain": domain}

    def predict_bytes(self, image_bytes, crop=None):
        self.calls.append((image_bytes, crop))
        if self.error:
            raise self.error
        return {"model_version": "fake-unittest-only", "calibrated": False,
                "review_required": True,
                "predicted_class": {"index": 0, "label": "fixture", "probability": 1.0},
                "probabilities": [{"index": 0, "label": "fixture", "probability": 1.0}],
                "input": {"width": 28, "height": 28, "mode": "L", "crop": crop},
                "preprocessing": {"version": "fake-v1"}, "timing_ms": {"total": 0.0}}


class GarmentLabApiTests(unittest.TestCase):
    def setUp(self):
        self.reference = FakePredictor()
        self.application = create_app(reference_predictor=self.reference)
        self.client = TestClient(self.application)
        self.image = png_bytes()

    def upload(self, path="/v1/reference/predict", **kwargs):
        return self.client.post(path, files={"file": ("sample.png", self.image, "image/png")}, **kwargs)

    def test_health_reports_reference_only_and_model_domains_separately(self):
        health = self.client.get("/health")
        self.assertEqual(health.json()["status"], "degraded_reference_only")
        self.assertFalse(health.json()["garment_ready"])
        models = self.client.get("/v1/models").json()
        self.assertTrue(models["reference"]["ready"])
        self.assertFalse(models["garment"]["ready"])
        self.assertIsNone(models["garment"]["metadata"])

    def test_real_route_has_no_reference_fallback(self):
        result = self.upload("/v1/predict")
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["error"]["code"], "model_not_ready")
        self.assertEqual(self.reference.calls, [])

    def test_reference_contract_passes_bytes_and_tags_request_id(self):
        result = self.upload()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["mode"], "fashion_mnist_reference")
        self.assertEqual(self.reference.calls, [(self.image, None)])
        self.assertEqual(result.headers["x-request-id"], result.json()["request_id"])
        self.assertTrue(result.json()["result"]["review_required"])

    def test_real_predictor_receives_exclusive_right_bottom_pixel_crop(self):
        garment = FakePredictor(domain="garment_crop_rgb")
        client = TestClient(create_app(reference_predictor=self.reference, garment_predictor=garment))
        result = client.post("/v1/predict", files={"file": ("sample.png", self.image, "image/png")}, data={"crop": "[1,2,20,22]"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["mode"], "real_garment")
        self.assertEqual(garment.calls, [(self.image, (1, 2, 20, 22))])
        self.assertEqual(self.reference.calls, [])

    def test_reference_rejects_crop(self):
        result = self.upload(data={"crop": "[1,2,20,22]"})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.json()["error"]["code"], "invalid_crop")
        self.assertEqual(self.reference.calls, [])

    def test_bad_crop_types_and_geometry_do_not_reach_predictor(self):
        garment = FakePredictor(domain="garment_crop_rgb")
        client = TestClient(create_app(garment_predictor=garment))
        for crop in ("not-json", "[true,0,2,2]", "[0,0,2.5,3]", "[2,0,1,3]", "[-1,0,2,3]", "[0,0,2]", '{"left":0}'):
            with self.subTest(crop=crop):
                result = client.post("/v1/predict", files={"file": ("sample.png", self.image, "image/png")}, data={"crop": crop})
                self.assertEqual(result.status_code, 400)
                self.assertEqual(result.json()["error"]["code"], "invalid_crop")
        self.assertEqual(garment.calls, [])

    def test_decode_or_model_input_error_is_safe_and_request_is_traceable(self):
        client = TestClient(create_app(reference_predictor=FakePredictor(ValueError("private-path-image-details"))))
        result = client.post("/v1/reference/predict", files={"file": ("x.png", self.image, "image/png")})
        self.assertEqual(result.status_code, 400)
        self.assertNotIn("private-path", result.text)
        self.assertEqual(result.json()["error"]["code"], "invalid_image")
        self.assertEqual(result.headers["x-request-id"], result.json()["request_id"])

    def test_predictor_domain_cannot_be_mislabeled_as_real_garment(self):
        with self.assertRaises(ValueError):
            create_app(garment_predictor=self.reference)

    def test_internal_failure_is_not_returned_as_prediction_or_traceback(self):
        client = TestClient(create_app(reference_predictor=FakePredictor(RuntimeError("secret-local-path"))))
        result = client.post("/v1/reference/predict", files={"file": ("x.png", self.image, "image/png")})
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["error"]["code"], "prediction_failed")
        self.assertNotIn("secret-local-path", result.text)

    def test_missing_empty_and_unsupported_files_have_structured_errors(self):
        missing = self.client.post("/v1/reference/predict")
        self.assertEqual(missing.status_code, 422)
        for body, mime, status in ((b"", "image/png", 400), (b"anything", "text/plain", 415)):
            with self.subTest(mime=mime, status=status):
                result = self.client.post("/v1/reference/predict", files={"file": ("file", body, mime)})
                self.assertEqual(result.status_code, status)
                self.assertIn("request_id", result.json())
        self.assertEqual(self.reference.calls, [])

    def test_file_size_limit_rejects_before_prediction(self):
        result = self.client.post("/v1/reference/predict", files={"file": ("large.png", b"x" * (MAX_FILE_BYTES + 1), "image/png")})
        self.assertEqual(result.status_code, 413)
        self.assertEqual(result.json()["error"]["code"], "upload_too_large")
        self.assertEqual(self.reference.calls, [])

    def test_chunked_body_without_content_length_is_also_bounded(self):
        messages = [{"type": "http.request", "body": b"x" * MAX_REQUEST_BYTES, "more_body": True},
                    {"type": "http.request", "body": b"x", "more_body": False}]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/v1/reference/predict",
                 "raw_path": b"/v1/reference/predict", "query_string": b"", "root_path": "",
                 "headers": [], "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)}
        asyncio.run(self.application(scope, receive, send))
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
        self.assertEqual(start["status"], 413)
        self.assertEqual(json.loads(body)["error"]["code"], "upload_too_large")
        self.assertEqual(self.reference.calls, [])

    def test_busy_service_does_not_queue_another_inference(self):
        self.application.state.prediction_gate = asyncio.Semaphore(0)
        result = self.upload()
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["error"]["code"], "service_busy")
        self.assertEqual(result.headers["retry-after"], "1")
        self.assertEqual(self.reference.calls, [])

    def test_ui_and_samples_available_without_any_loaded_model(self):
        client = TestClient(create_app())
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Ảnh quần áo thật", page.text)
        self.assertIn("Vùng chứa một trang phục", page.text)
        self.assertEqual(client.get("/health").json()["status"], "models_not_loaded")
        self.assertEqual(client.get("/static/samples/sample-tshirt.png").status_code, 200)
        self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])

    def test_no_permissive_cors_and_no_raw_image_content_in_request_log(self):
        with self.assertLogs("garmentlab.requests", level="INFO") as capture:
            result = self.client.post("/v1/reference/predict", files={"file": ("private-name.png", self.image, "image/png")}, headers={"Origin": "https://unrelated.example"})
        self.assertNotIn("access-control-allow-origin", result.headers)
        log = "\n".join(capture.output)
        self.assertIn(result.json()["request_id"], log)
        self.assertNotIn("private-name", log)
        self.assertNotIn("IHDR", log)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RELEASE = ROOT / "draft/garmentlab/reference_cnn"
SOURCE_CHECKPOINT = ROOT / "output/a1_final/evidence/tasks/final_cnn_C5_seed42/best_f1.pt"


@unittest.skipUnless((REFERENCE_RELEASE / "release.json").is_file() and SOURCE_CHECKPOINT.is_file(),
                     "Optional recorded reference release/checkpoint not present")
class RecordedReferenceApiIntegrationTests(unittest.TestCase):
    """CPU path parity for one saved input; not a new accuracy evaluation."""

    def test_saved_checkpoint_and_http_reference_have_matching_probabilities(self):
        import numpy as np
        import torch
        from a1.cnn import CustomCNN
        from garmentlab.inference import Predictor

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            predictor = Predictor(REFERENCE_RELEASE)
            checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=True)
            original = CustomCNN(dropout=0.3)
            original.load_state_dict(checkpoint["model_state_dict"], strict=True)
            original.eval()
            sample = (ROOT / "garmentlab/static/samples/sample-tshirt.png").read_bytes()
            pixels = np.asarray(Image.open(io.BytesIO(sample)), dtype=np.float32)
            tensor = torch.from_numpy(pixels.copy()).unsqueeze(0).unsqueeze(0) / 255.0
            with torch.inference_mode():
                expected = torch.softmax(original(tensor)[0], dim=0).numpy()
            response = TestClient(create_app(reference_predictor=predictor)).post(
                "/v1/reference/predict", files={"file": ("sample.png", sample, "image/png")})
            self.assertEqual(response.status_code, 200)
            result = response.json()["result"]
            actual = np.asarray([row["probability"] for row in result["probabilities"]])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(result["predicted_class"]["index"], int(expected.argmax()))
            self.assertEqual(result["model_version"], predictor.metadata["model_version"])
            self.assertFalse(result["calibrated"])
            self.assertTrue(result["review_required"])
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
