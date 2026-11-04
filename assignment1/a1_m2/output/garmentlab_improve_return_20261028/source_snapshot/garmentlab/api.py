"""Local draft API with separate benchmark-reference and real-garment routes."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 64 * 1024
MAX_CONCURRENT_PREDICTIONS = 2
STATIC = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger("garmentlab.requests")


def _error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "request_id": request_id,
        "error": {"code": code, "message": message},
    })


class RequestBoundary:
    """Bound even chunked bodies before multipart parsing; never log image data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        response_status = 500

        async def tagged_send(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-request-id", request_id.encode("ascii")),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"content-security-policy", b"default-src 'self'; img-src 'self' blob: data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"),
                ])
                message = {**message, "headers": headers}
            await send(message)

        try:
            for key, value in scope.get("headers", []):
                if key.lower() == b"content-length":
                    try:
                        declared = int(value)
                    except ValueError:
                        await _error(400, "invalid_request", "Content-Length không hợp lệ.", request_id)(scope, receive, tagged_send)
                        return
                    if declared < 0:
                        await _error(400, "invalid_request", "Content-Length không hợp lệ.", request_id)(scope, receive, tagged_send)
                        return
                    if declared > MAX_REQUEST_BYTES:
                        await _error(413, "upload_too_large", "Ảnh tối đa 5 MiB.", request_id)(scope, receive, tagged_send)
                        return
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                body.extend(message.get("body", b""))
                if len(body) > MAX_REQUEST_BYTES:
                    await _error(413, "upload_too_large", "Ảnh tối đa 5 MiB.", request_id)(scope, receive, tagged_send)
                    return
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, tagged_send)
        finally:
            LOGGER.info(json.dumps({"request_id": request_id, "method": scope.get("method"),
                "path": scope.get("path"), "status": response_status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}, ensure_ascii=False))


def _parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or not value.strip():
        return None
    try:
        crop = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Crop phải là JSON [left, top, right, bottom].") from error
    if not isinstance(crop, list) or len(crop) != 4 or any(type(x) is not int for x in crop):
        raise ValueError("Crop cần bốn tọa độ pixel nguyên [left, top, right, bottom].")
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("Crop cần left < right, top < bottom và tọa độ không âm.")
    return tuple(crop)


def create_app(reference_predictor=None, garment_predictor=None) -> FastAPI:
    """Inject predictors exposing metadata and predict_bytes(bytes, crop=None)."""
    for predictor, domain in ((reference_predictor, "fashion_mnist"),
                              (garment_predictor, "garment_crop_rgb")):
        if predictor is not None and predictor.metadata.get("domain") != domain:
            raise ValueError(f"Predictor domain must be {domain} for this route.")
    application = FastAPI(title="GarmentLab — local draft", version="0.1.0",
                          docs_url=None, redoc_url=None)
    application.state.reference_predictor = reference_predictor
    application.state.garment_predictor = garment_predictor
    application.state.prediction_gate = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
    application.add_middleware(RequestBoundary)

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        code = "invalid_request" if error.status_code < 500 else "service_unavailable"
        message = str(error.detail) if isinstance(error.detail, str) else "Yêu cầu không hợp lệ."
        return _error(error.status_code, code, message, request.state.request_id)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _error_details: RequestValidationError):
        return _error(422, "invalid_request", "Cần gửi một ảnh trong trường multipart 'file'.", request.state.request_id)

    def describe(predictor: Any, mode: str) -> dict:
        return {"ready": predictor is not None, "mode": mode,
                "metadata": dict(predictor.metadata) if predictor is not None else None}

    @application.get("/health")
    async def health(request: Request):
        reference_ready = application.state.reference_predictor is not None
        garment_ready = application.state.garment_predictor is not None
        status = "ready" if garment_ready else ("degraded_reference_only" if reference_ready else "models_not_loaded")
        return {"request_id": request.state.request_id, "status": status,
                "reference_ready": reference_ready, "garment_ready": garment_ready,
                "scope": "local_draft", "max_concurrent_predictions": MAX_CONCURRENT_PREDICTIONS}

    @application.get("/v1/models")
    async def models(request: Request):
        return {"request_id": request.state.request_id,
                "reference": describe(application.state.reference_predictor, "fashion_mnist_reference"),
                "garment": describe(application.state.garment_predictor, "real_garment")}

    async def predict(request: Request, file: UploadFile, crop_value: str | None, reference: bool):
        request_id = request.state.request_id
        predictor = application.state.reference_predictor if reference else application.state.garment_predictor
        try:
            if predictor is None:
                message = "Checkpoint Fashion-MNIST chưa được nạp." if reference else "Mô hình ảnh quần áo thật chưa sẵn sàng; bản nháp không dùng model Fashion-MNIST để thay thế."
                return _error(503, "model_not_ready", message, request_id)
            if file.content_type not in {"image/png", "image/jpeg"}:
                return _error(415, "unsupported_media_type", "Chỉ nhận ảnh PNG hoặc JPEG.", request_id)
            content = await file.read(MAX_FILE_BYTES + 1)
            if not content:
                return _error(400, "invalid_image", "Tệp ảnh trống.", request_id)
            if len(content) > MAX_FILE_BYTES:
                return _error(413, "upload_too_large", "Ảnh tối đa 5 MiB.", request_id)
            try:
                crop = _parse_crop(crop_value)
            except ValueError as error:
                return _error(400, "invalid_crop", str(error), request_id)
            if reference and crop is not None:
                return _error(400, "invalid_crop", "Chế độ tham chiếu dùng nguyên ảnh Fashion-MNIST 28×28, không crop.", request_id)
            gate = application.state.prediction_gate
            if gate.locked():
                response = _error(503, "service_busy", "Đang xử lý đủ số ảnh đồng thời. Vui lòng thử lại.", request_id)
                response.headers["Retry-After"] = "1"
                return response
            await gate.acquire()
            try:
                result = await run_in_threadpool(predictor.predict_bytes, content, crop=crop)
                if not isinstance(result, dict):
                    raise TypeError("Predictor result must be a dictionary")
                return JSONResponse({"request_id": request_id,
                    "mode": "fashion_mnist_reference" if reference else "real_garment",
                    "result": result})
            except ValueError:
                return _error(400, "invalid_image", "Ảnh hoặc vùng crop không đáp ứng yêu cầu đầu vào của mô hình.", request_id)
            except Exception:
                return _error(503, "prediction_failed", "Không thể hoàn tất dự đoán. Kiểm tra model và thử lại.", request_id)
            finally:
                gate.release()
        finally:
            await file.close()

    @application.post("/v1/reference/predict")
    async def reference_predict(request: Request, file: UploadFile = File(...), crop: str | None = Form(None)):
        return await predict(request, file, crop, reference=True)

    @application.post("/v1/predict")
    async def garment_predict(request: Request, file: UploadFile = File(...), crop: str | None = Form(None)):
        return await predict(request, file, crop, reference=False)

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC), name="static")
    return application


app = create_app()
