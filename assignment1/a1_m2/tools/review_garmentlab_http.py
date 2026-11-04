"""Measure local draft HTTP behaviour with public training fixtures; no model scoring."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--out", type=Path, default=ROOT / "draft/garmentlab/review/http_review.json")
    args = parser.parse_args()
    from urllib.parse import urlparse
    import httpx
    import numpy as np
    if urlparse(args.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This audit sends fixtures only to localhost.")
    image = (ROOT / "garmentlab/static/samples/sample-tshirt.png").read_bytes()
    with httpx.Client(base_url=args.base_url, timeout=15, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        def request(_=0, allow_busy=False):
            start = time.perf_counter()
            response = client.post("/v1/reference/predict", files={"file": ("fixture.png", image, "image/png")})
            elapsed = (time.perf_counter() - start) * 1000
            if allow_busy and response.status_code == 503:
                assert response.json()["error"]["code"] == "service_busy"
                return {"status": 503, "latency_ms": elapsed}
            response.raise_for_status()
            body = response.json()["result"]
            assert body["domain"] == "fashion_mnist" and body["calibrated"] is False
            assert body["review_required"] is True
            return {"status": 200, "latency_ms": elapsed} if allow_busy else elapsed
        first = request()
        for _ in range(10):
            request()
        sequential = [request() for _ in range(100)]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(lambda i: request(i, allow_busy=True), range(100)))
        wall = time.perf_counter() - start
        real = client.post("/v1/predict", files={"file": ("fixture.png", image, "image/png")})
        malformed = client.post("/v1/reference/predict", files={"file": ("bad.png", b"not-an-image", "image/png")})
        assert real.status_code == 503 and malformed.status_code == 400
    def describe(values):
        return {"calls": len(values), "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)), "raw_ms": values}
    receipt = {
        "status": "PASS_LOCAL_HTTP_REFERENCE_DRAFT", "health": health.json(),
        "scope": "Windows localhost HTTP request round trip, including multipart encoding/transport/decoding/model/result. Public official-training fixture, no accuracy evaluation or real-user traffic.",
        "model": "existing OPT3 CNN C5 seed42 benchmark reference; CPU 4 model threads",
        "first_observed_request_ms": first,
        "first_request_is_full_process_cold_start": False,
        "sequential": describe(sequential),
        "concurrency4_overload_drill": {
            "attempts": len(concurrent), "accepted": sum(x["status"] == 200 for x in concurrent),
            "busy_503": sum(x["status"] == 503 for x in concurrent),
            "successful_request_latency": describe([x["latency_ms"] for x in concurrent if x["status"] == 200]),
            "raw_attempts": concurrent, "wall_seconds": wall,
            "scope": "Intentional overload: API admits at most two predictions; excess returns service_busy with Retry-After. Not a no-errors throughput claim."
        },
        "real_photo_route_without_model_status": real.status_code,
        "malformed_image_status": malformed.status_code,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key not in {"raw_ms", "raw_attempts"}}
        return value
    print(json.dumps(compact(receipt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
