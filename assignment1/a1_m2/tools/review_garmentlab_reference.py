"""CPU parity and timing audit of the existing reference, not a new quality test."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "draft/garmentlab/reference_cnn")
    parser.add_argument("--out", type=Path, default=ROOT / "draft/garmentlab/review/reference_cpu_review.json")
    args = parser.parse_args()
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from PIL import Image
    from garmentlab.inference import Predictor, preprocess_image
    torch.set_num_threads(4)
    predictor = Predictor(args.release)
    # Fixed official-training samples: used only for export/transport equality, not test quality.
    raw_path = ROOT / "data/FashionMNIST/raw/train-images-idx3-ubyte"
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888":
        raise ValueError("Unexpected Fashion-MNIST training-image source.")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(60000, 28, 28)[:32]
    tensors = torch.from_numpy(images.astype(np.float32)[:, None] / 255.0)
    export_path = args.release / "model.onnx"
    torch.onnx.export(predictor.model, (tensors[:2],), str(export_path), dynamo=True,
                      input_names=["images"], output_names=["logits"], opset_version=18,
                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=128)},),
                      external_data=False)
    onnx.checker.check_model(str(export_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(export_path), sess_options=options, providers=["CPUExecutionProvider"])
    parity = []
    for n in [1, 3, 16, 32]:
        batch = tensors[:n]
        with torch.inference_mode():
            expected = predictor.model(batch).numpy()
        observed = session.run(["logits"], {"images": batch.numpy()})[0]
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise AssertionError("Non-finite logits cannot pass the parity audit.")
        error = float(np.max(np.abs(expected - observed)))
        disagreements = int(np.sum(expected.argmax(1) != observed.argmax(1)))
        if error > 1e-4 or disagreements:
            raise AssertionError("ONNX parity failed the declared FP32 audit tolerance.")
        parity.append({"batch": n, "max_abs_logit_error": error, "top1_disagreements": disagreements})
    stream = io.BytesIO()
    Image.fromarray(images[1]).save(stream, format="PNG")
    png = stream.getvalue()
    prepared, _ = preprocess_image(png, predictor.manifest)
    assert torch.equal(prepared, tensors[1:2])
    timings = {}
    for n in [1, 32]:
        batch = tensors[:n]
        for _ in range(10):
            with torch.inference_mode():
                predictor.model(batch)
            session.run(["logits"], {"images": batch.numpy()})
        for name, run in [
            ("pytorch", lambda: predictor.model(batch)),
            ("onnxruntime", lambda: session.run(["logits"], {"images": batch.numpy()})),
        ]:
            values = []
            with torch.inference_mode():
                for _ in range(100):
                    start = time.perf_counter()
                    run()
                    values.append((time.perf_counter() - start) * 1000)
            timings[f"{name}_batch{n}"] = {"median_ms": float(np.median(values)),
                                           "p95_ms": float(np.percentile(values, 95)),
                                           "measured_calls": 100, "raw_ms": values,
                                           "scope": "warm model call, prepared CPU inputs; includes runtime dispatch"}
    for _ in range(10):
        predictor.predict_bytes(png)
    values = []
    for _ in range(100):
        start = time.perf_counter()
        predictor.predict_bytes(png)
        values.append((time.perf_counter() - start) * 1000)
    timings["python_image_to_result"] = {"median_ms": float(np.median(values)),
                                          "p95_ms": float(np.percentile(values, 95)),
                                          "measured_calls": 100, "raw_ms": values,
                                          "scope": "PNG bytes decode/preprocess/forward/result; excludes HTTP and disk read"}
    receipt = {
        "status": "PASS_REFERENCE_PARITY_AND_CPU_TIMING", "release_id": predictor.manifest["release_id"],
        "scope": "Existing Fashion-MNIST model only; no new training or quality/real-photo evaluation.",
        "audit_input": {"partition": "official_training", "original_ids": list(range(32)),
                        "dataset_file_sha256": hashlib.sha256(raw).hexdigest()},
        "checkpoint_sha256": predictor.manifest["weights"]["sha256"],
        "onnx_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "export": {"dynamo": True, "opset": 18, "torch_export_batch_constraint": [1, 128],
                   "audited_runtime_batches": [1, 3, 16, 32],
                   "onnx_runtime_range_enforcement_claimed": False,
                   "fp32_max_abs_tolerance": 1e-4, "max_top1_disagreements": 0},
        "parity": parity, "timing": timings,
        "runtime": {"torch": torch.__version__, "onnxruntime": ort.__version__,
                    "onnx": onnx.__version__, "python": platform.python_version(),
                    "platform": platform.platform(), "processor": platform.processor(), "cpu_threads": 4},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "parity": parity,
                      "timing": {k: {a: b for a, b in v.items() if a != "raw_ms"} for k, v in timings.items()}}, indent=2))


if __name__ == "__main__":
    main()
