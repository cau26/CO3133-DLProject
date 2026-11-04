"""Package an existing checkpoint for draft inference; no training or new test score."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "draft/garmentlab/reference_cnn")
    args = parser.parse_args()
    source = ROOT / "output/a1_final/evidence/tasks/final_cnn_C5_seed42/best_f1.pt"
    expected_source = "020ba1edd924b2fd8ee5658ef0472570935ac4de86b5432701ebd23d267dbdf3"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source:
        raise ValueError("Reference source differs from the reviewed OPT3 checkpoint.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "weights.pt"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise FileExistsError("Refusing to overwrite different reference weights.")
    if not target.exists():
        shutil.copyfile(source, target)
    manifest = {
        "format_version": 1, "release_id": "opt3-cnn-c5-seed42-reference-v1",
        "domain": "fashion_mnist", "architecture": "cnn_c5", "status": "reference",
        "labels": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "preprocessing": {"version": "fashion_mnist_totensor_v1", "height": 28,
                          "width": 28, "channels": 1, "mode": "L", "resize": "none", "normalization": "zero_one"},
        "weights": {"file": "weights.pt", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "calibration": {"status": "not_fitted"},
        "provenance": {"source": str(source.relative_to(ROOT)).replace("\\", "/"),
                       "training_study": "g_m10_a1_optimization_v3_1_7days",
                       "checkpoint_epoch": 44, "new_training_performed": False},
        "limitations": ["Benchmark reference only; accepts original 28x28 grayscale Fashion-MNIST images.",
                        "No real-photo garment classifier is available in this draft.",
                        "Softmax values are uncalibrated; all outputs require review."]
    }
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    release_path = out / "release.json"
    if release_path.exists() and release_path.read_text(encoding="utf-8") != content:
        raise FileExistsError("Refusing to replace a different release manifest.")
    release_path.write_text(content, encoding="utf-8")
    from garmentlab.inference import Predictor
    predictor = Predictor(out)
    print(json.dumps({"status": "REFERENCE_READY", "directory": str(out),
                      "metadata": predictor.metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
