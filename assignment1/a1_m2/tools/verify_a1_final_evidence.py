"""Verify the public evidence subset and recompute all OPT3 prediction metrics.

Uses Python's standard library. It performs no training, model forward call,
checkpoint deserialization or official-test evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "output/a1_final/evidence"


def check_close(actual, expected, tolerance=1e-12):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"Metric mismatch: {actual} versus {expected}")


def main():
    receipt = json.loads((EVIDENCE / "COPY_RECEIPT.json").read_text(encoding="utf-8"))
    for item in receipt["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe evidence receipt path")
        payload = (EVIDENCE / relative).read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Evidence payload differs: {relative}")
    rows_checked = 0
    final = {}
    task_dirs = sorted((EVIDENCE / "tasks").iterdir())
    if len(task_dirs) != 61:
        raise ValueError("Expected 61 task directories")
    for task_dir in task_dirs:
        metrics = json.loads((task_dir / "metrics.json").read_text(encoding="utf-8"))
        with (task_dir / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        cm = [[0] * 10 for _ in range(10)]
        ids = set()
        nll = []
        for row in predictions:
            image_id, true, pred = int(row["original_id"]), int(row["true"]), int(row["pred"])
            if image_id in ids or not 0 <= true < 10 or not 0 <= pred < 10:
                raise ValueError("Duplicate ID or invalid class")
            ids.add(image_id)
            cm[true][pred] += 1
            loss = float(row["nll"])
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("Invalid saved NLL")
            nll.append(loss)
        count = len(predictions)
        accuracy = sum(cm[i][i] for i in range(10)) / count
        class_f1 = []
        for i in range(10):
            denominator = sum(cm[i]) + sum(cm[j][i] for j in range(10))
            class_f1.append(2 * cm[i][i] / denominator if denominator else 0.0)
        f1 = statistics.mean(class_f1)
        recorded = metrics["best_f1_metrics"]
        if count != recorded["n_examples"] or metrics["epochs_completed"] != 60:
            raise ValueError("Task coverage mismatch")
        check_close(accuracy, recorded["accuracy"])
        check_close(f1, recorded["macro_f1"])
        check_close(statistics.mean(nll), recorded["loss"], 1e-7)
        rows_checked += count
        if metrics["phase"] == "final":
            final.setdefault(metrics["family"], []).append((accuracy, f1))
    publication = json.loads((ROOT / "output/a1_final/data/publication_data.json").read_text(encoding="utf-8"))
    for summary in publication["final_exploratory_validation"]:
        values = final[summary["model"]]
        if len(values) != 3:
            raise ValueError("Expected three final seeds per family")
        for index, metric in enumerate(("accuracy", "macro_f1")):
            measurements = [row[index] for row in values]
            check_close(statistics.mean(measurements), summary[metric + "_mean"])
            check_close(statistics.stdev(measurements), summary[metric + "_sample_sd"])
    if rows_checked != 338400:
        raise ValueError("Expected 338,400 prediction rows")
    print(json.dumps({"status": "PASS_PUBLIC_EVIDENCE_AND_METRICS", "files": len(receipt["files"]),
                      "tasks": len(task_dirs), "prediction_rows": rows_checked, "final_seeds_per_family": 3,
                      "scope": "Byte hashes, saved-prediction metrics/NLL aggregation and final mean/sample SD; no model inference or checkpoint deserialization."}, indent=2))


if __name__ == "__main__":
    main()
