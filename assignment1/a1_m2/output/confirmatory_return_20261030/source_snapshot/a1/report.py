"""Generate versioned evidence indexes; preserve historical report artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import uuid

from .common import load_json, save_csv


def create_report(output_dir):
    out = Path(output_dir)
    if (out / "run_manifest.json").exists():
        return create_final_run_summary(out)
    raise ValueError("Historical report artifacts are immutable. Inspect the existing draft, or create a new final-protocol run.")

def create_final_run_summary(out):
    """Summarize existing evidence only; never call a pending run a finished report."""
    from .train import MODEL_NAMES
    destination = out / "summaries" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    completed, pending, rows = [], [], []
    for name in MODEL_NAMES:
        path = out / name / "metrics.json"
        if not path.is_file():
            pending.append(name)
            continue
        metrics = load_json(path)
        completed.append(name)
        row = {"model": name, "evaluation_split": "validation", "accuracy": metrics["accuracy"],
               "macro_f1": metrics["macro_f1"], "parameters": metrics["parameters"],
               "best_epoch": metrics["best_epoch"], "train_seconds": metrics["train_seconds"],
               "inference_ms_per_image": metrics["inference"]["per_image_ms_amortized"]}
        test_path = out / "final_test" / name / "metrics.json"
        if test_path.is_file():
            test = load_json(test_path)
            row.update(test_accuracy=test["accuracy"], test_macro_f1=test["macro_f1"])
        rows.append(row)
    # Equal CSV fields also support an interrupted test with only a subset complete.
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        save_csv(destination / "comparison.csv", [{key: row.get(key, "") for key in keys} for row in rows])
    text = ["# A1 saved evidence summary", "", "This is an automatically generated evidence index, not the final student report.",
            f"Completed validation models: {', '.join(completed) or 'none'}.",
            f"Pending validation models: {', '.join(pending) or 'none'}.",
            f"Freeze present: {(out / 'freeze_manifest.json').is_file()}.",
            f"All-five official test completion marker: {(out / 'final_test' / 'RESULTS_COMPLETE.json').is_file()}.",
            "", "Per-model metrics/configs, histories, figures and provenance remain in their original directories.",
            "Students must inspect learning curves/errors, justify design choices, and write the final interpretation."]
    (destination / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Evidence summary saved without overwriting previous summaries: {destination}", flush=True)
    return destination
