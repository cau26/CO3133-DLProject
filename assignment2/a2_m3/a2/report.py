"""Generate comparison tables and draft_report.md from actual saved metrics; never invent results."""
from pathlib import Path

from .common import VOC_CLASSES, load_json, save_csv


def create_report(output_dir, models=None):
    out = Path(output_dir)
    if models is None:
        models = [m for m in ["ssdlite320", "fasterrcnn_frozen", "fasterrcnn_finetune"]
                  if (out / m / "metrics.json").exists()]
    if not models:
        raise FileNotFoundError(f"No completed model metrics found in {out}")

    results = [load_json(out / m / "metrics.json") for m in models]
    split_hashes = {r["data"]["split_sha256"] for r in results}
    if len(split_hashes) != 1:
        raise ValueError("Models were evaluated on different data splits! Rerun with shared split.")

    rows = []
    for r in results:
        err = r["error_taxonomy"]
        rows.append({
            "model": r["model"],
            "architecture": r["architecture"],
            "freeze_backbone": r["freeze_backbone"],
            "val_map50": r["map50"],
            "val_map50_small": r["map50_small"],
            "val_map50_medium": r["map50_medium"],
            "val_map50_large": r["map50_large"],
            "total_parameters": r["total_parameters"],
            "trainable_parameters": r["trainable_parameters"],
            "best_epoch": r["best_epoch"],
            "epochs_completed": r["epochs_completed"],
            "train_seconds": r["train_seconds"],
            "validation_seconds": r["validation_seconds"],
            "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
            "inference_ms_per_image": r["inference"]["mean_latency_ms"],
            "correct_detections_at_030": err["correct_detections"],
            "missed_objects_at_030": err["missed_objects"],
            "wrong_class_at_030": err["wrong_class_detections"],
            "localization_errors_at_030": err["localization_errors"],
            "duplicate_detections_at_030": err["duplicate_detections"],
        })
    save_csv(out / "comparison.csv", rows)

    # Per-class comparison table
    per_class_rows = []
    for idx, cls_name in enumerate(VOC_CLASSES):
        row = {"class_id": idx + 1, "class_name": cls_name, "val_npos": results[0]["per_class_ap"][idx]["npos"]}
        for r in results:
            row[f"{r['model']}_ap50"] = r["per_class_ap"][idx]["ap50"]
        per_class_rows.append(row)
    save_csv(out / "per_class_comparison.csv", per_class_rows)

    md_table = [
        "| Model | Backbone mode | Val mAP@0.5 | Small AP | Med AP | Large AP | Trainable / Total Params | Best Epoch | Train (s) | Latency (ms/img) | Peak VRAM (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        mode = "Frozen ResNet-50" if r["freeze_backbone"] else ("ImageNet backbone" if r["model"] == "ssdlite320" else "Full fine-tune")
        md_table.append(
            f"| `{r['model']}` | {mode} | **{r['map50'] * 100.0:.2f}%** | "
            f"{r['map50_small'] * 100.0:.2f}% | {r['map50_medium'] * 100.0:.2f}% | {r['map50_large'] * 100.0:.2f}% | "
            f"{r['trainable_parameters']:,} / {r['total_parameters']:,} | "
            f"{r['best_epoch']}/{r['epochs_completed']} | {r['train_seconds']:.1f} | "
            f"{r['inference']['mean_latency_ms']:.2f} | {r['peak_gpu_memory_mb']:.0f} |"
        )
    comparison_md = "\n".join(md_table)
    (out / "comparison.md").write_text(comparison_md + "\n", encoding="utf-8")

    cfg = results[0]["config"]
    data = results[0]["data"]
    env = results[0]["environment"]
    parts = [
        "# Assignment 2 - M2 Draft Report Summary\n",
        f"**Group:** {cfg['group']}  \n**Course:** CO3133 - Deep Learning and Its Applications, Semester-261  \n"
        f"**Instructor:** Lê Thành Sách  \n**Members:** {'; '.join(cfg['members'])}\n",
        "> Generated automatically from verified run artifacts in `metrics.json`. No test-set results are included.\n",
        "## 1. Dataset and Finalized Split\n",
        f"- **Dataset:** {data['dataset']} (`{data['release']}`, MD5 `{data['archive_md5']}`).\n"
        f"- **Split:** {data['train_images']:,} training images ({data['train_non_difficult_objects']:,} usable non-difficult objects) "
        f"and {data['val_images']:,} validation images ({data['val_non_difficult_objects']:,} usable non-difficult objects; "
        f"{data['val_difficult_objects_ignored']:,} difficult objects retained for VOC ignore rules).\n"
        f"- **Split SHA-256:** `{data['split_sha256']}`.\n"
        f"- **Leakage prevention:** {data['leakage_checks']}.\n",
        "## 2. Quantitative Results (Validation Set)\n",
        comparison_md + "\n",
        f"**Hardware environment:** GPU `{env.get('gpu')}` ({env.get('gpu_total_memory_gb')} GB VRAM), "
        f"PyTorch `{env['versions'].get('torch')}`, Torchvision `{env['versions'].get('torchvision')}`.\n",
        "## 3. Per-Class AP@0.5 (%) Breakdown\n",
        "| Class | Val GT Objects | " + " | ".join(f"`{r['model']}` AP@0.5" for r in results) + " |",
        "|---|---:|" + "---:|" * len(results),
    ]
    for row in per_class_rows:
        ap_cells = " | ".join(f"{row[r['model'] + '_ap50'] * 100.0:.2f}%" for r in results)
        parts.append(f"| {row['class_name']} | {row['val_npos']} | {ap_cells} |")

    parts.append("\n## 4. Error Taxonomy (Validation Set, Confidence >= 0.30)\n")
    parts.append("| Model | Correct (TP) | Missed Objects (FN) | Missed Small (<2% area) | Wrong Class | Localization (0.1<=IoU<0.5) | Duplicate Detections | Background FP |")
    parts.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        e = r["error_taxonomy"]
        parts.append(
            f"| `{r['model']}` | {e['correct_detections']} | {e['missed_objects']} | {e['missed_small_objects']} | "
            f"{e['wrong_class_detections']} | {e['localization_errors']} | {e['duplicate_detections']} | {e['background_false_positives']} |"
        )

    (out / "draft_report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved comparison and draft report to {out / 'draft_report.md'}", flush=True)


def create_test_report(output_dir, models=None):
    out = Path(output_dir)
    if models is None:
        models = [m for m in ["ssdlite320", "fasterrcnn_frozen", "fasterrcnn_finetune"]
                  if (out / m / "test_metrics.json").exists()]
    if not models:
        raise FileNotFoundError(f"No completed test_metrics.json found in {out}")

    test_results = [load_json(out / m / "test_metrics.json") for m in models]

    val_vs_test_rows = []
    for t in test_results:
        err = t["error_taxonomy"]
        val_vs_test_rows.append({
            "model": t["model"],
            "selected_epoch": t["selected_epoch_from_val"],
            "val_map50": t["val_map50"],
            "test_map50": t["test_map50"],
            "val_to_test_delta": t["test_map50"] - t["val_map50"],
            "test_map50_small": t["test_map50_small"],
            "test_map50_medium": t["test_map50_medium"],
            "test_map50_large": t["test_map50_large"],
            "test_correct_at_030": err["correct_detections"],
            "test_missed_at_030": err["missed_objects"],
            "test_wrong_class_at_030": err["wrong_class_detections"],
            "test_localization_at_030": err["localization_errors"],
            "test_duplicates_at_030": err["duplicate_detections"],
            "test_eval_seconds": t["test_eval_seconds"],
        })
    save_csv(out / "val_vs_test_comparison.csv", val_vs_test_rows)

    per_class_rows = []
    for idx, cls_name in enumerate(VOC_CLASSES):
        row = {"class_id": idx + 1, "class_name": cls_name, "test_npos": test_results[0]["per_class_ap"][idx]["npos"]}
        for t in test_results:
            row[f"{t['model']}_test_ap50"] = t["per_class_ap"][idx]["ap50"]
        per_class_rows.append(row)
    save_csv(out / "test_per_class_comparison.csv", per_class_rows)

    md_lines = [
        "| Model | Selected Epoch | Val mAP@0.5 | Official Test mAP@0.5 | Delta (Test - Val) | Test Small AP | Test Med AP | Test Large AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in test_results:
        delta = (t["test_map50"] - t["val_map50"]) * 100.0
        md_lines.append(
            f"| `{t['model']}` | {t['selected_epoch_from_val']} | {t['val_map50'] * 100:.2f}% | "
            f"**{t['test_map50'] * 100:.2f}%** | {delta:+.2f} pp | "
            f"{t['test_map50_small'] * 100:.2f}% | {t['test_map50_medium'] * 100:.2f}% | {t['test_map50_large'] * 100:.2f}% |"
        )
    (out / "test_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved official test comparison to {out / 'test_comparison.md'}", flush=True)
