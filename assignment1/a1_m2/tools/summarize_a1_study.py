"""Audit 24 student-run validation tasks and summarize paired seed/budget effects. No inference."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a1.common import json_hash, load_json, protect_json, save_csv, save_json, sha256_file
from a1.study import ABLATIONS, MODELS, POLICY, progress, read_csv, relative_child, validate_study
from tools.verify_a1_results import Review, close, score_predictions


def mean_sd(values):
    values = list(values)
    if len(values) < 2 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Sample SD requires at least two finite independent seed observations")
    return {"n": len(values), "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def paired_effect(baseline, variant):
    if set(baseline) != set(variant) or len(baseline) != 3:
        raise ValueError("Paired comparisons require exactly the same three seed IDs")
    deltas = {seed: variant[seed] - baseline[seed] for seed in sorted(baseline)}
    return {**mean_sd(deltas.values()), "per_seed": deltas}


def history_rows(path):
    rows = read_csv(path)
    return [{k: int(v) if k in ("epoch", "gpu_peak_allocated_bytes") else float(v)
             for k, v in row.items()} for row in rows]


def audit_budget(directory, task, manifest, review, history, budget, reference_truth):
    import torch
    from a1.train import build_model, model_config, validate_budget_snapshot
    config = model_config(task["model"], review.config)
    binding = {"model_name": task["model"], "config": config, "config_sha256": json_hash(config),
               "split_sha256": review.summary["split_sha256"], "dataset_sha256": review.summary["dataset_sha256"],
               "source_sha256": review.run["source_sha256"], "run_id": review.run["run_id"]}
    selection = validate_budget_snapshot(directory, binding, history, budget)
    metrics = load_json(directory / f"budget{budget}_metrics.json")
    expected = {"model": task["model"], "evaluation_split": "validation", "prefix_budget": budget,
                "best_epoch": selection["selected_epoch"], "n_eval": 6000,
                "checkpoint": selection["checkpoint"], "checkpoint_sha256": selection["checkpoint_sha256"],
                "selection_history_sha256": selection["selection_history_sha256"], "config": config,
                "config_sha256": binding["config_sha256"], "split_sha256": binding["split_sha256"],
                "dataset_sha256": binding["dataset_sha256"], "source_sha256": binding["source_sha256"],
                "run_id": binding["run_id"], "test_evaluated": False,
                "validation_predictions": f"budget{budget}_validation_predictions.csv"}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f'{task["task_id"]}, budget{budget}: invalid binding {key}')
    if metrics["source"]["source_sha256"] != binding["source_sha256"]:
        raise ValueError("Budget evaluation source differs from run")
    predictions = directory / expected["validation_predictions"]
    if metrics["predictions_sha256"] != sha256_file(predictions):
        raise ValueError("Budget prediction bytes differ from saved fingerprint")
    scores = score_predictions(read_csv(predictions), "official_train_index", review.validation_ids)
    if scores["truth"] != [reference_truth[index] for index in scores["ids"]]:
        raise ValueError("Study validation truth labels differ from the verified historical split")
    for field in ("accuracy", "macro_f1"):
        close(scores[field], metrics[field], field, atol=1e-12, rtol=1e-12)
    close(selection["validation_loss"], metrics["loss"], "budget loss", atol=1e-5, rtol=1e-5)
    chosen = history[selection["selected_epoch"] - 1]
    for field in ("accuracy", "macro_f1"):
        close(chosen["val_" + field], metrics[field], "history budget " + field, atol=1e-5, rtol=1e-5)
    prefix = history[:budget]
    close(sum(row["train_seconds"] for row in prefix), metrics["train_seconds"], "prefix training sum", atol=1e-8)
    close(sum(row["val_seconds"] for row in prefix), metrics["epoch_validation_seconds"], "prefix validation sum", atol=1e-8)
    checkpoint = torch.load(directory / selection["checkpoint"], map_location="cpu", weights_only=True)
    model = build_model(task["model"], config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("Nonfinite prefix checkpoint weights")
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != metrics["parameters"]:
        raise ValueError("Budget parameter count differs from checkpoint architecture")
    for key in ("gpu", "versions", "cuda_runtime", "cudnn_version"):
        if metrics["environment"].get(key) != review._study_environment.get(key):
            raise ValueError("Recorded study hardware/software differs across seed tasks")
    endpoint = prefix[-1]
    return {"arm": task["arm"], "model": task["model"], "seed": task["seed"], "budget": budget,
            "selected_epoch": selection["selected_epoch"], "loss": metrics["loss"],
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "parameters": parameters,
            "train_seconds": metrics["train_seconds"], "epoch_validation_seconds": metrics["epoch_validation_seconds"],
            "endpoint_train_accuracy": endpoint["train_accuracy"], "endpoint_val_accuracy": endpoint["val_accuracy"],
            "endpoint_train_loss": endpoint["train_loss"], "endpoint_val_loss": endpoint["val_loss"],
            "last10_endpoint_loss_change": prefix[-1]["val_loss"] - prefix[-10]["val_loss"],
            "selected_in_last10": selection["selected_epoch"] > budget - 10,
            "checkpoint_sha256": metrics["checkpoint_sha256"], "run_id": binding["run_id"],
            "per_class_f1": [2 * scores["confusion"][k][k] / (sum(scores["confusion"][k]) +
                                sum(row[k] for row in scores["confusion"])) for k in range(10)]}


def aggregate_records(records, seeds=(42, 43, 44)):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["arm"], row["model"], row["budget"])].append(row)
    summary = []
    for (arm, model, budget), rows in sorted(grouped.items()):
        if len(rows) != 3 or sorted(row["seed"] for row in rows) != sorted(seeds):
            raise ValueError("Missing/duplicate seed observations; cannot report a complete n=3 summary")
        counts = {row["parameters"] for row in rows}
        if len(counts) != 1:
            raise ValueError("Architecture parameter count changes across seeds")
        result = {"arm": arm, "model": model, "budget": budget, "n_seeds": 3, "parameters": counts.pop()}
        for key in ("accuracy", "macro_f1", "loss", "train_seconds", "selected_epoch"):
            stats = mean_sd(row[key] for row in rows)
            result[key + "_mean"] = stats["mean"]
            result[key + "_sample_sd"] = stats["sample_sd"]
        result["selected_in_last10_count"] = sum(row["selected_in_last10"] for row in rows)
        summary.append(result)
    return summary


def compare_arms(records):
    index = {(r["arm"], r["model"], r["budget"], r["seed"]): r for r in records}
    if len(index) != len(records):
        raise ValueError("Duplicate arm/model/budget/seed record")
    effects = []
    for variant in ABLATIONS:
        for budget in (20, 60):
            result = {"contrast": variant["id"] + " minus baseline", "model": variant["model"], "budget": budget}
            for metric in ("accuracy", "macro_f1", "loss"):
                try:
                    baseline = {s: index[("baseline", variant["model"], budget, s)][metric] for s in (42, 43, 44)}
                    ablated = {s: index[(variant["id"], variant["model"], budget, s)][metric] for s in (42, 43, 44)}
                except KeyError as exc:
                    raise ValueError("Incomplete same-seed ablation pairing") from exc
                paired = paired_effect(baseline, ablated)
                result[metric + "_delta_mean"] = paired["mean"]
                result[metric + "_delta_sample_sd"] = paired["sample_sd"]
                for s, delta in paired["per_seed"].items():
                    result[f"{metric}_delta_seed{s}"] = delta
            effects.append(result)
    budgets = []
    for model in MODELS:
        result = {"contrast": "best-through-60 minus best-through-20", "model": model, "n_seeds": 3}
        for metric in ("accuracy", "macro_f1", "loss"):
            try:
                first = {s: index[("baseline", model, 20, s)][metric] for s in (42, 43, 44)}
                last = {s: index[("baseline", model, 60, s)][metric] for s in (42, 43, 44)}
            except KeyError as exc:
                raise ValueError("Incomplete baseline budget pairing") from exc
            effect = paired_effect(first, last)
            result[metric + "_delta_mean"] = effect["mean"]
            result[metric + "_delta_sample_sd"] = effect["sample_sd"]
            if metric == "loss" and any(v > 1e-5 for v in effect["per_seed"].values()):
                raise ValueError("Minimum through 60 cannot exceed minimum through 20 on the same history")
        budgets.append(result)
    return effects, budgets


def render_figures(records, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    baseline = [next(r for r in summaries if r["arm"] == "baseline" and r["model"] == m and r["budget"] == 60) for m in MODELS]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    x = np.arange(5)
    ax.errorbar(x, [r["accuracy_mean"] * 100 for r in baseline],
                yerr=[r["accuracy_sample_sd"] * 100 for r in baseline], fmt="o", capsize=5, color="#007D83")
    ax.set(xticks=x, xticklabels=MODELS, ylabel="Validation accuracy (%)",
           title="Supplementary validation: three training seeds, mean ± sample SD")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "baseline_seed_variability.png", dpi=180)
    fig.savefig(output / "baseline_seed_variability.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, variant in zip(axes, ABLATIONS):
        for seed in (42, 43, 44):
            values = [next(r["accuracy"] * 100 for r in records if r["arm"] == arm and
                           r["model"] == variant["model"] and r["budget"] == 60 and r["seed"] == seed)
                      for arm in ("baseline", variant["id"])]
            axis.plot([0, 1], values, "o-", label=f"seed{seed}")
        axis.set(xticks=[0, 1], xticklabels=["baseline", "ablated"], title=variant["id"], ylabel="Validation accuracy (%)")
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(output / "paired_ablations.png", dpi=180)
    plt.close(fig)


def summarize(output):
    output = Path(output).resolve()
    manifest = validate_study(output)
    state = progress(output, manifest)
    if state["completed_tasks"] != 24 or not (output / "STUDY_COMPLETE.json").is_file():
        raise ValueError(f'Incomplete study: {state["completed_tasks"]}/24. Resume training; no complete multi-seed table created.')
    marker = load_json(output / "STUDY_COMPLETE.json")
    if marker != {"study_id": manifest["protocol"]["study_id"], "manifest_sha256": manifest["manifest_sha256"],
                  "completed_tasks": 24, "evaluation_policy": POLICY}:
        raise ValueError("Study completion marker differs from its fixed plan")
    destination = output / "analysis"
    if (destination / "ANALYSIS_COMPLETE.json").exists():
        completed = load_json(destination / "ANALYSIS_COMPLETE.json")
        if completed["study_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Existing analysis belongs to a different study")
        for relative, expected in completed["inputs_sha256"].items():
            if sha256_file(relative_child(output, relative)) != expected:
                raise ValueError("Completed analysis input changed: " + relative)
        for relative, expected in completed["outputs_sha256"].items():
            if sha256_file(relative_child(destination, relative)) != expected:
                raise ValueError("Completed derived analysis changed: " + relative)
        return load_json(destination / "study_summary.json")
    protocol = manifest["protocol"]
    reference = relative_child(ROOT, protocol["reference_run"])
    reference_truth = {int(r["official_train_index"]): int(r["true"]) for r in read_csv(reference / "linear/validation_predictions.csv")}
    if len(reference_truth) != 6000:
        raise ValueError("Historical validation reference is incomplete")
    study_environment = load_json(output / "study_environment.json")
    if study_environment.get("device") != "cuda" or "A100" not in (study_environment.get("gpu") or ""):
        raise ValueError("Supplementary full-training records must identify an actual A100 CUDA runtime")
    records, checks = [], []
    for planned in manifest["runs"]:
        directory = relative_child(output, planned["run_relative"])
        review = Review(directory)
        review._study_environment = study_environment
        review.check("run metadata", review.metadata)
        if review.config != planned["config"]:
            raise ValueError("Actual run config differs from immutable study plan")
        review.check("split", review.split)
        review.check("source files", lambda: review.source_hashes(review.run["source"]["files_sha256"]))
        for model in planned["models"]:
            reviewed = review.model(model)
            if reviewed["status"] != "CONSISTENT" or review.errors or review.optional:
                raise ValueError("Run validation/checkpoint audit failed: " + json.dumps(review.errors + review.optional))
            metrics = load_json(directory / model / "metrics.json")
            for key in ("gpu", "versions", "cuda_runtime", "cudnn_version"):
                if metrics["environment"].get(key) != study_environment.get(key):
                    raise ValueError("Study task environment differs from the common recorded environment")
            task = next(t for t in manifest["tasks"] if t["run_key"] == planned["run_key"] and t["model"] == model)
            history = history_rows(directory / model / "history.csv")
            for budget in protocol["evaluation_budgets"]:
                records.append(audit_budget(directory / model, task, manifest, review, history, budget, reference_truth))
        checks.append({"run": planned["run_key"], "checks": review.checks, "errors": review.errors})
    if len(records) != 48:
        raise ValueError("Exactly 48 validated seed/model/budget records are required")
    summaries = aggregate_records(records, protocol["training_seeds"])
    effects, budgets = compare_arms(records)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"study_id": protocol["study_id"], "study_type": manifest["study_type"], "official_test_evaluated": False, "n_tasks": 24,
               "n_budget_records": 48, "training_seeds": protocol["training_seeds"], "split_seed": protocol["split_seed"],
               "split_sha256": protocol["expected_split_sha256"], "standard_deviation_ddof": 1,
               "records": records, "aggregates": summaries, "paired_ablations": effects, "paired_budgets": budgets,
               "checks": checks,
               "limitations": ["Exploratory post-test validation-only supplement. No upgraded official-test claim.",
                               "n=3 describes training-seed variability on one fixed validation split, not all data/split variability.",
                               "Optimizer/regularization recipes are not an architecture-only causal comparison.",
                               "Best-through60 minimum validation loss benefits mechanically from more checkpoint choices.",
                               "Paired ablations fix other recipe settings, but stochastic trajectories can differ.",
                               "No-position Transformer removes 6272 parameters; shared initial backbone weights are matched, not parameter count."]}
    save_json(destination / "study_summary.json", summary)
    save_csv(destination / "seed_metrics.csv", [{k: v for k, v in row.items() if k != "per_class_f1"} for row in records])
    save_csv(destination / "mean_sample_sd.csv", summaries)
    save_csv(destination / "paired_ablation_deltas.csv", effects)
    save_csv(destination / "budget20_vs60.csv", budgets)
    classes = load_json(reference / "data_summary.json")["classes"]
    save_csv(destination / "class_f1_by_seed.csv", [
        {"arm": r["arm"], "model": r["model"], "seed": r["seed"], "budget": r["budget"],
         "class_id": k, "class_name": classes[k], "validation_f1": f1}
        for r in records for k, f1 in enumerate(r["per_class_f1"])])
    render_figures(records, summaries, destination)
    lines = ["# Supplementary validation study", "", "Post-test exploratory supplement. Original official-test results remain unchanged.",
             "No new test evaluation. Each recipe has three training seeds on the same split.", "",
             "| Arm | Model | Budget | Accuracy mean±sampleSD (%) | Macro-F1 mean±sampleSD |",
             "|---|---|---:|---:|---:|"]
    lines.extend(f'| {r["arm"]} | {r["model"]} | {r["budget"]} | '
                 f'{r["accuracy_mean"]*100:.3f} ± {r["accuracy_sample_sd"]*100:.3f} | '
                 f'{r["macro_f1_mean"]:.5f} ± {r["macro_f1_sample_sd"]:.5f} |' for r in summaries)
    lines.extend(["", "## Paired ablations", "", "Differences are variant minus baseline at the same seed; accuracy deltas below use percentage points.",
                  "", "| Contrast | Budget | Accuracy delta mean±sampleSD (pp) |", "|---|---:|---:|"])
    lines.extend(f'| {r["contrast"]} | {r["budget"]} | {r["accuracy_delta_mean"]*100:.3f} ± '
                 f'{r["accuracy_delta_sample_sd"]*100:.3f} |' for r in effects)
    lines.extend(["", "## Interpretation limits", ""] + ["- " + s for s in summary["limitations"]])
    (destination / "study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    inputs = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*"))
              if p.is_file() and p.relative_to(output).parts[0] not in ("analysis", "smoke", "smoke_data", "sessions")
              and p.name not in ("artifact_manifest.json", "study_progress.json", ".execution.lock")}
    outputs = {p.relative_to(destination).as_posix(): sha256_file(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    protect_json(destination / "ANALYSIS_COMPLETE.json", {"study_manifest_sha256": manifest["manifest_sha256"],
                 "n_tasks": 24, "n_budget_records": 48, "inputs_sha256": inputs, "outputs_sha256": outputs,
                 "evaluation_policy": POLICY})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.out)
    print(f'ANALYSIS COMPLETE: {result["n_tasks"]}/24 tasks, {result["n_budget_records"]} prefix records, validation only')
    for row in result["aggregates"]:
        print(f'{row["arm"]:26s} {row["model"]:11s} budget{row["budget"]}: '
              f'acc {row["accuracy_mean"]*100:.3f}±{row["accuracy_sample_sd"]*100:.3f}% (n=3)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
