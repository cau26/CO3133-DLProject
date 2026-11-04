"""Frozen-choice evaluation and calibration for the student-run garment study.

Assessment predictions describe a fixed protocol. They never choose a recipe,
training seed, checkpoint, temperature, threshold or release quality criterion.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from .runtime import atomic_json, digest, utc_now
from .probabilities import log_probabilities as _log_probabilities
from .backend import configure_evaluation_backend, snapshot_evaluation_backend

ROLE_NAMES = ("fit", "selection", "calibration", "assessment")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _json_hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _arrays(logits, targets):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Logits must be a nonempty finite N x K array with K >= 2.")
    if y.shape != (len(z),) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError("Targets must be integer class indices matching the logits.")
    return z, y.astype(np.int64, copy=False)


def classification_metrics(logits, targets, class_names=None, *, temperature=1.0, bins=15):
    """All-class macro-F1; NLL/Brier/ECE use the full probability distribution."""
    z, y = _arrays(logits, targets)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be positive.")
    names = list(class_names) if class_names is not None else [str(i) for i in range(z.shape[1])]
    if len(names) != z.shape[1]:
        raise ValueError("Class names do not match logits.")
    log_probs = _log_probabilities(z, temperature)
    probs = np.exp(log_probs)
    predicted = probs.argmax(axis=1)
    confusion = np.zeros((z.shape[1], z.shape[1]), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    tp = confusion.diagonal().astype(np.float64)
    precision = np.divide(tp, confusion.sum(axis=0), out=np.zeros_like(tp), where=confusion.sum(axis=0) != 0)
    recall = np.divide(tp, confusion.sum(axis=1), out=np.zeros_like(tp), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    confidence = probs.max(axis=1)
    correct = predicted == y
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for i in range(bins):
        chosen = bin_ids == i
        count = int(chosen.sum())
        avg_conf = float(confidence[chosen].mean()) if count else None
        accuracy = float(correct[chosen].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - avg_conf)
        reliability.append({"lower": i / bins, "upper": (i + 1) / bins,
                            "count": count, "confidence": avg_conf, "accuracy": accuracy})
    one_hot = np.eye(z.shape[1])[y]
    nll = float(-log_probs[np.arange(len(y)), y].mean())
    return {"samples": len(y), "accuracy": float(correct.mean()), "macro_f1": float(f1.mean()),
            "loss": nll, "nll": nll, "brier_score": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
            "brier_definition": "mean sum of squared K-class probability errors, not divided by K",
            "ece": float(ece), "ece_bins": bins, "reliability_bins": reliability,
            "confusion": confusion.tolist(),
            "per_class": [{"index": i, "label": names[i], "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]), "support": int(confusion[i].sum())}
                for i in range(len(names))]}


def fit_temperature(logits, targets, *, minimum=0.05, maximum=20.0):
    """Deterministic bounded 1-D calibration; never accepts an assessment argument."""
    z, y = _arrays(logits, targets)
    if not 0 < minimum <= 1 <= maximum or minimum >= maximum:
        raise ValueError("Temperature bounds must bracket 1.")

    def objective(log_t):
        lp = _log_probabilities(z, math.exp(log_t))
        return float(-lp[np.arange(len(y)), y].mean())

    left, right = math.log(minimum), math.log(maximum)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    fc, fd = objective(c), objective(d)
    for _ in range(80):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(d)
    candidates = [0.0, math.log(minimum), math.log(maximum), (left + right) / 2]
    best = min(candidates, key=lambda value: (objective(value), abs(value)))
    temperature = math.exp(best)
    return {"temperature": temperature, "nll_before": objective(0.0), "nll_after": objective(best),
            "bounds": [minimum, maximum], "samples": len(y),
            "method": "deterministic bounded golden-section log-temperature search, with T=1 guard",
            "scope": "Fitted and described on calibration data only; in-sample NLL is not test calibration quality."}


def wilson_interval(successes, count, *, z=1.959963984540054):
    """Approximate two-sided 95% binomial interval; not a coverage guarantee."""
    if type(successes) is not int or type(count) is not int or count < 0 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial counts.")
    if count == 0:
        return [None, None]
    p, denominator = successes / count, 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def select_review_threshold(probabilities, targets, *, target_accuracy=0.95, min_accepted=30):
    """Maximum calibration coverage meeting the empirical accuracy/count target.

    Tied confidence values stay together. No eligible threshold returns None;
    the caller must keep that release as a non-serving candidate.
    """
    p, y = np.asarray(probabilities, dtype=np.float64), np.asarray(targets)
    if p.ndim != 2 or not len(p) or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("Probabilities must be a finite nonempty N x K array.")
    if ((p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-7)
            or y.shape != (len(p),) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError("Invalid probabilities or targets.")
    if not 0 < target_accuracy <= 1 or type(min_accepted) is not int or min_accepted < 1:
        raise ValueError("Invalid threshold-selection criteria.")
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    order = np.argsort(-confidence, kind="stable")
    sorted_conf = confidence[order]
    cumulative = np.cumsum(correct[order])
    boundaries = np.flatnonzero(np.r_[sorted_conf[:-1] != sorted_conf[1:], True])
    eligible = [int(i) for i in boundaries if i + 1 >= min_accepted and cumulative[i] / (i + 1) >= target_accuracy]
    count = eligible[-1] + 1 if eligible else 0
    successes = int(cumulative[count - 1]) if count else 0
    return {"eligible": bool(eligible), "review_threshold": float(sorted_conf[count - 1]) if count else None,
            "accepted_count": count, "samples": len(y), "coverage": count / len(y),
            "accepted_accuracy": successes / count if count else None,
            "accepted_correct": successes, "accepted_accuracy_wilson_95": wilson_interval(successes, count),
            "target_accuracy": target_accuracy, "minimum_accepted": min_accepted,
            "policy": "accept confidence >= threshold; maximize calibration coverage; keep confidence ties",
            "scope": "Threshold chosen on calibration data; empirical target and interval are not a future accuracy guarantee."}


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_hash(data, role):
    indices = np.flatnonzero(data["roles"] == ROLE_NAMES.index(role))
    return _json_hash({"role": role, "prepared_bindings": data["bindings"],
                      "records": [{"row_index": int(i), "record": data["records"][int(i)]} for i in indices]})


def _new_model(factory, names, seed):
    model = factory(num_classes=len(names), seed=seed, pretrained=False)
    return model.eval()


def _load_model(factory, names, seed, path, device):
    import torch
    model = _new_model(factory, names, seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def _infer(model, images, indices, *, device, batch_size, corruption=None, stop_check=None):
    import torch
    from .training import prepare_batch
    collected = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            if stop_check is not None and stop_check():
                raise TimeoutError("Evaluation deadline/budget reached before the next inference batch.")
            chosen = indices[start:start + batch_size]
            if corruption is None:
                tensor = prepare_batch(images, chosen, device=device)
            else:
                batch = _corrupt(np.asarray(images[chosen]), corruption, chosen)
                tensor = prepare_batch(batch, np.arange(len(batch)), device=device)
            logits = model(tensor).detach().cpu().numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError("Evaluation produced non-finite logits.")
            collected.append(logits)
    if not collected:
        raise ValueError("Evaluation partition is empty.")
    return np.concatenate(collected, axis=0)


def _cached_logits(directory, model, data, role, weights_sha, *, device, batch_size, corruption=None, stop_check=None):
    directory.mkdir(parents=True, exist_ok=True)
    key = {"weights_sha256": weights_sha, "split_sha256": _split_hash(data, role),
           "corruption": corruption, "inference_device": str(device), "batch_size": int(batch_size),
           "backend": snapshot_evaluation_backend(device), "schema": 2}
    npz, receipt = directory / "logits.npz", directory / "LOGITS_RECEIPT.json"
    indices = np.flatnonzero(data["roles"] == ROLE_NAMES.index(role))
    if npz.is_file() and receipt.is_file():
        recorded = _load_json(receipt)
        if recorded["binding"] != key or digest(npz) != recorded["sha256"]:
            raise ValueError("Saved evaluation logits failed integrity/binding checks.")
        with np.load(npz, allow_pickle=False) as archive:
            logits = archive["logits"].copy()
            if not np.array_equal(archive["indices"], indices) or not np.array_equal(archive["targets"], data["labels"][indices]):
                raise ValueError("Saved evaluation logits have different IDs/labels.")
        return logits, indices
    logits = _infer(model, data["images"], indices, device=device,
                    batch_size=batch_size, corruption=corruption, stop_check=stop_check)
    _save_npz(npz, logits=logits, targets=data["labels"][indices], indices=indices)
    atomic_json(receipt, {"binding": key, "sha256": digest(npz), "device": str(device),
        "scope": "Fixed-checkpoint inference; saved complete passes reused on resume. A crash before commit may repeat an incomplete pass."})
    return logits, indices


def _corrupt(images, spec, original_indices):
    from PIL import Image, ImageFilter
    kind, severity = spec["kind"], spec["severity"]
    output = []
    for image, row in zip(images, original_indices):
        if kind == "noise":
            generator = np.random.default_rng(20261028 + int(row) * 11 + severity)
            result = np.clip(image.astype(np.float32) + generator.normal(0, spec["sigma"] * 255, image.shape), 0, 255).astype(np.uint8)
        elif kind == "blur":
            result = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(spec["radius"])))
        elif kind == "translation":
            shift = spec["pixels"]
            result = np.zeros_like(image)
            result[shift:, shift:] = image[:-shift, :-shift]
        else:
            raise ValueError("Unknown corruption kind.")
        output.append(result)
    return np.stack(output)


def _write_predictions(directory, logits, targets, indices, records, temperature=1.0):
    probabilities = np.exp(_log_probabilities(np.asarray(logits, dtype=np.float64), temperature))
    _save_npz(directory / "probabilities.npz", probabilities=probabilities,
              targets=targets, indices=indices, temperature=np.asarray(temperature))
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["row_index", "annotation_id", "image_id", "source_group_id", "true", "predicted", "nll", "confidence"]
        fields += [f"probability_{i}" for i in range(probabilities.shape[1])]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        lp = _log_probabilities(np.asarray(logits, dtype=np.float64), temperature)
        for j, index in enumerate(indices):
            record = records[int(index)]
            row = {"row_index": int(index), "annotation_id": record.get("annotation_id"),
                   "image_id": record.get("image_id"), "source_group_id": record.get("source_group_id"),
                   "true": int(targets[j]), "predicted": int(probabilities[j].argmax()),
                   "nll": float(-lp[j, targets[j]]), "confidence": float(probabilities[j].max())}
            row.update({f"probability_{i}": float(p) for i, p in enumerate(probabilities[j])})
            writer.writerow(row)


def _verify_tasks(out, config, data):
    import torch
    manifest_path = out / "study_manifest.json"
    manifest = _load_json(manifest_path)
    binding = digest(manifest_path)
    expected = [(recipe, seed) for recipe in config["recipes"] for seed in config["seeds"]]
    if len(expected) != 6 or len(set(expected)) != 6:
        raise ValueError("Finalization requires exactly two recipes and three seeds (six complete tasks).")
    if (manifest.get("config") != config or list(manifest.get("labels", [])) != list(data["classnames"])
            or manifest.get("data") != data["bindings"]):
        raise ValueError("Study config or labels differ from finalization inputs.")
    tasks = []
    for recipe, seed in expected:
        task_id = f"{recipe}_seed{seed}"
        directory = out / "tasks" / task_id
        if not (directory / "COMPLETE.json").is_file():
            raise ValueError(f"Incomplete study: task {task_id} has no COMPLETE receipt. Assessment was not opened.")
        receipt = _load_json(directory / "COMPLETE.json")
        task = _load_json(directory / "task.json")
        if receipt.get("task_id") != task_id or receipt.get("binding") != binding:
            raise ValueError("Task receipt differs from immutable study binding.")
        declared_task = next((item for item in manifest.get("tasks", []) if item.get("task_id") == task_id), None)
        if task != declared_task:
            raise ValueError("Task differs from the immutable study manifest.")
        for name in ("task.json", "best.pt", "history.json", "epoch_state.pt"):
            item = receipt.get("files", {}).get(name)
            expected_digest = item.get("sha256") if isinstance(item, dict) else item
            if expected_digest != digest(directory / name):
                raise ValueError(f"Completed task artifact hash mismatch: {task_id}/{name}")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if (checkpoint.get("binding") != binding or checkpoint.get("task") != task
                or task.get("recipe") != recipe or task.get("seed") != seed
                or task.get("epochs") != config["epochs"]):
            raise ValueError("Checkpoint/task identity differs from declared experiment.")
        history = _load_json(directory / "history.json")
        if len(history) != config["epochs"] or [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Task history is incomplete.")
        if not 1 <= checkpoint.get("epoch", 0) <= config["epochs"]:
            raise ValueError("Selected checkpoint epoch is invalid.")
        best_row = min(history, key=lambda row: (-row["selection"]["macro_f1"], row["selection"]["loss"], row["epoch"]))
        if checkpoint["epoch"] != best_row["epoch"] or checkpoint["selection"] != best_row["selection"]:
            raise ValueError("Checkpoint violates the predeclared F1/CE/earliest selection rule.")
        selected = checkpoint["selection"]
        for key in ("loss", "accuracy", "macro_f1"):
            if not math.isfinite(selected[key]):
                raise ValueError("Saved selection metrics are not finite.")
        tasks.append({"task_id": task_id, "recipe": recipe, "seed": seed,
                      "directory": directory, "checkpoint": directory / "best.pt",
                      "weights_sha256": digest(directory / "best.pt"), "epoch": checkpoint["epoch"],
                      "selection": selected, "history": history, "fit_seconds": receipt["fit_seconds"]})
    return binding, manifest, tasks


def _data_gate(data, prepared):
    groups = {"source_group_id": {}, "evaluation_group_id": {}}
    errors = []
    counts = {}
    for role_index, role in enumerate(ROLE_NAMES):
        chosen = np.flatnonzero(data["roles"] == role_index)
        counts[role] = np.bincount(data["labels"][chosen], minlength=len(data["classnames"])).tolist()
        if not len(chosen) or any(n == 0 for n in counts[role]):
            errors.append(f"Empty role/class support: {role}")
        for index in chosen:
            record = data["records"][int(index)]
            for field in groups:
                group = record.get(field)
                if not group:
                    errors.append(f"Missing {field} identity")
                    continue
                if group in groups[field] and groups[field][group] != role:
                    errors.append(f"{field} crosses partitions")
                groups[field][group] = role
    readiness_path = Path(prepared) / "DATA_READINESS.json"
    readiness = _load_json(readiness_path) if readiness_path.is_file() else {}
    ready_value = readiness.get("training_ready")
    if ready_value is None and isinstance(readiness.get("readiness"), dict):
        ready_value = readiness["readiness"].get("training_ready")
    if ready_value is not True:
        errors.append("Prepared data readiness does not explicitly mark training_ready true")
    return {"status": "pass" if not errors else "fail", "class_counts": counts,
            "failures": sorted(set(errors)), "prepared_bindings": data["bindings"],
            "scope": "Cached data integrity, nonempty class support and source-group separation; not legal clearance or independent garment identity."}


def _corruption_specs():
    return [{"kind": kind, "severity": i + 1, key: value}
            for kind, key, values in [("noise", "sigma", [0.02, 0.05, 0.1]),
                                      ("blur", "radius", [0.5, 1.0, 2.0]),
                                      ("translation", "pixels", [2, 4, 8])]
            for i, value in enumerate(values)]


def _export_and_benchmark(model, data, indices, release_dir, config, *, smoke):
    import torch
    from .training import prepare_batch
    result = {"status": "fail", "scope": "CPU FP32 tensor-logit parity after choices frozen; no assessment-driven model selection.",
              "max_abs_tolerance": config.get("onnx_max_abs_error", 1e-4),
              "export_settings": {"dynamo": True, "optimize": False, "opset_version": 18,
                  "external_data": False, "onnxruntime_graph_optimization_level": "ORT_ENABLE_ALL",
                  "onnxruntime_providers": ["CPUExecutionProvider"],
                  "onnxruntime_intra_op_num_threads": 1, "onnxruntime_inter_op_num_threads": 1}}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        import onnxruntime as ort
        result["runtime"] = {"torch": str(torch.__version__), "onnxruntime": ort.__version__}
        model = model.to("cpu").eval()
        audit_indices = indices[:64]
        tensor = prepare_batch(data["images"], audit_indices, device="cpu")
        dummy = tensor[:min(2, len(tensor))]
        target = release_dir / "model.onnx"
        # Preserve BatchNorm in the exported graph: exporter-side folding exceeded
        # the fixed FP32 parity tolerance on the returned fine-tuned checkpoint.
        torch.onnx.export(model, (dummy,), str(target), dynamo=True, optimize=False,
            input_names=["images"], output_names=["logits"], external_data=False,
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=128)},),
            opset_version=18)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(target), sess_options=options, providers=["CPUExecutionProvider"])
        errors, disagreements = [], 0
        evidence = {}
        for batch in sorted(set([1, min(3, len(tensor)), len(tensor)])):
            sample = tensor[:batch].contiguous()
            with torch.inference_mode():
                expected = model(sample).cpu().numpy()
            actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise RuntimeError("ONNX output shape/nonfinite parity failure.")
            errors.append(float(np.max(np.abs(expected - actual))))
            disagreements += int(np.sum(expected.argmax(1) != actual.argmax(1)))
            evidence[f"pytorch_batch_{batch}"] = expected
            evidence[f"onnx_batch_{batch}"] = actual
        _save_npz(release_dir / "onnx_parity_outputs.npz", **evidence)
        result.update({"status": "pass" if max(errors) <= result["max_abs_tolerance"] and not disagreements else "fail",
                       "maximum_absolute_error": max(errors), "top1_disagreements": disagreements,
                       "batch_sizes": sorted(set([1, min(3, len(tensor)), len(tensor)])),
                       "distinct_assessment_cases": len(audit_indices), "model_sha256": digest(target),
                       "declared_batch_range": [1, 128], "tested_batch_range_is_only_the_recorded_sizes": True})
        if config.get("cpu_benchmark", True):
            sample = tensor[:1].contiguous()
            def pytorch_call():
                with torch.inference_mode():
                    model(sample)
            def onnx_call():
                session.run(None, {session.get_inputs()[0].name: sample.numpy()})
            timing = {}
            for label, call in [("pytorch", pytorch_call), ("onnxruntime", onnx_call)]:
                for _ in range(5):
                    call()
                times = []
                for _ in range(30):
                    start = time.perf_counter()
                    call()
                    times.append((time.perf_counter() - start) * 1000)
                timing[label] = {"individual_call_ms": times, "median_ms": float(np.median(times)),
                                 "p95_ms": float(np.percentile(times, 95))}
            atomic_json(release_dir / "CPU_BENCHMARK.json", {"scope": "Warm CPU forward-only, input tensor already in memory; excludes decoding/transfer/network. Not service latency.",
                "batch_size": 1, "cpu_threads": 1, "warmup_calls": 5, "measured_calls": 30,
                "processor": platform.processor(), "platform": platform.platform(),
                "torch": str(torch.__version__), "onnxruntime": ort.__version__, "smoke_only": smoke,
                "export_settings": result["export_settings"],
                "measurements": timing})
    except Exception as error:
        result.update({"status": "fail", "exception_type": type(error).__name__, "message": str(error)[:1500]})
    finally:
        torch.set_num_threads(old_threads)
    atomic_json(release_dir / "ONNX_PARITY.json", result)
    return result


def _figures(evaluation, tasks, assessment, names, deployment_id, corruption_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = evaluation / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for task in tasks:
        epochs = [row["epoch"] for row in task["history"]]
        for ax, key in zip(axes, ["loss", "accuracy", "macro_f1"]):
            line, = ax.plot(epochs, [row["selection"][key] for row in task["history"]], label=task["task_id"], alpha=.85)
            ax.plot(epochs, [row["train"][key] for row in task["history"]], color=line.get_color(), linestyle="--", alpha=.5)
            ax.set(xlabel="Epoch", ylabel=f"Selection {key}")
    fig.suptitle("Solid: selection; dashed: training (with training augmentation/dropout)", fontsize=10)
    axes[0].legend(fontsize=6)
    fig.savefig(figure_dir / "selection_curves.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    cm = np.asarray(assessment[deployment_id]["confusion"])
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True",
           title="Fixed deployment checkpoint: assessment confusion")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.savefig(figure_dir / "assessment_confusion.png", dpi=150)
    plt.close(fig)
    if corruption_rows:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        for kind in ("noise", "blur", "translation"):
            rows = [row for row in corruption_rows if row["condition"]["kind"] == kind]
            ax.plot([row["condition"]["severity"] for row in rows], [row["metrics"]["macro_f1"] for row in rows], marker="o", label=kind)
        ax.axhline(assessment[deployment_id]["macro_f1"], color="grey", linestyle="--", label="clean")
        ax.set(xlabel="Predeclared severity", ylabel="Macro-F1", title="Post-letterbox pixel stress test")
        ax.legend()
        fig.savefig(figure_dir / "corruption_stress.png", dpi=150)
        plt.close(fig)


def _diagnostic_figures(evaluation, data, indices, logits, temperature, raw, calibrated):
    """Descriptive, fixed-checkpoint diagnostics; never feed these back into selection."""
    import matplotlib.pyplot as plt
    folder = evaluation / "figures"
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    for label, metrics in [("Raw", raw), ("Temperature scaled", calibrated)]:
        rows = [row for row in metrics["reliability_bins"] if row["count"]]
        ax.plot([row["confidence"] for row in rows], [row["accuracy"] for row in rows], marker="o", label=label)
    ax.set(xlabel="Mean confidence in bin", ylabel="Accuracy in bin", xlim=(0, 1), ylim=(0, 1),
           title="Assessment reliability: fixed calibration policy")
    ax.legend()
    fig.savefig(folder / "assessment_reliability.png", dpi=150)
    plt.close(fig)

    probabilities = np.exp(_log_probabilities(np.asarray(logits, dtype=np.float64), temperature))
    predictions = probabilities.argmax(axis=1)
    correct = predictions == data["labels"][indices]
    chosen = list(np.flatnonzero(~correct)[:6]) + list(np.flatnonzero(correct)[:6])
    fig, axes = plt.subplots(3, 4, figsize=(12, 10), layout="constrained")
    credits = []
    for axis, position in zip(axes.flat, chosen):
        index = int(indices[position])
        record = data["records"][index]
        actual, predicted = int(data["labels"][index]), int(predictions[position])
        axis.imshow(data["images"][index])
        axis.set_title(f"ID {record.get('annotation_id', index)}\ntrue: {data['classnames'][actual]}\npred: {data['classnames'][predicted]} ({probabilities[position].max():.3f})", fontsize=8)
        credits.append({"row_index": index, "annotation_id": record.get("annotation_id"),
            "image_id": record.get("image_id"), "source_group_id": record.get("source_group_id"),
            "original_url": record.get("original_url"), "license_id": record.get("license_id"),
            "author": record.get("author") or "not_provided_in_annotations"})
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("First six errors and six correct cases by stable assessment order; descriptive examples", fontsize=10)
    fig.savefig(folder / "assessment_examples.jpg", dpi=140)
    plt.close(fig)
    with (folder / "ASSESSMENT_EXAMPLE_CREDITS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_index", "annotation_id", "image_id", "source_group_id", "original_url", "license_id", "author"])
        writer.writeheader()
        writer.writerows(credits)


def finalize_study(out, prepared, config, *, model_factory=None, smoke=False, stop_check=None):
    """Finalize all six complete tasks; a failed serving gate keeps a candidate.

    ``smoke=True`` permits tiny injectable software fixtures but can never create
    a ready real-garment release. It does not claim task/model quality.
    """
    import torch
    from .training import build_model, load_prepared
    if model_factory is not None and not smoke:
        raise ValueError("Injected model factories are allowed only with smoke=True; never create a real release from a fixture.")
    out, prepared = Path(out), Path(prepared)
    data = load_prepared(prepared)
    names = list(data["classnames"])
    factory = model_factory or build_model
    binding, manifest, tasks = _verify_tasks(out, config, data)
    if manifest.get("weight_source") == "injected_software_smoke_only" and not smoke:
        raise ValueError("An injected-model study is software smoke evidence only.")
    backend = configure_evaluation_backend()
    backend_path = out / "EVALUATION_BACKEND.json"
    if backend_path.exists():
        try:
            recorded_backend = _load_json(backend_path)
        except (ValueError, UnicodeError) as error:
            raise ValueError("Evaluation backend artifact changed or is unreadable.") from error
        if recorded_backend != backend:
            raise ValueError("Evaluation backend/runtime differs from its immutable receipt; use a separate recovery output.")
    else:
        atomic_json(backend_path, backend)
    backend_sha256 = digest(backend_path)
    final_path = out / "FINAL_SUMMARY.json"
    if final_path.is_file():
        previous = _load_json(final_path)
        if (previous.get("study_binding") != binding or previous.get("prepared_bindings") != data["bindings"]
                or previous.get("smoke_only") != smoke
                or previous.get("evaluation_backend_sha256") != backend_sha256):
            raise ValueError("Final summary binding differs; refusing to repeat assessment under new inputs.")
        for item in previous["artifacts"]:
            if digest(out / item["path"]) != item["sha256"]:
                raise ValueError("Finalized artifact changed after evaluation.")
        return previous
    evaluation = out / "evaluation"
    evaluation.mkdir(exist_ok=True)
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    batch_size = config.get("batch_size", 128)
    selection = {}
    for task in tasks:
        if stop_check is not None and stop_check():
            raise TimeoutError("Evaluation deadline/budget reached before the next selection task.")
        model = _load_model(factory, names, task["seed"], task["checkpoint"], device)
        task["total_parameters"] = sum(p.numel() for p in model.parameters())
        task["trainable_parameters"] = (sum(p.numel() for p in model.classifier[-1].parameters())
                                        if task["recipe"] == "frozen" else task["total_parameters"])
        logits, indices = _cached_logits(evaluation / task["task_id"] / "selection", model, data,
            "selection", task["weights_sha256"], device=device, batch_size=batch_size, stop_check=stop_check)
        metrics = classification_metrics(logits, data["labels"][indices], names)
        if (abs(metrics["macro_f1"] - task["selection"]["macro_f1"]) > 1e-8
                or abs(metrics["accuracy"] - task["selection"]["accuracy"]) > 1e-8
                or abs(metrics["loss"] - task["selection"]["loss"]) > 1e-5):
            raise ValueError("Recomputed checkpoint selection metrics do not match training evidence.")
        selection[task["task_id"]] = metrics
        atomic_json(evaluation / task["task_id"] / "selection" / "metrics.json", metrics)
        del model
    recipe_means = {recipe: float(np.mean([selection[t["task_id"]]["macro_f1"] for t in tasks if t["recipe"] == recipe]))
                    for recipe in config["recipes"]}
    winner = config["recipes"][0]
    for recipe in config["recipes"][1:]:
        if recipe_means[recipe] > recipe_means[winner] + 1e-12:
            winner = recipe
    deployment_seed = config.get("deployment_seed", 314)
    deployment = next((t for t in tasks if t["recipe"] == winner and t["seed"] == deployment_seed), None)
    if deployment is None:
        raise ValueError("Fixed deployment seed is absent; never select a seed using assessment.")
    # Calibrate and assess with the CPU FP32 backend used by the local service.
    # Selection verification above still uses the declared training backend.
    device = torch.device("cpu")
    batch_size = 1  # The HTTP endpoint serves one crop per call.
    model = _load_model(factory, names, deployment_seed, deployment["checkpoint"], device)
    calibration_dir = evaluation / "calibration"
    calibration_logits, calibration_indices = _cached_logits(calibration_dir, model, data,
        "calibration", deployment["weights_sha256"], device=device, batch_size=batch_size, stop_check=stop_check)
    calibration_targets = data["labels"][calibration_indices]
    temperature_fit = fit_temperature(calibration_logits, calibration_targets)
    temperature = temperature_fit["temperature"]
    calibrated_probs = np.exp(_log_probabilities(calibration_logits.astype(np.float64), temperature))
    threshold = select_review_threshold(calibrated_probs, calibration_targets,
        target_accuracy=config.get("calibration_target_accuracy", .95), min_accepted=config.get("calibration_min_accepted", 30))
    calibration = {"status": "fitted", "temperature": temperature,
        "review_threshold": threshold["review_threshold"], "fit_split_sha256": _split_hash(data, "calibration"),
        "threshold_selection_split_sha256": _split_hash(data, "calibration")}
    atomic_json(calibration_dir / "CALIBRATION.json", {"fit": temperature_fit, "threshold": threshold,
        "binding": calibration, "same_calibration_partition_used_for_temperature_and_threshold": True})
    _write_predictions(calibration_dir, calibration_logits, calibration_targets, calibration_indices, data["records"], temperature)
    data_gate = _data_gate(data, prepared)
    anchor = next(t for t in tasks if t["recipe"] == config["recipes"][0] and t["seed"] == deployment_seed)
    minimum = config.get("quality_min_selection_f1", .70)
    regression = config.get("quality_max_f1_regression", .005)
    deploy_f1 = selection[deployment["task_id"]]["macro_f1"]
    anchor_f1 = selection[anchor["task_id"]]["macro_f1"]
    quality_gate = {"status": "pass" if recipe_means[winner] >= minimum and deploy_f1 + regression >= anchor_f1 else "fail",
        "criterion": "selected recipe mean selection F1 >= preset floor; fixed deployment seed does not regress more than preset tolerance against frozen baseline same seed",
        "minimum_selection_macro_f1": minimum, "maximum_allowed_regression": regression,
        "recipe_mean": recipe_means[winner], "deployment_seed_selection_f1": deploy_f1, "anchor_seed_selection_f1": anchor_f1,
        "scope": "Selection data only; no test metric used to choose model or gate criterion."}
    calibration_gate = {"status": "pass" if threshold["eligible"] and temperature_fit["nll_after"] <= temperature_fit["nll_before"] + 1e-10 else "fail",
        "temperature_fit": temperature_fit, "threshold_selection": threshold,
        "scope": "In-sample calibration fit and empirical threshold eligibility, not guaranteed future accepted accuracy."}
    freeze = {"schema": 1, "study_binding": binding, "deployment_task_id": deployment["task_id"],
        "evaluation_backend_sha256": backend_sha256,
        "deployment_seed_rule": "preset config deployment_seed, never choose by assessment", "selected_recipe": winner,
        "recipe_selection_means": recipe_means, "calibration": calibration,
        "quality_criteria": {"minimum_selection_macro_f1": minimum, "maximum_f1_regression": regression},
        "corruptions": _corruption_specs(), "parity_max_abs_error": config.get("onnx_max_abs_error", 1e-4),
        "assessment_policy": "All model/calibration/threshold choices frozen before assessment inference. All six task assessment scores are descriptive; no test-driven selection.",
        "smoke_only": smoke}
    freeze_path = evaluation / "SELECTION_FREEZE.json"
    if freeze_path.exists() and _load_json(freeze_path) != freeze:
        raise ValueError("A prior selection freeze differs; do not reopen assessment under changed choices.")
    atomic_json(freeze_path, freeze)

    assessment = {}
    assessment_indices = np.flatnonzero(data["roles"] == 3)
    for task in tasks:
        if stop_check is not None and stop_check():
            raise TimeoutError("Evaluation deadline/budget reached before the next assessment task.")
        current = model if task["task_id"] == deployment["task_id"] else _load_model(factory, names, task["seed"], task["checkpoint"], device)
        directory = evaluation / task["task_id"] / "assessment"
        logits, indices = _cached_logits(directory, current, data, "assessment", task["weights_sha256"],
                                         device=device, batch_size=batch_size, stop_check=stop_check)
        assessment[task["task_id"]] = classification_metrics(logits, data["labels"][indices], names)
        atomic_json(directory / "metrics.json", assessment[task["task_id"]])
        _write_predictions(directory, logits, data["labels"][indices], indices, data["records"])
        if current is not model:
            del current
    deployment_dir = evaluation / deployment["task_id"] / "assessment"
    with np.load(deployment_dir / "logits.npz", allow_pickle=False) as archive:
        deployment_logits = archive["logits"].copy()
    calibrated_assessment = classification_metrics(deployment_logits, data["labels"][assessment_indices], names, temperature=temperature)
    calibrated_assessment["acceptance"] = _acceptance(deployment_logits, data["labels"][assessment_indices], temperature, threshold["review_threshold"])
    atomic_json(deployment_dir / "calibrated_metrics.json", calibrated_assessment)
    calibrated_directory = deployment_dir / "calibrated"
    calibrated_directory.mkdir(exist_ok=True)
    _write_predictions(calibrated_directory, deployment_logits,
        data["labels"][assessment_indices], assessment_indices, data["records"], temperature)
    corruptions = []
    for spec in _corruption_specs():
        if stop_check is not None and stop_check():
            raise TimeoutError("Evaluation deadline/budget reached before the next corruption pass.")
        directory = evaluation / "corruptions" / f"{spec['kind']}_severity{spec['severity']}"
        logits, indices = _cached_logits(directory, model, data, "assessment", deployment["weights_sha256"],
                                         device=device, batch_size=batch_size, corruption=spec, stop_check=stop_check)
        metrics = classification_metrics(logits, data["labels"][indices], names, temperature=temperature)
        metrics["acceptance"] = _acceptance(logits, data["labels"][indices], temperature, threshold["review_threshold"])
        atomic_json(directory / "metrics.json", metrics)
        _write_predictions(directory, logits, data["labels"][indices], indices, data["records"], temperature)
        corruptions.append({"condition": spec, "metrics": metrics,
            "macro_f1_delta_from_clean": metrics["macro_f1"] - calibrated_assessment["macro_f1"]})

    release_dir = out / "release_candidate"
    if stop_check is not None and stop_check():
        raise TimeoutError("Evaluation deadline/budget reached before model export.")
    release_dir.mkdir(exist_ok=True)
    shutil.copyfile(deployment["checkpoint"], release_dir / "weights.pt")
    weights_sha = digest(release_dir / "weights.pt")
    parity = _export_and_benchmark(model, data, assessment_indices, release_dir, config, smoke=smoke)
    gates = {"data": data_gate, "quality": quality_gate, "calibration": calibration_gate, "parity": parity}
    ready = not smoke and all(gate["status"] == "pass" for gate in gates.values())
    release_id = config.get("study_id", "garmentlab") + "-" + winner + f"-seed{deployment_seed}"
    receipt = {"format_version": 1, "release_id": release_id, "weights_sha256": weights_sha,
               "calibration": calibration, "gates": gates, "study_binding": binding,
               "selection_freeze_sha256": digest(freeze_path), "smoke_only": smoke,
               "scope": "Automated local research release checks; not a human sign-off, legal clearance or production readiness certification."}
    atomic_json(release_dir / "evaluation_receipt.json", receipt)
    if smoke:
        atomic_json(release_dir / "SMOKE_ONLY.json", {"status": "SMOKE_ONLY_NOT_A_REAL_MODEL_RELEASE", "gates": gates})
    else:
        release = {"format_version": 1, "release_id": release_id, "domain": "garment_crop_rgb",
            "architecture": "mobilenet_v3_small", "status": "ready" if ready else "candidate", "labels": names,
            "preprocessing": {"version": "garment_rgb_letterbox_v1", "height": int(data["images"].shape[1]),
                "width": int(data["images"].shape[2]), "channels": 3, "mode": "RGB", "resize": "letterbox_bilinear",
                "fill": [0, 0, 0], "normalization": "imagenet_v1"},
            "weights": {"file": "weights.pt", "sha256": weights_sha}, "calibration": calibration,
            "evaluation_receipt": {"file": "evaluation_receipt.json", "sha256": digest(release_dir / "evaluation_receipt.json")},
            "scope": "Single preselected garment crop in the supported label space; not arbitrary-photo object detection."}
        atomic_json(release_dir / "release.json", release)
        from .release import validate_release
        validate_release(release_dir, require_ready=ready)
    _figures(evaluation, tasks, assessment, names, deployment["task_id"], corruptions)
    _diagnostic_figures(evaluation, data, assessment_indices, deployment_logits, temperature,
                        assessment[deployment["task_id"]], calibrated_assessment)
    summary_by_recipe = {}
    for recipe in config["recipes"]:
        rows = [task for task in tasks if task["recipe"] == recipe]
        summary_by_recipe[recipe] = {}
        for phase, metrics in [("selection", selection), ("assessment", assessment)]:
            summary_by_recipe[recipe][phase] = {key: {"mean": float(np.mean([metrics[t["task_id"]][key] for t in rows])),
                "sample_sd": float(np.std([metrics[t["task_id"]][key] for t in rows], ddof=1)), "seeds": [t["seed"] for t in rows]}
                for key in ("accuracy", "macro_f1", "loss")}
    summary = {"status": "SMOKE_ONLY_COMPLETE" if smoke else ("COMPLETE_RESEARCH_ARTIFACTS" if ready else "COMPLETE_CANDIDATE_NOT_READY"),
        "completed_utc": utc_now().isoformat(), "study_binding": binding, "prepared_bindings": data["bindings"],
        "evaluation_backend_sha256": backend_sha256,
        "smoke_only": smoke, "ready_for_local_service": ready, "labels": names, "task_count": len(tasks),
        "selected_recipe": winner, "deployment_task_id": deployment["task_id"], "deployment_seed": deployment_seed,
        "recipe_summary": summary_by_recipe, "selection": selection, "assessment": assessment,
        "task_costs": [{key: task[key] for key in ("task_id", "total_parameters", "trainable_parameters", "fit_seconds")} for task in tasks],
        "cost_scope": "Frozen-task fit seconds exclude shared feature extraction. Read feature_cache/COMPLETE.json for that additional measured cost; do not compare head-only time to full fine-tuning time.",
        "deployment_calibrated_assessment": calibrated_assessment, "calibration": calibration,
        "corruptions": corruptions, "gates": gates, "inference_device": str(device), "inference_batch_size": batch_size,
        "release_directory": release_dir.relative_to(out).as_posix(), "selection_freeze": freeze_path.relative_to(out).as_posix(),
        "limitations": ["Assessment is conditional on the declared data subset and cropping procedure, not smartphone-photo validation.",
            "Seed sample SD is not uncertainty over independent datasets or garments.",
            "Calibration temperature and threshold share one calibration partition; its selected accepted accuracy is optimistic.",
            "Wilson intervals treat individual accepted crops as Bernoulli samples; repeated views/source groups can be dependent.",
            "Corruptions alter already letterboxed 224px RGB arrays; these are project-specific stress tests, not an official corruption benchmark.",
            "No human review, legal clearance, production traffic or uptime is asserted by automated passing gates."],
        "artifacts": [{"path": backend_path.relative_to(out).as_posix(), "sha256": backend_sha256,
                       "bytes": backend_path.stat().st_size}]}
    for base in (evaluation, release_dir):
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                summary["artifacts"].append({"path": path.relative_to(out).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    atomic_json(final_path, summary)
    return summary


def _acceptance(logits, targets, temperature, threshold):
    p = np.exp(_log_probabilities(np.asarray(logits, dtype=np.float64), temperature))
    accepted = p.max(axis=1) >= threshold if threshold is not None else np.zeros(len(p), dtype=bool)
    count = int(accepted.sum())
    correct = int((p.argmax(axis=1)[accepted] == targets[accepted]).sum())
    return {"threshold": threshold, "accepted_count": count, "samples": len(p), "coverage": count / len(p),
            "accepted_accuracy": correct / count if count else None,
            "accepted_accuracy_wilson_95": wilson_interval(correct, count),
            "scope": "Frozen calibration threshold evaluated descriptively; no threshold adjustment on this split."}
