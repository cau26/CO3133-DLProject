"""Choice-frozen v2 evaluation; the existing assessment is regression evidence.

Training, candidate/seed selection and calibration choices never consult
assessment metrics. A successful technical release is not a 95% accuracy claim.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import evaluate as legacy
from .backend import configure_evaluation_backend
from .calibration_v2 import fit_calibration_v2, partition_calibration, review_mask
from .probabilities import log_probabilities
from .runtime import atomic_json, digest, utc_now
from .study_v2 import validate_config, verify_final_tasks
from .training import build_model, load_prepared

ASSESSMENT_SCOPE = "POST_TEST_EXPLORATORY_REGRESSION"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _immutable(path, value):
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Immutable v2 evaluation receipt differs: {path.name}")
    else:
        atomic_json(path, value)


def _stop(check, stage):
    if check is not None and check():
        raise TimeoutError("Evaluation stopped before " + stage + "; completed passes are reusable")


def acceptance_metrics(logits, targets, calibration):
    """Crop-weighted descriptive acceptance; honors review_all at confidence1."""
    p = np.exp(log_probabilities(logits, calibration["temperature"]))
    accepted = ~review_mask(p, calibration)
    count = int(accepted.sum())
    correct = int((p.argmax(1)[accepted] == np.asarray(targets)[accepted]).sum())
    return {"policy_mode": calibration["policy"]["mode"], "threshold": calibration["review_threshold"],
            "samples": len(p), "accepted_count": count, "accepted_correct": correct,
            "coverage": count / len(p), "accepted_accuracy": correct / count if count else None,
            "scope": "Descriptive crop-weighted regression score with frozen policy; not the independent-component bound or a future accuracy guarantee."}


def _release_manifest(config, names, data, task, calibration, release, ready=False):
    release_id = config["study_id"] + "-" + task["candidate_id"] + f"-seed{task['seed']}"
    return {"format_version": 1, "release_id": release_id, "domain": "garment_crop_rgb",
            "architecture": "mobilenet_v3_small", "status": "ready" if ready else "candidate", "labels": names,
            "preprocessing": {"version": "garment_rgb_letterbox_v1", "height": int(data["images"].shape[1]),
                "width": int(data["images"].shape[2]), "channels": 3, "mode": "RGB", "resize": "letterbox_bilinear",
                "fill": [0, 0, 0], "normalization": "imagenet_v1"},
            "weights": {"file": "weights.pt", "sha256": digest(release / "weights.pt")},
            "calibration": calibration,
            "evaluation_receipt": {"file": "evaluation_receipt.json", "sha256": digest(release / "evaluation_receipt.json")},
            "scope": "One selected six-class garment crop. Local technical readiness is separate from statistical risk eligibility and production validation."}


def _serving_check(out, release, manifest, data, indices, logits, calibration):
    """Only an unexposed temporary copy may carry ready during preflight."""
    from .inference import Predictor
    from .release import validate_release
    from tools.reexport_garmentlab import _api_check, _png_cases, _predictor_check
    probabilities = np.exp(log_probabilities(logits, calibration["temperature"]))
    accepted = ~review_mask(probabilities, calibration)
    cases = _png_cases(data, indices, probabilities, accepted, calibration["review_threshold"])
    with tempfile.TemporaryDirectory(prefix=".serving-v2-", dir=out) as temporary:
        staging = Path(temporary)
        for name in ("weights.pt", "model.onnx", "evaluation_receipt.json"):
            shutil.copyfile(release / name, staging / name)
        staged_manifest = dict(manifest, status="ready")
        atomic_json(staging / "release.json", staged_manifest)
        validate_release(staging, require_ready=True)
        predictor = Predictor(staging)
        png = _predictor_check(predictor, cases, 1e-4)
        api = asyncio.run(_api_check(predictor, cases, 1e-4))
    report = {"status": "pass" if png["status"] == api["status"] == "pass" else "fail",
              "png_predictor": png, "http_contract": api,
              "scope": "Actual local CPU Predictor and in-process HTTP contracts on recorded prepared crops. No public deployment or arbitrary-photo validation."}
    return report


def finalize_study_v2(out, prepared, config, *, model_factory=None, smoke=False, stop_check=None):
    """Finalize a complete controlled study without assessment-driven choices."""
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _finalize(out, prepared, config, model_factory=model_factory, smoke=smoke, stop_check=stop_check)
    finally:
        torch.set_num_threads(previous_threads)


def _finalize(out, prepared, config, *, model_factory, smoke, stop_check):
    import torch
    if (model_factory is not None) != bool(smoke):
        raise ValueError("Injected models require explicit smoke=True; real evaluation needs its actual architecture")
    config = validate_config(config)
    out, prepared = Path(out).resolve(strict=True), Path(prepared).resolve(strict=True)
    frozen, descriptors = verify_final_tasks(out)
    manifest = _json(out / "study_v2_manifest.json")
    if manifest["config"] != config or frozen["smoke_only"] != smoke:
        raise ValueError("Finalizer config or smoke scope differs from the completed training study")
    data = load_prepared(prepared)
    if data["bindings"] != frozen["data"] or data["classnames"] != frozen["labels"]:
        raise ValueError("Prepared data or class map differs from frozen choices")
    data_gate = legacy._data_gate(data, prepared)
    if data_gate["status"] != "pass" and not smoke:
        raise ValueError("Prepared data failed v2 technical gates before any model inference")
    if config.get("onnx_max_abs_error", 1e-4) != 1e-4:
        raise ValueError("V2 export retains the fixed absolute-logit tolerance1e-4")
    if not smoke and (not str(config["device"]).startswith("cuda") or not torch.cuda.is_available()
                      or "A100" not in torch.cuda.get_device_name(torch.device(config["device"]))):
        raise RuntimeError("Real selection verification requires the declared A100 environment")
    backend = configure_evaluation_backend()
    backend_path = out / "EVALUATION_BACKEND_V2.json"
    _immutable(backend_path, backend)
    source_paths = [Path(__file__), Path(legacy.__file__), Path(__file__).with_name("calibration_v2.py"),
                    Path(__file__).with_name("uncertainty_v2.py"),
                    Path(__file__).with_name("probabilities.py"), Path(__file__).with_name("inference.py"),
                    Path(__file__).with_name("release.py"), Path(__file__).with_name("api.py"),
                    Path(__file__).with_name("training.py"), Path(__file__).with_name("study_v2.py"),
                    Path(__file__).resolve().parents[1] / "tools/reexport_garmentlab.py"]
    identity = {"schema": "garmentlab_finalization_inputs_v2", "study_binding": frozen["study_binding"],
                "final_selection_sha256": digest(out / "FINAL_SELECTION.json"), "prepared_bindings": data["bindings"],
                "config_sha256": legacy._json_hash(config), "backend_sha256": digest(backend_path),
                "source_sha256": {p.name: digest(p) for p in source_paths}, "smoke_only": smoke,
                "assessment_scope": ASSESSMENT_SCOPE}
    identity_path = out / "EVALUATION_INPUTS_V2.json"
    _immutable(identity_path, identity)
    for path in source_paths:
        target = out / "evaluation_source_snapshot" / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(path):
            raise ValueError("Evaluation source snapshot differs")
        if not target.exists():
            target.write_bytes(path.read_bytes())
    final_path = out / "FINAL_SUMMARY_V2.json"
    if final_path.exists():
        previous = _json(final_path)
        if previous["evaluation_inputs_sha256"] != digest(identity_path):
            raise ValueError("Completed finalization inputs differ; refusing another assessment under changed choices")
        for item in previous["artifacts"]:
            path = out / item["path"]
            if not path.resolve().is_relative_to(out) or not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Finalized v2 artifact changed")
        return previous
    _stop(stop_check, "selection verification")
    evaluation = out / "evaluation_v2"
    evaluation.mkdir(exist_ok=True)
    tasks = []
    for descriptor in descriptors:
        task = dict(descriptor)
        path = (out / descriptor["checkpoint"]).resolve(strict=True)
        if not path.is_relative_to(out) or digest(path) != descriptor["weights_sha256"]:
            raise ValueError("Final checkpoint path or hash differs")
        task["checkpoint"] = path
        tasks.append(task)
    factory = model_factory or build_model
    names = data["classnames"]
    selection = {}
    for task in tasks:
        _stop(stop_check, "the next selection pass")
        model = legacy._load_model(factory, names, task["seed"], task["checkpoint"], config["device"])
        logits, indices = legacy._cached_logits(evaluation / task["task_id"] / "selection", model, data,
            "selection", task["weights_sha256"], device=config["device"], batch_size=config["batch_size"], stop_check=stop_check)
        metrics = legacy.classification_metrics(logits, data["labels"][indices], names)
        if any(abs(metrics[key] - task["selection"][key]) > tolerance
               for key, tolerance in [("accuracy", 1e-8), ("macro_f1", 1e-8), ("loss", 1e-5)]):
            raise ValueError("V2 selection recomputation differs from training evidence; no tolerance is relaxed")
        selection[task["task_id"]] = metrics
        atomic_json(evaluation / task["task_id"] / "selection/metrics.json", metrics)
        task["total_parameters"] = sum(p.numel() for p in model.parameters())
        del model
    deployment = next(t for t in tasks if t["task_id"] == frozen["deployment_task_id"])
    anchor = next(t for t in tasks if t["candidate_id"] == "C0" and t["seed"] == config["deployment_seed"])
    candidate_id = frozen["deployment_candidate_id"]
    mean_selection = float(np.mean([selection[t["task_id"]]["macro_f1"] for t in tasks if t["candidate_id"] == candidate_id]))
    minimum = config.get("quality_min_selection_f1", .70)
    regression = config["deployment_max_selection_f1_regression"]
    quality_gate = {"status": "pass" if mean_selection >= minimum and selection[deployment["task_id"]]["macro_f1"] >= selection[anchor["task_id"]]["macro_f1"] - regression else "fail",
                    "minimum_selection_macro_f1": minimum, "maximum_selection_regression": regression,
                    "deployment_candidate_mean_selection_macro_f1": mean_selection,
                    "deployment_guard": frozen["deployment_guard"], "scope": "Selection-only quality criteria and fixed-seed fallback; no assessment-driven choice."}
    calibration_indices = np.flatnonzero(data["roles"] == 2)
    records = [data["records"][int(i)] for i in calibration_indices]
    partition = partition_calibration(records, config.get("calibration", {}))
    partition_path = evaluation / "CALIBRATION_PARTITION.json"
    _immutable(partition_path, {"partition": partition, "global_array_indices": calibration_indices.tolist(),
                               "original_calibration_role_sha256": legacy._split_hash(data, "calibration")})
    _stop(stop_check, "calibration inference")
    model = legacy._load_model(factory, names, deployment["seed"], deployment["checkpoint"], "cpu")
    calibration_dir = evaluation / "calibration"
    calibration_logits, actual_indices = legacy._cached_logits(calibration_dir, model, data, "calibration",
        deployment["weights_sha256"], device="cpu", batch_size=1, stop_check=stop_check)
    if not np.array_equal(actual_indices, calibration_indices):
        raise ValueError("Calibration indices changed after freezing metadata partition")
    fitted = fit_calibration_v2(calibration_logits, data["labels"][calibration_indices], records,
        config.get("calibration", {}), partition=partition)
    calibration = fitted["binding"]
    calibration_path = calibration_dir / "CALIBRATION_V2.json"
    _immutable(calibration_path, fitted)
    legacy._write_predictions(calibration_dir, calibration_logits, data["labels"][calibration_indices],
                              calibration_indices, data["records"], calibration["temperature"])
    calibration_gate = {"status": "pass", "temperature_fit_valid": True,
                        "policy_mode": calibration["policy"]["mode"],
                        "target_bound_satisfied_under_assumptions": fitted["risk_control"]["target_bound_satisfied_under_assumptions"],
                        "unconditional_accuracy_guarantee": False,
                        "scope": "Valid disjoint calibration fit and explicit review policy. Review-all is technically valid even when no risk threshold qualifies; passing this gate is not a95% accuracy guarantee."}
    freeze = {"schema": "garmentlab_choices_before_regression_assessment_v2", "evaluation_inputs_sha256": digest(identity_path),
              "final_selection_sha256": digest(out / "FINAL_SELECTION.json"), "deployment_task_id": deployment["task_id"],
              "calibration_partition_sha256": digest(partition_path), "calibration_fit_sha256": digest(calibration_path),
              "calibration": calibration, "quality_criteria": quality_gate,
              "corruptions": legacy._corruption_specs(), "parity_max_abs_error": config.get("onnx_max_abs_error", 1e-4),
              "assessment_scope": ASSESSMENT_SCOPE, "smoke_only": smoke,
              "scope": "All model, seed, calibration and policy choices frozen before regression assessment inference. Previously inspected assessment cannot become a new blind test."}
    freeze_path = evaluation / "SELECTION_FREEZE.json"
    _immutable(freeze_path, freeze)
    assessment, assessment_indices = {}, np.flatnonzero(data["roles"] == 3)
    for task in tasks:
        _stop(stop_check, "the next regression assessment pass")
        current = model if task["task_id"] == deployment["task_id"] else legacy._load_model(factory, names, task["seed"], task["checkpoint"], "cpu")
        directory = evaluation / task["task_id"] / "assessment"
        logits, indices = legacy._cached_logits(directory, current, data, "assessment", task["weights_sha256"],
                                                device="cpu", batch_size=1, stop_check=stop_check)
        metrics = legacy.classification_metrics(logits, data["labels"][indices], names)
        metrics["evaluation_scope"] = ASSESSMENT_SCOPE
        assessment[task["task_id"]] = metrics
        atomic_json(directory / "metrics.json", metrics)
        legacy._write_predictions(directory, logits, data["labels"][indices], indices, data["records"])
        if current is not model:
            del current
    deployment_dir = evaluation / deployment["task_id"] / "assessment"
    from .uncertainty_v2 import paired_group_bootstrap
    predictions = {}
    for task in tasks:
        with np.load(evaluation / task["task_id"] / "assessment/logits.npz", allow_pickle=False) as saved:
            predictions[task["task_id"]] = saved["logits"].argmax(1)
    uncertainty = paired_group_bootstrap(data["labels"][assessment_indices], predictions, tasks,
        [data["records"][int(i)] for i in assessment_indices], selected_candidate=frozen["selection_candidate_id"],
        classes=len(names), replicates=config.get("bootstrap_replicates", 2000))
    atomic_json(evaluation / "GROUPED_UNCERTAINTY.json", uncertainty)
    with np.load(deployment_dir / "logits.npz", allow_pickle=False) as archive:
        deployment_logits = archive["logits"].copy()
    calibrated = legacy.classification_metrics(deployment_logits, data["labels"][assessment_indices], names,
                                                temperature=calibration["temperature"])
    calibrated["acceptance"] = acceptance_metrics(deployment_logits, data["labels"][assessment_indices], calibration)
    calibrated["evaluation_scope"] = ASSESSMENT_SCOPE
    atomic_json(deployment_dir / "calibrated_metrics.json", calibrated)
    calibrated_dir = deployment_dir / "calibrated"
    calibrated_dir.mkdir(exist_ok=True)
    legacy._write_predictions(calibrated_dir, deployment_logits, data["labels"][assessment_indices], assessment_indices,
                              data["records"], calibration["temperature"])
    corruptions = []
    for spec in legacy._corruption_specs():
        _stop(stop_check, "the next fixed corruption pass")
        folder = evaluation / "corruptions" / f"{spec['kind']}_severity{spec['severity']}"
        logits, indices = legacy._cached_logits(folder, model, data, "assessment", deployment["weights_sha256"],
                                                device="cpu", batch_size=1, corruption=spec, stop_check=stop_check)
        score = legacy.classification_metrics(logits, data["labels"][indices], names, temperature=calibration["temperature"])
        score["acceptance"] = acceptance_metrics(logits, data["labels"][indices], calibration)
        score["evaluation_scope"] = ASSESSMENT_SCOPE
        atomic_json(folder / "metrics.json", score)
        legacy._write_predictions(folder, logits, data["labels"][indices], indices, data["records"], calibration["temperature"])
        corruptions.append({"condition": spec, "metrics": score, "macro_f1_delta_from_clean": score["macro_f1"]-calibrated["macro_f1"]})
    _stop(stop_check, "CPU export")
    release = out / "release_candidate"
    release.mkdir(exist_ok=True)
    shutil.copyfile(deployment["checkpoint"], release / "weights.pt")
    if digest(release / "weights.pt") != deployment["weights_sha256"]:
        raise ValueError("Exported weights differ from the frozen deployment checkpoint")
    parity = legacy._export_and_benchmark(model, data, assessment_indices, release, config, smoke=smoke)
    gates = {"data": data_gate, "quality": quality_gate,
             "calibration": calibration_gate, "parity": parity}
    release_id = config["study_id"] + "-" + deployment["candidate_id"] + f"-seed{deployment['seed']}"

    def write_receipt():
        atomic_json(release / "evaluation_receipt.json", {"format_version": 1, "release_id": release_id,
            "weights_sha256": digest(release / "weights.pt"), "calibration": calibration, "gates": gates,
            "study_binding": frozen["study_binding"], "selection_freeze_sha256": digest(freeze_path), "smoke_only": smoke,
            "scope": "Local technical checks, distinct from component-representative statistical eligibility. Not production certification."})

    write_receipt()
    manifest_value = None
    if smoke:
        serving = {"status": "not_applicable_smoke", "scope": "Synthetic fixture can never create a real ready model"}
        atomic_json(release / "SMOKE_ONLY.json", {"status": "SMOKE_ONLY_NOT_A_REAL_MODEL_RELEASE"})
    else:
        manifest_value = _release_manifest(config, names, data, deployment, calibration, release)
        atomic_json(release / "release.json", manifest_value)
        serving = {"status": "fail", "scope": "Prerequisite technical gates did not all pass"}
        if all(g["status"] == "pass" for g in gates.values()):
            _stop(stop_check, "actual local serving preflight")
            try:
                serving = _serving_check(out, release, manifest_value, data, assessment_indices, deployment_logits, calibration)
            except Exception as error:
                serving = {"status": "fail", "exception_type": type(error).__name__, "message": str(error)[:1500],
                           "scope": "Actual serving preflight failed; original candidate never activated"}
    atomic_json(release / "SERVING_CHECK.json", serving)
    gates["serving"] = serving
    ready = not smoke and all(g["status"] == "pass" for g in gates.values())
    write_receipt()
    if not smoke:
        manifest_value = _release_manifest(config, names, data, deployment, calibration, release, ready=ready)
        atomic_json(release / "release.json", manifest_value)
        from .release import validate_release
        validate_release(release, require_ready=ready)
    legacy._figures(evaluation, tasks, assessment, names, deployment["task_id"], corruptions)
    legacy._diagnostic_figures(evaluation, data, assessment_indices, deployment_logits, calibration["temperature"],
                              assessment[deployment["task_id"]], calibrated)
    by_candidate = {}
    for candidate in sorted({t["candidate_id"] for t in tasks}):
        rows = [t for t in tasks if t["candidate_id"] == candidate]
        by_candidate[candidate] = {}
        for phase, values in [("selection", selection), ("assessment_regression", assessment)]:
            by_candidate[candidate][phase] = {key: {"mean": float(np.mean([values[t["task_id"]][key] for t in rows])),
                "sample_sd": float(np.std([values[t["task_id"]][key] for t in rows], ddof=1)), "seeds": [t["seed"] for t in rows]}
                for key in ("accuracy", "macro_f1", "loss")}
    summary = {"schema": "garmentlab_improved_study_summary_v2", "status": "SMOKE_ONLY_COMPLETE" if smoke else ("COMPLETE_RESEARCH_ARTIFACTS" if ready else "COMPLETE_CANDIDATE_NOT_READY"),
        "completed_utc": utc_now().isoformat(), "study_binding": frozen["study_binding"], "prepared_bindings": data["bindings"],
        "evaluation_inputs_sha256": digest(identity_path), "final_selection_sha256": digest(out / "FINAL_SELECTION.json"),
        "evaluation_backend_sha256": digest(backend_path), "smoke_only": smoke, "ready_for_local_service": ready,
        "assessment_scope": ASSESSMENT_SCOPE, "labels": names, "task_count": len(tasks),
        "selected_candidate_id": frozen["selection_candidate_id"], "deployment_candidate_id": candidate_id,
        "deployment_task_id": deployment["task_id"], "deployment_seed": deployment["seed"],
        "candidate_summary": by_candidate, "selection": selection, "assessment": assessment,
        "grouped_uncertainty": uncertainty,
        "deployment_calibrated_assessment": calibrated, "calibration": calibration, "calibration_v2": fitted,
        "target_bound_satisfied_under_assumptions": fitted["risk_control"]["target_bound_satisfied_under_assumptions"],
        "unconditional_accuracy_guarantee": False, "corruptions": corruptions, "gates": gates,
        "inference_device": "cpu", "inference_batch_size": 1, "release_directory": "release_candidate",
        "limitations": ["The original assessment327 was inspected before this upgrade; all new scores on it are exploratory regression evidence.",
            "Group-representative binomial bounds are conditional on stated independence/population assumptions and do not guarantee crop-weighted95% accuracy.",
            "Data size, six-label scope and manual-crop requirements are unchanged. No smartphone-photo or production validation is asserted.",
            "A review-all policy is valid but provides zero autonomous coverage. Training-seed SD is not uncertainty over independent datasets."],
        "artifacts": []}
    for path in [backend_path, identity_path, *sorted((out / "evaluation_source_snapshot").rglob("*")),
                 *sorted(evaluation.rglob("*")), *sorted(release.rglob("*"))]:
        if path.is_file():
            summary["artifacts"].append({"path": path.relative_to(out).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size})
    atomic_json(final_path, summary)
    return summary
