"""Derive a CPU export from the verified recovery return; never train or mutate it.

The default manifest pin identifies the student's 28 October recovery archive.
An additive release has its own ID, runtime, receipts and complete provenance.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

PINNED_MANIFEST_SHA256 = "dba432eecf76e3510cfb6ce6be3d683d7097ae343d948603bd689fd8b0d1e4ce"
REFERENCE_RETURN_ZIP_SHA256 = "662b2fe873223abf25fe48c4ece51c1c3d5738af97a1602d893525de0d53f2a4"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=invalid)


def _member(root, name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Manifest artifact must use a nonempty relative POSIX path")
    path, windows = PurePosixPath(name), PureWindowsPath(name)
    if path.is_absolute() or windows.drive or windows.root or ":" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Manifest artifact path escapes its run directory")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise ValueError("Manifest artifacts cannot traverse links or junctions")
    target = target.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Manifest artifact must be a regular file inside the run")
    return target


def verify_run(run, *, expected_manifest_sha256):
    """Check the pinned export and every payload before loading any checkpoint."""
    original = Path(run)
    if original.is_symlink() or (hasattr(original, "is_junction") and original.is_junction()):
        raise ValueError("Run root cannot be a link or junction")
    root = original.resolve(strict=True)
    manifest_path = _member(root, "EXPORT_MANIFEST.json")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Returned export manifest SHA-256 differs from the supplied pin")
    manifest = read_json(manifest_path)
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("Export manifest must declare payloads")
    paths = set()
    for item in payloads:
        name = item["path"]
        if name in paths or name == "EXPORT_MANIFEST.json":
            raise ValueError("Duplicate or self-referential manifest artifact")
        paths.add(name)
        target = _member(root, name)
        if type(item.get("bytes")) is not int or target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
            raise ValueError(f"Manifest payload size/hash mismatch: {name}")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
            raise ValueError("Extracted run contains a link or junction")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != paths | {"EXPORT_MANIFEST.json"}:
        raise ValueError("Extracted file set differs from export manifest")
    return {"root": root, "manifest_sha256": expected_manifest_sha256,
            "payload_count": len(payloads), "manifest": manifest}


def assert_new_output(run, prepared, out):
    out = Path(out).resolve()
    for source in (Path(run).resolve(), Path(prepared).resolve()):
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Output must be separate from and cannot contain either input directory")
    if out.exists():
        raise FileExistsError("Output already exists; choose a new directory, never overwrite a derived release")
    return out


def compare_logits(actual, expected, *, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2 or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Parity arrays must have matching finite nonempty N x K shapes")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    disagreements = int(np.sum(actual.argmax(1) != expected.argmax(1)))
    return {"status": "pass" if error <= tolerance and disagreements == 0 else "fail",
            "maximum_absolute_error": error, "max_abs_tolerance": tolerance,
            "top1_disagreements": disagreements, "samples": len(actual)}


def compare_archived_oracle(local_npz, archived_npz, *, tolerance):
    rows = []
    with np.load(local_npz, allow_pickle=False) as local, np.load(archived_npz, allow_pickle=False) as archived:
        for batch in (1, 3, 64):
            key = f"pytorch_batch_{batch}"
            actual = local[f"onnx_batch_{batch}"]
            if actual.shape[0] != batch or archived[key].shape[0] != batch:
                raise ValueError("Archived oracle batch shape does not match fixed original audit")
            row = compare_logits(actual, archived[key], tolerance=tolerance)
            rows.append({"batch_size": batch, **row})
    return {"status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
            "scope": "New ONNX CPU outputs against the unchanged archived PyTorch 2.11 CPU oracle, original batches 1/3/64.",
            "max_abs_tolerance": tolerance, "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
            "top1_disagreements": sum(row["top1_disagreements"] for row in rows), "batches": rows}


def _snapshot(out):
    names = ["tools/reexport_garmentlab.py", "garmentlab/evaluate.py", "garmentlab/backend.py",
             "garmentlab/training.py", "garmentlab/inference.py", "garmentlab/probabilities.py",
             "garmentlab/release.py", "garmentlab/runtime.py", "garmentlab/api.py", "a1/cnn.py"]
    values = {}
    for name in names:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        values[name] = sha256_file(target)
    return values


def _save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _original_evidence(run, out):
    original = out / "evidence/original_candidate"
    original.mkdir(parents=True)
    for source in sorted((run / "release_candidate").iterdir()):
        if source.is_file():
            shutil.copyfile(source, original / source.name)
    names = ["EXPORT_MANIFEST.json", "FINAL_SUMMARY.json", "RUN_STATUS.json", "RECOVERY_MANIFEST.json",
             "EVALUATION_BACKEND.json", "study_manifest.json", "study_state.json", "ORIGINAL_PAYLOAD_MAP.json",
             "evaluation/SELECTION_FREEZE.json"]
    for name in names:
        target = out / "evidence/returned_run" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, target)


def _full_assessment(model, data, indices, archived_logits, release, calibration, tolerance, out):
    import onnxruntime as ort
    from garmentlab.evaluate import _infer
    from garmentlab.probabilities import log_probabilities
    from garmentlab.training import prepare_batch
    local = _infer(model, data["images"], indices, device="cpu", batch_size=1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(release / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    onnx = np.concatenate([session.run(None, {session.get_inputs()[0].name: prepare_batch(data["images"], [i]).numpy()})[0] for i in indices])
    probabilities = {}
    decisions = {}
    for name, logits in [("archived", archived_logits), ("local_pytorch", local), ("new_onnx", onnx)]:
        probabilities[name] = np.exp(log_probabilities(logits, calibration["temperature"]))
        decisions[name] = probabilities[name].max(1) >= calibration["review_threshold"]
    rows = {}
    for name, logits in [("local_pytorch", local), ("new_onnx", onnx)]:
        item = compare_logits(logits, archived_logits, tolerance=tolerance)
        item.update({"maximum_probability_error": float(np.max(np.abs(probabilities[name] - probabilities["archived"]))),
                     "maximum_confidence_error": float(np.max(np.abs(probabilities[name].max(1) - probabilities["archived"].max(1)))),
                     "accepted_decision_disagreements": int(np.sum(decisions[name] != decisions["archived"])),
                     "accepted_count": int(decisions[name].sum())})
        item["status"] = "pass" if item["status"] == "pass" and item["accepted_decision_disagreements"] == 0 else "fail"
        rows[name] = item
    values = {"indices": indices, "targets": data["labels"][indices], "archived_logits": archived_logits,
              "local_pytorch_logits": local, "new_onnx_logits": onnx,
              "temperature": np.asarray(calibration["temperature"]), "threshold": np.asarray(calibration["review_threshold"])}
    values.update({f"{k}_probabilities": v for k, v in probabilities.items()})
    values.update({f"{k}_accepted": v for k, v in decisions.items()})
    _save_npz(out / "FULL_ASSESSMENT_CPU_CHECK.npz", **values)
    report = {"status": "pass" if all(row["status"] == "pass" for row in rows.values()) else "fail",
              "samples": len(indices), "batch_size": 1, "temperature": calibration["temperature"],
              "review_threshold": calibration["review_threshold"], "checks": rows,
              "scope": "Numerical/serving equivalence only; original model, metric, seed and threshold choices unchanged. No calibration refit.",
              "arrays_sha256": sha256_file(out / "FULL_ASSESSMENT_CPU_CHECK.npz")}
    return report, probabilities["archived"], decisions["archived"]


def _png_cases(data, indices, probabilities, accepted, threshold):
    from PIL import Image
    from garmentlab.training import prepare_batch
    choices = [int(np.flatnonzero(data["labels"][indices] == label)[0]) for label in range(len(data["classnames"]))]
    for mask in (accepted, ~accepted):
        matches = np.flatnonzero(mask)
        if len(matches):
            choices.append(int(matches[0]))
    choices.append(int(np.argmin(np.abs(probabilities.max(1) - threshold))))
    cases = []
    for position in sorted(set(choices)):
        index = int(indices[position])
        stream = io.BytesIO()
        Image.fromarray(np.asarray(data["images"][index])).save(stream, format="PNG")
        cases.append({"position": position, "array_index": index, "png": stream.getvalue(),
                      "expected_tensor_sha256": hashlib.sha256(prepare_batch(data["images"], [index]).contiguous().numpy().tobytes()).hexdigest(),
                      "expected_class": int(probabilities[position].argmax()),
                      "expected_confidence": float(probabilities[position].max()),
                      "expected_review": not bool(accepted[position])})
    return cases


def _predictor_check(predictor, cases, tolerance):
    from garmentlab.inference import preprocess_image
    rows = []
    for case in cases:
        tensor, _ = preprocess_image(case["png"], predictor.manifest, crop=[0, 0, 224, 224])
        tensor_sha = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        result = predictor.predict_bytes(case["png"], crop=[0, 0, 224, 224])
        error = abs(result["predicted_class"]["probability"] - case["expected_confidence"])
        ok = (tensor_sha == case["expected_tensor_sha256"] and result["predicted_class"]["index"] == case["expected_class"]
              and result["predicted_class"]["label"] == predictor.manifest["labels"][case["expected_class"]]
              and result["review_required"] == case["expected_review"]
              and error <= tolerance and result["calibrated"] is True and result["release_status"] == "ready")
        rows.append({"array_index": case["array_index"], "predicted_index": result["predicted_class"]["index"],
                     "expected_index": case["expected_class"], "review_required": result["review_required"],
                     "expected_review": case["expected_review"], "confidence_error": error, "input_tensor_sha256": tensor_sha,
                     "prepared_tensor_sha256": case["expected_tensor_sha256"], "passed": ok})
    return {"status": "pass" if all(row["passed"] for row in rows) else "fail", "cases": rows,
            "scope": "PNG encoding of prepared 224px crops through actual decode/crop/preprocess/Predictor; no arbitrary-photo quality claim."}


async def _api_check(predictor, cases, tolerance):
    import httpx
    from garmentlab.api import create_app
    app = create_app(garment_predictor=predictor)
    rows = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/health")
        rows.append({"name": "health", "passed": response.status_code == 200 and response.json()["garment_ready"] is True})
        response = await client.get("/v1/models")
        rows.append({"name": "models", "passed": response.status_code == 200 and response.json()["garment"]["ready"] is True})
        representatives = []
        for reviewed in (False, True):
            representatives += next(([case] for case in cases if case["expected_review"] is reviewed), [])
        for case in representatives:
            response = await client.post("/v1/predict", files={"file": ("case.png", case["png"], "image/png")}, data={"crop": "[0,0,224,224]"})
            result = response.json().get("result", {})
            best = result.get("predicted_class", {})
            passed = (response.status_code == 200 and result.get("model_version") == predictor.manifest["release_id"]
                      and best.get("index") == case["expected_class"] and result.get("review_required") is case["expected_review"]
                      and best.get("label") == predictor.manifest["labels"][case["expected_class"]]
                      and abs(best.get("probability", -1) - case["expected_confidence"]) <= tolerance
                      and result.get("release_status") == "ready" and result.get("calibrated") is True)
            rows.append({"name": "predict_review" if case["expected_review"] else "predict_accept", "array_index": case["array_index"], "passed": passed, "status_code": response.status_code})
        response = await client.post("/v1/predict", files={"file": ("case.png", cases[0]["png"], "image/png")})
        rows.append({"name": "missing_crop_rejected", "passed": response.status_code == 400})
        response = await client.post("/v1/reference/predict", files={"file": ("case.png", cases[0]["png"], "image/png")})
        rows.append({"name": "absent_reference_not_substituted", "passed": response.status_code == 503})
    return {"status": "pass" if all(row["passed"] for row in rows) else "fail", "checks": rows,
            "scope": "In-process HTTP ASGI contract with actual Predictor; no network deployment, uptime or production traffic."}


def reexport(run, prepared, out, *, expected_manifest_sha256=PINNED_MANIFEST_SHA256):
    """Return a separate ready release only after numerical and serving checks pass."""
    from garmentlab.runtime import atomic_json, utc_now
    from garmentlab.backend import configure_evaluation_backend, snapshot_evaluation_backend
    from garmentlab.evaluate import _verify_tasks, _split_hash, _load_model, _export_and_benchmark
    from garmentlab.training import load_prepared, build_model
    from garmentlab.release import validate_release
    from garmentlab.inference import Predictor
    import torch
    checked = verify_run(run, expected_manifest_sha256=expected_manifest_sha256)
    run = checked["root"]
    prepared = Path(prepared).resolve(strict=True)
    out = assert_new_output(run, prepared, out)
    summary = read_json(run / "FINAL_SUMMARY.json")
    status = read_json(run / "RUN_STATUS.json")
    if status.get("status") != "COMPLETE" or summary.get("smoke_only") is not False:
        raise ValueError("Only a completed real recovery evaluation can be re-exported")
    original = read_json(run / "release_candidate/release.json")
    validate_release(run / "release_candidate", require_ready=False)
    receipt = read_json(run / "release_candidate/evaluation_receipt.json")
    freeze = read_json(run / "evaluation/SELECTION_FREEZE.json")
    study = read_json(run / "study_manifest.json")
    config = study["config"]
    study_state = read_json(run / "study_state.json")
    if study_state.get("status") != "ALL_TASKS_COMPLETE" or study_state.get("completed") != 6 or study_state.get("planned") != 6:
        raise ValueError("All six original tasks must have complete receipts")
    if any(summary["gates"][k]["status"] != "pass" for k in ("data", "quality", "calibration")):
        raise ValueError("Original data/quality/calibration gates must all pass")
    if summary["gates"] != receipt["gates"] or receipt["calibration"] != original["calibration"] or original["calibration"] != freeze["calibration"]:
        raise ValueError("Original frozen release/calibration/gate bindings differ")
    if sha256_file(run / "release_candidate/evaluation_receipt.json") != original["evaluation_receipt"]["sha256"]:
        raise ValueError("Original release receipt SHA binding differs")
    data = load_prepared(prepared)
    binding, _, tasks = _verify_tasks(run, config, data)
    if study_state.get("binding") != binding or original["labels"] != data["classnames"] or original["preprocessing"] != data["index"]["preprocessing"]:
        raise ValueError("Original labels/preprocessing or task completion binding differs")
    if summary["prepared_bindings"] != data["bindings"] or freeze["study_binding"] != binding or receipt["study_binding"] != binding or summary["study_binding"] != binding:
        raise ValueError("Prepared arrays or study differ from original frozen evidence")
    if receipt["selection_freeze_sha256"] != sha256_file(run / "evaluation/SELECTION_FREEZE.json"):
        raise ValueError("Selection freeze SHA differs from original receipt")
    deployment_id = summary["deployment_task_id"]
    if deployment_id != freeze["deployment_task_id"] or summary["deployment_seed"] != config["deployment_seed"]:
        raise ValueError("Original deployment selection differs from the preset seed")
    deployment = next(t for t in tasks if t["task_id"] == deployment_id)
    means = {recipe: float(np.mean([t["selection"]["macro_f1"] for t in tasks if t["recipe"] == recipe])) for recipe in config["recipes"]}
    if means != freeze["recipe_selection_means"] or deployment["recipe"] != max(config["recipes"], key=lambda recipe: means[recipe]):
        raise ValueError("Frozen recipe differs from the original selection-only rule")
    weights_sha = deployment["weights_sha256"]
    if weights_sha != original["weights"]["sha256"] or weights_sha != receipt["weights_sha256"]:
        raise ValueError("Selected checkpoint is not the original released candidate")
    calibration = deepcopy(original["calibration"])
    tolerance = config["onnx_max_abs_error"]
    if tolerance != freeze["parity_max_abs_error"] or tolerance != 1e-4:
        raise ValueError("The original unchanged 1e-4 parity criterion is required")
    indices = np.flatnonzero(data["roles"] == 3)
    assessment_dir = run / "evaluation" / deployment_id / "assessment"
    with np.load(assessment_dir / "logits.npz", allow_pickle=False) as archived:
        archived_logits = archived["logits"].copy()
        if not np.array_equal(archived["indices"], indices):
            raise ValueError("Prepared assessment order differs from archived oracle indices")
    if len(indices) != 327 or archived_logits.shape != (327, len(data["classnames"])):
        raise ValueError("This derived export requires the unchanged original 327 assessment cases")
    logits_receipt = read_json(assessment_dir / "LOGITS_RECEIPT.json")
    if logits_receipt["binding"]["split_sha256"] != _split_hash(data, "assessment") or logits_receipt["binding"]["weights_sha256"] != weights_sha:
        raise ValueError("Archived assessment oracle data/weight bindings differ")
    if calibration["fit_split_sha256"] != _split_hash(data, "calibration") or calibration["threshold_selection_split_sha256"] != _split_hash(data, "calibration"):
        raise ValueError("Calibration partition binding differs")

    out.mkdir(parents=True, exist_ok=False)
    release = out / "release"
    release.mkdir()
    release_manifest = deepcopy(original)
    release_manifest.update({"release_id": original["release_id"] + "-export2-cpu", "status": "candidate"})
    gates = {k: deepcopy(summary["gates"][k]) for k in ("data", "quality", "calibration")}
    gates["parity"] = {"status": "fail", "scope": "Derived export checks not completed", "max_abs_tolerance": tolerance}
    derived = {"status": "INCOMPLETE_CANDIDATE", "ready_for_local_service": False,
               "input_run": str(run), "output": str(out), "release_id": release_manifest["release_id"],
               "scope": "Separate CPU export validation; no retraining, calibration refit or assessment-driven model selection."}
    receipt_path = release / "evaluation_receipt.json"

    def write_release(status_value):
        new_receipt = {"format_version": 1, "release_id": release_manifest["release_id"],
            "weights_sha256": weights_sha, "calibration": calibration, "gates": gates,
            "study_binding": binding, "selection_freeze_sha256": receipt["selection_freeze_sha256"],
            "smoke_only": False, "scope": "Derived local CPU export of frozen research model. Original data/quality/calibration gates retained with provenance; no production certification."}
        atomic_json(receipt_path, new_receipt)
        release_manifest["status"] = status_value
        release_manifest["evaluation_receipt"] = {"file": "evaluation_receipt.json", "sha256": sha256_file(receipt_path)}
        atomic_json(release / "release.json", release_manifest)

    previous_threads = torch.get_num_threads()
    try:
        _original_evidence(run, out)
        source_hashes = _snapshot(out)
        shutil.copyfile(deployment["checkpoint"], release / "weights.pt")
        configure_evaluation_backend()
        torch.set_num_threads(1)
        backend = snapshot_evaluation_backend("cpu")
        atomic_json(out / "EVALUATION_BACKEND.json", backend)
        provenance = {"schema": "garmentlab_derived_cpu_export_v1", "created_utc": utc_now().isoformat(),
            "original_manifest_sha256": checked["manifest_sha256"], "verified_payload_count": checked["payload_count"],
            "reference_return_zip_sha256": REFERENCE_RETURN_ZIP_SHA256 if expected_manifest_sha256 == PINNED_MANIFEST_SHA256 else None,
            "reference_zip_scope": "Known received archive associated with the pinned manifest; this CLI verifies the extracted manifest and payloads, not the ZIP container bytes.",
            "original_study_binding": binding, "original_weights_sha256": weights_sha,
            "prepared_bindings": data["bindings"], "calibration_unchanged": calibration,
            "original_parity": summary["gates"]["parity"], "original_runtime": study["environment"],
            "derived_backend": backend, "derived_platform": platform.platform(), "source_sha256": source_hashes,
            "no_training_performed": True, "no_original_artifact_mutated": True,
            "oracle_input_binding": {"assessment_indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
                "assessment_split_sha256": _split_hash(data, "assessment"), "first_cases": 64,
                "scope": "Original parity source uses first64 frozen assessment indices. Original NPZ has no independent input hash; association is verified through source, frozen data binding and archived assessment indices."}}
        atomic_json(out / "PROVENANCE.json", provenance)
        write_release("candidate")
        model = _load_model(build_model, data["classnames"], config["deployment_seed"], deployment["checkpoint"], "cpu")
        local = _export_and_benchmark(model, data, indices, release, config, smoke=False)
        shutil.copyfile(release / "ONNX_PARITY.json", release / "LOCAL_EXPORT_PARITY.json")
        oracle = compare_archived_oracle(release / "onnx_parity_outputs.npz", run / "release_candidate/onnx_parity_outputs.npz", tolerance=tolerance)
        atomic_json(out / "ORIGINAL_ORACLE_PARITY.json", oracle)
        gates["parity"] = {"status": "pass" if local["status"] == oracle["status"] == "pass" else "fail",
            "scope": "Both current CPU PyTorch and archived original PyTorch 2.11 oracles must satisfy unchanged1e-4 with zero top1 disagreement.",
            "max_abs_tolerance": tolerance, "maximum_absolute_error": max(local.get("maximum_absolute_error", oracle["maximum_absolute_error"]), oracle["maximum_absolute_error"]),
            "top1_disagreements": local.get("top1_disagreements", -1) + oracle["top1_disagreements"],
            "local_cpu": local, "archived_original_pytorch_2_11": oracle, "model_sha256": sha256_file(release / "model.onnx")}
        atomic_json(release / "ONNX_PARITY.json", gates["parity"])
        write_release("candidate")
        if gates["parity"]["status"] != "pass":
            raise ValueError("Derived export does not satisfy local and archived-oracle numerical gates")
        full, probs, accepted = _full_assessment(model, data, indices, archived_logits, release, calibration, tolerance, out)
        atomic_json(out / "FULL_ASSESSMENT_CPU_CHECK.json", full)
        if full["status"] != "pass":
            raise ValueError("Full assessment numerical or confidence/acceptance equivalence failed")
        cases = _png_cases(data, indices, probs, accepted, calibration["review_threshold"])
        # Internal provisional manifest permits testing the real ready-only Predictor.
        # Any validation or HTTP failure below immediately demotes this manifest.
        write_release("ready")
        validate_release(release, require_ready=True)
        predictor = Predictor(release)
        png = _predictor_check(predictor, cases, tolerance)
        atomic_json(out / "PNG_PREDICTOR_CHECK.json", png)
        if png["status"] != "pass":
            raise ValueError("PNG-to-Predictor contract differs from frozen assessment decisions")
        api = asyncio.run(_api_check(predictor, cases, tolerance))
        atomic_json(out / "API_CONTRACT_CHECK.json", api)
        if api["status"] != "pass":
            raise ValueError("HTTP contract validation failed")
        validate_release(release, require_ready=True)
        # Ensure no original payload was changed during the entire operation.
        verify_run(run, expected_manifest_sha256=expected_manifest_sha256)
        derived.update({"status": "READY_LOCAL_CPU_EXPORT", "ready_for_local_service": True,
            "completed_utc": utc_now().isoformat(), "release_directory": str(release),
            "weights_sha256": weights_sha, "model_sha256": sha256_file(release / "model.onnx"),
            "local_parity": local, "archived_oracle_parity": oracle, "full_assessment": full,
            "png_predictor": png, "api_contract": api, "calibration": calibration,
            "original_study_metrics_unchanged": True,
            "limitations": ["This is an additive Windows CPU export, not a rerun in the original Colab torch2.11 environment.",
                "Original candidate failure remains preserved. This does not retroactively change the received archive status.",
                "Readiness is for the declared local service on preselected six-class garment crops, not production or arbitrary photos.",
                "CPU forward timings exclude decoding, HTTP and network; use only the newly measured CPU_BENCHMARK.json."]})
    except (Exception, KeyboardInterrupt) as error:
        if (release / "weights.pt").is_file():
            write_release("candidate")
        derived.update({"status": "FAILED_CANDIDATE_NOT_READY", "ready_for_local_service": False,
                        "exception_type": type(error).__name__, "message": str(error), "gates": gates})
        (out / "LAST_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        torch.set_num_threads(previous_threads)
        artifacts = []
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "LOCAL_RELEASE_SUMMARY.json":
                artifacts.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        derived["artifacts"] = artifacts
        atomic_json(out / "LOCAL_RELEASE_SUMMARY.json", derived)
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Verified extracted original Recovery return")
    parser.add_argument("--prepared", required=True, type=Path, help="Original SHA-bound prepared view")
    parser.add_argument("--out", required=True, type=Path, help="New separate directory; existing paths are refused")
    parser.add_argument("--expected-manifest-sha256", default=PINNED_MANIFEST_SHA256)
    args = parser.parse_args()
    result = reexport(args.run, args.prepared, args.out, expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps({k: result.get(k) for k in ("status", "ready_for_local_service", "output", "message")}, ensure_ascii=True))
    return 0 if result["ready_for_local_service"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
