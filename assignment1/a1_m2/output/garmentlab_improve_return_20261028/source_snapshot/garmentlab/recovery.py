"""Recover completed GarmentLab checkpoints into a new evaluation-only run.

The original archive and training evidence stay unchanged. Failed evaluation
outputs are forensic evidence, never a cache for the corrected finalizer.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import stat
import traceback
import zipfile

from .runtime import RunSession, atomic_json, digest, export_results, utc_now

ORIGINAL_RESULTS_SHA256 = "5ed04a70628b8eec2db3a233cd112e5e5fb3b864a9922d622642a5d155913952"
ARRAY_NAMES = ("images_uint8.npy", "labels.npy", "roles.npy")
ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _no_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Recovery paths cannot traverse symlinks or junctions")


def _safe_name(name):
    pure = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or pure.is_absolute()
            or ".." in pure.parts or any(":" in part for part in pure.parts)):
        raise ValueError("Unsafe result ZIP path")
    return pure.as_posix().rstrip("/")


def inspect_results(results_zip, expected_sha256):
    """Verify the pinned ZIP, CRC, every manifest entry and complete task evidence."""
    path = Path(results_zip)
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("An explicit results SHA-256 is required")
    expected_sha256 = expected_sha256.lower()
    if digest(path) != expected_sha256:
        raise ValueError("Original result ZIP hash differs; never edit an expected hash to bypass this check")
    with zipfile.ZipFile(path) as archive:
        infos, total = {}, 0
        canonical = set()
        for item in archive.infolist():
            name = _safe_name(item.orig_filename)
            key = name.casefold()
            if key in canonical or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
                raise ValueError("Duplicate, encrypted or symlink result ZIP entry")
            canonical.add(key)
            if item.is_dir():
                continue
            if item.file_size > 512 * 1024**2:
                raise ValueError("A result payload exceeds the recovery limit")
            total += item.file_size
            infos[name] = item
        if len(infos) > 10000 or total > 4 * 1024**3:
            raise ValueError("Result ZIP exceeds recovery limits")
        if archive.testzip() is not None:
            raise ValueError("Original result ZIP CRC check failed")
        if "EXPORT_MANIFEST.json" not in infos:
            raise ValueError("Original result export manifest is missing")
        export = json.loads(archive.read(infos["EXPORT_MANIFEST.json"]))
        entries = {}
        for entry in export.get("payloads", []):
            name = _safe_name(entry.get("path", ""))
            if name in entries or name not in infos:
                raise ValueError("Export manifest has duplicate or missing payloads")
            if entry.get("bytes") != infos[name].file_size or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")):
                raise ValueError("Export manifest size/hash declaration is invalid")
            h = hashlib.sha256()
            with archive.open(infos[name]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != entry["sha256"]:
                raise ValueError(f"Original payload hash failed: {name}")
            entries[name] = entry
        if set(infos) != set(entries) | {"EXPORT_MANIFEST.json"}:
            raise ValueError("Result ZIP contains unmanifested payloads")
        read = lambda name: json.loads(archive.read(infos[name]))
        study, state = read("study_manifest.json"), read("study_state.json")
        config = study["config"]
        expected_tasks = {f"{recipe}_seed{seed}" for recipe in config["recipes"] for seed in config["seeds"]}
        tasks = study["tasks"]
        if (state.get("status") != "ALL_TASKS_COMPLETE" or state.get("completed") != len(expected_tasks)
                or state.get("planned") != len(expected_tasks) or len(tasks) != len(expected_tasks)
                or {task["task_id"] for task in tasks} != expected_tasks):
            raise ValueError("Recovery requires every registered training task to be complete")
        for task in tasks:
            task_id = task["task_id"]
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", task_id):
                raise ValueError("Invalid training task ID")
            prefix = f"tasks/{task_id}/"
            complete = read(prefix+"COMPLETE.json")
            history = read(prefix+"history.json")
            if (complete.get("binding") != state.get("binding") or complete.get("epochs") != config["epochs"]
                    or complete.get("task_id") != task_id or len(history) != config["epochs"]):
                raise ValueError("Completed task metadata differs from the original study")
            if [row["epoch"] for row in history] != list(range(1, config["epochs"]+1)):
                raise ValueError("Original training history is incomplete")
            chosen = min(history, key=lambda row: (-row["selection"]["macro_f1"], row["selection"]["loss"], row["epoch"]))
            if complete.get("best_epoch") != chosen["epoch"] or complete.get("selection") != chosen["selection"]:
                raise ValueError("Original selected checkpoint metadata violates its registered rule")
            for name, item in complete.get("files", {}).items():
                if prefix+name not in entries or entries[prefix+name]["sha256"] != item["sha256"]:
                    raise ValueError("Task completion hash differs from the export")
            if not {prefix+name for name in ("best.pt", "epoch_state.pt", "task.json", "history.json")} <= entries.keys():
                raise ValueError("A completed task is missing checkpoint evidence")
        clock = read("RUN_CLOCK.json")
        elapsed = clock.get("active_process_seconds")
        limit = config.get("max_total_process_seconds")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int,float)) or not math.isfinite(elapsed)
                or elapsed < 0 or not isinstance(limit, (int,float)) or not math.isfinite(limit) or limit <= 0):
            raise ValueError("Original process-budget evidence is invalid")
        ready, index = read("data_evidence/DATA_READINESS.json"), read("data_evidence/index_manifest.json")
        if ready.get("training_ready") is not True:
            raise ValueError("Original data readiness did not pass")
        for name in ARRAY_NAMES:
            declared = ready["files"][name]
            declared = declared["sha256"] if isinstance(declared, dict) else declared
            if study["data"][name] != declared:
                raise ValueError("Original array bindings disagree")
        for name in ("index_manifest.json", "DATA_READINESS.json"):
            if entries["data_evidence/"+name]["sha256"] != study["data"][name]:
                raise ValueError("Original data sidecar binding disagrees")
        data_config_name = "source/configs/garmentlab_data.json"
        if data_config_name not in entries:
            data_config_name = "source_snapshot/configs/garmentlab_data.json"
        if data_config_name not in entries:
            raise ValueError("Original data configuration is missing")
        original_run = read("RUN_MANIFEST.json") if "RUN_MANIFEST.json" in entries else {}
    if digest(path) != expected_sha256:
        raise ValueError("Original result ZIP changed during inspection")
    return {"zip_sha256": expected_sha256, "zip_bytes": path.stat().st_size, "entries": entries,
            "study": study, "state": state, "config": config, "readiness": ready, "index": index,
            "data_config_name": data_config_name, "original_run": original_run,
            "original_process_seconds": elapsed, "recovery_budget_seconds": max(0.0, min(7200.0, limit-elapsed))}


def _destination(name):
    if name in {"study_manifest.json", "study_state.json", "RUN_MANIFEST.json"} or name.startswith(
            ("tasks/", "data_evidence/", "source_snapshot/", "source/", "feature_cache/")):
        return name
    return "original_failure/"+name


def _patch_snapshot(source_dir):
    source = Path(source_dir)
    files = sorted([*source.joinpath("garmentlab").glob("*.py"), source/"tools/recover_garmentlab.py",
                    source/"SOURCE_MANIFEST.json"])
    result = {path.relative_to(source).as_posix(): digest(path) for path in files if path.is_file()}
    if not result:
        raise ValueError("Recovery source snapshot is empty")
    return result


def _write_preserved(path, data, expected_hash):
    path = Path(path)
    _no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected_hash:
            raise ValueError(f"Preserved recovery artifact changed: {path.name}")
        return
    temporary = path.with_name(path.name+".copying")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as stream:
        stream.write(data)
    if digest(temporary) != expected_hash:
        raise ValueError("Preserved copy hash failed")
    temporary.replace(path)


def _initialize(out, results_zip, evidence, source_dir):
    patch = _patch_snapshot(source_dir)
    payload_map = {name: _destination(name) for name in evidence["entries"]}
    payload_map["EXPORT_MANIFEST.json"] = "original_failure/EXPORT_MANIFEST.json"
    if len(set(value.casefold() for value in payload_map.values())) != len(payload_map):
        raise ValueError("Original payload paths collide after recovery mapping")
    identity = {"schema": "garmentlab_evaluation_recovery_v1", "original_results_sha256": evidence["zip_sha256"],
        "original_results_bytes": evidence["zip_bytes"], "original_source_fingerprint": evidence["original_run"].get("source_fingerprint"),
        "original_training_binding": evidence["state"]["binding"], "original_process_seconds": evidence["original_process_seconds"],
        "recovery_budget_seconds": evidence["recovery_budget_seconds"], "stop_by_utc": evidence["config"]["stop_by_utc"],
        "patch_files_sha256": patch, "original_payload_map": payload_map,
        "scope": "New evaluation-only run; no training or checkpoint changes; original failed evaluation is quarantined"}
    manifest_path = out/"RECOVERY_MANIFEST.json"
    if manifest_path.exists():
        if _json(manifest_path) != identity:
            raise ValueError("Recovery input/source identity changed; use a new recovery directory")
    else:
        unexpected = {p.name for p in out.iterdir()} - {"RUN_CLOCK.json", "CONTROLLER_LEASE.json"}
        if unexpected:
            raise ValueError("Recovery must use a new output directory")
        atomic_json(manifest_path, identity)
    with zipfile.ZipFile(results_zip) as archive:
        for original, destination in payload_map.items():
            data = archive.read(original)
            expected = (evidence["entries"][original]["sha256"] if original in evidence["entries"]
                        else hashlib.sha256(data).hexdigest())
            _write_preserved(out/destination, data, expected)
    for name, expected in patch.items():
        _write_preserved(out/"recovery_source_snapshot"/name, (Path(source_dir)/name).read_bytes(), expected)
    atomic_json(out/"ORIGINAL_PAYLOAD_MAP.json", {"original_results_sha256": evidence["zip_sha256"],
                "mapping": payload_map, "note": "The original export manifest describes mapped preserved bytes, not the new runtime status paths."})
    return identity


def _environment():
    import torch
    import torchvision
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": "cuda" if torch.cuda.is_available() else "cpu"}


def _check_environment(expected, observed, *, enforce, smoke):
    differences = {key: {"expected": expected.get(key), "observed": observed.get(key)}
                   for key in ("torch", "torchvision", "gpu") if expected.get(key) != observed.get(key)}
    eligible = not differences and observed.get("device") == "cuda" and "A100" in str(observed.get("gpu"))
    if enforce and not smoke and not eligible:
        raise RuntimeError("Recovery finalization requires the original torch/torchvision versions and the same A100 GPU: "+json.dumps(differences))
    return {"eligible_for_real_finalization": eligible, "differences": differences, "observed": observed,
            "expected": expected, "smoke_only": smoke, "scope": "Environment inspection only; no synthetic or real model pass"}


def _prepare_worker(work, cache, config, result_queue):
    try:
        from .images import prepare_dataset
        path = prepare_dataset(work, cache, config)
        result_queue.put({"prepared": str(path)})
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})


def _run_preparer(work, cache, config, session, prepare_fn=None):
    if session.should_stop(60):
        raise TimeoutError("Recovery budget/deadline reservation reached before preparation")
    if prepare_fn is not None:
        path = prepare_fn(work, cache, config)
        if session.should_stop(0):
            raise TimeoutError("Recovery preparation exceeded its process budget")
        return Path(path)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    worker = context.Process(target=_prepare_worker, args=(str(work),str(cache),str(config),result_queue))
    worker.start()
    try:
        while worker.is_alive():
            worker.join(timeout=1)
            if session.should_stop(60):
                worker.terminate()
                worker.join(timeout=10)
                raise TimeoutError("Recovery preparation stopped at its budget/deadline; cache is retained")
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise RuntimeError("Data preparation worker ended without a result") from exc
        if "error" in result:
            raise RuntimeError("Data preparation failed:\n"+result["error"])
        return Path(result["prepared"])
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
        result_queue.close()


def reconstitute_prepared_view(rebuilt, view, out, evidence):
    """Reuse verified arrays; restore original JSON bytes only after semantic checks."""
    rebuilt, view, out = Path(rebuilt), Path(view), Path(out)
    _no_symlinks(rebuilt)
    _no_symlinks(view)
    old_ready, old_index = evidence["readiness"], evidence["index"]
    new_ready, new_index = _json(rebuilt/"DATA_READINESS.json"), _json(rebuilt/"index_manifest.json")
    if (new_ready.get("training_ready") is not True or new_ready.get("binding") != old_ready.get("binding")
            or new_index != old_index):
        raise ValueError("Rebuilt data bindings or records/class order/roles differ from the original")
    for key, value in old_ready.get("sources", {}).items():
        if new_ready.get("sources", {}).get(key, {}).get("sha256") != value.get("sha256"):
            raise ValueError("Rebuilt raw-source hash differs from the original")
    for name in ARRAY_NAMES:
        _no_symlinks(rebuilt/name)
        if digest(rebuilt/name) != evidence["study"]["data"][name]:
            raise ValueError(f"Rebuilt array differs from the original: {name}")
    # Require the rebuilt local receipt to be truthful before restoring originals.
    for name, item in new_ready.get("files", {}).items():
        expected = item.get("sha256") if isinstance(item, dict) else item
        _safe_name(name)
        if digest(rebuilt/name) != expected:
            raise ValueError("Rebuilt preparation receipt contains an invalid payload hash")
    view.mkdir(parents=True, exist_ok=True)
    methods = {}
    for name in ARRAY_NAMES:
        source, target = rebuilt/name, view/name
        _no_symlinks(target)
        if target.exists():
            if digest(target) != evidence["study"]["data"][name]:
                raise ValueError("Previously reconstituted array changed")
            methods[name] = "existing_verified_array"
        else:
            try:
                os.link(source, target)
                methods[name] = "hardlink_to_verified_rebuilt_array"
            except OSError:
                shutil.copyfile(source, target)
                methods[name] = "byte_copy_of_verified_rebuilt_array"
    for source in (out/"data_evidence").iterdir():
        if source.is_file():
            _write_preserved(view/source.name, source.read_bytes(), digest(source))
    from .training import load_prepared
    data = load_prepared(view)
    if data["bindings"] != evidence["study"]["data"]:
        raise ValueError("Reconstituted prepared view does not reproduce original study bindings")
    if hasattr(data["images"], "_mmap"):
        data["images"]._mmap.close()
    receipt = {"schema": "garmentlab_prepared_reconstitution_v1", "original_results_sha256": evidence["zip_sha256"],
        "original_study_data": evidence["study"]["data"], "rebuilt_data_binding": new_ready.get("binding"),
        "rebuilt_index_sha256": digest(rebuilt/"index_manifest.json"),
        "rebuilt_readiness_sha256": digest(rebuilt/"DATA_READINESS.json"),
        "original_index_semantically_equal": True, "raw_sources_and_three_arrays_hash_identical": True,
        "array_materialization": methods, "metadata_policy": "Exact archived sidecar bytes restored after semantic/array/source verification; original hashes and receipts were not edited",
        "prepared_view": str(view.resolve()), "no_training_performed": True,
        "receipt_scope": "Initial reconstitution; every resume revalidates sources, arrays, semantic records and original study bindings"}
    receipt_path = out/"PREPARED_RECONSTITUTION.json"
    if receipt_path.exists():
        initial = _json(receipt_path)
        for key in ("original_results_sha256", "original_study_data", "rebuilt_data_binding",
                    "original_index_semantically_equal", "raw_sources_and_three_arrays_hash_identical"):
            if initial.get(key) != receipt[key]:
                raise ValueError("Initial reconstitution identity changed on resume")
    else:
        atomic_json(receipt_path, receipt)
    atomic_json(out/"PREPARED_POINTER.json", {"prepared_dir": str(view.resolve()), "work_dir": str(view.parent.resolve())})
    return view


def _export_recovery(out):
    standard = export_results(out)
    target = Path(out)/"exports/g_m10_garmentlab_recovery_run01_results.zip"
    standard.replace(target)
    return target


def recover_results(results_zip, out_dir, work_dir, cache_dir, *, expected_results_sha256,
                    stage="all", source_dir=None, model_factory=None, smoke=False,
                    prepare_fn=None, finalize_fn=None, environment_fn=None):
    """Run check/prepare/finalize/export stages without any training entry point."""
    if stage not in {"check", "prepare", "finalize", "export", "all"}:
        raise ValueError("Invalid recovery stage")
    if (model_factory is not None or prepare_fn is not None or finalize_fn is not None or environment_fn is not None) and not smoke:
        raise ValueError("Dependency injection is restricted to smoke-only software tests")
    for path in (out_dir, work_dir, cache_dir):
        _no_symlinks(path)
    out, work, cache = Path(out_dir).resolve(), Path(work_dir).resolve(), Path(cache_dir).resolve()
    evidence = inspect_results(results_zip, expected_results_sha256)
    if out.exists() and not (out/"RECOVERY_MANIFEST.json").is_file():
        if set(p.name for p in out.iterdir()) - {"CONTROLLER_LEASE.json", "RUN_CLOCK.json"}:
            raise ValueError("Refusing to overwrite an existing original/unrelated run directory")
    out.mkdir(parents=True, exist_ok=True)
    acquired, result_zip, summary = False, None, None
    status, error = {"status": "RECOVERY_RUNNING", "stage": stage, "updated_utc": utc_now().isoformat()}, None
    try:
        with RunSession(out, total_seconds=evidence["recovery_budget_seconds"],
                        stop_by_utc=evidence["config"]["stop_by_utc"]) as session:
            acquired = True
            _initialize(out, results_zip, evidence, source_dir or ROOT)
            if stage == "export":
                status = _json(out/"RUN_STATUS.json") if (out/"RUN_STATUS.json").exists() else {"status": "RECOVERY_EXPORT_ONLY"}
            else:
                atomic_json(out/"RUN_STATUS.json", status)
                if session.should_stop(0):
                    raise TimeoutError("Recovery cumulative budget or original absolute deadline has been reached")
                observed = (environment_fn or _environment)()
                check = _check_environment(evidence["study"]["environment"], observed,
                                           enforce=stage in {"all", "finalize"}, smoke=smoke)
                atomic_json(out/"RECOVERY_DEVICE_CHECK.json", check)
                prepared = work/"prepared_view"
                if stage in {"all", "prepare"}:
                    session.stage = "recover_prepare"
                    session.tick()
                    rebuilt = _run_preparer(work/"rebuild", cache, out/evidence["data_config_name"], session, prepare_fn)
                    prepared = reconstitute_prepared_view(rebuilt, prepared, out, evidence)
                elif stage == "finalize":
                    pointer = out/"PREPARED_POINTER.json"
                    if not pointer.exists():
                        raise ValueError("Prepare the verified recovery view before finalization")
                    prepared = Path(_json(pointer)["prepared_dir"])
                    _no_symlinks(prepared)
                    from .training import load_prepared
                    data = load_prepared(prepared)
                    if data["bindings"] != evidence["study"]["data"]:
                        raise ValueError("Prepared recovery view binding changed")
                    data["images"]._mmap.close()
                if stage in {"all", "finalize"}:
                    session.stage = "recover_finalize"
                    session.tick()
                    if session.should_stop(60):
                        raise TimeoutError("Recovery deadline/budget reservation reached before finalization")
                    if finalize_fn is None:
                        from .evaluate import finalize_study
                        finalizer = finalize_study
                    else:
                        finalizer = finalize_fn
                    options = {"smoke": smoke, "stop_check": lambda: session.should_stop(60)}
                    if model_factory is not None:
                        options["model_factory"] = model_factory
                    summary = finalizer(out, prepared, evidence["config"], **options)
                    status = {"status": "COMPLETE", "finalization": summary, "updated_utc": utc_now().isoformat(),
                              "scope": "Evaluation-only recovery; original training artifacts unchanged", "smoke_only": smoke}
                else:
                    status = {"status": "RECOVERY_PREPARED" if stage == "prepare" else "RECOVERY_CHECKED",
                              "updated_utc": utc_now().isoformat(), "smoke_only": smoke}
                atomic_json(out/"RUN_STATUS.json", status)
    except (Exception, KeyboardInterrupt) as exc:
        if not acquired:
            raise
        error = exc
        status = {"status": "INCOMPLETE_STOPPED" if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else "FAILED",
                  "stage": stage, "exception_type": type(exc).__name__, "message": str(exc),
                  "updated_utc": utc_now().isoformat(), "scope": "Evaluation-only recovery; no training was invoked"}
        atomic_json(out/"RUN_STATUS.json", status)
        (out/"RECOVERY_LAST_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if acquired:
            result_zip = _export_recovery(out)
    return {"status": status, "out_dir": str(out), "result_zip": str(result_zip) if result_zip else None,
            "prepared_view": str(work/"prepared_view"), "error": str(error) if error else None}
