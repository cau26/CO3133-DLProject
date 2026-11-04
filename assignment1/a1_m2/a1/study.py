"""Fixed, resumable supplementary validation study. This module never opens test data."""
from __future__ import annotations

import copy
import csv
import hashlib
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .common import json_hash, load_json, protect_json, save_json, sha256_file, source_metadata

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("linear", "mlp", "cnn", "gru", "transformer")
POLICY = "validation_only_post_test_study"
PROTOCOL_TYPE = "post_test_exploratory_validation_only"
TOOL_FILES = ("tools/run_a1_study.py", "tools/summarize_a1_study.py")
ABLATIONS = [
    {"id": "cnn_no_dropout", "model": "cnn", "overrides": {"dropout": 0.0}},
    {"id": "gru_columns", "model": "gru", "overrides": {"sequence_axis": "columns"}},
    {"id": "transformer_no_position", "model": "transformer", "overrides": {"use_position_embedding": False}},
]


def relative_child(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Expected a safe relative POSIX path")
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Unsafe study-relative path")
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Path leaves study root")
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def protocol_record(path):
    protocol = load_json(path)
    if (protocol.get("schema_version") != 1 or protocol.get("study_type") != PROTOCOL_TYPE
            or protocol.get("official_test_already_seen") is not True):
        raise ValueError("The supplementary study must acknowledge prior official-test exposure")
    if not re.fullmatch(r"[a-z0-9_]{1,64}", protocol.get("study_id", "")):
        raise ValueError("Invalid study_id")
    seeds = protocol.get("training_seeds")
    if (type(seeds) is not list or len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("This locked design requires three distinct uint32 training seeds")
    if seeds != [42, 43, 44]:
        raise ValueError("The declared paired analysis is fixed to training seeds42/43/44")
    if (protocol.get("split_seed") != 42 or type(protocol["split_seed"]) is not int
            or protocol.get("epochs") != 60 or protocol.get("evaluation_budgets") != [20, 60]
            or protocol.get("baseline_models") != list(MODELS)
            or protocol.get("ablations") != ABLATIONS):
        raise ValueError("Protocol differs from the declared 24-task/60-epoch controlled study")
    for field in ("expected_split_sha256", "expected_dataset_sha256", "reference_freeze_sha256", "reference_artifact_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", protocol.get(field, "")):
            raise ValueError(f"Invalid {field}")
    return protocol


def reference_record(protocol):
    """Verify historical bytes without importing/executing the historical training code."""
    reference = relative_child(ROOT, protocol["reference_run"])
    freeze_path = reference / "freeze_manifest.json"
    if sha256_file(freeze_path) != protocol["reference_freeze_sha256"]:
        raise ValueError("Historical freeze differs from the fixed protocol reference")
    frozen = load_json(freeze_path)
    body = {k: v for k, v in frozen.items() if k != "manifest_sha256"}
    if frozen.get("manifest_sha256") != json_hash(body):
        raise ValueError("Historical freeze self-hash mismatch")
    if (frozen["run_id"] != protocol["reference_run_id"]
            or frozen["split_sha256"] != protocol["expected_split_sha256"]
            or frozen["dataset_sha256"] != protocol["expected_dataset_sha256"]):
        raise ValueError("Historical run/dataset/split reference mismatch")
    for relative, expected in frozen["files_sha256"].items():
        if sha256_file(relative_child(reference, relative)) != expected:
            raise ValueError("Historical locked file changed: " + relative)
    export_path = reference / "artifact_manifest.json"
    if sha256_file(export_path) != protocol["reference_artifact_manifest_sha256"]:
        raise ValueError("Historical export manifest changed")
    export = load_json(export_path)
    for relative, expected in export.items():
        if sha256_file(relative_child(reference, relative)) != expected:
            raise ValueError("Historical exported file changed: " + relative)
    return {"run_id": frozen["run_id"], "freeze_sha256": sha256_file(freeze_path),
            "source_sha256": frozen["source_sha256"], "locked_files": len(frozen["files_sha256"]),
            "exported_files": len(export), "export_manifest_sha256": sha256_file(export_path)}


def source_record():
    return {"training_source": source_metadata(),
            "study_tools_sha256": {name: sha256_file(ROOT / name) for name in TOOL_FILES}}


def build_design(protocol):
    base = load_json(relative_child(ROOT, protocol["base_config"]))
    recorded = load_json(relative_child(ROOT, protocol["reference_run"]) / "config.json")
    comparison = copy.deepcopy(base)
    comparison["device"] = recorded["device"]
    if comparison != recorded:
        raise ValueError("Baseline recipe changed from the already reported original study")
    runs, tasks = [], []
    for arm in [{"id": "baseline", "models": list(MODELS), "overrides": {}}] + [
            {"id": a["id"], "models": [a["model"]], "overrides": a["overrides"]} for a in ABLATIONS]:
        for seed in protocol["training_seeds"]:
            run_key = f'{arm["id"]}_seed{seed}'
            cfg = copy.deepcopy(base)
            cfg.update(seed=seed, split_seed=protocol["split_seed"], epochs=protocol["epochs"], device="cuda",
                       evaluation_budgets=protocol["evaluation_budgets"], evaluation_policy=POLICY,
                       study_id=protocol["study_id"], study_arm=arm["id"],
                       expected_split_sha256=protocol["expected_split_sha256"],
                       expected_dataset_sha256=protocol["expected_dataset_sha256"])
            cfg["per_model"]["gru"]["sequence_axis"] = "rows"
            cfg["per_model"]["transformer"]["use_position_embedding"] = True
            if arm["id"] != "baseline":
                cfg["per_model"][arm["models"][0]].update(arm["overrides"])
            record = {"run_key": run_key, "arm": arm["id"], "seed": seed,
                      "models": arm["models"], "run_relative": "runs/" + run_key,
                      "config_relative": "configs/" + run_key + ".json",
                      "config": cfg, "config_sha256": json_hash(cfg)}
            runs.append(record)
            tasks.extend({"task_id": f'{run_key}.{model}', "run_key": run_key, "arm": arm["id"],
                          "seed": seed, "model": model, "run_relative": record["run_relative"],
                          "config_relative": record["config_relative"]} for model in arm["models"])
    assert len(runs) == 12 and len(tasks) == 24
    return runs, tasks


def prepare_study(output, protocol_path):
    output = Path(output).resolve()
    protocol = protocol_record(protocol_path)
    reference = relative_child(ROOT, protocol["reference_run"])
    for protected in (reference, ROOT / "outputs/a1_t4"):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Study output must be separate from historical results and their parent directories")
    if (output / "study_manifest.json").exists():
        return validate_study(output, protocol)
    if (output / "runs").exists():
        raise ValueError("Unbound runs cannot be adopted; use a new supplementary output directory")
    reference_identity = reference_record(protocol)
    runs, tasks = build_design(protocol)
    for run in runs:
        protect_json(relative_child(output, run["config_relative"]), run["config"])
    body = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "study_type": PROTOCOL_TYPE, "protocol": protocol, "protocol_sha256": json_hash(protocol),
            "reference": reference_identity, "source": source_record(), "runs": runs, "tasks": tasks,
            "official_test_policy": "Never opened or evaluated by this supplementary study"}
    body["manifest_sha256"] = json_hash(body)
    protect_json(output / "study_manifest.json", body)
    return body


def validate_study(output, protocol=None):
    output = Path(output).resolve()
    manifest = load_json(output / "study_manifest.json")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("schema_version") != 1 or manifest.get("manifest_sha256") != json_hash(body):
        raise ValueError("Study manifest self-hash/schema mismatch")
    saved = manifest["protocol"]
    if manifest["study_type"] != PROTOCOL_TYPE or manifest["protocol_sha256"] != json_hash(saved):
        raise ValueError("Study nature/protocol binding mismatch")
    if protocol is not None and saved != protocol:
        raise ValueError("Protocol changed after study creation; preserve it and use a new study")
    current = source_record()
    if (manifest["source"]["training_source"]["source_sha256"] != current["training_source"]["source_sha256"]
            or manifest["source"]["study_tools_sha256"] != current["study_tools_sha256"]):
        raise ValueError("Source changed after study creation; restore the exact source bundle")
    if reference_record(saved) != manifest["reference"]:
        raise ValueError("Historical reference changed")
    runs, tasks = build_design(saved)
    if manifest["runs"] != runs or manifest["tasks"] != tasks:
        raise ValueError("Task/config matrix changed")
    actual_runs = output / "runs"
    if actual_runs.is_dir():
        unexpected = {p.name for p in actual_runs.iterdir() if p.is_dir()} - {r["run_key"] for r in runs}
        if unexpected:
            raise ValueError("Unplanned run directories require a separate study: " + ", ".join(sorted(unexpected)))
    if (output / "final_test").exists():
        raise ValueError("Official-test artifacts are forbidden in this supplementary study")
    for run in runs:
        cfg = load_json(relative_child(output, run["config_relative"]))
        if cfg != run["config"] or json_hash(cfg) != run["config_sha256"]:
            raise ValueError("Planned configuration changed: " + run["run_key"])
    return manifest


def task_state(output, task, manifest):
    run = relative_child(output, task["run_relative"])
    directory = run / task["model"]
    if (run / "final_test").exists() or (run / "freeze_manifest.json").exists():
        raise ValueError("Supplementary run contains forbidden freeze/test artifacts")
    if not directory.exists():
        return "PENDING"
    if not (directory / "metrics.json").is_file():
        if not (directory / "last.pt").is_file():
            raise ValueError(f'{task["task_id"]}: incomplete directory has no resumable epoch checkpoint; preserve it for diagnosis')
        return "INCOMPLETE"
    # Never classify a truncated or mixed run as complete just because metrics.json exists.
    cfg = next(r["config"] for r in manifest["runs"] if r["run_key"] == task["run_key"])
    from .train import model_config
    effective = model_config(task["model"], cfg)
    metrics = load_json(directory / "metrics.json")
    run_manifest = load_json(run / "run_manifest.json")
    if (load_json(run / "config.json") != cfg or run_manifest["config_sha256"] != json_hash(cfg)
            or run_manifest["source_sha256"] != manifest["source"]["training_source"]["source_sha256"]
            or metrics.get("run_id") != run_manifest["run_id"]
            or metrics.get("model") != task["model"]
            or metrics["source"]["source_sha256"] != run_manifest["source_sha256"]):
        raise ValueError("Completed task is not bound to the planned run/source")
    if (metrics["config"] != effective or metrics["config_sha256"] != json_hash(effective)
            or metrics["epochs_completed"] != cfg["epochs"] or metrics["evaluation_split"] != "validation"
            or metrics["test_evaluated"] is not False
            or metrics["checkpoint_sha256"] != sha256_file(directory / "best.pt")):
        raise ValueError("Completed task metadata is inconsistent: " + task["task_id"])
    summary = load_json(run / "data_summary.json")
    if (summary["split_sha256"] != manifest["protocol"]["expected_split_sha256"]
            or summary["dataset_sha256"] != manifest["protocol"]["expected_dataset_sha256"]
            or summary.get("split_seed") != 42 or summary.get("seed") != task["seed"]
            or summary.get("split_metadata_schema_version") != 2):
        raise ValueError("Completed task uses a different dataset/split")
    for budget in cfg["evaluation_budgets"]:
        for name in (f"best_through_{budget}.pt", f"best_through_{budget}_selection.json",
                     f"budget{budget}_metrics.json", f"budget{budget}_validation_predictions.csv"):
            if not (directory / name).is_file():
                raise ValueError("Completed task missing preserved budget artifact: " + name)
    if len(read_csv(directory / "history.csv")) != cfg["epochs"]:
        raise ValueError("Completed task has a truncated epoch history")
    return "COMPLETE"


def progress(output, manifest):
    rows = [{**task, "state": task_state(output, task, manifest)} for task in manifest["tasks"]]
    return {"study_id": manifest["protocol"]["study_id"], "total_tasks": len(rows),
            "completed_tasks": sum(row["state"] == "COMPLETE" for row in rows), "tasks": rows}


@contextmanager
def execution_lock(output):
    """Local OS lock releases on process death; use one Colab runtime per study."""
    # Keep this local, rather than depending on advisory-lock support in Drive FUSE.
    key = hashlib.sha256(str(Path(output).resolve()).encode("utf-8")).hexdigest()[:24]
    path = ROOT / ".private-build/study_locks" / (key + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("Another process is already training this study") from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def register_environment(output, manifest, device):
    import torch
    from .common import environment, set_seed
    if device.type != "cuda" or not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Full study training requires an actual NVIDIA A100 CUDA runtime")
    source = source_metadata()
    if source["working_tree_dirty"] is not False or source["git_commit"] is None:
        raise ValueError("Full study training requires the clean source bundle checkout")
    if source["git_commit"] != manifest["source"]["training_source"]["git_commit"]:
        raise ValueError("Study source commit changed")
    set_seed(manifest["protocol"]["training_seeds"][0])
    torch.set_num_threads(manifest["runs"][0]["config"]["cpu_threads"])
    current = environment(device)
    current["gpu_driver_info"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    path = Path(output) / "study_environment.json"
    if path.exists():
        previous = load_json(path)
        fields = ("gpu", "versions", "cuda_runtime", "cudnn_version", "python", "torch_threads", "gpu_driver_info")
        if any(previous.get(k) != current.get(k) for k in fields):
            raise ValueError("GPU/software differs from the saved study. Restore the original runtime before resuming.")
    else:
        protect_json(path, current)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    protect_json(Path(output) / "sessions" / (session + ".json"), current)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    pip_path = Path(output) / "study_pip_freeze.txt"
    if not pip_path.exists():
        with pip_path.open("x", encoding="utf-8") as stream:
            stream.write(packages)
    with (Path(output) / "sessions" / (session + "_pip_freeze.txt")).open("x", encoding="utf-8") as stream:
        stream.write(packages)
    return current


def execute_study(output, protocol_path, data, device, max_tasks=None, invoke=None):
    """Only called by the student's deliberate --stage train invocation."""
    output, data = Path(output).resolve(), Path(data).resolve()
    manifest = prepare_study(output, protocol_path)
    executed = 0
    with execution_lock(output):
        register_environment(output, manifest, device)
        for index, task in enumerate(manifest["tasks"], 1):
            state = task_state(output, task, manifest)
            print(f'[{index}/24] {task["task_id"]}: {state}', flush=True)
            if state == "COMPLETE":
                continue
            if max_tasks is not None and executed >= max_tasks:
                break
            command = [sys.executable, str(ROOT / "run_a1.py"), "--stage", "train", "--models", task["model"],
                       "--config", str(relative_child(output, task["config_relative"])), "--out",
                       str(relative_child(output, task["run_relative"])), "--data", str(data), "--device", "cuda"]
            if state == "INCOMPLETE":
                command.append("--resume")
            if invoke is None:
                subprocess.run(command, cwd=ROOT, check=True)
            else:
                invoke(command)
            if task_state(output, task, manifest) != "COMPLETE":
                raise ValueError("Training returned without a complete budgeted task")
            executed += 1
            save_json(output / "study_progress.json", progress(output, manifest))
        state = progress(output, manifest)
        save_json(output / "study_progress.json", state)
        if state["completed_tasks"] == state["total_tasks"]:
            marker = {"study_id": manifest["protocol"]["study_id"], "manifest_sha256": manifest["manifest_sha256"],
                      "completed_tasks": 24, "evaluation_policy": POLICY}
            protect_json(output / "STUDY_COMPLETE.json", marker)
            print("STUDY COMPLETE: 24/24 validation-only tasks", flush=True)
        return state
