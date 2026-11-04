"""Matched confirmation using the unchanged OPT3 training engine.

Each seed/family has two training trajectories: the historical C0 recipe and
the frozen OPT3-selected recipe. CE- and F1-selected checkpoints come from the
same trajectory. The final-epoch checkpoint is fixed before training and is the
primary recipe comparison. This isolates the combined recipe from checkpoint
selection; it does not isolate individual LR, decay, dropout or schedule effects.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
import socket
import statistics
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from a1opt.engine import _csv_bytes, _evaluate, train_task
from a1opt.models import build_model
from a1opt.recipes import FAMILIES, effective_config_hash
from a1opt.storage import (atomic_bytes, atomic_json, atomic_torch_save, json_hash,
                           load_json, sha256_file, verify_complete)
from a1opt.study import load_training_data, split_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1009, 2027, 3037)
HISTORICAL_FASHION_SEEDS = frozenset((42, 43, 44, 101, 202))
DATASET_SHA256 = "1f243a60b4b748a44c48b9a6f6907be2a08bee3e4e226f0e86fbde8e147b745a"
SPLIT_SHA256 = "0289c4940126abe338a376ea49eb866788d57cd67fa3ca17fe80139abb8c8fba"
POLICIES = ("ce", "f1", "final")
LIMITATIONS = [
    "Previously inspected old validation selects CE/F1 checkpoints; its metrics are exploratory.",
    "Fresh training seeds do not create fresh images or undo historical evaluation exposure.",
    "The official test partition is not loaded and is not a new blind benchmark.",
    "The paired factor design isolates combined recipe changes from checkpoint selection, not individual hyperparameters.",
    "Across-seed sample SD is conditional on one fixed split, not a population confidence interval.",
    "A genuinely independent image benchmark requires an externally collected, prospectively locked audit set.",
]


def source_identity():
    files = sorted(p for folder in ("a1", "a1opt", "a1confirm")
                   for p in (ROOT / folder).glob("*.py"))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}
    return {"files_sha256": hashes, "sha256": json_hash(hashes),
            "text_hash_policy": "UTF-8 source bytes with CRLF normalized to LF"}


def environment_identity(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU software checks and CUDA training are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    return {"python_major_minor": list(sys.version_info[:2]), "device_type": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "numpy", "scikit-learn")}}


def make_protocol(publication_path, families=FAMILIES, seeds=DEFAULT_SEEDS,
                  time_budget_seconds=17100):
    """Freeze existing recipes without sampling or observing a new objective."""
    families, seeds = list(families), list(seeds)
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError("families must be a nonempty unique subset of the five model families.")
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds)
            or HISTORICAL_FASHION_SEEDS.intersection(seeds)):
        raise ValueError("Use at least three unique fresh integer seeds, excluding historical Fashion-MNIST seeds.")
    if (isinstance(time_budget_seconds, bool) or not isinstance(time_budget_seconds, (int, float))
            or not math.isfinite(time_budget_seconds) or not 0 < time_budget_seconds <= 604800):
        raise ValueError("The locked time budget must be positive and at most seven days.")
    publication_path = Path(publication_path)
    publication = load_json(publication_path)
    anchors = {row["family"]: row for row in publication["all_candidates"]
               if row["candidate_id"] == "C0"}
    selected = {row["family"]: row for row in publication["selected_recipes"]}
    recipes = {}
    for family in families:
        recipes[family] = {}
        for arm, candidate in (("C0", anchors[family]), ("selected", selected[family])):
            candidate = copy.deepcopy(candidate)
            if candidate["effective_sha256"] != effective_config_hash(family, candidate["config"]):
                raise ValueError(f"Published recipe fingerprint differs: {family}/{arm}")
            if candidate["config"].get("epochs") != 60:
                raise ValueError("Confirmation retains the frozen 60-epoch training horizon.")
            candidate["config"].update(num_workers=0, cpu_threads=4)
            recipes[family][arm] = candidate
    return {"schema_version": 1, "study_type": "matched_post_test_recipe_checkpoint_confirmation",
            "families": families, "seeds": seeds, "epochs": 60, "recipes": recipes,
            "publication_sha256": sha256_file(publication_path),
            "expected_dataset_sha256": DATASET_SHA256,
            "expected_split_sha256": SPLIT_SHA256,
            "checkpoint_policies": list(POLICIES), "primary_comparison": "selected_minus_C0_at_fixed_final_epoch",
            "primary_family": "cnn" if "cnn" in families else families[0],
            "secondary_families": [f for f in families if f != ("cnn" if "cnn" in families else families[0])],
            "secondary_comparisons": ["recipe_at_ce", "recipe_at_f1", "selector_within_recipe", "factor_interaction"],
            "time_budget_seconds": float(time_budget_seconds), "finalization_reserve_seconds": 120,
            "training_pairing": "same split, engine, training seed, initialization policy and loader seed; no early stopping",
            "task_order": "family then seed; recipe order alternates by seed index",
            "old_validation_scope": "previously_seen_exploratory_selection_set",
            "official_test_evaluated": False, "limitations": LIMITATIONS}


def tasks_for(manifest):
    protocol = manifest["protocol"]
    tasks = []
    for family in protocol["families"]:
        for index, seed in enumerate(protocol["seeds"]):
            for arm in (("C0", "selected") if index % 2 == 0 else ("selected", "C0")):
                recipe = protocol["recipes"][family][arm]
                tasks.append({"task_id": f"{family}_{arm}_seed{seed}", "family": family,
                              "phase": "matched_confirmation", "seed": seed,
                              "epochs": protocol["epochs"], "arm": arm,
                              "candidate_id": recipe["candidate_id"],
                              "config": copy.deepcopy(recipe["config"])})
    return tasks


class RunLease:
    """Token-checked advisory Drive lease; one runtime owns the study."""
    def __init__(self, out, accrue=False):
        self.out, self.accrue = Path(out), accrue
        self.path = self.out / "CONFIRMATION_OWNER.json"
        self.guard = self.out / "CONFIRMATION_OWNER_TRANSACTION.lock"
        self.token = uuid.uuid4().hex

    @contextlib.contextmanager
    def _guard(self):
        if any((self.out / name).exists() for name in
               ("study_manifest.json", "run_manifest.json", "freeze_manifest.json", "student_run_export.json")):
            raise ValueError("This is a historical experiment folder; use a separate confirmation directory.")
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError("Ownership transaction is busy/interrupted; preserve its lock for diagnosis.") from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"token": self.token, "pid": os.getpid()}, stream)
            yield
        finally:
            self.guard.unlink(missing_ok=True)

    def __enter__(self):
        with self._guard():
            if self.path.exists():
                previous = load_json(self.path)
                if previous.get("active") and time.time() - previous["heartbeat_unix"] < 600:
                    raise RuntimeError("Another runtime owns this study, or the previous heartbeat is younger than ten minutes.")
            self.record = {"token": self.token, "active": True, "host": socket.gethostname(),
                           "pid": os.getpid(), "heartbeat_unix": time.time()}
            atomic_json(self.path, self.record)
            self.last = time.monotonic()
        return self

    def _heartbeat(self):
        if load_json(self.path)["token"] != self.token:
            raise RuntimeError("Study ownership changed; refusing concurrent writes.")
        now = time.monotonic()
        if self.accrue and (self.out / "clock.json").exists():
            clock = load_json(self.out / "clock.json")
            clock["active_run_seconds"] += now - self.last
            atomic_json(self.out / "clock.json", clock)
        self.last = now
        self.record["heartbeat_unix"] = time.time()
        atomic_json(self.path, self.record)

    def heartbeat(self):
        with self._guard():
            self._heartbeat()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            with self._guard():
                self._heartbeat()
                self.record["active"] = False
                atomic_json(self.path, self.record)
        except Exception:
            if exc_type is None:
                raise
        return False


def _verify_manifest(out):
    out = Path(out)
    manifest = load_json(out / "confirmation_manifest.json")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != 1 or manifest["manifest_sha256"] != json_hash(body):
        raise ValueError("Confirmation manifest checksum/schema differs.")
    if manifest["source"] != source_identity():
        raise ValueError("Confirmation source changed; restore the locked source to resume.")
    if sha256_file(out / "split_ids.npz") != manifest["split_file_sha256"]:
        raise ValueError("Confirmation split artifact changed.")
    clock = load_json(out / "clock.json")
    if (clock["budget_seconds"] != manifest["protocol"]["time_budget_seconds"]
            or not math.isfinite(clock["active_run_seconds"]) or clock["active_run_seconds"] < 0):
        raise ValueError("Confirmation clock is invalid or its locked budget changed.")
    return manifest


def prepare(out, publication_path, data_dir, device="cuda", families=FAMILIES,
            seeds=DEFAULT_SEEDS, time_budget_seconds=17100):
    out = Path(out)
    protocol = make_protocol(publication_path, families, seeds, time_budget_seconds)
    with RunLease(out):
        if (out / "confirmation_manifest.json").exists():
            manifest = _verify_manifest(out)
            if manifest["protocol"] != protocol or manifest["environment"] != environment_identity(device):
                raise ValueError("Requested protocol/environment differs from the existing study.")
            return manifest
        if any((out / name).exists() for name in
               ("tasks", "study_manifest.json", "run_manifest.json", "freeze_manifest.json", "STUDY_COMPLETE.json")):
            raise ValueError("Use a new confirmation folder, separate from every historical run.")
        preparation = {"protocol": protocol, "source": source_identity(),
                       "environment": environment_identity(device)}
        atomic_json(out / "PREPARATION_BINDING.json", preparation, immutable=True)
        dataset, _ = load_training_data(data_dir, DATASET_SHA256)
        with np.load(ROOT / "assets/a1_optimization_reference/split_indices.npz", allow_pickle=False) as saved:
            train, validation = saved["train"].copy(), saved["validation"].copy()
        if (len(train), len(validation)) != (54000, 6000) or len(dataset) != 60000:
            raise ValueError("The confirmation study retains the exact original 54k/6k split.")
        if (not np.array_equal(np.sort(np.r_[train, validation]), np.arange(60000))
                or split_hash(train, validation) != SPLIT_SHA256):
            raise ValueError("Original training/validation split fingerprint differs.")
        stream = io.BytesIO()
        np.savez_compressed(stream, train=train, validation=validation)
        if (out / "split_ids.npz").exists():
            with np.load(out / "split_ids.npz", allow_pickle=False) as previous:
                if not np.array_equal(previous["train"], train) or not np.array_equal(previous["validation"], validation):
                    raise ValueError("An interrupted preparation has a different split.")
        else:
            atomic_bytes(out / "split_ids.npz", stream.getvalue())
        atomic_json(out / "clock.json", {"budget_seconds": protocol["time_budget_seconds"],
                    "active_run_seconds": 0.0,
                    "scope": "Controller active run time; setup/smoke excluded; epoch-boundary stop can overrun by an epoch."},
                    immutable=True)
        manifest = {"schema_version": 1, **preparation,
                    "split_file_sha256": sha256_file(out / "split_ids.npz")}
        manifest["manifest_sha256"] = json_hash(manifest)
        atomic_json(out / "confirmation_manifest.json", manifest, immutable=True)
        return manifest


def _binding(manifest):
    return {"confirmation_manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": manifest["source"]["sha256"],
            "dataset_sha256": manifest["protocol"]["expected_dataset_sha256"],
            "split_sha256": manifest["protocol"]["expected_split_sha256"],
            "environment": manifest["environment"]}


def verify_trajectory(out, task, binding):
    out = Path(out)
    engine = verify_complete(out / "engine", binding)
    if engine["task_sha256"] != json_hash(task):
        raise ValueError("Completed trajectory task changed.")
    record = load_json(out / "TRAJECTORY_COMPLETE.json")
    if record["task_sha256"] != json_hash(task) or record["binding_sha256"] != json_hash(binding):
        raise ValueError("Trajectory binding differs.")
    for name, digest in record["artifacts_sha256"].items():
        path = (out / name).resolve()
        if out.resolve() not in path.parents or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Trajectory artifact missing/changed: {name}")
    return record


def train_trajectory(out, task, dataset, train_ids, val_ids, binding, device,
                     stop_at=None, heartbeat=None, fail_after_epoch=None):
    """Run one shared trajectory and export CE/F1/fixed-final checkpoints.

    The low-level function also supports synthetic software tests. The public
    controller constructs the locked 60-epoch, official-train-only tasks.
    """
    out, device = Path(out), torch.device(device)
    if (out / "TRAJECTORY_COMPLETE.json").exists():
        return verify_trajectory(out, task, binding)
    engine = train_task(out / "engine", task, dataset, dataset, train_ids, val_ids,
                        binding, device, stop_at=stop_at, heartbeat=heartbeat,
                        fail_after_epoch=fail_after_epoch)
    if engine is None:
        return None
    pointer = load_json(out / "engine/COMMIT.json")
    state_path = (out / "engine" / pointer["state_file"]).resolve()
    if (state_path.parent != (out / "engine").resolve()
            or sha256_file(state_path) != pointer["sha256"]):
        raise ValueError("Final committed state is unsafe or changed.")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (state["epoch"] != task["epochs"] or state["task_sha256"] != json_hash(task)
            or state["binding_sha256"] != json_hash(binding)):
        raise ValueError("Final committed state is not the locked completed trajectory.")
    final_payload = {"model_state_dict": state["model"], "task": task, "binding": binding,
                     "epoch": task["epochs"], "selection_rule": "fixed_final_epoch_locked_before_training",
                     "trusted_own_source_artifact_only": True}
    atomic_torch_save(out / "final_epoch.pt", final_payload)
    generator = torch.Generator().manual_seed(task["seed"] + 8017)
    loader = DataLoader(Subset(dataset, list(map(int, val_ids))), shuffle=False,
                        batch_size=task["config"].get("batch_size", 128), num_workers=0,
                        generator=generator, pin_memory=device.type == "cuda")
    model = build_model(task["family"], task["config"], task["seed"]).to(device)
    paths = {"ce": "engine/best_ce.pt", "f1": "engine/best_f1.pt", "final": "final_epoch.pt"}
    cells = []
    for policy, relative in paths.items():
        payload = torch.load(out / relative, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        metrics, predictions = _evaluate(model, loader, device, list(map(int, val_ids)))
        expected = (engine["metrics"]["best_ce_metrics"] if policy == "ce" else
                    engine["metrics"]["best_f1_metrics"] if policy == "f1" else
                    {key: state["history"][-1]["val_" + key] for key in ("loss", "accuracy", "macro_f1")})
        if any(not math.isclose(metrics[key], expected[key], abs_tol=1e-7, rel_tol=0)
               for key in ("loss", "accuracy", "macro_f1")):
            raise ValueError(f"Reloaded {policy} checkpoint metrics differ.")
        atomic_bytes(out / f"predictions_{policy}.csv", _csv_bytes(predictions))
        cells.append({"task_id": task["task_id"], "family": task["family"], "seed": task["seed"],
                      "arm": task["arm"], "candidate_id": task["candidate_id"], "policy": policy,
                      "checkpoint_epoch": payload["epoch"], "checkpoint": relative,
                      "checkpoint_sha256": sha256_file(out / relative), **metrics,
                      "evaluation_scope": "previously_seen_old_validation_exploratory"})
    atomic_json(out / "cells.json", cells)
    artifacts = ["engine/COMPLETE.json", "engine/best_ce.pt", "engine/best_f1.pt", "final_epoch.pt",
                 "cells.json", *[f"predictions_{policy}.csv" for policy in POLICIES]]
    record = {"status": "COMPLETE", "task_sha256": json_hash(task), "binding_sha256": json_hash(binding),
              "same_training_trajectory_for_all_policies": True, "cells": cells,
              "artifacts_sha256": {name: sha256_file(out / name) for name in artifacts}}
    atomic_json(out / "TRAJECTORY_COMPLETE.json", record, immutable=True)
    return verify_trajectory(out, task, binding)


def _records(out, manifest):
    binding = _binding(manifest)
    return {task["task_id"]: verify_trajectory(Path(out) / "tasks" / task["task_id"], task, binding)
            for task in tasks_for(manifest)
            if (Path(out) / "tasks" / task["task_id"] / "TRAJECTORY_COMPLETE.json").exists()}


def summarize_cells(cells):
    """Aggregate prespecified paired contrasts, without a significance claim."""
    index = {(row["family"], row["seed"], row["arm"], row["policy"]): row for row in cells}
    rows, contrasts = [], []
    for family in dict.fromkeys(row["family"] for row in cells):
        seeds = sorted({row["seed"] for row in cells if row["family"] == family})
        paired = [s for s in seeds if all((family, s, arm, policy) in index
                  for arm in ("C0", "selected") for policy in POLICIES)]
        for arm in ("C0", "selected"):
            for policy in POLICIES:
                available = [index[(family, seed, arm, policy)] for seed in seeds
                             if (family, seed, arm, policy) in index]
                if not available:
                    continue
                row = {"family": family, "arm": arm, "policy": policy, "n_seeds": len(available)}
                for metric in ("macro_f1", "accuracy", "loss"):
                    values = [r[metric] for r in available]
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
                rows.append(row)
        definitions = {
            "recipe_fixed_final_primary": [("selected", "final", 1), ("C0", "final", -1)],
            "recipe_at_ce": [("selected", "ce", 1), ("C0", "ce", -1)],
            "recipe_at_f1": [("selected", "f1", 1), ("C0", "f1", -1)],
            "selector_within_C0": [("C0", "f1", 1), ("C0", "ce", -1)],
            "selector_within_selected": [("selected", "f1", 1), ("selected", "ce", -1)],
            "recipe_selector_interaction": [("selected", "f1", 1), ("selected", "ce", -1),
                                             ("C0", "f1", -1), ("C0", "ce", 1)],
        }
        for name, terms in definitions.items():
            values = [sum(sign * index[(family, seed, arm, policy)]["macro_f1"]
                          for arm, policy, sign in terms) for seed in paired]
            contrasts.append({"family": family, "contrast": name, "seeds": paired,
                              "n_paired_seeds": len(paired), "macro_f1_delta_per_seed": values,
                              "macro_f1_delta_mean": statistics.mean(values) if values else None,
                              "macro_f1_delta_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                              "interpretation": "paired exploratory old-validation difference; not an independent generalization estimate"})
    return {"cells": rows, "paired_contrasts": contrasts, "limitations": LIMITATIONS,
            "official_test_evaluated": False}


def status(out):
    out = Path(out)
    manifest = _verify_manifest(out)
    records = _records(out, manifest)
    expected = tasks_for(manifest)
    complete = len(records) == len(expected) and (out / "STUDY_COMPLETE.json").exists()
    if complete:
        marker = load_json(out / "STUDY_COMPLETE.json")
        actual = {task_id: sha256_file(out / "tasks" / task_id / "TRAJECTORY_COMPLETE.json")
                  for task_id in records}
        if (marker["manifest_sha256"] != manifest["manifest_sha256"]
                or marker["trajectory_receipts_sha256"] != actual
                or marker["analysis_sha256"] != sha256_file(out / "analysis/summary.json")
                or marker["evaluation_lock_sha256"] != sha256_file(out / "EVALUATION_LOCK.json")):
            raise ValueError("Study completion artifact changed.")
    return {"status": "COMPLETE" if complete else "INCOMPLETE",
            "completed_trajectories": len(records), "maximum_trajectories": len(expected),
            "completed_checkpoint_cells": 3 * len(records), "maximum_checkpoint_cells": 3 * len(expected),
            "clock": load_json(out / "clock.json"), "official_test_evaluated": False,
            "limitations": LIMITATIONS}


def run(out, data_dir, device="cuda", max_tasks=None):
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer.")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Full confirmation training is student-run on an A100; CPU is for software checks.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    out = Path(out)
    with RunLease(out, accrue=False) as lease:
        manifest = _verify_manifest(out)
        if manifest["environment"] != environment_identity(device):
            raise ValueError("Resume environment differs from the locked environment.")
        if status(out)["status"] == "COMPLETE":
            return status(out)
        dataset, _ = load_training_data(data_dir, manifest["protocol"]["expected_dataset_sha256"])
        with np.load(out / "split_ids.npz", allow_pickle=False) as saved:
            train_ids, val_ids = saved["train"].copy(), saved["validation"].copy()
        records = _records(out, manifest)
        lease.accrue = True
        lease.last = time.monotonic()
        completed_now = 0
        for task in tasks_for(manifest):
            if task["task_id"] in records:
                continue
            lease.heartbeat()
            clock = load_json(out / "clock.json")
            left = clock["budget_seconds"] - clock["active_run_seconds"]
            reserve = manifest["protocol"]["finalization_reserve_seconds"]
            if left <= reserve:
                atomic_json(out / "STOP_REASON.json", {"status": "INCOMPLETE_TIME_BUDGET", "seconds_left": left})
                return status(out)
            result = train_trajectory(out / "tasks" / task["task_id"], task, dataset, train_ids, val_ids,
                                      _binding(manifest), device, stop_at=time.monotonic() + left - reserve,
                                      heartbeat=lease.heartbeat)
            lease.heartbeat()
            if result is None:
                atomic_json(out / "STOP_REASON.json", {"status": "INCOMPLETE_EPOCH_BOUNDARY", "task_id": task["task_id"]})
                return status(out)
            records[task["task_id"]] = result
            completed_now += 1
            cells = [cell for record in records.values() for cell in record["cells"]]
            atomic_json(out / "analysis/summary.json", summarize_cells(cells))
            if max_tasks is not None and completed_now >= max_tasks:
                break
        if len(records) == len(tasks_for(manifest)):
            roster = [{**{key: cell[key] for key in ("task_id", "family", "seed", "arm", "policy", "checkpoint_epoch", "checkpoint_sha256")},
                       "checkpoint": f"tasks/{cell['task_id']}/{cell['checkpoint']}"}
                      for record in records.values() for cell in record["cells"]]
            lock = {"schema_version": 1, "confirmation_manifest_sha256": manifest["manifest_sha256"],
                    "checkpoints": roster, "primary_policy": "final", "primary_contrast": "selected_minus_C0",
                    "primary_family": manifest["protocol"]["primary_family"],
                    "audit_labels_accessed_by_this_package": False,
                    "scope": "All models and contrasts frozen before an external prospective audit; no per-audit winner selection."}
            lock["lock_sha256"] = json_hash(lock)
            atomic_json(out / "EVALUATION_LOCK.json", lock, immutable=True)
            atomic_json(out / "STUDY_COMPLETE.json", {"status": "COMPLETE", "manifest_sha256": manifest["manifest_sha256"],
                        "trajectory_receipts_sha256": {name: sha256_file(out / "tasks" / name / "TRAJECTORY_COMPLETE.json") for name in records},
                        "analysis_sha256": sha256_file(out / "analysis/summary.json"),
                        "evaluation_lock_sha256": sha256_file(out / "EVALUATION_LOCK.json"),
                        "evaluation_scope": "matched_exploratory_old_validation_only"}, immutable=True)
        return status(out)


def smoke(out, data_dir, device="cuda"):
    out, device = Path(out), torch.device(device)
    with RunLease(out):
        manifest = _verify_manifest(out)
        dataset, _ = load_training_data(data_dir, manifest["protocol"]["expected_dataset_sha256"])
        with np.load(out / "split_ids.npz", allow_pickle=False) as saved:
            ids = saved["train"][:8]
        images = torch.stack([dataset[int(i)][0] for i in ids]).to(device)
        labels = torch.tensor([dataset[int(i)][1] for i in ids], device=device)
        torch.set_num_threads(4)
        checks = []
        for family, recipes in manifest["protocol"]["recipes"].items():
            for arm, recipe in recipes.items():
                config = recipe["config"]
                model = build_model(family, config, manifest["protocol"]["seeds"][0]).to(device)
                optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config["optimizer"]]
                optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if logits.shape != (8, 10) or not torch.isfinite(loss):
                    raise FloatingPointError("Smoke forward is invalid.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Smoke optimizer produced nonfinite weights.")
                checks.append({"family": family, "arm": arm, "finite_forward_backward_optimizer": True})
        receipt = {"scope": "One eight-training-image software step per recipe; no study trajectory or performance result.",
                   "checks": checks, "official_test_evaluated": False}
        atomic_json(out / "smoke_checks.json", receipt)
        return receipt


def export_results(out):
    out = Path(out)
    with RunLease(out):
        current = status(out)
        manifest = _verify_manifest(out)
        for relative in manifest["source"]["files_sha256"]:
            atomic_bytes(out / "source_snapshot" / relative, (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
        atomic_json(out / "STUDY_PROGRESS.json", current)
        excluded = {"artifact_manifest.json", "export_receipt.json", "CONFIRMATION_OWNER.json", "CONFIRMATION_OWNER_TRANSACTION.lock"}
        payloads = [p for p in sorted(out.rglob("*")) if p.is_file() and not p.is_symlink()
                    and p.name not in excluded and not p.name.endswith(".tmp")]
        hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in payloads}
        atomic_json(out / "artifact_manifest.json", hashes)
        archive = out.parent / (out.name + "_results.zip")
        temporary = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in [*payloads, out / "artifact_manifest.json"]:
                zipped.write(path, path.relative_to(out).as_posix())
        with zipfile.ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Export CRC check failed.")
            for name, digest in hashes.items():
                if hashlib.sha256(zipped.read(name)).hexdigest() != digest:
                    raise ValueError(f"Export payload differs: {name}")
        os.replace(temporary, archive)
        receipt = {**current, "archive": str(archive), "sha256": sha256_file(archive),
                   "bytes": archive.stat().st_size, "payload_count": len(hashes)}
        atomic_json(out / "export_receipt.json", receipt)
        return receipt
