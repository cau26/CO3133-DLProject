"""Small, resumable training engine for the separately versioned A1 study.

The student runs the real Fashion-MNIST jobs. Tests use tiny synthetic data.
Validation chooses max macro-F1, then min CE, then the earliest epoch. A second
checkpoint records min CE. The test dataset is never accepted by this API.
"""
from __future__ import annotations

import csv
import io
import math
import random
import time
import uuid
import warnings
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .storage import (atomic_bytes, atomic_json, atomic_torch_save, json_hash,
                      load_json, sha256_file, verify_complete)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _summary(loss_sum, n, confusion):
    if n == 0:
        raise ValueError("An empty dataset cannot be evaluated.")
    diagonal = confusion.diagonal().double()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = torch.where(denominator > 0, 2 * diagonal / denominator,
                     torch.zeros_like(diagonal))
    return {"loss": float(loss_sum / n),
            "accuracy": float(diagonal.sum() / n), "macro_f1": float(f1.mean()),
            "n_examples": int(n)}


@torch.no_grad()
def _evaluate(model, loader, device, original_ids=None, classes=10):
    model.eval()
    confusion = torch.zeros((classes, classes), dtype=torch.int64)
    loss_sum, n, rows = 0.0, 0, []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite validation loss.")
        predicted = logits.argmax(1)
        truth, pred, per_loss = targets.cpu(), predicted.cpu(), losses.cpu()
        confusion += torch.bincount(truth * classes + pred,
                                     minlength=classes * classes).reshape(classes, classes)
        loss_sum += float(per_loss.double().sum())
        if original_ids is not None:
            for offset in range(len(truth)):
                rows.append({"original_id": int(original_ids[n + offset]),
                             "true": int(truth[offset]), "pred": int(pred[offset]),
                             "nll": float(per_loss[offset])})
        n += len(truth)
    return _summary(loss_sum, n, confusion), rows


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _schedule_clock(config, epochs, steps_per_epoch, base_lr):
    """Resolve the exact declared update clock; never silently clamp warmup."""
    if type(epochs) is not int or epochs < 1 or type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("Schedule epochs and steps_per_epoch must be positive integers.")
    if isinstance(base_lr, bool) or not isinstance(base_lr, Real) or not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("Schedule base_lr must be a positive finite real number.")
    schedule = config.get("schedule", "constant")
    if schedule not in ("constant", "cosine", "warmup_cosine"):
        raise ValueError(f"Unknown schedule: {schedule}")
    min_ratio = config.get("min_lr_ratio", .05)
    if (isinstance(min_ratio, bool) or not isinstance(min_ratio, Real)
            or not math.isfinite(min_ratio) or not 0 <= min_ratio <= 1):
        raise ValueError("min_lr_ratio must be a finite real number in [0,1].")
    warmup_epochs = config.get("warmup_epochs", 5 if schedule == "warmup_cosine" else 0)
    if type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer.")
    if schedule == "warmup_cosine":
        if not 0 < warmup_epochs < epochs:
            raise ValueError("warmup_cosine requires 0 < warmup_epochs < epochs.")
    elif warmup_epochs != 0:
        raise ValueError("warmup_epochs must be zero for constant/cosine schedules.")
    return {"total_updates": epochs * steps_per_epoch,
            "warmup_updates": warmup_epochs * steps_per_epoch,
            "min_ratio": float(min_ratio), "base_lr": float(base_lr), "schedule": schedule}


def _load_transaction(out, task_hash, binding_hash):
    pointers = [out / "COMMIT.json", out / "COMMIT.previous.json"]
    errors = []
    for pointer in pointers:
        if not pointer.exists():
            continue
        try:
            record = load_json(pointer)
            state_path = out / record["state_file"]
            if state_path.resolve().parent != out.resolve():
                raise ValueError("Unsafe transaction filename.")
            if sha256_file(state_path) != record["sha256"]:
                raise ValueError("Transaction checksum differs.")
            # Trusted self-produced checkpoint; numpy/Python RNG states need pickle.
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state["task_sha256"] != task_hash or state["binding_sha256"] != binding_hash:
                raise ValueError("Transaction source/config/split/environment binding differs.")
            if pointer.name != "COMMIT.json":
                atomic_json(out / "RECOVERY.json", {
                    "status": "PREVIOUS_COMMITTED_GENERATION_RECOVERED",
                    "epoch": state["epoch"], "rejected": errors,
                    "discarded_work_included_in_fit_timer": False})
            return state, record
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
            errors.append(f"{pointer.name}: {type(error).__name__}: {error}")
    if errors or any(out.glob("state-*.pt")):
        raise ValueError("No valid committed transaction; preserve this folder for diagnosis. "
                         + " | ".join(errors))
    return None, None


def _commit(out, state, previous):
    # A replayed epoch never overwrites an old or corrupt generation. Pointer
    # replacement alone commits a new transaction; orphan generations are inert.
    filename = f"state-{state['epoch']:04d}-{uuid.uuid4().hex}.pt"
    digest = atomic_torch_save(out / filename, state)
    record = {"state_file": filename, "sha256": digest, "epoch": state["epoch"],
              "task_sha256": state["task_sha256"],
              "binding_sha256": state["binding_sha256"]}
    if previous:
        atomic_json(out / "COMMIT.previous.json", previous)
    atomic_json(out / "COMMIT.json", record)
    keep = {filename, previous["state_file"] if previous else filename}
    for old in out.glob("state-*.pt"):
        if old.name not in keep:
            try:
                old.unlink()
            except OSError as error:
                warnings.warn(f"Committed state is safe; retained an older checkpoint: {old}: {error}")
    return record


def _checkpoint_payload(weights, task, binding, metrics, epoch):
    return {"model_state_dict": weights, "task": task, "binding": binding,
            "metrics": metrics, "epoch": epoch,
            "trusted_own_source_artifact_only": True}


def train_task(out: Path, task: dict, train_dataset, val_dataset, train_ids, val_ids,
               binding: dict, device, stop_at=None, fail_after_epoch=None, heartbeat=None):
    """Train/resume one locked task; return completion or None at a time boundary.

    ``stop_at`` is a time.monotonic() deadline checked at epoch boundaries. The
    test-only ``fail_after_epoch`` injects a crash after that epoch is committed.
    Existing COMPLETE results are integrity-checked and never overwritten.
    """
    from .models import build_model
    from .recipes import schedule_lr

    out, device = Path(out), torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    task_hash, binding_hash = json_hash(task), json_hash(binding)
    if (out / "COMPLETE.json").exists():
        complete = verify_complete(out, binding)
        if complete["task_sha256"] != task_hash:
            raise ValueError("Completed task config differs.")
        return complete
    if (out / "FAILED.json").exists():
        raise RuntimeError(f"Task has a recorded numerical failure: {out / 'FAILED.json'}")
    atomic_json(out / "TASK.json", {"task": task, "binding": binding}, immutable=True)
    config = task["config"]
    epochs, batch = int(task.get("epochs", 60)), int(config.get("batch_size", 128))
    if epochs < 1 or batch < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Train and validation subsets must be nonempty.")
    if len(set(map(int, train_ids))) != len(train_ids) or len(set(map(int, val_ids))) != len(val_ids):
        raise ValueError("Duplicate original IDs in a task split.")
    if set(map(int, train_ids)) & set(map(int, val_ids)):
        raise ValueError("Train/validation IDs overlap.")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    _seed(int(task["seed"]))
    model = build_model(task["family"], config, int(task["seed"])).to(device)
    base_lr = float(config.get("learning_rate", config.get("lr", .001)))
    weight_decay = float(config.get("weight_decay", config.get("WD", 0.0)))
    optimizer_name = config.get("optimizer", "adam").lower()
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    train_generator = torch.Generator().manual_seed(int(task["seed"]) + 1017)
    val_generator = torch.Generator().manual_seed(int(task["seed"]) + 8017)
    loader_kwargs = {"batch_size": batch, "num_workers": int(config.get("num_workers", 0)),
                     "worker_init_fn": _worker_seed, "persistent_workers": False,
                     "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_dataset, list(map(int, train_ids))), shuffle=True,
                              generator=train_generator, **loader_kwargs)
    val_loader = DataLoader(Subset(val_dataset, list(map(int, val_ids))), shuffle=False,
                            generator=val_generator, **loader_kwargs)
    schedule_clock = _schedule_clock(config, epochs, len(train_loader), base_lr)
    total_updates = schedule_clock["total_updates"]
    schedule, warmup = schedule_clock["schedule"], schedule_clock["warmup_updates"]
    state, previous = _load_transaction(out, task_hash, binding_hash)
    if state is None:
        state = {"schema": 1, "epoch": 0, "updates": 0, "history": [],
                 "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                 "rng": _rng_state(), "train_generator": train_generator.get_state(),
                 "val_generator": val_generator.get_state(), "best_f1": None,
                 "best_ce": None, "fit_wall_seconds": 0.0,
                 "task_sha256": task_hash, "binding_sha256": binding_hash,
                 "schedule_clock": schedule_clock}
        previous = _commit(out, state, None)
    else:
        if state["schedule_clock"] != schedule_clock:
            raise ValueError("Resume schedule clock differs.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                # Non-capturable Adam keeps its scalar step on CPU. Moment
                # tensors must follow model parameters onto the resumed GPU.
                if torch.is_tensor(value) and key != "step":
                    optimizer_state[key] = value.to(device)
        train_generator.set_state(state["train_generator"])
        val_generator.set_state(state["val_generator"])
        _restore_rng(state["rng"])
    if heartbeat is not None:
        heartbeat()
    clip = float(config.get("gradient_clip", config.get("clip_grad", config.get("clip", 1.0))))
    try:
        for epoch in range(state["epoch"] + 1, epochs + 1):
            if stop_at is not None and time.monotonic() >= stop_at:
                return None
            started = time.monotonic()
            model.train()
            confusion = torch.zeros((10, 10), dtype=torch.int64)
            loss_sum, n, norms, clipped, lr_first, lr_last = 0.0, 0, [], 0, None, None
            train_started = time.monotonic()
            for images, targets in train_loader:
                update_index = state["updates"] + 1
                lr = base_lr if schedule == "constant" else schedule_lr(
                    base_lr, update_index, total_updates, warmup_updates=warmup,
                    min_ratio=schedule_clock["min_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                lr_first = lr if lr_first is None else lr_first
                lr_last = lr
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training CE.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else math.inf,
                                                      error_if_nonfinite=True)
                norm_value = float(norm)
                norms.append(norm_value)
                clipped += int(clip > 0 and norm_value > clip)
                optimizer.step()
                predicted, truth = logits.detach().argmax(1).cpu(), targets.cpu()
                confusion += torch.bincount(truth * 10 + predicted, minlength=100).reshape(10, 10)
                loss_sum += float(loss.detach()) * len(truth)
                n += len(truth)
                state["updates"] += 1
            train_seconds = time.monotonic() - train_started
            val_started = time.monotonic()
            val, _ = _evaluate(model, val_loader, device)
            val_seconds = time.monotonic() - val_started
            train = _summary(loss_sum, n, confusion)
            record = {"epoch": epoch, "updates": state["updates"],
                      **{f"train_{key}": value for key, value in train.items()},
                      **{f"val_{key}": value for key, value in val.items()},
                      "lr_first": lr_first, "lr_last": lr_last,
                      "gradient_norm_mean": float(np.mean(norms)),
                      "gradient_norm_max": max(norms),
                      "clip_fraction": clipped / len(norms),
                      "train_seconds": train_seconds, "validation_seconds": val_seconds}
            state["history"].append(record)
            best = state["best_f1"]
            if best is None or (val["macro_f1"], -val["loss"]) > (
                    best["metrics"]["macro_f1"], -best["metrics"]["loss"]):
                state["best_f1"] = {"epoch": epoch, "metrics": val, "weights": _cpu_state(model)}
            best = state["best_ce"]
            if best is None or val["loss"] < best["metrics"]["loss"]:
                state["best_ce"] = {"epoch": epoch, "metrics": val, "weights": _cpu_state(model)}
            state.update(epoch=epoch, model=_cpu_state(model), optimizer=optimizer.state_dict(),
                         rng=_rng_state(), train_generator=train_generator.get_state(),
                         val_generator=val_generator.get_state())
            state["fit_wall_seconds"] += time.monotonic() - started
            previous = _commit(out, state, previous)
            if heartbeat is not None:
                heartbeat()
            if fail_after_epoch == epoch:
                raise RuntimeError("Injected test interruption after committed epoch.")
            print(f"{task['task_id']} epoch {epoch}/{epochs}: val F1={val['macro_f1']:.5f}, "
                  f"CE={val['loss']:.5f}", flush=True)
    except FloatingPointError as error:
        atomic_json(out / "FAILED.json", {"status": "NUMERICAL_FAILURE", "error": str(error),
                                          "task_sha256": task_hash, "binding_sha256": binding_hash,
                                          "last_committed_epoch": previous["epoch"]})
        raise
    except RuntimeError as error:
        if "non-finite" in str(error).lower() or "nonfinite" in str(error).lower():
            atomic_json(out / "FAILED.json", {"status": "NUMERICAL_FAILURE", "error": str(error),
                                              "task_sha256": task_hash, "binding_sha256": binding_hash,
                                              "last_committed_epoch": previous["epoch"]})
        raise
    if state["epoch"] != epochs or state["updates"] != total_updates:
        raise AssertionError("Incomplete task reached finalization.")
    finalization_started = time.monotonic()
    for key, filename in (("best_f1", "best_f1.pt"), ("best_ce", "best_ce.pt")):
        best = state[key]
        atomic_torch_save(out / filename, _checkpoint_payload(best["weights"], task, binding,
                                                             best["metrics"], best["epoch"]))
    # Reload the exact exported F1-selected checkpoint before reporting completion.
    checkpoint = torch.load(out / "best_f1.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected, predictions = _evaluate(model, val_loader, device, list(map(int, val_ids)))
    for key in ("loss", "accuracy", "macro_f1"):
        if not math.isclose(selected[key], state["best_f1"]["metrics"][key], rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"Reloaded selected checkpoint differs for {key}.")
    metrics = {"task_id": task["task_id"], "family": task["family"], "phase": task["phase"],
               "seed": task["seed"], "parameters": sum(p.numel() for p in model.parameters()),
               "selection_rule": "max_macro_f1_then_min_ce_then_earliest_epoch",
               "best_f1_epoch": state["best_f1"]["epoch"], "best_f1_metrics": selected,
               "best_ce_epoch": state["best_ce"]["epoch"],
               "best_ce_metrics": state["best_ce"]["metrics"],
               "epochs_completed": state["epoch"], "optimizer_updates": state["updates"],
               "n_train": len(train_ids), "n_validation": len(val_ids),
               "fit_wall_seconds": state["fit_wall_seconds"],
               "fit_timer_scope": "completed epochs: train + validation + in-memory checkpoint selection; "
                                  "excludes setup, transaction writes, finalization, discarded crash work",
               "training_loop_seconds": sum(row["train_seconds"] for row in state["history"]),
               "validation_seconds": sum(row["validation_seconds"] for row in state["history"]),
               "scope": "validation_only_exploratory_no_official_test",
               "task_sha256": task_hash, "binding_sha256": binding_hash}
    if task["phase"] == "final":
        from a1.train import measure_inference
        # The same first validation batch for every final seed/family. This is
        # model-forward timing after selection, never an official-test call.
        inference_batch = next(iter(val_loader))[0]
        metrics["inference"] = measure_inference(model, inference_batch, device,
                                                 warmup=10, repeats=50)
        metrics["inference"]["interpretation"] = (
            "batch-amortized model-forward milliseconds; not single-request service latency")
    atomic_bytes(out / "history.csv", _csv_bytes(state["history"]))
    atomic_bytes(out / "predictions.csv", _csv_bytes(predictions))
    metrics["finalization_seconds_before_metrics_write"] = time.monotonic() - finalization_started
    atomic_json(out / "metrics.json", metrics)
    artifact_names = ["TASK.json", "history.csv", "predictions.csv", "metrics.json",
                      "best_f1.pt", "best_ce.pt", "COMMIT.json", previous["state_file"]]
    complete = {"status": "COMPLETE", "task_sha256": task_hash, "binding_sha256": binding_hash,
                "metrics": metrics,
                "artifacts_sha256": {name: sha256_file(out / name) for name in artifact_names}}
    atomic_json(out / "COMPLETE.json", complete, immutable=True)
    return verify_complete(out, binding)
