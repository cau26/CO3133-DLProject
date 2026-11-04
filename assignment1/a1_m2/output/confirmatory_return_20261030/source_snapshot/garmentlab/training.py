"""Fixed, resumable student-run MobileNet comparison; no HPO or test selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch
from torch import nn

from .runtime import atomic_json, digest, utc_now, utc_parse

ROLE_NAMES = ("fit", "selection", "calibration", "assessment")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TrainingStopped(RuntimeError):
    """A recorded budget or caller deadline requires an incomplete, resumable stop."""


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _protect_json(path, value):
    path = Path(path)
    if path.exists() and _json(path) != value:
        raise ValueError(f"Saved binding differs: {path.name}. Use a new run directory.")
    if not path.exists():
        atomic_json(path, value)


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.025 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def _state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tensor_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def load_prepared(prepared_dir):
    """Validate materialized arrays and split records without evaluating any model."""
    root = Path(prepared_dir)
    ready = _json(root / "DATA_READINESS.json")
    if ready.get("training_ready") is not True:
        raise ValueError("Prepared data has not passed the declared training-readiness checks")
    index = _json(root / "index_manifest.json")
    records = index.get("records")
    classnames = index.get("labels")
    if classnames is None and isinstance(index.get("class_map"), list):
        classnames = [item["label"] for item in index["class_map"]]
    if (not isinstance(records, list) or not records or not isinstance(classnames, list)
            or not 2 <= len(classnames) <= 1000 or len(set(classnames)) != len(classnames)):
        raise ValueError("Prepared index requires records and ordered unique labels")
    images = np.load(root / "images_uint8.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", allow_pickle=False)
    roles = np.load(root / "roles.npy", allow_pickle=False)
    n = len(records)
    if images.shape != (n, 224, 224, 3) or images.dtype != np.uint8:
        raise ValueError("Prepared images must be uint8 [N,224,224,3]")
    if (labels.shape != (n,) or labels.dtype.kind not in "iu" or labels.min() < 0
            or labels.max() >= len(classnames) or roles.shape != (n,) or roles.dtype.kind not in "iu"
            or not np.isin(roles, range(4)).all()):
        raise ValueError("Prepared label/role arrays are invalid")
    groups = {}
    for position, record in enumerate(records):
        if record.get("array_index", position) != position:
            raise ValueError("Prepared records must follow array_index order")
        role = record.get("role", record.get("partition"))
        if role != ROLE_NAMES[int(roles[position])]:
            raise ValueError("Prepared record role differs from roles.npy")
        if "class_index" in record and record["class_index"] != int(labels[position]):
            raise ValueError("Prepared record class_index differs from labels.npy")
        if "label" in record and record["label"] != classnames[int(labels[position])]:
            raise ValueError("Prepared record label differs from the ordered class map")
        group = record.get("source_group_id")
        if group is not None:
            if group in groups and groups[group] != role:
                raise ValueError("A source group appears across data roles")
            groups[group] = role
    for role in range(4):
        if not np.any(roles == role):
            raise ValueError(f"Prepared data has no {ROLE_NAMES[role]} examples")
        if set(labels[roles == role].tolist()) != set(range(len(classnames))):
            raise ValueError(f"Prepared {ROLE_NAMES[role]} partition is missing a class")
    bindings = {name: digest(root / name) for name in
                ("images_uint8.npy", "labels.npy", "roles.npy", "index_manifest.json", "DATA_READINESS.json")}
    # Materialization receipts may declare array hashes; refuse mismatching declarations.
    declared = ready.get("files", ready.get("arrays", {}))
    if isinstance(declared, dict):
        for name, item in declared.items():
            expected = item.get("sha256") if isinstance(item, dict) else item
            if name in bindings and isinstance(expected, str) and expected != bindings[name]:
                raise ValueError(f"Prepared array hash mismatch: {name}")
    return {"images": images, "labels": labels.astype(np.int64), "roles": roles,
            "records": records, "classnames": classnames, "bindings": bindings, "index": index}


def build_model(num_classes=6, seed=314, pretrained=True):
    """ImageNet backbone and projection, with a paired fresh six-class last layer."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from a1.cnn import GlobalAveragePool2d
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.avgpool = GlobalAveragePool2d()
        model.classifier[2] = nn.Dropout(0.2)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def prepare_batch(images, indices, device="cpu", flip_mask=None):
    values = np.array(images[np.asarray(indices, dtype=np.int64)], dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(values).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0
    if flip_mask is not None:
        mask = torch.as_tensor(flip_mask, device=device, dtype=torch.bool)
        tensor = torch.where(mask[:, None, None, None], tensor.flip(-1), tensor)
    mean = tensor.new_tensor(MEAN)[None, :, None, None]
    std = tensor.new_tensor(STD)[None, :, None, None]
    return (tensor - mean) / std


def _features(model, batch):
    value = model.avgpool(model.features(batch)).flatten(1)
    return model.classifier[:2](value)


def extract_features(model, images, device="cpu", batch_size=128, flip=False, indices=None, stop_check=None):
    """FP32 embeddings before classifier dropout; no label-dependent model fitting."""
    ids = np.arange(len(images)) if indices is None else np.asarray(indices, dtype=np.int64)
    model = model.to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            if stop_check is not None and stop_check():
                raise TrainingStopped("Feature extraction stopped at the caller deadline")
            rows = ids[start:start + batch_size]
            batch = prepare_batch(images, rows, device, np.full(len(rows), flip))
            encoded = _features(model, batch)
            if not torch.isfinite(encoded).all():
                raise ValueError("Feature extraction produced non-finite embeddings")
            values.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(values) if values else np.empty((0, model.classifier[-1].in_features), np.float32)


def set_training_mode(model, recipe):
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    if recipe == "frozen":
        model.features.eval()
        model.avgpool.eval()
        model.classifier[:2].eval()
    return model


def configure_optimizer(model, recipe, config):
    if recipe not in ("frozen", "finetune"):
        raise ValueError("Unknown training recipe")
    for parameter in model.parameters():
        parameter.requires_grad_(recipe == "finetune")
    head = list(model.classifier[-1].parameters())
    for parameter in head:
        parameter.requires_grad_(True)
    if recipe == "frozen":
        groups = [{"params": head, "lr": config["head_lr"], "name": "head"}]
    else:
        head_ids = {id(parameter) for parameter in head}
        groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids],
                   "lr": config["backbone_lr"], "name": "backbone"},
                  {"params": head, "lr": config["head_lr"], "name": "head"}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    schedule = lambda epoch: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / config["epochs"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def classification_metrics(true, predicted, loss_sum, num_classes):
    true, predicted = np.asarray(true, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    if not len(true):
        raise ValueError("Cannot calculate metrics for an empty partition")
    matrix = np.bincount(true * num_classes + predicted, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    support = matrix.sum(1) + matrix.sum(0)
    f1 = np.divide(2 * np.diag(matrix), support, out=np.zeros(num_classes, dtype=float), where=support != 0)
    return {"loss": float(loss_sum / len(true)), "accuracy": float(np.trace(matrix) / len(true)),
            "macro_f1": float(f1.mean()), "confusion": matrix.tolist(), "examples": int(len(true))}


def run_epoch(model, data, indices, config, recipe, *, optimizer=None, cache=None,
              sampler=None, augmentation=None, stop_check=None):
    training = optimizer is not None
    device = torch.device(config["device"])
    ids = np.asarray(indices, dtype=np.int64)
    if training:
        set_training_mode(model, recipe)
        ids = ids[torch.randperm(len(ids), generator=sampler).numpy()]
    else:
        model.eval()
    total_loss, truth, predictions = 0.0, [], []
    with torch.enable_grad() if training else torch.inference_mode():
        for start in range(0, len(ids), config["batch_size"]):
            if stop_check is not None and stop_check():
                raise TrainingStopped("Deadline/budget reservation reached during an epoch")
            rows = ids[start:start + config["batch_size"]]
            flips = (torch.rand(len(rows), generator=augmentation).numpy() < 0.5) if training else np.zeros(len(rows), bool)
            targets = torch.as_tensor(data["labels"][rows], dtype=torch.long, device=device)
            if recipe == "frozen":
                features = np.array(cache["original"][rows], copy=True)
                if training and np.any(flips):
                    features[flips] = cache["flipped"][rows[flips]]
                inputs = torch.from_numpy(features).to(device)
                logits = model.classifier[2:](inputs)
            else:
                logits = model(prepare_batch(data["images"], rows, device, flips))
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise ValueError("Training/evaluation loss or logits are not finite")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0), error_if_nonfinite=True)
                optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            truth.extend(targets.cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    return classification_metrics(truth, predictions, total_loss, len(data["classnames"]))


def _rng(sampler, augmentation):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"kind": numpy[0], "values": numpy[1].tolist(),
            "position": numpy[2], "has_gauss": numpy[3], "cached_gauss": numpy[4]},
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": sampler.get_state(), "augmentation": augmentation.get_state()}


def _restore_rng(value, sampler, augmentation):
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state((numpy["kind"], np.array(numpy["values"], dtype=np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gauss"]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])
    sampler.set_state(value["sampler"])
    augmentation.set_state(value["augmentation"])


def _environment(config):
    def version(name):
        try:
            return str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            return "not_installed"
    cuda = str(config["device"]).startswith("cuda")
    return {"python": platform.python_version(), "torch": str(torch.__version__), "torchvision": version("torchvision"),
            "numpy": str(np.__version__), "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if cuda and torch.cuda.is_available() else None,
            "device": str(config["device"]), "precision": "fp32", "bn_running_statistics": "fixed_imagenet"}


class _Clock:
    def __init__(self, out, config, external):
        self.path, self.config, self.external = Path(out) / "TRAINING_CLOCK.json", config, external
        self.previous = _json(self.path)["active_seconds"] if self.path.exists() else 0.0
        self.started, self.last = time.monotonic(), 0.0

    @property
    def elapsed(self):
        return self.previous + time.monotonic() - self.started

    def tick(self, force=False):
        if force or time.monotonic() - self.last > 10:
            atomic_json(self.path, {"active_seconds": self.elapsed, "max_active_seconds": self.config["max_active_seconds"],
                        "scope": "Cumulative training-process time including model/cache setup; abrupt tail after last heartbeat may be unrecorded."})
            self.last = time.monotonic()

    def stopped(self):
        self.tick()
        return (self.elapsed >= self.config["max_active_seconds"]
                or (utc_parse(self.config["stop_by_utc"]) - utc_now()).total_seconds()
                    <= self.config["finalization_reserve_seconds"]
                or (self.external is not None and self.external()))


def _feature_cache(data, out, model, config, binding, clock):
    folder = Path(out) / "feature_cache"
    folder.mkdir(exist_ok=True)
    encoder = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("classifier.3.")}
    identity = {"binding": binding, "encoder_sha256": _tensor_digest(encoder), "dtype": "float32",
                "scope": "Original fit/selection embeddings; horizontally flipped fit embeddings only"}
    _protect_json(folder / "identity.json", identity)
    complete_path = folder / "COMPLETE.json"
    if complete_path.exists():
        complete = _json(complete_path)
        if any(complete.get(key) != value for key, value in identity.items()):
            raise ValueError("Feature-cache completion binding changed")
        for name in ("original", "flipped"):
            if complete.get("files", {}).get(name + ".npy", {}).get("sha256") != digest(folder / (name + ".npy")):
                raise ValueError("A completed feature-cache file changed")
        return {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                for name in ("original", "flipped")}
    progress_path = folder / "progress.json"
    progress = _json(progress_path) if progress_path.exists() else {"original": {"completed": 0, "chunks": []},
                                                                   "flipped": {"completed": 0, "chunks": []}}
    model.eval()
    began = time.monotonic()
    previous_seconds = progress.get("active_seconds", 0.0)
    for name, role_ids in (("original", np.flatnonzero(data["roles"] <= 1)),
                           ("flipped", np.flatnonzero(data["roles"] == 0))):
        path = folder / (name + ".npy")
        saved = progress[name]
        shape = (len(data["labels"]), model.classifier[-1].in_features)
        if not path.exists():
            if saved["completed"]:
                raise ValueError("Feature-cache progress refers to a missing array")
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            array[:] = np.nan
            array.flush()
        else:
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError("Feature-cache shape or precision changed")
        for chunk in saved["chunks"]:
            rows = role_ids[chunk["start"]:chunk["end"]]
            if hashlib.sha256(array[rows].tobytes()).hexdigest() != chunk["sha256"]:
                raise ValueError("A committed feature-cache chunk changed")
        for start in range(saved["completed"], len(role_ids), config["feature_batch_size"]):
            if clock.stopped():
                raise TrainingStopped("Feature cache is partial; safely resume the same run")
            rows = role_ids[start:start + config["feature_batch_size"]]
            values = extract_features(model, data["images"], config["device"], config["feature_batch_size"],
                                      name == "flipped", rows, clock.stopped)
            array[rows] = values
            array.flush()
            saved["completed"] = start + len(rows)
            saved["chunks"].append({"start": start, "end": start + len(rows),
                                    "sha256": hashlib.sha256(values.tobytes()).hexdigest()})
            progress["active_seconds"] = previous_seconds + time.monotonic() - began
            atomic_json(progress_path, progress)
            print(f"CACHE {name}: {saved['completed']}/{len(role_ids)}", flush=True)
        if not np.isfinite(array[role_ids]).all():
            raise ValueError("Completed feature cache contains non-finite rows")
        del array
    receipt = {**identity, "files": {name + ".npy": {"sha256": digest(folder / (name + ".npy"))}
                                            for name in ("original", "flipped")},
               "wall_seconds": previous_seconds + time.monotonic() - began,
               "timing_scope": "Cumulative cache invocations to committed batches; abrupt uncommitted tail may be unrecorded.",
               "cost_note": "Feature extraction is a shared one-off cost, reported separately from frozen-head fitting."}
    atomic_json(complete_path, receipt)
    return {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in ("original", "flipped")}


def _completed(folder, task, binding):
    marker = folder / "COMPLETE.json"
    if not marker.exists():
        return False
    complete = _json(marker)
    if complete.get("binding") != binding or complete.get("task_id") != task["task_id"]:
        raise ValueError("Completed task binding changed")
    for name in ("task.json", "best.pt", "history.json", "epoch_state.pt"):
        if complete.get("files", {}).get(name, {}).get("sha256") != digest(folder / name):
            raise ValueError(f"Completed task artifact changed: {task['task_id']}/{name}")
    return True


def _train_task(data, out, task, config, binding, cache, clock, factory):
    folder = Path(out) / "tasks" / task["task_id"]
    folder.mkdir(parents=True, exist_ok=True)
    _protect_json(folder / "task.json", task)
    if _completed(folder, task, binding):
        return
    if clock.stopped():
        raise TrainingStopped("No new task started after the budget/deadline reservation")
    model = factory(num_classes=len(data["classnames"]), seed=task["seed"], pretrained=True).to(config["device"])
    if isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2].p = config["head_dropout"]
    initial_head = _tensor_digest(model.classifier[-1].state_dict())
    encoder = {name: tensor for name, tensor in model.state_dict().items() if not name.startswith("classifier.3.")}
    if _tensor_digest(encoder) != _json(Path(out) / "feature_cache/identity.json")["encoder_sha256"]:
        raise ValueError("Task initialization differs from the shared pretrained feature encoder")
    other_recipe = "finetune" if task["recipe"] == "frozen" else "frozen"
    other_marker = Path(out) / "tasks" / f"{other_recipe}_seed{task['seed']}" / "COMPLETE.json"
    if other_marker.exists() and _json(other_marker).get("initial_head_sha256") != initial_head:
        raise ValueError("Paired recipes do not have identical initial head weights")
    optimizer, scheduler = configure_optimizer(model, task["recipe"], config)
    sampler = torch.Generator().manual_seed(task["seed"] + 10000)
    augmentation = torch.Generator().manual_seed(task["seed"] + 20000)
    random.seed(task["seed"])
    np.random.seed(task["seed"])
    torch.manual_seed(task["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(task["seed"])
    history, best, next_epoch, fit_seconds = [], None, 1, 0.0
    state_path = folder / "epoch_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("binding") != binding or state.get("task") != task or state.get("initial_head_sha256") != initial_head:
            raise ValueError("Resume checkpoint source/config/data/environment/task binding changed")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for parameter, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    values[key] = value.to(parameter.device)
        scheduler.load_state_dict(state["scheduler"])
        history, best, fit_seconds = state["history"], state["best"], state["fit_seconds"]
        next_epoch = state["epoch"] + 1
        _restore_rng(state["rng"], sampler, augmentation)
        # Epoch state is authoritative if a crash interrupted the auxiliary file writes.
        atomic_json(folder / "history.json", history)
        if best is not None:
            _atomic_torch(folder / "best.pt", state["best_checkpoint"])
    fit_ids, selection_ids = np.flatnonzero(data["roles"] == 0), np.flatnonzero(data["roles"] == 1)
    for epoch in range(next_epoch, config["epochs"] + 1):
        if clock.stopped():
            raise TrainingStopped("Epoch resume checkpoint saved; budget/deadline reservation reached")
        began = time.monotonic()
        rates = [group["lr"] for group in optimizer.param_groups]
        train = run_epoch(model, data, fit_ids, config, task["recipe"], optimizer=optimizer, cache=cache,
                          sampler=sampler, augmentation=augmentation, stop_check=clock.stopped)
        selection = run_epoch(model, data, selection_ids, config, task["recipe"], cache=cache, stop_check=clock.stopped)
        elapsed = time.monotonic() - began
        fit_seconds += elapsed
        history.append({"epoch": epoch, "train": train, "selection": selection, "lr": rates, "wall_seconds": elapsed})
        improved = (best is None or selection["macro_f1"] > best["selection"]["macro_f1"]
                    or (selection["macro_f1"] == best["selection"]["macro_f1"]
                        and selection["loss"] < best["selection"]["loss"]))
        if improved:
            best = {"epoch": epoch, "selection": selection}
            best_checkpoint = {"model_state_dict": _state_cpu(model), "epoch": epoch, "selection": selection,
                               "binding": binding, "task": task, "initial_head_sha256": initial_head}
        else:
            best_checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        scheduler.step()
        state = {"model_state_dict": _state_cpu(model), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": _rng(sampler, augmentation), "epoch": epoch,
                 "history": history, "best": best, "best_checkpoint": best_checkpoint,
                 "fit_seconds": fit_seconds, "binding": binding, "task": task, "initial_head_sha256": initial_head,
                 "training_active_seconds": clock.elapsed}
        # Commit the authoritative resume transaction before updating derived artifacts.
        _atomic_torch(state_path, state)
        _atomic_torch(folder / "best.pt", best_checkpoint)
        atomic_json(folder / "history.json", history)
        clock.tick(True)
        print(f"{task['task_id']} epoch {epoch}/{config['epochs']} trainCE={train['loss']:.4f} "
              f"selectionF1={selection['macro_f1']:.4f} best={best['selection']['macro_f1']:.4f} "
              f"seconds={elapsed:.1f}", flush=True)
    complete = {"task_id": task["task_id"], "binding": binding, "epochs": config["epochs"],
                "best_epoch": best["epoch"], "selection": best["selection"], "fit_seconds": fit_seconds,
                "scope": "Completed epochs train and selection passes; excludes model/cache setup and checkpoint writes.",
                "initial_head_sha256": initial_head,
                "files": {name: {"sha256": digest(folder / name)}
                          for name in ("task.json", "best.pt", "history.json", "epoch_state.pt")}}
    atomic_json(folder / "COMPLETE.json", complete)
    del model, optimizer, scheduler
    if str(config["device"]).startswith("cuda"):
        torch.cuda.empty_cache()


def train_study(prepared_dir, out_dir, config=None, *, stop_check=None, model_factory=None):
    """Run the fixed six tasks, or return INCOMPLETE with epoch/cache resume evidence."""
    if config is None:
        config = _json(Path(__file__).resolve().parents[1] / "configs/garmentlab_deadline.json")
    config = json.loads(json.dumps(config))
    if config.get("precision") != "fp32" or config.get("recipes") != ["frozen", "finetune"]:
        raise ValueError("This fixed study requires FP32 and frozen/finetune recipes")
    for key in ("epochs", "batch_size", "feature_batch_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer config: {key}")
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Training seeds must be unique")
    if any(type(seed) is not int or not 0 <= seed < 2 ** 32 - 20000 for seed in config["seeds"]):
        raise ValueError("Training seeds must be bounded nonnegative integers")
    for key in ("head_lr", "backbone_lr", "gradient_clip", "max_active_seconds"):
        value = config.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive finite config: {key}")
    if (type(config.get("weight_decay")) not in (float, int) or not math.isfinite(config["weight_decay"])
            or config["weight_decay"] < 0):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (type(config.get("head_dropout")) not in (float, int) or not math.isfinite(config["head_dropout"])
            or not 0 <= config["head_dropout"] < 1):
        raise ValueError("head_dropout must be finite and in [0,1)")
    cuda = str(config["device"]).startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full training was not started")
    if cuda and model_factory is None and "A100" not in torch.cuda.get_device_name(torch.device(config["device"])):
        raise RuntimeError("This fixed full training run requires an NVIDIA A100")
    if not cuda and model_factory is None:
        raise ValueError("CPU training is permitted only through the injected tiny software-smoke model")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    factory = model_factory or build_model
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = _Clock(out, config, stop_check)
    data = load_prepared(prepared_dir)
    tasks = [{"task_id": f"{recipe}_seed{seed}", "recipe": recipe, "seed": seed,
              "epochs": config["epochs"], "labels": data["classnames"], "architecture": "mobilenet_v3_small"}
             for seed in config["seeds"] for recipe in config["recipes"]]
    manifest = {"schema_version": 1, "config": config, "labels": data["classnames"], "data": data["bindings"],
                "source_sha256": digest(Path(__file__)), "environment": _environment(config), "tasks": tasks,
                "protocol": "Paired head initialization, paired shuffle/flip streams, fixed BN statistics; selection-only checkpoint choice.",
                "weight_source": "MobileNet_V3_Small_Weights.IMAGENET1K_V1" if model_factory is None else "injected_software_smoke_only"}
    _protect_json(out / "study_manifest.json", manifest)
    binding = digest(out / "study_manifest.json")
    status = {"status": "INCOMPLETE", "completed": 0, "planned": len(tasks), "binding": binding}
    try:
        if all(_completed(out / "tasks" / task["task_id"], task, binding) for task in tasks):
            status.update(status="ALL_TASKS_COMPLETE", completed=len(tasks))
            return status
        if clock.stopped():
            raise TrainingStopped("Budget/deadline reservation reached before setup")
        cache_model = factory(num_classes=len(data["classnames"]), seed=0, pretrained=True).to(config["device"])
        cache = _feature_cache(data, out, cache_model, config, binding, clock)
        del cache_model
        for task in tasks:
            _train_task(data, out, task, config, binding, cache, clock, factory)
            status["completed"] += 1
            atomic_json(out / "study_state.json", status)
        status["status"] = "ALL_TASKS_COMPLETE"
    except TrainingStopped as exc:
        status["stop_reason"] = str(exc)
        status["completed"] = sum(_completed(out / "tasks" / task["task_id"], task, binding) for task in tasks)
    finally:
        clock.tick(True)
        status["training_active_seconds"] = clock.elapsed
        atomic_json(out / "study_state.json", status)
    return status
