"""Shared, resumable training and checkpoint verification for five model families."""
import copy
import csv
import io
import random
import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from .common import (atomic_torch_save, environment, json_hash, load_json, protect_json,
                     save_csv, save_json, set_seed, sha256_file, source_metadata)
from .data import make_loaders, resolve_split_seed
from .linear import LinearClassifier
from .mlp import MLP
from .cnn import CustomCNN
from .gru import RowGRU
from .transformer import PatchTransformer

MODEL_NAMES = ("linear", "mlp", "cnn", "gru", "transformer")


def evaluation_budgets(config):
    """Optional prefix budgets are explicit, sorted and bounded by the actual run."""
    if "evaluation_budgets" not in config:
        return []
    budgets = config["evaluation_budgets"]
    if type(budgets) is not list or not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError("evaluation_budgets must be a nonempty list of positive integers")
    if budgets != sorted(set(budgets)):
        raise ValueError("evaluation_budgets must be sorted and unique")
    if type(config.get("epochs")) is not int or any(value > config["epochs"] for value in budgets):
        raise ValueError("Every evaluation budget must be <= epochs")
    return budgets


def model_config(name, config):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    resolved = copy.deepcopy(config)
    overrides = resolved.pop("per_model", {}).get(name, {})
    forbidden = {"seed", "split_seed", "evaluation_budgets", "validation_size", "device", "num_workers", "cpu_threads",
                 "group", "members", "per_model"}
    if forbidden.intersection(overrides):
        raise ValueError(f"Per-model overrides cannot change shared experiment fields: {forbidden.intersection(overrides)}")
    resolved.update(overrides)
    if "seed" in resolved:
        resolve_split_seed(resolved)
    if "sequence_axis" in resolved and (not isinstance(resolved["sequence_axis"], str)
                                       or resolved["sequence_axis"] not in ("rows", "columns")):
        raise ValueError("sequence_axis must be 'rows' or 'columns'")
    if "use_position_embedding" in resolved and type(resolved["use_position_embedding"]) is not bool:
        raise ValueError("use_position_embedding must be a boolean")
    dimensions = ({"gru_hidden_size": 128, "gru_num_layers": 1} if name == "gru" else
                  {"patch_size": 4, "d_model": 128, "nhead": 4, "transformer_layers": 2,
                   "dim_feedforward": 256} if name == "transformer" else {})
    for key, default in dimensions.items():
        value = resolved.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if name == "transformer" and (28 % resolved.get("patch_size", 4)
                                   or resolved.get("d_model", 128) % resolved.get("nhead", 4)):
        raise ValueError("patch_size must divide 28 and nhead must divide d_model")
    evaluation_budgets(resolved)
    return resolved


def build_model(name, config):
    config = model_config(name, config)
    if name == "linear":
        return LinearClassifier()
    if name == "mlp":
        return MLP(config.get("hidden_size", 256))
    if name == "cnn":
        return CustomCNN(config.get("cnn_channels", [32, 64, 128]), config.get("dropout", 0.2))
    if name == "gru":
        return RowGRU(config.get("gru_hidden_size", 128), config.get("gru_num_layers", 1),
                      config.get("dropout", 0.0), config.get("sequence_axis", "rows"))
    return PatchTransformer(config.get("patch_size", 4), config.get("d_model", 128),
                            config.get("nhead", 4), config.get("transformer_layers", 2),
                            config.get("dim_feedforward", 256), config.get("dropout", 0.1),
                            config.get("use_position_embedding", True))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, loss_fn, device, optimizer=None, gradient_clip=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples = 0.0, 0
    truths, predictions = [], []
    # Validation never computes gradients or changes parameters.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss: stopped without marking the run complete")
            if training:
                loss.backward()
                if gradient_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n  # sample-weighted, handles a short final batch
            total_samples += n
            truths.append(labels.detach().cpu().numpy())
            predictions.append(logits.argmax(1).detach().cpu().numpy())
    if not total_samples:
        raise ValueError("Cannot evaluate an empty DataLoader")
    true = np.concatenate(truths)
    pred = np.concatenate(predictions)
    return {"loss": total_loss / total_samples,
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=list(range(10)),
                                      average="macro", zero_division=0))}, true, pred


def measure_inference(model, batch, device, warmup=10, repeats=50):
    # Model-only latency: input already on the selected device; excludes loading.
    images = batch.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(images)
        synchronize(device)
    seconds = (time.perf_counter() - start) / repeats
    return {"batch_size": len(images), "warmup_batches": warmup, "measured_batches": repeats,
            "batch_ms": seconds * 1000, "per_image_ms_amortized": seconds * 1000 / len(images),
            "scope": "model forward only; input already on device; excludes DataLoader and transfers"}


def plot_results(name, history, true, pred, data, out, split="validation"):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        for curve_split in ["train", "val"]:
            ax.plot(epochs, [r[f"{curve_split}_{metric}"] for r in history], label=curve_split)
        ax.set(xlabel="Epoch", ylabel=metric, title=f"{name.upper()}: {metric}")
        ax.legend()
        ax.grid(alpha=0.2)
    if history:
        fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)
    names = data["summary"]["classes"]
    cm = confusion_matrix(true, pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8.5, 7.5), layout="constrained")
    img = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10), names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(10), names, fontsize=9)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{name.upper()}: {split} confusion matrix")
    fig.colorbar(img, ax=ax, shrink=0.75)
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    save_csv(out / "confusion_matrix.csv", [
        {"true_label": i, **{f"predicted_{j}": int(cm[i, j]) for j in range(10)}}
        for i in range(10)])
    selected = np.r_[np.flatnonzero(true == pred)[:5], np.flatnonzero(true != pred)[:5]]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, selected):
        image, _ = data[split][int(idx)]
        ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {names[true[idx]]}\nPred: {names[pred[idx]]}", fontsize=9,
                     color="#166534" if true[idx] == pred[idx] else "#b91c1c")
    fig.suptitle(f"{name.upper()}: first correct and incorrect {split} examples")
    fig.savefig(out / "examples.png", dpi=150)
    plt.close(fig)
    errors = [{"true_label": i, "true_class": names[i], "predicted_label": j,
               "predicted_class": names[j], "count": int(cm[i, j]),
               "rate_within_true_class": float(cm[i, j] / cm[i].sum())}
              for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    return sorted(errors, key=lambda row: row["count"], reverse=True)[:5]


def make_optimizer(model, config):
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if config.get("optimizer", "adam") not in optimizers:
        raise ValueError("Supported optimizers are adam and adamw")
    return optimizers[config.get("optimizer", "adam")](
        model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "shuffle_generator": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator.set_state(state["shuffle_generator"].cpu())


def assert_checkpoint(checkpoint, name, config, data, provenance, run_id):
    expected = {"model_name": name, "config_sha256": json_hash(config),
                "split_sha256": data["summary"]["split_sha256"],
                "dataset_sha256": data["summary"]["dataset_sha256"],
                "source_sha256": provenance["source_sha256"], "run_id": run_id}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} differs; refusing to mix experiments")


def _prefix_history(history, budget):
    prefix = [{"epoch": row["epoch"], "val_loss": row["val_loss"]}
              for row in history if row["epoch"] <= budget]
    if [row["epoch"] for row in prefix] != list(range(1, budget + 1)):
        raise ValueError(f"Budget {budget} requires an uninterrupted complete history prefix")
    return prefix


def _selection_document(binding, history, budget, checkpoint_path):
    prefix = _prefix_history(history, budget)
    selected = min(prefix, key=lambda row: row["val_loss"])
    return {"schema_version": 1, "prefix_budget": budget, "selected_epoch": selected["epoch"],
            "validation_loss": selected["val_loss"], "selection_history": prefix,
            "selection_history_sha256": json_hash(prefix),
            "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha256_file(checkpoint_path),
            **{key: binding[key] for key in ("model_name", "config_sha256", "split_sha256",
                                           "dataset_sha256", "source_sha256", "run_id")},
            "selection_rule": "minimum validation loss through prefix; earliest epoch wins ties"}


def save_budget_snapshot(out, binding, history, budget, best_state):
    """Write before last.pt's epoch boundary; identical pre-boundary retries preserve bytes."""
    out = Path(out)
    path = out / f"best_through_{budget}.pt"
    sidecar = out / f"best_through_{budget}_selection.json"
    prefix = _prefix_history(history, budget)
    selected = min(prefix, key=lambda row: row["val_loss"])
    expected = {**binding, "model_state": best_state, "epoch": selected["epoch"],
                "val_loss": selected["val_loss"], "prefix_budget": budget,
                "selection_history_sha256": json_hash(prefix)}
    if path.exists():
        actual = torch.load(path, map_location="cpu", weights_only=True)
        non_state = {key: value for key, value in actual.items() if key != "model_state"}
        expected_non_state = {key: value for key, value in expected.items() if key != "model_state"}
        if non_state != expected_non_state or set(actual["model_state"]) != set(best_state):
            raise ValueError(f"Existing immutable budget snapshot differs: {path}")
        if not all(torch.equal(actual["model_state"][key], value.detach().cpu()) for key, value in best_state.items()):
            raise ValueError(f"Existing immutable budget weights differ: {path}")
    else:
        if sidecar.exists():
            raise FileNotFoundError(f"Budget sidecar exists without its checkpoint: {path}")
        atomic_torch_save(expected, path)
    protect_json(sidecar, _selection_document(binding, history, budget, path))


def validate_budget_snapshot(out, binding, history, budget):
    out = Path(out)
    path, sidecar = out / f"best_through_{budget}.pt", out / f"best_through_{budget}_selection.json"
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Completed budget {budget} snapshot/selection missing; cannot reconstruct it from later best.pt")
    selection = load_json(sidecar)
    if selection != _selection_document(binding, history, budget, path):
        raise ValueError(f"Budget {budget} snapshot hash or selection history changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {**binding, "epoch": selection["selected_epoch"], "val_loss": selection["validation_loss"],
                "prefix_budget": budget, "selection_history_sha256": selection["selection_history_sha256"]}
    if {key: value for key, value in checkpoint.items() if key != "model_state"} != required:
        raise ValueError(f"Budget {budget} checkpoint binding differs from the saved prefix")
    return selection


def evaluate_budget_snapshots(name, model, val_loader, data, config, out, device, binding, history, provenance, hardware):
    """Evaluate preserved prefix choices on validation only, after full training completes."""
    out = Path(out)
    for budget in evaluation_budgets(config):
        selection = validate_budget_snapshot(out, binding, history, budget)
        metrics_path = out / f"budget{budget}_metrics.json"
        predictions_path = out / f"budget{budget}_validation_predictions.csv"
        if metrics_path.exists():
            saved = load_json(metrics_path)
            if (saved.get("checkpoint_sha256") != selection["checkpoint_sha256"]
                    or saved.get("run_id") != binding["run_id"] or saved.get("config") != config
                    or saved.get("evaluation_split") != "validation" or saved.get("prefix_budget") != budget
                    or saved.get("n_eval") != len(data["validation"])
                    or saved.get("split_sha256") != binding["split_sha256"]
                    or saved.get("source_sha256") != binding["source_sha256"]
                    or not np.isclose(saved.get("loss", float("inf")), selection["validation_loss"], atol=1e-5, rtol=1e-5)
                    or saved.get("selection_history_sha256") != selection["selection_history_sha256"]
                    or not predictions_path.is_file()
                    or saved.get("predictions_sha256") != sha256_file(predictions_path)):
                raise ValueError(f"Completed budget {budget} evaluation artifacts changed")
            continue
        checkpoint = torch.load(out / selection["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        synchronize(device)
        start = time.perf_counter()
        stats, true, pred = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
        synchronize(device)
        evaluation_seconds = time.perf_counter() - start
        if not np.isclose(stats["loss"], selection["validation_loss"], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Budget {budget} checkpoint reload changed its selected validation loss")
        rows = [{"official_train_index": int(idx), "true": int(t), "predicted": int(p)}
                for idx, t, p in zip(data["val_ids"], true, pred)]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        content = buffer.getvalue().encode("utf-8")
        if predictions_path.exists():
            if predictions_path.read_bytes() != content:
                raise ValueError(f"Existing incomplete budget predictions differ: {predictions_path}")
        else:
            with predictions_path.open("xb") as stream:
                stream.write(content)
        prefix = [row for row in history if row["epoch"] <= budget]
        metrics = {"model": name, "evaluation_split": "validation", "prefix_budget": budget,
                   "best_epoch": selection["selected_epoch"], "n_eval": len(true), **stats,
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "checkpoint": selection["checkpoint"], "checkpoint_sha256": selection["checkpoint_sha256"],
                   "selection_history_sha256": selection["selection_history_sha256"],
                   "checkpoint_rule": selection["selection_rule"],
                   "validation_predictions": predictions_path.name, "predictions_sha256": sha256_file(predictions_path),
                   "train_seconds": sum(row["train_seconds"] for row in prefix),
                   "epoch_validation_seconds": sum(row["val_seconds"] for row in prefix),
                   "checkpoint_validation_seconds": evaluation_seconds,
                   "timing_scope": "prefix training sums from actual epochs; checkpoint validation measured after full run; no inference latency measurement",
                   "config": config, "config_sha256": binding["config_sha256"],
                   "split_sha256": binding["split_sha256"], "dataset_sha256": binding["dataset_sha256"],
                   "source_sha256": binding["source_sha256"],
                   "data": data["summary"], "run_id": binding["run_id"],
                   "source": provenance, "environment": hardware, "test_evaluated": False}
        protect_json(metrics_path, metrics)


def train_model(name, data, config, output_dir, device, resume=False):
    root, cfg = Path(output_dir), model_config(name, config)
    if (root / "freeze_manifest.json").exists():
        raise FileExistsError("Run is frozen; training requires a new output directory")
    out = root / name
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Completed model output is immutable: {out}")
    if out.exists() and not resume:
        raise FileExistsError(f"Existing model directory (including incomplete runs): {out}. Use --resume or a new --out")
    if resume and not (out / "last.pt").is_file():
        raise FileNotFoundError(f"No last.pt to resume: {out}")
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    set_seed(cfg["seed"])
    train_loader, val_loader = make_loaders(data, cfg)
    model = build_model(name, cfg).to(device)
    loss_fn, optimizer = nn.CrossEntropyLoss(), make_optimizer(model, cfg)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, cfg["scheduler_step_size"],
                 gamma=cfg.get("scheduler_gamma", 0.5)) if cfg.get("scheduler_step_size") else None)
    provenance, hardware = source_metadata(), environment(device)
    print(f"{name}: optimizer={cfg.get('optimizer', 'adam')}, lr={cfg['learning_rate']:g}, "
          f"batch_size={cfg['batch_size']}, epochs={cfg['epochs']}, device={device}, resume={resume}", flush=True)
    run_id = load_json(root / "run_manifest.json")["run_id"]
    binding = {"model_name": name, "config": cfg, "config_sha256": json_hash(cfg),
               "split_sha256": data["summary"]["split_sha256"],
               "dataset_sha256": data["summary"]["dataset_sha256"],
               "source_sha256": provenance["source_sha256"], "run_id": run_id}
    protect_json(out / "model_config.json", cfg)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    prior_wall_seconds, start_epoch = 0.0, 1
    if resume:
        last = torch.load(out / "last.pt", map_location=device, weights_only=True)
        assert_checkpoint(last, name, cfg, data, provenance, run_id)
        if (last["environment"]["device"] != str(device) or last["environment"]["versions"] != hardware["versions"]
                or last["environment"]["gpu"] != hardware["gpu"]):
            raise ValueError("Resume requires the same device/GPU family and package versions")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        if scheduler:
            scheduler.load_state_dict(last["scheduler_state"])
        history, best_loss, best_epoch = last["history"], last["best_loss"], last["best_epoch"]
        best_state, prior_wall_seconds = last["best_model_state"], last["fit_wall_seconds"]
        start_epoch = last["epoch"] + 1
        for budget in evaluation_budgets(cfg):
            if budget <= last["epoch"]:
                validate_budget_snapshot(out, binding, history, budget)
        restore_rng(last["rng"], train_loader.generator)
        # last.pt is the epoch transaction boundary; undo any newer partial best.pt.
        atomic_torch_save({**binding, "model_state": best_state, "epoch": best_epoch,
                           "val_loss": best_loss}, out / "best.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    overall_start = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_lr = optimizer.param_groups[0]["lr"]
        synchronize(device)
        start = time.perf_counter()
        train_stats, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, cfg.get("gradient_clip"))
        synchronize(device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_stats, _, _ = run_epoch(model, val_loader, loss_fn, device)
        synchronize(device)
        val_seconds = time.perf_counter() - start
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
               **{f"val_{k}": v for k, v in val_stats.items()},
               "learning_rate": epoch_lr, "train_seconds": train_seconds, "val_seconds": val_seconds,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        history.append(row)
        if val_stats["loss"] < best_loss:
            best_loss, best_epoch = val_stats["loss"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            atomic_torch_save({**binding, "model_state": best_state, "epoch": epoch,
                               "val_loss": best_loss}, out / "best.pt")
        if scheduler:
            scheduler.step()
        if epoch in evaluation_budgets(cfg):
            save_budget_snapshot(out, binding, history, epoch, best_state)
        fit_wall_seconds = prior_wall_seconds + time.perf_counter() - overall_start
        atomic_torch_save({**binding, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                           "scheduler_state": scheduler.state_dict() if scheduler else None,
                           "epoch": epoch, "history": history, "best_loss": best_loss,
                           "best_epoch": best_epoch, "best_model_state": best_state,
                           "rng": capture_rng(train_loader.generator), "fit_wall_seconds": fit_wall_seconds,
                           "environment": hardware}, out / "last.pt")
        save_csv(out / "history.csv", history)
        print(f"{name:11s} epoch {epoch:02d}/{cfg['epochs']}: train_loss={train_stats['loss']:.4f} "
              f"train_acc={train_stats['accuracy']:.4f} "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
              f"val_f1={val_stats['macro_f1']:.4f} lr={epoch_lr:g}", flush=True)
    fit_wall_seconds = prior_wall_seconds + time.perf_counter() - overall_start
    # Restore the authoritative last.pt history even after a crash while writing CSV/plots.
    save_csv(out / "history.csv", history)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    assert_checkpoint(checkpoint, name, cfg, data, provenance, run_id)
    model.load_state_dict(checkpoint["model_state"])
    stats, true, pred = run_epoch(model, val_loader, loss_fn, device)
    if not np.isclose(stats["loss"], best_loss, atol=1e-5, rtol=1e-5):
        raise ValueError("Checkpoint reload changed validation loss")
    evaluate_budget_snapshots(name, model, val_loader, data, cfg, out, device, binding, history, provenance, hardware)
    model.load_state_dict(checkpoint["model_state"])
    save_csv(out / "validation_predictions.csv", [
        {"official_train_index": int(idx), "true": int(t), "predicted": int(p)}
        for idx, t, p in zip(data["val_ids"], true, pred)])
    top_errors = plot_results(name, history, true, pred, data, out)
    latency = measure_inference(model, next(iter(val_loader))[0], device,
                                cfg.get("inference_warmup_batches", 10), cfg.get("inference_measured_batches", 50))
    metrics = {"model": name, "evaluation_split": "validation", "n_eval": len(true),
               **stats, "parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": best_epoch, "checkpoint_rule": "lowest validation loss; first if tied",
               "checkpoint": "best.pt", "checkpoint_sha256": sha256_file(out / "best.pt"),
               "epochs_completed": len(history), "run_id": run_id,
               "train_seconds": sum(r["train_seconds"] for r in history),
               "validation_seconds": sum(r["val_seconds"] for r in history),
               "fit_wall_seconds": fit_wall_seconds,
               "gpu_peak_allocated_bytes": max(row["gpu_peak_allocated_bytes"] for row in history),
               "train_time_scope": "actual epoch loops including data loading/updates/metrics; excludes validation; cold first epoch retained",
               "inference": latency, "top_errors": top_errors, "config": cfg,
               "config_sha256": json_hash(cfg), "data": data["summary"], "environment": hardware,
               "source": provenance, "test_evaluated": False,
               "human_review_status": "Pending: student must review actual results before submission"}
    protect_json(out / "metrics.json", metrics)
    return metrics


def evaluate_checkpoint(name, data, output_dir, device):
    """Reload a saved checkpoint and verify its validation metrics without training."""
    out = Path(output_dir) / name
    saved = load_json(out / "metrics.json")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    assert_checkpoint(checkpoint, name, saved["config"], data, source_metadata(), saved["run_id"])
    if checkpoint["split_sha256"] != data["summary"]["split_sha256"]:
        raise ValueError("Checkpoint and requested split do not match")
    if checkpoint.get("model_name") != name or checkpoint.get("config") != saved["config"]:
        raise ValueError("Checkpoint model/config does not match its own saved metrics")
    if saved.get("checkpoint_sha256") != sha256_file(out / "best.pt"):
        raise ValueError("Checkpoint hash differs from completed metrics")
    model = build_model(name, checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    val_loader = make_loaders(data, checkpoint["config"])[1]
    stats, _, _ = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
    checks = {key: bool(np.isclose(stats[key], saved[key], atol=1e-5, rtol=1e-5))
              for key in ["loss", "accuracy", "macro_f1"]}
    verification = {"model": name, "recomputed": stats, "matches_saved": checks,
                    "device": str(device), "split_sha256": data["summary"]["split_sha256"]}
    verification["source"] = source_metadata()
    verification["environment"] = environment(device)
    verify_out = out / "verifications" / uuid.uuid4().hex
    verify_out.mkdir(parents=True, exist_ok=False)
    save_json(verify_out / "verification.json", verification)
    print(f"{name}: recomputed {stats}; matches saved metrics: {all(checks.values())}", flush=True)
    if not all(checks.values()):
        raise ValueError("Recomputed metrics differ. Inspect config, data, software and hardware.")
    return verification


def smoke_test(config, output_dir, device, images=None, labels=None, models=None):
    """One forward/backward/update and checkpoint reload per model; never an experiment."""
    selected = MODEL_NAMES if models is None else models
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(name not in MODEL_NAMES for name in selected)
            or len(selected) != len(set(selected))):
        raise ValueError("models must be a nonempty list/tuple of unique known model names")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(config["seed"])
    synthetic = images is None and labels is None
    if (images is None) != (labels is None):
        raise ValueError("Provide both images and labels, or neither for synthetic data")
    if synthetic:
        images = torch.rand(4, 1, 28, 28)
        labels = torch.arange(4, dtype=torch.long)
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28) or labels.shape != (len(images),):
        raise ValueError("Smoke inputs must have [N,1,28,28] images and [N] labels")
    rows = []
    for name in selected:
        set_seed(config["seed"])
        cfg = model_config(name, config)
        model = build_model(name, cfg).to(device)
        optimizer = make_optimizer(model, cfg)
        model.train()
        batch, targets = images.to(device), labels.to(device)
        before = next(model.parameters()).detach().clone()
        logits = model(batch)
        if logits.shape != (len(images), 10) or not torch.isfinite(logits).all():
            raise AssertionError(f"{name}: invalid output logits")
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise AssertionError(f"{name}: non-finite loss or gradients")
        optimizer.step()
        if torch.equal(before, next(model.parameters()).detach()):
            raise AssertionError(f"{name}: optimizer did not update the first parameter")
        model.eval()
        with torch.inference_mode():
            expected = model(batch)
        path = out / f"{name}_smoke.pt"
        torch.save({"model_name": name, "config": cfg, "model_state": model.state_dict(),
                    "purpose": "smoke only; not experiment evidence"}, path)
        reloaded = build_model(name, cfg).to(device)
        reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state"])
        reloaded.eval()
        with torch.inference_mode():
            actual = reloaded(batch)
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5):
            raise AssertionError(f"{name}: checkpoint reload mismatch")
        rows.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                     "batch_shape": list(images.shape), "finite_loss": True, "optimizer_updated": True,
                     "checkpoint_reload_matches": True})
        print(f"SMOKE ONLY {name}: backward/update/reload passed", flush=True)
    summary = {"purpose": "smoke only; not training, validation or test experiment evidence",
               "data_kind": "synthetic random tensors" if synthetic else "caller-provided single batch",
               "models": rows, "source": source_metadata(), "environment": environment(device)}
    save_json(out / "smoke_checks.json", summary)
    return summary
