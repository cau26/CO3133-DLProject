"""One training/evaluation implementation shared by Linear and MLP."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from .common import environment, load_json, save_csv, save_json, set_seed, source_metadata
from .data import make_loaders
from .linear import LinearClassifier
from .mlp import MLP


def build_model(name, config):
    return LinearClassifier() if name == "linear" else MLP(config["hidden_size"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
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
            if training:
                loss.backward()
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n  # sample-weighted, handles a short final batch
            total_samples += n
            truths.append(labels.detach().cpu().numpy())
            predictions.append(logits.argmax(1).detach().cpu().numpy())
    true = np.concatenate(truths)
    pred = np.concatenate(predictions)
    return {"loss": total_loss / total_samples,
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=list(range(10)),
                                      average="macro", zero_division=0))}, true, pred


def measure_inference(model, batch, device):
    # Model-only latency: input already on the selected device; excludes loading.
    images = batch.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(images)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(50):
            model(images)
        synchronize(device)
    seconds = (time.perf_counter() - start) / 50
    return {"batch_size": len(images), "warmup_batches": 10, "measured_batches": 50,
            "batch_ms": seconds * 1000, "per_image_ms_amortized": seconds * 1000 / len(images),
            "scope": "model forward only; input already on device; excludes DataLoader and transfers"}


def plot_results(name, history, true, pred, data, out):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            ax.plot(epochs, [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric, title=f"{name.upper()}: {metric}")
        ax.legend()
        ax.grid(alpha=0.2)
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
    ax.set(xlabel="Predicted", ylabel="True", title=f"{name.upper()}: validation confusion matrix")
    fig.colorbar(img, ax=ax, shrink=0.75)
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    selected = np.r_[np.flatnonzero(true == pred)[:5], np.flatnonzero(true != pred)[:5]]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, selected):
        image, _ = data["validation"][int(idx)]
        ax.imshow(image[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {names[true[idx]]}\nPred: {names[pred[idx]]}", fontsize=9,
                     color="#166534" if true[idx] == pred[idx] else "#b91c1c")
    fig.suptitle(f"{name.upper()}: first correct and incorrect validation examples")
    fig.savefig(out / "examples.png", dpi=150)
    plt.close(fig)
    errors = [{"true_label": i, "true_class": names[i], "predicted_label": j,
               "predicted_class": names[j], "count": int(cm[i, j]),
               "rate_within_true_class": float(cm[i, j] / cm[i].sum())}
              for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    return sorted(errors, key=lambda row: row["count"], reverse=True)[:5]


def train_model(name, data, config, output_dir, device):
    set_seed(config["seed"])
    train_loader, val_loader = make_loaders(data, config)
    model = build_model(name, config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch = [], float("inf"), 0
    provenance = source_metadata()
    hardware = environment(device)
    synchronize(device)
    overall_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        synchronize(device)
        start = time.perf_counter()
        train_stats, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        synchronize(device)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_stats, _, _ = run_epoch(model, val_loader, loss_fn, device)
        synchronize(device)
        val_seconds = time.perf_counter() - start
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
               **{f"val_{k}": v for k, v in val_stats.items()},
               "train_seconds": train_seconds, "val_seconds": val_seconds}
        history.append(row)
        save_csv(out / "history.csv", history)
        if val_stats["loss"] < best_loss:
            best_loss, best_epoch = val_stats["loss"], epoch
            torch.save({"model_state": model.state_dict(), "model_name": name,
                        "config": config, "epoch": epoch, "val_loss": best_loss,
                        "split_sha256": data["summary"]["split_sha256"],
                        "source_sha256": provenance["source_sha256"]}, out / "best.pt")
        print(f"{name:6s} epoch {epoch:02d}/{config['epochs']}: "
              f"train_loss={train_stats['loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
              f"val_f1={val_stats['macro_f1']:.4f}", flush=True)
    fit_wall_seconds = time.perf_counter() - overall_start
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    stats, true, pred = run_epoch(model, val_loader, loss_fn, device)
    assert abs(stats["loss"] - best_loss) < 1e-5, "Checkpoint reload changed validation loss"
    save_csv(out / "validation_predictions.csv", [
        {"official_train_index": int(idx), "true": int(t), "predicted": int(p)}
        for idx, t, p in zip(data["val_ids"], true, pred)])
    top_errors = plot_results(name, history, true, pred, data, out)
    latency = measure_inference(model, next(iter(val_loader))[0], device)
    metrics = {"model": name, "evaluation_split": "validation", "n_eval": len(true),
               **stats, "parameters": sum(p.numel() for p in model.parameters()),
               "best_epoch": best_epoch, "checkpoint_rule": "lowest validation loss; first if tied",
               "checkpoint": "best.pt", "epochs_completed": len(history),
               "train_seconds": sum(r["train_seconds"] for r in history),
               "validation_seconds": sum(r["val_seconds"] for r in history),
               "fit_wall_seconds": fit_wall_seconds,
               "train_time_scope": "training loop including data loading, metric collection and updates",
               "inference": latency, "top_errors": top_errors,
               "config": config, "data": data["summary"], "environment": hardware,
               "source": provenance, "test_evaluated": False,
               "human_review_status": "Pending: group must inspect code, rerun and verify before submission"}
    save_json(out / "metrics.json", metrics)
    return metrics


def evaluate_checkpoint(name, data, output_dir, device):
    """Reload a saved checkpoint and verify its validation metrics without training."""
    out = Path(output_dir) / name
    saved = load_json(out / "metrics.json")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    if checkpoint["split_sha256"] != data["summary"]["split_sha256"]:
        raise ValueError("Checkpoint and requested split do not match")
    model = build_model(name, checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    val_loader = make_loaders(data, checkpoint["config"])[1]
    stats, _, _ = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
    checks = {key: bool(np.isclose(stats[key], saved[key], atol=1e-5, rtol=1e-5))
              for key in ["loss", "accuracy", "macro_f1"]}
    verification = {"model": name, "recomputed": stats, "matches_saved": checks,
                    "device": str(device), "split_sha256": data["summary"]["split_sha256"]}
    save_json(out / "checkpoint_verification.json", verification)
    print(f"{name}: recomputed {stats}; matches saved metrics: {all(checks.values())}", flush=True)
    if not all(checks.values()):
        raise ValueError("Recomputed metrics differ. Inspect config, data, software and hardware.")
