"""Produce real dataset statistics and figures before training."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import save_csv, save_json
from .data import make_loaders


def run_eda(data, config, output_dir):
    out = Path(output_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)
    labels = data["full_train"].targets.numpy()
    names = data["summary"]["classes"]
    rows = []
    for k, name in enumerate(names):
        rows.append({"label": k, "class": name,
                     "official_train": int((labels == k).sum()),
                     "train": int((labels[data["train_ids"]] == k).sum()),
                     "validation": int((labels[data["val_ids"]] == k).sum()),
                     "test": int((data["test"].targets.numpy() == k).sum())})
    save_csv(out / "class_counts.csv", rows)
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for j, (split, color) in enumerate([("train", "#2563eb"),
                                       ("validation", "#f59e0b"), ("test", "#64748b")]):
        ax.bar(x + (j - 1) * 0.26, [r[split] for r in rows], 0.26,
               label=split, color=color)
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("Fashion-MNIST class distribution (stratified split)")
    ax.legend()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(11, 8), layout="constrained")
    for k in range(10):
        # Two deterministic training examples of every class.
        ids = data["train_ids"][labels[data["train_ids"]] == k][:2]
        for j, idx in enumerate(ids):
            ax = axes.flat[k + 10 * j]
            ax.imshow(data["full_train"].data[idx], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{k}: {names[k]}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Two training examples per class")
    fig.savefig(out / "samples.png", dpi=150)
    plt.close(fig)
    batch, targets = next(iter(make_loaders(data, config)[0]))
    assert batch.ndim == 4 and tuple(batch.shape[1:]) == (1, 28, 28)
    assert targets.ndim == 1 and batch.min() >= 0 and batch.max() <= 1
    info = {"images_shape": list(batch.shape), "labels_shape": list(targets.shape),
            "images_dtype": str(batch.dtype), "labels_dtype": str(targets.dtype),
            "pixel_min": float(batch.min()), "pixel_max": float(batch.max()),
            "train_class_max_min_ratio": max(r["train"] for r in rows) / min(r["train"] for r in rows)}
    save_json(out / "batch_info.json", info)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.imshow(batch[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(names[int(targets[i])], fontsize=10)
        ax.axis("off")
    fig.suptitle("A batch after preprocessing: float32, [0,1]")
    fig.savefig(out / "batch_preview.png", dpi=150)
    plt.close(fig)
    print(f"EDA saved: {out}; batch {tuple(batch.shape)}", flush=True)
