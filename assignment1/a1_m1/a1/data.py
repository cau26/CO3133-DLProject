"""Fashion-MNIST download, stratified split, Dataset and DataLoader.

All models reuse the SAME saved train/validation indices. The official test
set is kept separate and is not used to select checkpoints in this draft.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .common import save_json


def prepare_data(data_dir, output_dir, config):
    # HTTPS copy published by the dataset authors. Torchvision verifies the
    # official per-file MD5 checksums before extracting these IDX files.
    datasets.FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ]
    transform = transforms.ToTensor()  # uint8 [0,255] -> float32 [0,1]
    full_train = datasets.FashionMNIST(str(data_dir), train=True,
                                      download=True, transform=transform)
    test = datasets.FashionMNIST(str(data_dir), train=False,
                                download=True, transform=transform)
    labels = full_train.targets.numpy()
    train_ids, val_ids = train_test_split(
        np.arange(len(full_train)), test_size=config["validation_size"],
        random_state=config["seed"], stratify=labels)
    train_ids = np.sort(train_ids)
    val_ids = np.sort(val_ids)
    assert np.intersect1d(train_ids, val_ids).size == 0, "Train/validation overlap"
    assert np.array_equal(np.sort(np.r_[train_ids, val_ids]), np.arange(len(full_train)))
    assert len(full_train) == 60000 and len(test) == 10000
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "split_indices.npz", train=train_ids, validation=val_ids)
    split_hash = hashlib.sha256(train_ids.astype("<i8").tobytes()
                                + val_ids.astype("<i8").tobytes()).hexdigest()
    summary = {
        "dataset": "Fashion-MNIST", "source": "https://github.com/zalandoresearch/fashion-mnist",
        "license": "MIT", "official_train": len(full_train), "train": len(train_ids),
        "archive_md5_expected": dict(datasets.FashionMNIST.resources),
        "validation": len(val_ids), "test": len(test), "seed": config["seed"],
        "stratified": True, "split_sha256": split_hash,
        "input_shape": [1, 28, 28], "classes": full_train.classes,
        "preprocessing": "ToTensor: float32 [0,1]; no augmentation in this baseline",
        "split_unit": "individual image; official test partition retained",
        "leakage_checks": "train/validation IDs disjoint; complete coverage of official train",
        "test_usage": "descriptive class counts only; no model selection or test evaluation",
        "duplicate_audit": "not performed; split-ID checks do not prove absence of duplicate images",
    }
    save_json(out / "data_summary.json", summary)
    return {"full_train": full_train, "test": test,
            "train": Subset(full_train, train_ids.tolist()),
            "validation": Subset(full_train, val_ids.tolist()),
            "train_ids": train_ids, "val_ids": val_ids, "summary": summary}


def make_loaders(data, config):
    # A new seeded generator per model makes training sample order comparable.
    generator = torch.Generator().manual_seed(config["seed"])
    common = {"batch_size": config["batch_size"], "num_workers": config["num_workers"]}
    train = DataLoader(data["train"], shuffle=True, generator=generator, **common)
    val = DataLoader(data["validation"], shuffle=False, **common)
    return train, val
