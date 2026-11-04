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

from .common import load_json, protect_json


def split_digest(train_ids, val_ids):
    return hashlib.sha256(train_ids.astype("<i8").tobytes()
                          + val_ids.astype("<i8").tobytes()).hexdigest()


def resolve_split_seed(config):
    """Legacy configs use seed for both purposes; explicit split_seed fixes the split."""
    for key in ("seed", "split_seed"):
        if key in config and (type(config[key]) is not int or not 0 <= config[key] < 2 ** 32):
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    return config.get("split_seed", config["seed"])


def validate_split(train_ids, val_ids, size, validation_size):
    if train_ids.ndim != 1 or val_ids.ndim != 1:
        raise ValueError("Saved split must contain one-dimensional index arrays")
    if train_ids.dtype.kind not in "iu" or val_ids.dtype.kind not in "iu":
        raise ValueError("Saved split indices must be integers")
    if len(val_ids) != validation_size or len(train_ids) != size - validation_size:
        raise ValueError("Saved split sizes differ from the immutable configuration")
    if not np.array_equal(np.sort(np.r_[train_ids, val_ids]), np.arange(size)):
        raise ValueError("Split contains overlapping, duplicate, missing or out-of-range IDs")
    if not np.array_equal(train_ids, np.sort(train_ids)) or not np.array_equal(val_ids, np.sort(val_ids)):
        raise ValueError("Split IDs must retain their saved sorted order")


def prepare_data(data_dir, output_dir, config, include_test=True):
    split_seed = resolve_split_seed(config)
    separate_seed_schema = "split_seed" in config
    # HTTPS copy published by the dataset authors. Torchvision verifies the
    # official per-file MD5 checksums before extracting these IDX files.
    datasets.FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ]
    transform = transforms.ToTensor()  # uint8 [0,255] -> float32 [0,1]
    full_train = datasets.FashionMNIST(str(data_dir), train=True,
                                      download=True, transform=transform)
    test = (datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
            if include_test else None)
    labels = full_train.targets.numpy()
    if len(full_train) != 60000 or (test is not None and len(test) != 10000):
        raise ValueError("Unexpected Fashion-MNIST official partition size")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_path, summary_path = out / "split_indices.npz", out / "data_summary.json"
    if split_path.exists() != summary_path.exists():
        raise FileExistsError("Incomplete split artifacts. Preserve this directory and choose a new --out.")
    dataset_hash = hashlib.sha256(full_train.data.numpy().tobytes() + labels.astype("<i8").tobytes()).hexdigest()
    if split_path.exists():
        with np.load(split_path, allow_pickle=False) as saved:
            train_ids, val_ids = saved["train"].copy(), saved["validation"].copy()
        stored_summary = load_json(summary_path)
        stored_separate_schema = "split_seed" in stored_summary
        if stored_separate_schema != separate_seed_schema:
            raise ValueError("Saved split metadata schema differs; legacy/new runs cannot be silently adopted")
        if separate_seed_schema and (stored_summary.get("split_metadata_schema_version") != 2
                                      or stored_summary.get("split_seed") != split_seed):
            raise ValueError("Saved split_seed/schema differs from the requested configuration")
        if not separate_seed_schema and "split_metadata_schema_version" in stored_summary:
            raise ValueError("Unexpected split metadata version in legacy run")
        if stored_summary.get("seed") != config["seed"]:
            raise ValueError("Saved training/legacy seed differs from the requested configuration")
        if stored_summary.get("dataset_sha256") != dataset_hash:
            raise ValueError("Saved run dataset fingerprint differs or is missing; use a new --out")
        if stored_summary.get("split_sha256") != split_digest(train_ids, val_ids):
            raise ValueError("Saved split fingerprint does not match the stored indices")
    else:
        train_ids, val_ids = train_test_split(
            np.arange(len(full_train)), test_size=config["validation_size"],
            random_state=split_seed, stratify=labels)
        train_ids, val_ids = np.sort(train_ids), np.sort(val_ids)
    validate_split(train_ids, val_ids, len(full_train), config["validation_size"])
    split_hash = split_digest(train_ids, val_ids)
    summary = {
        "dataset": "Fashion-MNIST", "source": "https://github.com/zalandoresearch/fashion-mnist",
        "license": "MIT", "official_train": len(full_train), "train": len(train_ids),
        "archive_md5_expected": dict(datasets.FashionMNIST.resources),
        "validation": len(val_ids), "test": 10000, "seed": config["seed"],
        "dataset_sha256": dataset_hash,
        "stratified": True, "split_sha256": split_hash,
        "input_shape": [1, 28, 28], "classes": full_train.classes,
        "preprocessing": "ToTensor: float32 [0,1]; no augmentation in this baseline",
        "split_unit": "individual image; official test partition retained",
        "leakage_checks": "train/validation IDs disjoint; complete coverage of official train",
        "test_usage": "reserved; descriptive EDA counts allowed; evaluation requires an explicit five-model freeze",
        "duplicate_audit": "not performed; split-ID checks do not prove absence of duplicate images",
    }
    # Preserve the complete legacy summary shape when split_seed was not supplied.
    if separate_seed_schema:
        summary.update(split_metadata_schema_version=2, split_seed=split_seed)
        if config.get("evaluation_policy") == "validation_only_post_test_study":
            summary["test_usage"] = "official test already seen historically; supplementary study never evaluates it"
    if not split_path.exists():
        with split_path.open("xb") as stream:
            np.savez_compressed(stream, train=train_ids, validation=val_ids)
    protect_json(summary_path, summary)
    return {"full_train": full_train, "test": test,
            "train": Subset(full_train, train_ids.tolist()),
            "validation": Subset(full_train, val_ids.tolist()),
            "train_ids": train_ids, "val_ids": val_ids, "summary": summary}


def make_loaders(data, config):
    # A new seeded generator per model makes training sample order comparable.
    generator = torch.Generator().manual_seed(config["seed"])
    common = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
              "pin_memory": str(config.get("device", "cpu")).startswith("cuda")}
    train = DataLoader(data["train"], shuffle=True, generator=generator, **common)
    val = DataLoader(data["validation"], shuffle=False, **common)
    return train, val
