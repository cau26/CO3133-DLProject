"""Shared helpers for Assignment 2: files, seeds, environment and source-code provenance."""
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}
ID_TO_CLASS = {i + 1: name for i, name in enumerate(VOC_CLASSES)}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_metadata():
    files = [*sorted(ROOT.glob("*.py")), *sorted((ROOT / "a2").glob("*.py"))]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
            text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL,
            text=True, timeout=5).strip())
    except (subprocess.SubprocessError, OSError):
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "source_sha256": combined,
        "files_sha256": hashes,
    }


def environment(device):
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    gpu_name = None
    gpu_memory_gb = None
    cuda_version = torch.version.cuda
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        gpu_memory_gb = round(props.total_memory / (1024 ** 3), 2)
    pkgs = {}
    for name in ["torch", "torchvision", "numpy", "pandas", "matplotlib", "Pillow", "scikit-learn"]:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "gpu": gpu_name,
        "gpu_total_memory_gb": gpu_memory_gb,
        "cuda_version": cuda_version,
        "versions": pkgs,
    }
