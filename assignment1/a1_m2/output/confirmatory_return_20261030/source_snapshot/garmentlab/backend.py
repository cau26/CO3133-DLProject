"""Explicit FP32 evaluation policy matching the immutable training implementation."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform

import numpy as np
import torch

POLICY_ID = "garmentlab_training_fp32_deterministic_v1"


def snapshot_evaluation_backend(device=None):
    """Describe current process settings; this function does not change flags."""
    try:
        vision = str(importlib.metadata.version("torchvision"))
    except importlib.metadata.PackageNotFoundError:
        vision = "not_installed"
    cuda_available = torch.cuda.is_available()
    selected_device = str(device) if device is not None else None
    cuda_index = (torch.device(device).index or 0) if selected_device and selected_device.startswith("cuda") else 0
    return {
        "schema": 1,
        "policy_id": POLICY_ID,
        "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": selected_device,
        "flags": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "default_dtype": str(torch.get_default_dtype()),
            "cuda_autocast_enabled": bool(torch.is_autocast_enabled()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": vision, "numpy": str(np.__version__),
            "cuda": str(torch.version.cuda), "cudnn": torch.backends.cudnn.version(),
            "cuda_available": bool(cuda_available),
            "gpu": str(torch.cuda.get_device_name(cuda_index)) if cuda_available else None,
            "gpu_capability": list(torch.cuda.get_device_capability(cuda_index)) if cuda_available else None,
            "cpu_threads": int(torch.get_num_threads()),
            "cpu_interop_threads": int(torch.get_num_interop_threads()),
        },
    }


def configure_evaluation_backend():
    """Apply the exact four backend settings made by ``train_study``.

    Run this inside the finalization process, not only inside a parent notebook.
    It does not alter the training source, weights, metric tolerances or policy.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return snapshot_evaluation_backend()
