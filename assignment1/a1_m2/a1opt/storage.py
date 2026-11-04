"""Atomic local/Drive writes and verification for the student's optimization run.

Only checkpoints produced by this source are loaded. A hash proves integrity,
not trust: never pass a checkpoint downloaded from an unknown author to torch.load.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _replace(temporary, path):
    # Windows sync/antivirus services can briefly hold a recently read JSON.
    # Retry only that platform's PermissionError; never delete the destination
    # or weaken atomic replacement/readback. Persistent permission failures fail.
    delays = (0, .05, .1, .2, .4, .8) if os.name == "nt" else (0,)
    for index, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if index == len(delays) - 1:
                raise


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, path)
        if sha256_file(path) != hashlib.sha256(payload).hexdigest():
            raise OSError(f"Write/readback mismatch: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if load_json(path) != value:
            raise ValueError(f"Immutable document differs: {path}; choose a new run folder.")
        return
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n").encode("utf-8"))
    if load_json(path) != value:
        raise OSError(f"JSON readback mismatch: {path}")


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                            dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        expected = sha256_file(temporary)
        _replace(temporary, path)
        if sha256_file(path) != expected:
            raise OSError(f"Checkpoint readback mismatch: {path}")
        return expected
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_complete(out, binding=None):
    """Return a completed task only after checking every listed artifact byte."""
    out = Path(out)
    complete = load_json(out / "COMPLETE.json")
    if complete.get("status") != "COMPLETE":
        raise ValueError(f"Invalid completion status: {out}")
    if binding is not None and complete.get("binding_sha256") != json_hash(binding):
        raise ValueError(f"Completion binding differs: {out}")
    artifacts = complete.get("artifacts_sha256", {})
    if not artifacts:
        raise ValueError(f"No bound artifacts in completion: {out}")
    root = out.resolve()
    for relative, expected in artifacts.items():
        path = (out / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Missing/unsafe completed artifact: {relative}")
        if sha256_file(path) != expected:
            raise ValueError(f"Completed artifact changed: {path}")
    return complete
