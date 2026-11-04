"""Validate local model releases and atomically record activation or rollback.

Hashes establish consistency with local files, not authorship or authenticity.
The registry is a local deployment record; it is not evidence of production use.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"[a-fA-F0-9]{64}\Z")
_GATES = ("data", "quality", "calibration", "parity")
_DOMAINS = {
    "fashion_mnist": ("cnn_c5", 1, "L"),
    "garment_crop_rgb": ("mobilenet_v3_small", 3, "RGB"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"JSON document exceeds 16 MiB: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                       parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _version(document: dict, name: str) -> None:
    if type(document.get("format_version")) is not int or document["format_version"] != 1:
        raise ValueError(f"Unsupported {name} format_version")


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _artifact(root: Path, relative: Any) -> Path:
    """Accept only a relative regular file whose resolved target stays in root."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("Artifact file must be a nonempty relative path")
    windows = PureWindowsPath(relative)
    normalized = relative.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if (windows.drive or windows.root or posix.is_absolute() or ":" in relative
            or any(part in ("", ".", "..") for part in normalized.split("/"))):
        raise ValueError("Artifact path must not be absolute or contain traversal")
    target = (root / Path(*posix.parts)).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Artifact must be a regular file within the release directory")
    return target


def _bound_artifact(root: Path, value: Any, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain file and sha256")
    expected = _digest(value.get("sha256"), f"{name}.sha256")
    target = _artifact(root, value.get("file"))
    actual = _sha256(target)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return target, actual


def _preprocessing(value: Any, domain: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("preprocessing must be an object")
    for name in ("height", "width"):
        if type(value.get(name)) is not int or not 1 <= value[name] <= 4096:
            raise ValueError(f"preprocessing.{name} must be an integer in [1,4096]")
    _, channels, mode = _DOMAINS[domain]
    if type(value.get("channels")) is not int or value["channels"] != channels:
        raise ValueError("preprocessing.channels does not match the domain")
    if value.get("mode") != mode:
        raise ValueError("preprocessing.mode does not match the domain")
    if domain == "fashion_mnist" and (value["height"], value["width"]) != (28, 28):
        raise ValueError("The Fashion-MNIST reference requires 28x28 input")
    for name in ("mean", "std"):
        if name not in value:
            continue
        numbers = value[name]
        if (not isinstance(numbers, list) or len(numbers) != channels
                or any(type(number) not in (int, float) or not math.isfinite(number)
                       or (name == "std" and number <= 0) for number in numbers)):
            raise ValueError(f"preprocessing.{name} must contain {channels} finite values"
                             + (" greater than zero" if name == "std" else ""))


def _calibration(value: Any) -> None:
    if not isinstance(value, dict) or value.get("status") != "fitted":
        raise ValueError("Ready release calibration must have fitted status")
    temperature, threshold = value.get("temperature"), value.get("review_threshold")
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("Calibration temperature must be finite and positive")
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("Calibration review_threshold must be finite and in [0,1]")
    for name in ("fit_split_sha256", "threshold_selection_split_sha256"):
        _digest(value.get(name), f"calibration.{name}")
    policy = value.get("policy")
    if policy is not None:
        if (not isinstance(policy, dict) or policy.get("mode") not in ("review_all", "confidence_threshold")
                or type(policy.get("bound_eligible")) is not bool):
            raise ValueError("Calibration policy requires a known mode and boolean bound_eligible")
        if policy["mode"] == "review_all":
            if threshold != 1.0 or policy["bound_eligible"] is not False:
                raise ValueError("review_all requires threshold 1 and no eligible risk bound")
        elif policy["bound_eligible"] is not True:
            raise ValueError("An explicit confidence policy requires an eligible recorded risk bound")


def _receipt(root: Path, binding: Any, release_id: str, weights_sha256: str,
             calibration: dict) -> dict:
    receipt_path, _ = _bound_artifact(root, binding, "evaluation_receipt")
    receipt = _load_json(receipt_path)
    _version(receipt, "evaluation receipt")
    if receipt.get("release_id") != release_id:
        raise ValueError("Evaluation receipt release_id does not match the release")
    if _digest(receipt.get("weights_sha256"), "receipt.weights_sha256") != weights_sha256:
        raise ValueError("Evaluation receipt weights binding does not match")
    _calibration(receipt.get("calibration"))
    if receipt["calibration"] != calibration:
        raise ValueError("Evaluation receipt calibration does not match the release")
    gates = receipt.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Evaluation receipt must contain data/quality/calibration/parity gates")
    for name in _GATES:
        gate = gates.get(name)
        status = gate.get("status") if isinstance(gate, dict) else gate
        if status != "pass":
            raise ValueError(f"Evaluation gate '{name}' must pass before activation")
    return receipt


def validate_release(directory: str | os.PathLike, require_ready: bool = True) -> dict:
    """Check schema, paths, file hashes and declared evaluation gates.

    A reference release can be inspected with ``require_ready=False``. A real-data
    candidate can also be inspected, but setting status to ``ready`` always requires
    a hash-bound receipt with four passing gates, even in inspection mode. Passing
    these checks does not independently reproduce or authenticate the evaluation.
    """
    if type(require_ready) is not bool:
        raise ValueError("require_ready must be a boolean")
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Release directory must be a directory")
    manifest_path = _artifact(root, "release.json")
    manifest = _load_json(manifest_path)
    _version(manifest, "release")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or _ID.fullmatch(release_id) is None:
        raise ValueError("release_id must be a simple versioned identifier")
    domain = manifest.get("domain")
    if not isinstance(domain, str) or domain not in _DOMAINS:
        raise ValueError("Unsupported release domain")
    if manifest.get("architecture") != _DOMAINS[domain][0]:
        raise ValueError("Architecture does not match the release domain")
    labels = manifest.get("labels")
    if (not isinstance(labels, list) or not 2 <= len(labels) <= 1000
            or any(not isinstance(label, str) or not label or label.strip() != label
                   for label in labels)
            or len({label.casefold() for label in labels}) != len(labels)):
        raise ValueError("labels must be a unique list of at least two nonempty strings")
    if domain == "fashion_mnist" and len(labels) != 10:
        raise ValueError("The Fashion-MNIST reference requires ten labels")
    _preprocessing(manifest.get("preprocessing"), domain)
    weights_path, weights_sha256 = _bound_artifact(root, manifest.get("weights"), "weights")
    status = manifest.get("status")
    receipt = None
    reference_only = domain == "fashion_mnist"
    if reference_only:
        if status != "reference":
            raise ValueError("Fashion-MNIST artifacts must have reference status")
        if require_ready:
            raise ValueError("A reference release cannot be activated as a ready real-data model")
    else:
        if status not in ("candidate", "ready"):
            raise ValueError("A real-data release must have candidate or ready status")
        if status == "ready":
            calibration = manifest.get("calibration")
            _calibration(calibration)
            receipt = _receipt(root, manifest.get("evaluation_receipt"), release_id,
                               weights_sha256, calibration)
        elif require_ready:
            raise ValueError("A candidate release is not ready for activation")
    return {
        "manifest": manifest,
        "directory": str(root),
        "weights_path": str(weights_path),
        "manifest_sha256": _sha256(manifest_path),
        "evaluation_receipt": receipt,
        "reference_only": reference_only,
    }


def _entry(release: dict) -> dict:
    return {"release_id": release["manifest"]["release_id"],
            "directory": release["directory"],
            "manifest_sha256": release["manifest_sha256"],
            "weights_sha256": release["manifest"]["weights"]["sha256"].lower()}


def _validate_entry(value: Any) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("directory"), str):
        raise ValueError("Invalid registry release entry")
    if not Path(value["directory"]).is_absolute():
        raise ValueError("Registry release directory must be absolute")
    actual = _entry(validate_release(value["directory"], require_ready=True))
    if actual != value:
        raise ValueError("Registered release has changed since activation")
    return actual


def _state(path: Path) -> dict:
    if not path.exists():
        return {"format_version": 1, "active": None, "previous": None, "ledger": []}
    if path.is_symlink() or not path.is_file():
        raise ValueError("Registry active.json must be a regular nonsymlink file")
    value = _load_json(path)
    _version(value, "registry")
    if set(("active", "previous", "ledger")) - value.keys() or not isinstance(value["ledger"], list):
        raise ValueError("Invalid registry state")
    for key in ("active", "previous"):
        if value[key] is not None and not isinstance(value[key], dict):
            raise ValueError(f"Invalid registry {key} entry")
    return value


@contextmanager
def _locked(registry: Path):
    registry.mkdir(parents=True, exist_ok=True)
    lock = registry / ".release.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Release registry is busy; an existing lock must be investigated") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _atomic_state(path: Path, state: dict) -> None:
    temporary = path.parent / f".active-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _transition(state: dict, next_release: dict, action: str) -> dict:
    previous = state["active"]
    event = {"event_id": uuid.uuid4().hex,
             "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
             "action": action, "from": previous, "to": next_release}
    return {"format_version": 1, "active": next_release, "previous": previous,
            "ledger": state["ledger"] + [event]}


def activate_release(release_dir: str | os.PathLike, registry_dir: str | os.PathLike) -> dict:
    """Activate a verified ready release; pointer and local ledger share one atomic file."""
    candidate = _entry(validate_release(release_dir, require_ready=True))
    registry = Path(registry_dir).resolve()
    pointer = registry / "active.json"
    with _locked(registry):
        state = _state(pointer)
        for event in state["ledger"]:
            if not isinstance(event, dict) or not isinstance(event.get("to"), dict):
                raise ValueError("Invalid registry ledger")
            known = event["to"]
            if known.get("release_id") == candidate["release_id"] and known != candidate:
                raise ValueError("release_id cannot be reused for a different release binding")
        _validate_entry(candidate)
        if state["active"] != candidate:
            state = _transition(state, candidate, "activate")
            _atomic_state(pointer, state)
    return {**state, "registry_path": str(pointer)}


def rollback_release(registry_dir: str | os.PathLike) -> dict:
    """Restore the previous verified ready release and record the actual local transition."""
    registry = Path(registry_dir).resolve()
    pointer = registry / "active.json"
    with _locked(registry):
        state = _state(pointer)
        if state["active"] is None or state["previous"] is None:
            raise ValueError("No previous release is available for rollback")
        restored = _validate_entry(state["previous"])
        state = _transition(state, restored, "rollback")
        _atomic_state(pointer, state)
    return {**state, "registry_path": str(pointer)}
