"""Local study accounting and export for the student-run Colab workflow."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
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


def utc_now():
    return datetime.now(timezone.utc)


def utc_parse(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Deadline must include a timezone.")
    return result.astimezone(timezone.utc)


class RunSession:
    """A single-writer lease with cumulative process time, including setup.

    The lease reduces accidental concurrent notebook execution. Drive Fuse is not
    a distributed lock service. Only one runtime may write to a study directory.
    """

    def __init__(self, out_dir, *, total_seconds=54000, stop_by_utc="2026-10-28T12:59:00Z"):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.clock = self.out / "RUN_CLOCK.json"
        self.lock = self.out / "CONTROLLER_LEASE.json"
        self.token = uuid.uuid4().hex
        self.limit = total_seconds
        self.stop_by = utc_parse(stop_by_utc)
        self.started = time.monotonic()
        self.previous = 0.0
        self.stage = "initializing"
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def __enter__(self):
        if self.clock.exists():
            self.previous = json.loads(self.clock.read_text(encoding="utf-8"))["active_process_seconds"]
        if self.lock.exists():
            old = json.loads(self.lock.read_text(encoding="utf-8"))
            age = (utc_now() - utc_parse(old["heartbeat_utc"])).total_seconds()
            if age < 600:
                raise RuntimeError("Another controller lease is active. Close the other run or wait for its lease to expire.")
            stale = self.lock.with_name("STALE_LEASE_" + uuid.uuid4().hex + ".json")
            self.lock.replace(stale)
        descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"owner": self.token, "heartbeat_utc": utc_now().isoformat()}, stream)
        self.tick()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    @property
    def active_seconds(self):
        return self.previous + time.monotonic() - self.started

    def tick(self):
        if json.loads(self.lock.read_text(encoding="utf-8")).get("owner") != self.token:
            raise RuntimeError("Controller lease ownership changed; stop writing this study.")
        atomic_json(self.lock, {"owner": self.token, "heartbeat_utc": utc_now().isoformat(), "stage": self.stage})
        atomic_json(self.clock, {"active_process_seconds": self.active_seconds,
                                "budget_seconds": self.limit, "stop_by_utc": self.stop_by.isoformat(),
                                "stage": self.stage, "updated_utc": utc_now().isoformat(),
                                "scope": "Cumulative controller process time including preparation/training/finalization. Idle GPU allocation outside the process and abrupt tail after last heartbeat are not counted."})

    def _heartbeat(self):
        while not self.stop_event.wait(20):
            try:
                self.tick()
            except Exception as exc:
                self.error = str(exc)
                self.stop_event.set()

    def should_stop(self, reserve_seconds=1800):
        if self.error:
            raise RuntimeError(self.error)
        return (self.active_seconds + reserve_seconds >= self.limit
                or (self.stop_by - utc_now()).total_seconds() <= reserve_seconds)

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        try:
            self.tick()
        finally:
            if self.lock.exists() and json.loads(self.lock.read_text(encoding="utf-8")).get("owner") == self.token:
                self.lock.unlink()


def export_results(out_dir, *, source_dir=None):
    """Export completed and partial evidence, excluding datasets/features/temp files."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lease = out / "CONTROLLER_LEASE.json"
    if lease.exists():
        current = json.loads(lease.read_text(encoding="utf-8"))
        if (utc_now() - utc_parse(current["heartbeat_utc"])).total_seconds() < 600:
            raise RuntimeError("Controller is still running. Export after it stops.")
    status_path = out / "RUN_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "INCOMPLETE_NO_RUN_STATUS"}
    payloads = {}
    for path in out.rglob("*"):
        relative = path.relative_to(out)
        if (not path.is_file() or path.is_symlink() or "exports" in relative.parts
                or ("feature_cache" in relative.parts and path.suffix != ".json")
                or path.suffix in {".tmp", ".zip", ".npy", ".mmap"}
                or path.name in {"CONTROLLER_LEASE.json", "notebook_console.log"}
                or path.name.startswith("STALE_LEASE_") or "__pycache__" in relative.parts):
            continue
        payloads[relative.as_posix()] = path
    if source_dir is not None:
        source = Path(source_dir)
        for name in ["SOURCE_MANIFEST.json", "configs/garmentlab_deadline.json", "configs/garmentlab_data.json"]:
            path = source / name
            if path.is_file():
                payloads["source/" + name] = path
    entries = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(payloads.items())]
    manifest = {"schema": "garmentlab_deadline_results_v1", "status": status,
                "exported_utc": utc_now().isoformat(), "payloads": entries,
                "excluded": "Source photographs, prepared image arrays, feature arrays, live lease and temp files. Feature-cache identity/cost receipts and resume checkpoints are included; the working cache remains on Drive."}
    folder = out / "exports"
    folder.mkdir(exist_ok=True)
    target = folder / "g_m10_garmentlab_a100_deadline_run01_results.zip"
    temp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in sorted(payloads.items()):
            archive.write(path, name)
        archive.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Result ZIP failed CRC verification.")
        for entry in entries:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError("A result changed during export. Retry after the writer stops.")
    os.replace(temp, target)
    return target
