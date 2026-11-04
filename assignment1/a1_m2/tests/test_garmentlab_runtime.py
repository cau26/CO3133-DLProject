"""Clock, deadline and partial-export checks; no GPU or model training."""
import hashlib
import json
from datetime import timedelta
from pathlib import Path
import zipfile

import pytest

from garmentlab.runtime import RunSession, atomic_json, export_results, utc_now, utc_parse


def test_timezone_required():
    with pytest.raises(ValueError):
        utc_parse("2026-10-28T23:59:00")
    assert utc_parse("2026-10-28T23:59:00+07:00").hour == 16


def test_clock_cumulative_and_live_lease_rejected(tmp_path):
    stop = (utc_now() + timedelta(days=1)).isoformat()
    with RunSession(tmp_path, total_seconds=9999, stop_by_utc=stop) as first:
        assert not first.should_stop(0)
        with pytest.raises(RuntimeError, match="lease"):
            with RunSession(tmp_path, stop_by_utc=stop):
                pass
    saved = json.loads((tmp_path / "RUN_CLOCK.json").read_text())
    with RunSession(tmp_path, total_seconds=9999, stop_by_utc=stop) as second:
        assert second.active_seconds >= saved["active_process_seconds"]
    assert not (tmp_path / "CONTROLLER_LEASE.json").exists()


def test_elapsed_budget_and_absolute_deadline_stop(tmp_path):
    atomic_json(tmp_path / "RUN_CLOCK.json", {"active_process_seconds": 500})
    with RunSession(tmp_path, total_seconds=550, stop_by_utc=(utc_now() + timedelta(days=1)).isoformat()) as session:
        assert session.should_stop(60)
    with RunSession(tmp_path, total_seconds=9999, stop_by_utc=(utc_now() - timedelta(seconds=1)).isoformat()) as session:
        assert session.should_stop(0)


def test_stale_lease_recovered_but_evidence_preserved(tmp_path):
    atomic_json(tmp_path / "CONTROLLER_LEASE.json", {"owner": "old", "heartbeat_utc": (utc_now() - timedelta(minutes=12)).isoformat()})
    with RunSession(tmp_path, stop_by_utc=(utc_now() + timedelta(days=1)).isoformat()):
        assert len(list(tmp_path.glob("STALE_LEASE_*.json"))) == 1


def test_changed_lease_stops_writer_and_is_not_deleted(tmp_path):
    with pytest.raises(RuntimeError, match="ownership"):
        with RunSession(tmp_path, stop_by_utc=(utc_now() + timedelta(days=1)).isoformat()) as session:
            atomic_json(tmp_path / "CONTROLLER_LEASE.json", {"owner": "other", "heartbeat_utc": utc_now().isoformat()})
            session.should_stop(0)  # Explicit write check happens on tick.
            session.tick()
    assert json.loads((tmp_path / "CONTROLLER_LEASE.json").read_text())["owner"] == "other"


def test_partial_export_excludes_arrays_and_binds_bytes(tmp_path):
    atomic_json(tmp_path / "RUN_STATUS.json", {"status": "INCOMPLETE", "completed_tasks": 1})
    (tmp_path / "best.pt").write_bytes(b"software-fixture-only")
    (tmp_path / "images.npy").write_bytes(b"must-not-export")
    (tmp_path / "feature_cache").mkdir()
    (tmp_path / "feature_cache" / "receipt.json").write_text("{}")
    (tmp_path / "feature_cache" / "features.npy").write_bytes(b"large-private-array")
    target = export_results(tmp_path)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("EXPORT_MANIFEST.json"))
        assert manifest["status"]["status"] == "INCOMPLETE"
        assert "images.npy" not in archive.namelist()
        assert "feature_cache/receipt.json" in archive.namelist()
        assert "feature_cache/features.npy" not in archive.namelist()
        for entry in manifest["payloads"]:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
    export_results(tmp_path)  # Re-export never recursively includes an earlier ZIP.


def test_export_refuses_live_writer_and_ignores_mutable_console(tmp_path):
    with RunSession(tmp_path, stop_by_utc=(utc_now() + timedelta(days=1)).isoformat()):
        with pytest.raises(RuntimeError, match="still running"):
            export_results(tmp_path)
    (tmp_path / "notebook_console.log").write_text("mutable transport log")
    with zipfile.ZipFile(export_results(tmp_path)) as archive:
        assert "notebook_console.log" not in archive.namelist()
