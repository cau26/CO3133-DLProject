import json
from pathlib import Path
import zipfile
import pytest
import torch

from tools import run_garmentlab_improve as controller
from test_garmentlab_evaluate_v2 import fixture
from test_garmentlab_training import tiny_factory


@pytest.fixture(autouse=True)
def bounded_cpu():
    threads, mkldnn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(1); torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(threads); torch.backends.mkldnn.enabled = mkldnn


def test_controller_runs_actual_tiny_pipeline_then_exports_and_resumes(tmp_path):
    prepared, config = fixture(tmp_path)
    path = tmp_path / "config.json"; path.write_text(json.dumps(config), encoding="utf-8")
    data_config = controller.ROOT / "configs/garmentlab_data.json"
    out, work, cache = [tmp_path / n for n in ("out", "work", "cache")]
    result = controller.run_workflow(out, work, cache, path, data_config, smoke=True,
        model_factory=tiny_factory, prepare_fn=lambda *_: prepared)
    assert result["status"]["status"] == "COMPLETE"
    assert result["status"]["finalization"]["smoke_only"] is True
    assert result["status"]["finalization"]["ready_for_local_service"] is False
    with zipfile.ZipFile(result["result_zip"]) as archive:
        assert archive.testzip() is None
        assert "FINAL_SUMMARY_V2.json" in archive.namelist()
    summary = (out / "FINAL_SUMMARY_V2.json").read_bytes()
    replay = controller.run_workflow(out, work, cache, path, data_config, stage="finalize", smoke=True,
        model_factory=tiny_factory)
    assert replay["status"]["status"] == "COMPLETE"
    assert (out / "FINAL_SUMMARY_V2.json").read_bytes() == summary


def test_budget_stop_exports_explicit_partial_and_injection_cannot_claim_real(tmp_path):
    prepared, config = fixture(tmp_path)
    config["max_active_seconds"] = 1
    path = tmp_path / "config.json"; path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="software smoke"):
        controller.run_workflow(tmp_path/"no", tmp_path/"w", tmp_path/"c", path,
            controller.ROOT/"configs/garmentlab_data.json", model_factory=tiny_factory)
    result = controller.run_workflow(tmp_path/"out", tmp_path/"work", tmp_path/"cache", path,
        controller.ROOT/"configs/garmentlab_data.json", smoke=True, model_factory=tiny_factory,
        prepare_fn=lambda *_: (_ for _ in ()).throw(AssertionError("Preparation must not start after budget")))
    assert result["status"]["status"] == "INCOMPLETE_STOPPED"
    assert Path(result["result_zip"]).is_file()
    assert not (tmp_path/"out/FINAL_SUMMARY_V2.json").exists()
