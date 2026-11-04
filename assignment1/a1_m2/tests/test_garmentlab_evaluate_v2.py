"""Real tiny controlled-study/finalization/replay; never a quality claim."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from garmentlab import evaluate as legacy, evaluate_v2, study_v2
from garmentlab.runtime import atomic_json, digest
from test_garmentlab_training import make_prepared, tiny_factory


@pytest.fixture(autouse=True)
def cpu_settings():
    values = (torch.get_num_threads(), torch.backends.mkldnn.enabled,
              torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(values[0])
    torch.backends.mkldnn.enabled = values[1]
    torch.use_deterministic_algorithms(values[2])
    torch.backends.cudnn.benchmark = values[3]
    torch.backends.cuda.matmul.allow_tf32 = values[4]
    torch.backends.cudnn.allow_tf32 = values[5]


def fixture(tmp_path):
    prepared = make_prepared(tmp_path / "prepared")
    path = prepared / "index_manifest.json"
    value = json.loads(path.read_text())
    for row in value["records"]:
        row["evaluation_group_id"] = row["source_group_id"]
        row["annotation_id"] = row["array_index"]
    atomic_json(path, value)
    cfg = json.loads(study_v2.CONFIG.read_text(encoding="utf-8"))
    cfg.update(device="cpu", epochs=2, batch_size=2, stop_by_utc="2099-01-01T00:00:00Z")
    cfg["candidates"] = [c for c in cfg["candidates"] if c["id"] in ("C0", "C1", "C3")]
    cfg["calibration"] = {"minimum_accepted_groups": 30}
    return prepared, cfg


def test_real_tiny_study_finalization_freezes_before_assessment_and_replays_without_model(tmp_path, monkeypatch):
    prepared, cfg = fixture(tmp_path)
    out = tmp_path / "study"
    result = study_v2.run_study(prepared, out, cfg, model_factory=tiny_factory, smoke=True)
    assert result["status"] == "TRAINING_COMPLETE_CHOICES_FROZEN"
    final_selection = (out / "FINAL_SELECTION.json").read_bytes()
    calls = []
    actual = legacy._infer

    def check_choices_before_forward(model, images, indices, **kwargs):
        values = np.asarray(indices)
        if np.any(values >= 12):
            assert (out / "evaluation_v2/CALIBRATION_PARTITION.json").exists()
        if np.any(values >= 18):
            freeze = json.loads((out / "evaluation_v2/SELECTION_FREEZE.json").read_text())
            assert freeze["calibration"]["policy"]["mode"] == "review_all"
            assert freeze["assessment_scope"] == "POST_TEST_EXPLORATORY_REGRESSION"
        calls.append(values.tolist())
        return actual(model, images, indices, **kwargs)

    monkeypatch.setattr(legacy, "_infer", check_choices_before_forward)
    summary = evaluate_v2.finalize_study_v2(out, prepared, cfg, model_factory=tiny_factory, smoke=True)
    assert summary["status"] == "SMOKE_ONLY_COMPLETE" and not summary["ready_for_local_service"]
    assert summary["assessment_scope"] == "POST_TEST_EXPLORATORY_REGRESSION"
    assert len(summary["assessment"]) in (3, 6) and len(summary["corruptions"]) == 9
    assert summary["calibration"]["policy"]["mode"] == "review_all"
    assert summary["target_bound_satisfied_under_assumptions"] is False
    assert summary["deployment_calibrated_assessment"]["acceptance"]["accepted_count"] == 0
    assert summary["gates"]["parity"]["status"] == "pass", summary["gates"]["parity"]
    assert not (out / "release_candidate/release.json").exists()
    assert (out / "release_candidate/SMOKE_ONLY.json").exists()
    assert (out / "FINAL_SELECTION.json").read_bytes() == final_selection
    assert calls and any(max(c) >= 18 for c in calls)
    saved = (out / "FINAL_SUMMARY_V2.json").read_bytes()
    for entry in summary["artifacts"]:
        assert digest(out / entry["path"]) == entry["sha256"]

    def no_forward(*args, **kwargs):
        pytest.fail("Complete replay must not build a model or execute another inference pass")

    monkeypatch.setattr(legacy, "_load_model", no_forward)
    monkeypatch.setattr(legacy, "_cached_logits", no_forward)
    repeated = evaluate_v2.finalize_study_v2(out, prepared, cfg, model_factory=tiny_factory, smoke=True)
    assert repeated == summary and (out / "FINAL_SUMMARY_V2.json").read_bytes() == saved
    # Completed-output tampering is rejected before any new inference.
    path = out / "evaluation_v2/calibration/CALIBRATION_V2.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate_v2.finalize_study_v2(out, prepared, cfg, model_factory=tiny_factory, smoke=True)


def test_incomplete_study_never_opens_calibration_or_assessment(tmp_path, monkeypatch):
    prepared, cfg = fixture(tmp_path)
    out = tmp_path / "incomplete"
    study_v2.plan_study(prepared, out, cfg, smoke=True)
    monkeypatch.setattr(legacy, "_infer", lambda *a, **k: pytest.fail("No incomplete-study forward"))
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate_v2.finalize_study_v2(out, prepared, cfg, model_factory=tiny_factory, smoke=True)
    assert not (out / "evaluation_v2").exists()


def test_frozen_study_config_and_model_scope_cannot_be_changed(tmp_path):
    prepared, cfg = fixture(tmp_path)
    out = tmp_path / "out"
    study_v2.run_study(prepared, out, cfg, model_factory=tiny_factory, smoke=True)
    altered = deepcopy(cfg)
    altered["calibration"]["target_accuracy"] = .9
    with pytest.raises(ValueError, match="config or smoke scope"):
        evaluate_v2.finalize_study_v2(out, prepared, altered, model_factory=tiny_factory, smoke=True)
    with pytest.raises(ValueError, match="Injected models"):
        evaluate_v2.finalize_study_v2(out, prepared, cfg, model_factory=tiny_factory, smoke=False)


def test_policy_acceptance_never_accepts_exactly_one_when_review_all():
    binding = {"temperature": 1., "review_threshold": 1., "policy": {"mode": "review_all"}}
    score = evaluate_v2.acceptance_metrics(np.array([[1000., -1000.], [-1000., 1000.]]), np.array([0, 1]), binding)
    assert score["accepted_count"] == 0 and score["coverage"] == 0 and score["accepted_accuracy"] is None
    binding["policy"]["mode"] = "confidence_threshold"
    assert evaluate_v2.acceptance_metrics(np.array([[1000., -1000.]]), np.array([0]), binding)["accepted_count"] == 1
