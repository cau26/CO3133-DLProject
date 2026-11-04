"""Numerical/unit and tiny synthetic pipeline tests; no real training-quality claim."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from garmentlab import evaluate, training
from garmentlab.runtime import atomic_json, digest


def test_macro_f1_uses_all_declared_classes_and_confusion_counts():
    probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .8, .1], [.1, .1, .8]])
    result = evaluate.classification_metrics(np.log(probabilities), np.array([0, 1, 2, 2]))
    assert result["accuracy"] == .75
    assert result["macro_f1"] == pytest.approx(7 / 9)
    assert result["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    absent = evaluate.classification_metrics(np.log(probabilities[:1]), np.array([0]))
    assert absent["macro_f1"] == pytest.approx(1 / 3)


def test_nll_brier_and_ece_have_explicit_hand_checkable_definitions():
    probabilities = np.array([[.8, .2], [.6, .4]])
    result = evaluate.classification_metrics(np.log(probabilities), np.array([0, 1]), bins=1)
    assert result["nll"] == pytest.approx((-math.log(.8) - math.log(.4)) / 2)
    assert result["brier_score"] == pytest.approx(.4)
    assert result["ece"] == pytest.approx(.2)
    assert result["reliability_bins"][0]["count"] == 2


def test_temperature_is_bounded_positive_and_does_not_change_argmax():
    logits = np.array([[5., 0.], [5., 0.], [0., 5.], [0., 5.]])
    targets = np.array([0, 1, 1, 0])
    result = evaluate.fit_temperature(logits, targets)
    assert .05 <= result["temperature"] <= 20
    assert result["temperature"] > 1
    assert result["nll_after"] <= result["nll_before"]
    assert np.array_equal(logits.argmax(1), (logits / result["temperature"]).argmax(1))
    assert result == evaluate.fit_temperature(logits, targets)


def test_threshold_keeps_confidence_ties_and_uses_maximum_eligible_coverage():
    probabilities = np.array([[.99, .01], [.98, .02], [.9, .1], [.9, .1]])
    result = evaluate.select_review_threshold(probabilities, np.array([0, 0, 1, 0]), target_accuracy=1, min_accepted=2)
    assert result["eligible"] is True
    assert result["review_threshold"] == .98
    assert result["accepted_count"] == 2 and result["coverage"] == .5
    assert result["accepted_accuracy"] == 1


def test_no_eligible_threshold_is_none_not_an_accidentally_accepting_one():
    probabilities = np.array([[1., 0.], [1., 0.], [1., 0.]])
    result = evaluate.select_review_threshold(probabilities, np.array([1, 1, 1]), min_accepted=1)
    assert result["eligible"] is False
    assert result["review_threshold"] is None and result["coverage"] == 0
    accepted = evaluate._acceptance(np.array([[1000., -1000.]]), np.array([1]), 1., None)
    assert accepted["accepted_count"] == 0


def test_small_sample_binomial_uncertainty_does_not_guarantee_target():
    low, high = evaluate.wilson_interval(30, 30)
    assert low < .95 and high == pytest.approx(1.)
    assert evaluate.wilson_interval(0, 0) == [None, None]
    with pytest.raises(ValueError):
        evaluate.wilson_interval(31, 30)


@pytest.mark.parametrize("logits,targets", [([], []), ([[float("nan"), 0]], [0]),
    ([[0, 1]], [2]), ([[0, 1]], [True]), ([[0, 1]], [0.5])])
def test_invalid_numerical_inputs_fail(logits, targets):
    with pytest.raises(ValueError):
        evaluate.classification_metrics(logits, np.asarray(targets))


def test_corruptions_are_deterministic_per_sample_and_keep_shape_dtype():
    images = np.full((2, 224, 224, 3), 128, dtype=np.uint8)
    noise = {"kind": "noise", "severity": 1, "sigma": .02}
    together = evaluate._corrupt(images, noise, [7, 12])
    separate = evaluate._corrupt(images[1:], noise, [12])
    assert np.array_equal(together[1], separate[0])
    assert together.dtype == np.uint8 and together.shape == images.shape
    moved = evaluate._corrupt(images, {"kind": "translation", "severity": 1, "pixels": 2}, [7, 12])
    assert not moved[:, :2].any() and not moved[:, :, :2].any()


class TinyTransfer(nn.Module):
    def __init__(self, classes, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(999)
            self.features = nn.Sequential(nn.Conv2d(3, 4, 1, stride=16), nn.BatchNorm2d(4), nn.ReLU())
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Linear(4, 8), nn.Hardswish(), nn.Dropout(.2), nn.Linear(8, classes))
            torch.manual_seed(seed)
            self.classifier[-1] = nn.Linear(8, classes)

    def forward(self, images):
        return self.classifier(self.avgpool(self.features(images)).flatten(1))


def tiny_factory(num_classes=6, seed=314, pretrained=True):
    return TinyTransfer(num_classes, seed)


def synthetic_prepared(root):
    root.mkdir()
    rng = np.random.default_rng(46)
    np.save(root / "images_uint8.npy", rng.integers(0, 256, size=(24, 224, 224, 3), dtype=np.uint8))
    np.save(root / "labels.npy", np.tile(np.arange(6, dtype=np.int64), 4))
    roles = np.repeat(np.arange(4, dtype=np.uint8), 6)
    np.save(root / "roles.npy", roles)
    atomic_json(root / "index_manifest.json", {"labels": [f"class_{i}" for i in range(6)],
        "records": [{"array_index": i, "role": training.ROLE_NAMES[int(role)], "source_group_id": f"group-{i}",
                     "evaluation_group_id": f"component-{i}", "annotation_id": i, "image_id": i}
                    for i, role in enumerate(roles)]})
    atomic_json(root / "DATA_READINESS.json", {"training_ready": True, "scope": "Synthetic software fixture only"})
    return root


def smoke_config():
    result = json.loads((Path(__file__).resolve().parents[1] / "configs/garmentlab_deadline.json").read_text(encoding="utf-8"))
    result.update(device="cpu", epochs=2, batch_size=2, feature_batch_size=3,
                  stop_by_utc="2099-01-01T00:00:00Z", cpu_benchmark=False)
    return result


@pytest.fixture
def cpu_settings():
    previous, previous_mkldnn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(previous)
    torch.backends.mkldnn.enabled = previous_mkldnn


def test_six_task_train_then_finalize_smoke_freezes_before_assessment_and_never_releases_ready(tmp_path, monkeypatch, cpu_settings):
    prepared, out, config = synthetic_prepared(tmp_path / "prepared"), tmp_path / "out", smoke_config()
    assert training.train_study(prepared, out, config, model_factory=tiny_factory)["status"] == "ALL_TASKS_COMPLETE"
    original_infer = evaluate._infer
    assessed = []

    def check_freeze(model, images, indices, **kwargs):
        if np.any(np.asarray(indices) >= 18):
            assert (out / "evaluation/SELECTION_FREEZE.json").is_file()
            assessed.extend(np.asarray(indices).tolist())
        return original_infer(model, images, indices, **kwargs)

    monkeypatch.setattr(evaluate, "_infer", check_freeze)
    result = evaluate.finalize_study(out, prepared, config, model_factory=tiny_factory, smoke=True)
    assert result["status"] == "SMOKE_ONLY_COMPLETE" and result["task_count"] == 6
    assert result["ready_for_local_service"] is False
    assert result["deployment_seed"] == 314
    assert assessed and set(assessed) == set(range(18, 24))
    assert not (out / "release_candidate/release.json").exists()
    assert (out / "release_candidate/SMOKE_ONLY.json").is_file()
    assert result["gates"]["calibration"]["status"] == "fail"  # only 6 calibration examples, below preset30
    assert result["calibration"]["review_threshold"] is None
    assert len(result["assessment"]) == 6 and len(result["corruptions"]) == 9
    assert (out / "evaluation/figures/assessment_confusion.png").is_file()
    assert (out / "evaluation/figures/assessment_reliability.png").is_file()
    assert (out / "evaluation/figures/assessment_examples.jpg").is_file()
    assert len(result["task_costs"]) == 6
    assert result["gates"]["parity"]["status"] == "pass", result["gates"]["parity"]
    assert (out / "release_candidate/model.onnx").is_file()
    assert result == evaluate.finalize_study(out, prepared, config, model_factory=tiny_factory, smoke=True)
    with (out / result["artifacts"][0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate.finalize_study(out, prepared, config, model_factory=tiny_factory, smoke=True)


def test_incomplete_study_refuses_assessment_before_loading_any_evaluation_model(tmp_path, monkeypatch, cpu_settings):
    prepared, out, config = synthetic_prepared(tmp_path / "prepared"), tmp_path / "out", smoke_config()
    training.train_study(prepared, out, config, model_factory=tiny_factory)
    (out / "tasks/finetune_seed5772/COMPLETE.json").unlink()
    monkeypatch.setattr(evaluate, "_infer", lambda *args, **kwargs: pytest.fail("Incomplete study must not infer"))
    with pytest.raises(ValueError, match="Incomplete study"):
        evaluate.finalize_study(out, prepared, config, model_factory=tiny_factory, smoke=True)
    assert not (out / "evaluation/SELECTION_FREEZE.json").exists()


def test_changed_prepared_hash_binding_cannot_be_adopted_for_finalization(tmp_path, cpu_settings):
    prepared, out, config = synthetic_prepared(tmp_path / "prepared"), tmp_path / "out", smoke_config()
    training.train_study(prepared, out, config, model_factory=tiny_factory)
    index = json.loads((prepared / "index_manifest.json").read_text(encoding="utf-8"))
    index["records"][0]["annotation_id"] = 12345
    atomic_json(prepared / "index_manifest.json", index)
    with pytest.raises(ValueError, match="Study config or labels"):
        evaluate.finalize_study(out, prepared, config, model_factory=tiny_factory, smoke=True)


def test_fixture_factory_cannot_create_a_real_release(tmp_path):
    with pytest.raises(ValueError, match="smoke=True"):
        evaluate.finalize_study(tmp_path / "out", tmp_path / "prepared", smoke_config(), model_factory=tiny_factory)


def test_actual_mobilenet_random_weights_export_dynamic_batches(tmp_path, cpu_settings):
    # Exercise the deployed architecture/exporter, without training or downloading weights.
    model = training.build_model(num_classes=6, seed=314, pretrained=False).eval()
    rng = np.random.default_rng(314)
    data = {"images": rng.integers(0, 256, (3, 224, 224, 3), dtype=np.uint8)}
    config = {"onnx_max_abs_error": 1e-4, "cpu_benchmark": False}
    result = evaluate._export_and_benchmark(model, data, np.arange(3), tmp_path, config, smoke=True)
    assert result["status"] == "pass", result
    assert result["batch_sizes"] == [1, 3]
    assert result["top1_disagreements"] == 0
    assert result["max_abs_tolerance"] == 1e-4
    assert result["export_settings"]["optimize"] is False
    assert result["export_settings"]["onnxruntime_graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert result["runtime"]["torch"] == str(torch.__version__)
    # The failing trained-model export folded these layers before ORT execution.
    import onnx
    graph = onnx.load(tmp_path / "model.onnx")
    assert any(node.op_type == "BatchNormalization" for node in graph.graph.node)
