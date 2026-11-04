"""Tiny synthetic software checks, never full MobileNet training or quality evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from garmentlab import training
from garmentlab.runtime import atomic_json, digest


class TinyTransfer(nn.Module):
    def __init__(self, num_classes, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(999)  # Simulate the same pretrained encoder for every seed.
            self.features = nn.Sequential(nn.Conv2d(3, 4, 1, stride=16), nn.BatchNorm2d(4), nn.ReLU())
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Linear(4, 8), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(8, num_classes))
            torch.manual_seed(seed)
            self.classifier[-1] = nn.Linear(8, num_classes)

    def forward(self, images):
        return self.classifier(self.avgpool(self.features(images)).flatten(1))


def tiny_factory(num_classes=6, seed=314, pretrained=True):
    return TinyTransfer(num_classes, seed)


def config():
    path = Path(training.__file__).resolve().parents[1] / "configs/garmentlab_deadline.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result.update(device="cpu", epochs=2, batch_size=2, feature_batch_size=3,
                  stop_by_utc="2099-01-01T00:00:00Z", max_active_seconds=36000)
    return result


def make_prepared(root):
    root.mkdir()
    rng = np.random.default_rng(46)
    np.save(root / "images_uint8.npy", rng.integers(0, 256, size=(24, 224, 224, 3), dtype=np.uint8))
    np.save(root / "labels.npy", np.tile(np.arange(6, dtype=np.int64), 4))
    roles = np.repeat(np.arange(4, dtype=np.uint8), 6)
    np.save(root / "roles.npy", roles)
    atomic_json(root / "index_manifest.json", {"labels": [f"class_{i}" for i in range(6)],
                "records": [{"array_index": i, "role": training.ROLE_NAMES[int(role)],
                             "source_group_id": f"group-{i}"} for i, role in enumerate(roles)]})
    atomic_json(root / "DATA_READINESS.json", {"training_ready": True,
                "scope": "Synthetic software fixture only; no data/model quality claim"})
    return root


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    previous_mkldnn = torch.backends.mkldnn.enabled
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False  # Avoid this Windows CPU backend's tiny-BN backward crash.
    yield
    torch.set_num_threads(previous)
    torch.backends.mkldnn.enabled = previous_mkldnn


def test_paired_initial_head_and_frozen_parameter_bn_policy():
    frozen, finetune = tiny_factory(seed=314), tiny_factory(seed=314)
    assert torch.equal(frozen.classifier[-1].weight, finetune.classifier[-1].weight)
    frozen_opt, _ = training.configure_optimizer(frozen, "frozen", config())
    fine_opt, _ = training.configure_optimizer(finetune, "finetune", config())
    assert len(frozen_opt.param_groups) == 1 and len(fine_opt.param_groups) == 2
    assert not any(parameter.requires_grad for parameter in frozen.features.parameters())
    assert all(parameter.requires_grad for parameter in finetune.parameters())
    for model, recipe in ((frozen, "frozen"), (finetune, "finetune")):
        training.set_training_mode(model, recipe)
        assert not model.features[1].training
        assert model.classifier[2].training  # Shared head dropout remains active.


def test_feature_path_matches_full_model_and_flip_path(tmp_path):
    data = training.load_prepared(make_prepared(tmp_path / "prepared"))
    model = tiny_factory().eval()
    rows = np.array([0, 2, 4])
    for flip in (False, True):
        features = training.extract_features(model, data["images"], batch_size=2, indices=rows, flip=flip)
        batch = training.prepare_batch(data["images"], rows, flip_mask=[flip] * len(rows))
        with torch.inference_mode():
            torch.testing.assert_close(model.classifier[2:](torch.from_numpy(features)), model(batch), atol=1e-6, rtol=1e-6)


def test_six_tiny_tasks_write_bound_safe_checkpoints_and_never_read_eval_roles(tmp_path, monkeypatch):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    actual = training.prepare_batch
    seen = []
    def fit_selection_only(images, indices, device="cpu", flip_mask=None):
        seen.extend(np.asarray(indices).tolist())
        assert np.asarray(indices).max() < 12  # Never forward calibration or assessment.
        return actual(images, indices, device, flip_mask)
    monkeypatch.setattr(training, "prepare_batch", fit_selection_only)
    result = training.train_study(prepared, out, config(), model_factory=tiny_factory)
    assert result["status"] == "ALL_TASKS_COMPLETE" and result["completed"] == 6
    assert seen and max(seen) < 12
    binding = digest(out / "study_manifest.json")
    for seed in config()["seeds"]:
        heads = []
        for recipe in ("frozen", "finetune"):
            folder = out / "tasks" / f"{recipe}_seed{seed}"
            checkpoint = torch.load(folder / "epoch_state.pt", map_location="cpu", weights_only=True)
            assert checkpoint["binding"] == binding and checkpoint["epoch"] == 2
            assert checkpoint["task"]["recipe"] == recipe
            assert len(checkpoint["history"]) == 2
            assert checkpoint["scheduler"]["last_epoch"] == 2
            assert int(checkpoint["model_state_dict"]["features.1.num_batches_tracked"]) == 0
            assert torch.count_nonzero(checkpoint["model_state_dict"]["features.1.running_mean"]) == 0
            marker = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
            assert marker["files"]["best.pt"]["sha256"] == digest(folder / "best.pt")
            heads.append(marker["initial_head_sha256"])
        assert heads[0] == heads[1]


def test_epoch_resume_reproduces_uninterrupted_weights_and_metric_history(tmp_path):
    prepared = make_prepared(tmp_path / "prepared")
    uninterrupted, resumed = tmp_path / "whole", tmp_path / "resume"
    settings = config()
    settings["seeds"] = [314]
    assert training.train_study(prepared, uninterrupted, settings, model_factory=tiny_factory)["status"] == "ALL_TASKS_COMPLETE"
    def stop_after_epoch_commit():
        return (resumed / "tasks/frozen_seed314/epoch_state.pt").exists()
    partial = training.train_study(prepared, resumed, settings, model_factory=tiny_factory,
                                   stop_check=stop_after_epoch_commit)
    assert partial["status"] == "INCOMPLETE"
    assert not (resumed / "tasks/frozen_seed314/COMPLETE.json").exists()
    final = training.train_study(prepared, resumed, settings, model_factory=tiny_factory)
    assert final["status"] == "ALL_TASKS_COMPLETE"
    for recipe in ("frozen", "finetune"):
        left = torch.load(uninterrupted / f"tasks/{recipe}_seed314/epoch_state.pt", weights_only=True)
        right = torch.load(resumed / f"tasks/{recipe}_seed314/epoch_state.pt", weights_only=True)
        for key in left["model_state_dict"]:
            assert torch.equal(left["model_state_dict"][key], right["model_state_dict"][key])
        for a, b in zip(left["history"], right["history"]):
            assert a["train"] == b["train"] and a["selection"] == b["selection"] and a["lr"] == b["lr"]


def test_completed_artifact_tamper_and_changed_config_fail_closed(tmp_path):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    settings = config()
    settings["seeds"] = [314]
    training.train_study(prepared, out, settings, model_factory=tiny_factory)
    changed = copy.deepcopy(settings)
    changed["head_lr"] = 0.01
    with pytest.raises(ValueError, match="binding differs"):
        training.train_study(prepared, out, changed, model_factory=tiny_factory)
    with (out / "tasks/frozen_seed314/best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        training.train_study(prepared, out, settings, model_factory=tiny_factory)


def test_partial_feature_cache_resumes_and_detects_committed_chunk_tampering(tmp_path):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    def after_cache_batch():
        return (out / "feature_cache/progress.json").exists()
    partial = training.train_study(prepared, out, config(), stop_check=after_cache_batch, model_factory=tiny_factory)
    assert partial["status"] == "INCOMPLETE" and partial["completed"] == 0
    assert not (out / "feature_cache/COMPLETE.json").exists()
    array = np.load(out / "feature_cache/original.npy", mmap_mode="r+")
    array[0, 0] += 1
    array.flush()
    del array
    with pytest.raises(ValueError, match="committed feature-cache chunk changed"):
        training.train_study(prepared, out, config(), model_factory=tiny_factory)


def test_valid_partial_cache_resumes_without_replacing_committed_chunks(tmp_path):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    settings = config()
    settings["seeds"] = [314]
    partial = training.train_study(prepared, out, settings, model_factory=tiny_factory,
                                   stop_check=lambda: (out / "feature_cache/progress.json").exists())
    assert partial["status"] == "INCOMPLETE"
    first_chunk = json.loads((out / "feature_cache/progress.json").read_text())["original"]["chunks"][0]
    assert training.train_study(prepared, out, settings, model_factory=tiny_factory)["status"] == "ALL_TASKS_COMPLETE"
    progress = json.loads((out / "feature_cache/progress.json").read_text())
    assert progress["original"]["chunks"][0] == first_chunk
    assert progress["original"]["completed"] == 12 and progress["flipped"]["completed"] == 6


def test_verified_completed_study_reuses_evidence_without_another_model(tmp_path):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    settings = config()
    settings["seeds"] = [314]
    training.train_study(prepared, out, settings, model_factory=tiny_factory)
    def no_new_model(**kwargs):
        raise AssertionError("Completed evidence must not start another model")
    reused = training.train_study(prepared, out, settings, model_factory=no_new_model, stop_check=lambda: True)
    assert reused["status"] == "ALL_TASKS_COMPLETE" and reused["completed"] == 2


def test_changed_runtime_environment_refuses_resume(tmp_path, monkeypatch):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    settings = config()
    settings["seeds"] = [314]
    training.train_study(prepared, out, settings, model_factory=tiny_factory)
    actual = training._environment
    monkeypatch.setattr(training, "_environment", lambda config: {**actual(config), "torch": "changed-runtime"})
    with pytest.raises(ValueError, match="binding differs"):
        training.train_study(prepared, out, settings, model_factory=tiny_factory)


def test_deadline_stop_does_not_create_any_training_task(tmp_path):
    prepared, out = make_prepared(tmp_path / "prepared"), tmp_path / "out"
    settings = config()
    settings["stop_by_utc"] = "2000-01-01T00:00:00Z"
    result = training.train_study(prepared, out, settings, model_factory=tiny_factory)
    assert result["status"] == "INCOMPLETE" and result["completed"] == 0
    assert not (out / "tasks").exists()
    assert not (out / "feature_cache").exists()


def test_prepared_roles_and_group_overlap_are_not_silently_adopted(tmp_path):
    prepared = make_prepared(tmp_path / "prepared")
    index = json.loads((prepared / "index_manifest.json").read_text(encoding="utf-8"))
    index["records"][6]["source_group_id"] = index["records"][0]["source_group_id"]
    atomic_json(prepared / "index_manifest.json", index)
    with pytest.raises(ValueError, match="source group"):
        training.load_prepared(prepared)


def test_full_cpu_training_without_a_tiny_injected_model_is_blocked(tmp_path):
    with pytest.raises(ValueError, match="CPU training"):
        training.train_study(tmp_path, tmp_path / "out", config())


def test_real_cuda_training_refuses_a_non_a100_before_loading_data(tmp_path, monkeypatch):
    settings = config()
    settings["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA T4")
    with pytest.raises(RuntimeError, match="A100"):
        training.train_study(tmp_path, tmp_path / "out", settings)
