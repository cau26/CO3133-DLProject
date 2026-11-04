"""Prospective confirmation software checks; synthetic training only."""
import copy
import csv
import hashlib
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import a1confirm.study as study
from a1opt.storage import atomic_bytes, atomic_json, json_hash, load_json, sha256_file

PUBLICATION = study.ROOT / "output/a1_final/data/publication_data.json"


def _synthetic_inputs(arm="C0"):
    generator = torch.Generator().manual_seed(765)
    dataset = TensorDataset(torch.randn(40, 1, 28, 28, generator=generator), torch.arange(40) % 10)
    task = {"task_id": f"linear_{arm}_seed1009", "family": "linear", "arm": arm,
            "candidate_id": "C0" if arm == "C0" else "C1", "phase": "synthetic_confirmation",
            "seed": 1009, "epochs": 3,
            "config": {"batch_size": 7, "cpu_threads": 1, "num_workers": 0,
                       "optimizer": "adam", "learning_rate": .001, "weight_decay": 0,
                       "gradient_clip": 1, "schedule": "cosine"}}
    return task, dataset, list(range(30)), list(range(30, 40)), {"scope": "synthetic_only"}


def _train(out, **kwargs):
    task, dataset, train, val, binding = _synthetic_inputs()
    return study.train_trajectory(out, task, dataset, train, val, binding, "cpu", **kwargs)


def test_frozen_protocol_uses_real_C0_and_selected_without_sampling():
    protocol = study.make_protocol(PUBLICATION)
    assert protocol["seeds"] == [1009, 2027, 3037]
    assert protocol["families"] == list(study.FAMILIES)
    assert protocol["primary_family"] == "cnn"
    assert set(protocol["secondary_families"]) == set(study.FAMILIES) - {"cnn"}
    assert protocol["recipes"]["cnn"]["selected"]["candidate_id"] == "C5"
    assert protocol["recipes"]["cnn"]["C0"]["config"]["optimizer"] == "adamw"
    assert protocol["recipes"]["cnn"]["C0"]["config"]["dropout"] == .2
    assert protocol["recipes"]["cnn"]["selected"]["config"]["optimizer"] == "adamw"
    tasks = study.tasks_for({"protocol": protocol})
    assert len(tasks) == 30 and len({task["task_id"] for task in tasks}) == 30
    for family in study.FAMILIES:
        for seed in protocol["seeds"]:
            pair = [t for t in tasks if t["family"] == family and t["seed"] == seed]
            assert {t["arm"] for t in pair} == {"C0", "selected"}
            assert all(t["epochs"] == 60 for t in pair)
    assert [t["arm"] for t in tasks[:4]] == ["C0", "selected", "selected", "C0"]
    assert protocol["official_test_evaluated"] is False


@pytest.mark.parametrize("seeds", [(42, 1009, 2027), (1009, 1009, 2027), (1009, 2027), (True, 2027, 3037)])
def test_reused_or_invalid_seeds_fail_before_training(seeds):
    with pytest.raises(ValueError, match="fresh integer seeds"):
        study.make_protocol(PUBLICATION, seeds=seeds)


def test_recipe_tampering_is_detected(tmp_path):
    publication = load_json(PUBLICATION)
    publication["selected_recipes"][0]["config"]["learning_rate"] *= 2
    path = tmp_path / "recipes.json"
    import json
    path.write_text(json.dumps(publication), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe fingerprint differs"):
        study.make_protocol(path)


def test_shared_trajectory_exports_three_policies_and_fixed_last_weights(tmp_path):
    record = _train(tmp_path)
    assert record["same_training_trajectory_for_all_policies"]
    assert {row["policy"] for row in record["cells"]} == {"ce", "f1", "final"}
    engine = load_json(tmp_path / "engine/COMPLETE.json")
    by_policy = {row["policy"]: row for row in record["cells"]}
    assert by_policy["ce"]["checkpoint_epoch"] == engine["metrics"]["best_ce_epoch"]
    assert by_policy["f1"]["checkpoint_epoch"] == engine["metrics"]["best_f1_epoch"]
    assert by_policy["final"]["checkpoint_epoch"] == 3
    pointer = load_json(tmp_path / "engine/COMMIT.json")
    state = torch.load(tmp_path / "engine" / pointer["state_file"], weights_only=False)
    final = torch.load(tmp_path / "final_epoch.pt", weights_only=False)
    assert all(torch.equal(weight, final["model_state_dict"][name]) for name, weight in state["model"].items())
    assert final["selection_rule"] == "fixed_final_epoch_locked_before_training"
    for row in record["cells"]:
        with (tmp_path / f"predictions_{row['policy']}.csv").open(newline="", encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        assert len(predictions) == 10
        assert {int(p["original_id"]) for p in predictions} == set(range(30, 40))
        assert sum(p["true"] == p["pred"] for p in predictions) / 10 == row["accuracy"]
        assert row["evaluation_scope"] == "previously_seen_old_validation_exploratory"


def test_resume_and_completion_skip_preserve_all_three_checkpoint_weights(tmp_path):
    direct = _train(tmp_path / "direct")
    assert _train(tmp_path / "resume", stop_at=time.monotonic() - 1) is None
    with pytest.raises(RuntimeError, match="Injected test interruption"):
        _train(tmp_path / "resume", fail_after_epoch=1)
    resumed = _train(tmp_path / "resume")
    for policy in study.POLICIES:
        left = next(row for row in direct["cells"] if row["policy"] == policy)
        right = next(row for row in resumed["cells"] if row["policy"] == policy)
        assert left["macro_f1"] == right["macro_f1"]
        assert left["checkpoint_epoch"] == right["checkpoint_epoch"]
        a = torch.load(tmp_path / "direct" / left["checkpoint"], weights_only=False)["model_state_dict"]
        b = torch.load(tmp_path / "resume" / right["checkpoint"], weights_only=False)["model_state_dict"]
        assert all(torch.equal(value, b[key]) for key, value in a.items())
    before = (tmp_path / "resume/TRAJECTORY_COMPLETE.json").read_bytes()
    assert _train(tmp_path / "resume") == resumed
    assert (tmp_path / "resume/TRAJECTORY_COMPLETE.json").read_bytes() == before


def test_changed_prediction_evidence_fails_closed(tmp_path):
    _train(tmp_path)
    (tmp_path / "predictions_ce.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact missing/changed"):
        _train(tmp_path)


def _cells():
    cells = []
    for seed, shift in ((1009, 0), (2027, .001), (3037, -.001)):
        for arm, policy_values in (("C0", {"ce": .90, "f1": .91, "final": .88}),
                                   ("selected", {"ce": .92, "f1": .94, "final": .93})):
            for policy, value in policy_values.items():
                cells.append({"family": "cnn", "seed": seed, "arm": arm, "policy": policy,
                              "macro_f1": value + shift, "accuracy": value + shift, "loss": 1 - value})
    return cells


def test_factor_contrasts_are_paired_and_do_not_conflate_selector_and_recipe():
    summary = study.summarize_cells(_cells())
    contrasts = {row["contrast"]: row for row in summary["paired_contrasts"]}
    expected = {"recipe_fixed_final_primary": .05, "recipe_at_ce": .02, "recipe_at_f1": .03,
                "selector_within_C0": .01, "selector_within_selected": .02, "recipe_selector_interaction": .01}
    for name, value in expected.items():
        assert contrasts[name]["macro_f1_delta_mean"] == pytest.approx(value)
        assert contrasts[name]["n_paired_seeds"] == 3
        assert contrasts[name]["macro_f1_delta_sample_sd"] == pytest.approx(0, abs=1e-15)
    partial = study.summarize_cells([row for row in _cells() if not
                                    (row["seed"] == 3037 and row["arm"] == "selected")])
    assert all(row["n_paired_seeds"] == 2 for row in partial["paired_contrasts"])
    assert summary["official_test_evaluated"] is False


def _plan_without_download(tmp_path, monkeypatch):
    class TrainPartition:
        def __len__(self):
            return 60000
    monkeypatch.setattr(study, "load_training_data", lambda *args: (TrainPartition(), None))
    monkeypatch.setattr(study, "environment_identity", lambda device: {"device_type": "cpu", "software_test": True})
    return study.prepare(tmp_path, PUBLICATION, "unused", "cpu")


def test_manifest_resume_locks_config_and_split_and_partial_export_is_honest(tmp_path, monkeypatch):
    manifest = _plan_without_download(tmp_path / "run", monkeypatch)
    assert _plan_without_download(tmp_path / "run", monkeypatch) == manifest
    with pytest.raises(ValueError, match="protocol/environment differs"):
        study.prepare(tmp_path / "run", PUBLICATION, "unused", "cpu", seeds=(1009, 2027, 4051))
    progress = study.status(tmp_path / "run")
    assert progress["status"] == "INCOMPLETE" and progress["completed_trajectories"] == 0
    receipt = study.export_results(tmp_path / "run")
    assert receipt["status"] == "INCOMPLETE" and receipt["maximum_trajectories"] == 30
    with zipfile.ZipFile(receipt["archive"]) as zipped:
        import json
        hashes = json.loads(zipped.read("artifact_manifest.json"))
        assert zipped.testzip() is None
        assert len(hashes) == receipt["payload_count"]
        assert all(hashlib.sha256(zipped.read(name)).hexdigest() == digest for name, digest in hashes.items())
        assert "source_snapshot/a1confirm/study.py" in hashes
        assert "CONFIRMATION_OWNER.json" not in zipped.namelist()
    (tmp_path / "run/split_ids.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="split artifact changed"):
        study.status(tmp_path / "run")


def test_training_rejects_cpu_before_any_data_access(tmp_path):
    with pytest.raises(RuntimeError, match="A100"):
        study.run(tmp_path, "must-not-download", "cpu")


def test_live_owner_blocks_concurrent_writer(tmp_path):
    with study.RunLease(tmp_path):
        with pytest.raises(RuntimeError, match="Another runtime"):
            with study.RunLease(tmp_path):
                pass
    assert load_json(tmp_path / "CONFIRMATION_OWNER.json")["active"] is False


def test_historical_folder_is_not_mutated_by_confirmation_lease(tmp_path):
    marker = tmp_path / "study_manifest.json"
    marker.write_text('{"historical":true}', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="historical experiment folder"):
        with study.RunLease(tmp_path):
            pass
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_controller_partial_resume_full_roster_and_cached_completion(tmp_path, monkeypatch):
    # A deliberately tiny synthetic manifest exercises controller transitions;
    # public prepare() still enforces the real 60-epoch, 54k/6k contract.
    protocol = study.make_protocol(PUBLICATION, families=("linear",))
    protocol["epochs"] = 3
    for recipe in protocol["recipes"]["linear"].values():
        recipe["config"].update(epochs=3, batch_size=7, cpu_threads=1)
    stream = io.BytesIO()
    np.savez_compressed(stream, train=np.arange(30), validation=np.arange(30, 40))
    atomic_bytes(tmp_path / "split_ids.npz", stream.getvalue())
    environment = {"device_type": "cuda", "scope": "synthetic_controller_mock"}
    manifest = {"schema_version": 1, "protocol": protocol, "source": study.source_identity(),
                "environment": environment, "split_file_sha256": sha256_file(tmp_path / "split_ids.npz")}
    manifest["manifest_sha256"] = json_hash(manifest)
    atomic_json(tmp_path / "confirmation_manifest.json", manifest)
    atomic_json(tmp_path / "clock.json", {"budget_seconds": protocol["time_budget_seconds"], "active_run_seconds": 0})
    _, dataset, _, _, _ = _synthetic_inputs()
    monkeypatch.setattr(study, "load_training_data", lambda *args: (dataset, None))
    monkeypatch.setattr(study, "environment_identity", lambda *args: environment)
    class ControllerTorchProxy:
        class cuda:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def get_device_name(*args):
                return "synthetic A100 controller gate"
        def __getattr__(self, name):
            return getattr(torch, name)
    monkeypatch.setattr(study, "torch", ControllerTorchProxy())
    original_train = study.train_trajectory
    def cpu_training(out, task, data, train, val, binding, device, **kwargs):
        return original_train(out, task, data, train, val, binding, "cpu", **kwargs)
    monkeypatch.setattr(study, "train_trajectory", cpu_training)
    partial = study.run(tmp_path, "unused", "cuda", max_tasks=1)
    assert partial["status"] == "INCOMPLETE" and partial["completed_trajectories"] == 1
    first = next((tmp_path / "tasks").glob("*/TRAJECTORY_COMPLETE.json"))
    first_bytes = first.read_bytes()
    assert not (tmp_path / "EVALUATION_LOCK.json").exists()
    complete = study.run(tmp_path, "unused", "cuda")
    assert complete["status"] == "COMPLETE" and complete["completed_trajectories"] == 6
    lock = load_json(tmp_path / "EVALUATION_LOCK.json")
    assert len(lock["checkpoints"]) == 18 and lock["primary_policy"] == "final"
    assert lock["lock_sha256"] == json_hash({key: value for key, value in lock.items() if key != "lock_sha256"})
    assert all(sha256_file(tmp_path / row["checkpoint"]) == row["checkpoint_sha256"] for row in lock["checkpoints"])
    assert first.read_bytes() == first_bytes
    monkeypatch.setattr(study, "train_trajectory", lambda *args, **kwargs: pytest.fail("Completed study retrained."))
    assert study.run(tmp_path, "unused", "cuda")["status"] == "COMPLETE"
    (tmp_path / "EVALUATION_LOCK.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completion artifact changed"):
        study.status(tmp_path)
