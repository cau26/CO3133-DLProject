"""Tiny synthetic checks only: these never download or train Fashion-MNIST."""
import copy
import time

import pytest
import torch
from torch.utils.data import TensorDataset

from a1opt.engine import train_task
from a1opt.storage import atomic_json, load_json, verify_complete


def _inputs():
    generator = torch.Generator().manual_seed(27)
    data = TensorDataset(torch.randn(40, 1, 28, 28, generator=generator),
                         torch.arange(40) % 10)
    task = {"task_id": "synthetic-linear", "family": "linear", "phase": "synthetic_test",
            "seed": 113, "epochs": 3,
            "config": {"batch_size": 5, "cpu_threads": 1, "num_workers": 0,
                       "optimizer": "adam", "learning_rate": .001,
                       "weight_decay": 0.0, "gradient_clip": 1.0, "schedule": "cosine"}}
    return task, data, list(range(30)), list(range(30, 40)), {"source": "synthetic-test-only"}


def _run(out, **kwargs):
    task, data, train_ids, val_ids, binding = _inputs()
    return train_task(out, task, data, data, train_ids, val_ids, binding, "cpu", **kwargs)


def _weights(out):
    return torch.load(out / "best_f1.pt", weights_only=False)["model_state_dict"]


def test_epoch_zero_and_resume_match_uninterrupted(tmp_path):
    direct = _run(tmp_path / "direct")
    beats = []
    assert _run(tmp_path / "resumed", stop_at=time.monotonic() - 1,
                heartbeat=lambda: beats.append(1)) is None
    assert load_json(tmp_path / "resumed" / "COMMIT.json")["epoch"] == 0
    with pytest.raises(RuntimeError, match="Injected test interruption"):
        _run(tmp_path / "resumed", fail_after_epoch=1)
    resumed = _run(tmp_path / "resumed")
    for key, value in _weights(tmp_path / "direct").items():
        assert torch.equal(value, _weights(tmp_path / "resumed")[key])
    for key in ("best_f1_epoch", "best_ce_epoch", "best_f1_metrics", "best_ce_metrics"):
        assert direct["metrics"][key] == resumed["metrics"][key]
    assert resumed["metrics"]["optimizer_updates"] == 18
    assert beats == [1]
    before = (tmp_path / "resumed" / "COMPLETE.json").read_bytes()
    assert _run(tmp_path / "resumed") == resumed
    assert (tmp_path / "resumed" / "COMPLETE.json").read_bytes() == before


def test_corrupted_current_recovers_only_previous_commit(tmp_path):
    direct = _run(tmp_path / "direct")
    with pytest.raises(RuntimeError, match="Injected test interruption"):
        _run(tmp_path / "recover", fail_after_epoch=2)
    pointer = load_json(tmp_path / "recover" / "COMMIT.json")
    (tmp_path / "recover" / pointer["state_file"]).write_bytes(b"corrupted")
    recovered = _run(tmp_path / "recover")
    receipt = load_json(tmp_path / "recover" / "RECOVERY.json")
    assert receipt["epoch"] == 1
    assert direct["metrics"]["best_f1_metrics"] == recovered["metrics"]["best_f1_metrics"]
    for key, value in _weights(tmp_path / "direct").items():
        assert torch.equal(value, _weights(tmp_path / "recover")[key])


def test_integrity_and_binding_are_enforced(tmp_path):
    complete = _run(tmp_path / "complete")
    assert complete["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="binding differs"):
        verify_complete(tmp_path / "complete", {"source": "other"})
    (tmp_path / "complete" / "predictions.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        _run(tmp_path / "complete")


def test_no_uncommitted_checkpoint_is_loaded(tmp_path):
    folder = tmp_path / "orphan"
    folder.mkdir()
    torch.save({"not": "committed"}, folder / "state-0000.pt")
    with pytest.raises(ValueError, match="No valid committed transaction"):
        _run(folder)


def test_immutable_json_and_overlapping_split(tmp_path):
    path = tmp_path / "immutable.json"
    atomic_json(path, {"value": 1}, immutable=True)
    atomic_json(path, {"value": 1}, immutable=True)
    with pytest.raises(ValueError, match="Immutable document differs"):
        atomic_json(path, {"value": 2}, immutable=True)
    task, data, train_ids, _, binding = _inputs()
    with pytest.raises(ValueError, match="overlap"):
        train_task(tmp_path / "bad", task, data, data, train_ids, [0, 31], binding, "cpu")


def test_final_phase_times_selected_weights_on_validation_only(tmp_path,monkeypatch):
    import a1.train
    task,data,train_ids,val_ids,binding=_inputs()
    task['phase']='final';task['epochs']=1
    calls=[]
    def fake_measure(model,batch,device,warmup,repeats):
        assert model.training is False
        assert warmup==10 and repeats==50 and len(batch)==5
        calls.append(1)
        return {'batch_size':5,'batch_ms':1.0,'per_image_ms_amortized':.2}
    monkeypatch.setattr(a1.train,'measure_inference',fake_measure)
    result=train_task(tmp_path/'final',task,data,data,train_ids,val_ids,binding,'cpu')
    assert result['metrics']['inference']['batch_ms']==1.0 and calls==[1]
