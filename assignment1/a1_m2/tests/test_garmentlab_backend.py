"""CPU-only regression checks for fresh-process evaluation backend/cache binding."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from garmentlab import backend, evaluate, training
from garmentlab.runtime import atomic_json, digest


@pytest.fixture(autouse=True)
def restore_backend_flags():
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    yield
    torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
    torch.backends.cudnn.benchmark = previous["benchmark"]
    torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]


def incorrect_fresh_process_flags():
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class TinyInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return images.mean((-2, -1))[:, :2]


def data_fixture():
    return {"images": np.full((4, 224, 224, 3), 100, dtype=np.uint8),
            "labels": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "roles": np.asarray([0, 1, 2, 3], dtype=np.uint8),
            "records": [{"role": role, "source_group_id": f"fixture-{i}"}
                        for i, role in enumerate(training.ROLE_NAMES)],
            "classnames": ["first", "second"], "bindings": {"fixture": "a" * 64}}


def test_configure_restores_exact_training_flags_from_wrong_process_defaults():
    incorrect_fresh_process_flags()
    result = backend.configure_evaluation_backend()
    assert result["flags"]["deterministic_algorithms"] is True
    assert result["flags"]["deterministic_warn_only"] is False
    assert result["flags"]["cudnn_benchmark"] is False
    assert result["flags"]["cuda_matmul_allow_tf32"] is False
    assert result["flags"]["cudnn_allow_tf32"] is False
    assert result["flags"]["float32_matmul_precision"] == "highest"
    # train_study does not set this separate legacy flag; do not invent a policy change.
    assert result["flags"]["cudnn_deterministic"] is False
    assert result["runtime"]["torch"] == str(torch.__version__)
    assert result["runtime"]["cuda"] == str(torch.version.cuda)
    assert "cudnn" in result["runtime"] and len(result["backend_source_sha256"]) == 64


def test_snapshot_is_read_only_and_json_serializable():
    incorrect_fresh_process_flags()
    first = backend.snapshot_evaluation_backend("cpu")
    assert first == backend.snapshot_evaluation_backend(torch.device("cpu"))
    assert first["flags"]["cuda_matmul_allow_tf32"] is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    json.dumps(first, allow_nan=False)


def test_old_wrong_backend_cache_is_refused_after_configuration(tmp_path):
    incorrect_fresh_process_flags()
    model, data = TinyInference().eval(), data_fixture()
    evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    saved_hash = digest(tmp_path / "logits.npz")
    backend.configure_evaluation_backend()
    with pytest.raises(ValueError, match="integrity/binding"):
        evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    assert model.calls == 1 and digest(tmp_path / "logits.npz") == saved_hash


def test_legacy_schema_one_cache_cannot_silently_reuse(tmp_path):
    backend.configure_evaluation_backend()
    model, data = TinyInference().eval(), data_fixture()
    evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    receipt = json.loads((tmp_path / "LOGITS_RECEIPT.json").read_text(encoding="utf-8"))
    receipt["binding"]["schema"] = 1
    receipt["binding"].pop("backend")
    receipt["binding"].pop("batch_size")
    atomic_json(tmp_path / "LOGITS_RECEIPT.json", receipt)
    with pytest.raises(ValueError, match="integrity/binding"):
        evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    assert model.calls == 1


def test_cache_binds_batch_size_and_allows_identical_reuse(tmp_path):
    backend.configure_evaluation_backend()
    model, data = TinyInference().eval(), data_fixture()
    first = evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    second = evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    np.testing.assert_array_equal(first[0], second[0])
    assert model.calls == 1
    with pytest.raises(ValueError, match="integrity/binding"):
        evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=1)
    receipt = json.loads((tmp_path / "LOGITS_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["binding"]["schema"] == 2 and receipt["binding"]["batch_size"] == 2
    assert receipt["binding"]["backend"]["device"] == "cpu"


def test_cache_runtime_change_is_refused(tmp_path, monkeypatch):
    backend.configure_evaluation_backend()
    model, data = TinyInference().eval(), data_fixture()
    evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    original = evaluate.snapshot_evaluation_backend
    def changed_runtime(device=None):
        result = copy.deepcopy(original(device))
        result["runtime"]["torch"] = "different-runtime"
        return result
    monkeypatch.setattr(evaluate, "snapshot_evaluation_backend", changed_runtime)
    with pytest.raises(ValueError, match="integrity/binding"):
        evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64, device="cpu", batch_size=2)
    assert model.calls == 1


def _intercept_finalizer(monkeypatch):
    data = data_fixture()
    task = {"task_id": "frozen_seed314", "seed": 314, "recipe": "frozen",
            "checkpoint": "synthetic-unused-path.pt", "weights_sha256": "b" * 64}
    monkeypatch.setattr(training, "load_prepared", lambda _: data)
    monkeypatch.setattr(evaluate, "_verify_tasks", lambda *args: ("binding", {"weight_source": "injected_software_smoke_only"}, [task]))
    model = nn.Sequential(nn.Linear(2, 2))
    model.classifier = nn.Sequential(nn.Linear(2, 2))
    monkeypatch.setattr(evaluate, "_load_model", lambda *args: model)
    return data


def test_finalizer_configures_and_records_backend_before_first_cache_lookup(tmp_path, monkeypatch):
    incorrect_fresh_process_flags()
    _intercept_finalizer(monkeypatch)
    out = tmp_path / "run"
    def checked_cache(*args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark
        assert (out / "EVALUATION_BACKEND.json").is_file()
        raise RuntimeError("cache lookup intercepted before any model evaluation")
    monkeypatch.setattr(evaluate, "_cached_logits", checked_cache)
    with pytest.raises(RuntimeError, match="cache lookup intercepted"):
        evaluate.finalize_study(out, tmp_path, {"device": "cpu", "batch_size": 2},
                                model_factory=lambda **kwargs: None, smoke=True)
    recorded = json.loads((out / "EVALUATION_BACKEND.json").read_text(encoding="utf-8"))
    assert recorded == backend.snapshot_evaluation_backend()


def test_finalizer_cannot_overwrite_a_different_backend_receipt(tmp_path, monkeypatch):
    _intercept_finalizer(monkeypatch)
    out = tmp_path / "run"
    receipt = backend.configure_evaluation_backend()
    receipt["runtime"]["torch"] = "previous-runtime"
    atomic_json(out / "EVALUATION_BACKEND.json", receipt)
    before = (out / "EVALUATION_BACKEND.json").read_bytes()
    monkeypatch.setattr(evaluate, "_cached_logits", lambda *args, **kwargs: pytest.fail("Must not look up a cache under a mismatching backend"))
    with pytest.raises(ValueError, match="backend/runtime differs"):
        evaluate.finalize_study(out, tmp_path, {"device": "cpu", "batch_size": 2},
                                model_factory=lambda **kwargs: None, smoke=True)
    assert (out / "EVALUATION_BACKEND.json").read_bytes() == before


def test_inference_deadline_stop_commits_no_partial_logits(tmp_path):
    backend.configure_evaluation_backend()
    model, data = TinyInference().eval(), data_fixture()
    with pytest.raises(TimeoutError, match="deadline/budget"):
        evaluate._cached_logits(tmp_path, model, data, "selection", "b" * 64,
                                device="cpu", batch_size=2, stop_check=lambda: True)
    assert model.calls == 0
    assert not (tmp_path / "logits.npz").exists()
    assert not (tmp_path / "LOGITS_RECEIPT.json").exists()
