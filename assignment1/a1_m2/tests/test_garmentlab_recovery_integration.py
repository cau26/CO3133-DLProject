"""Real finalizer integration using six tiny synthetic CPU fits only."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest
import torch

from garmentlab import evaluate, recovery, training
from garmentlab.runtime import atomic_json, digest, export_results


def _helpers():
    path = Path(__file__).with_name("test_garmentlab_evaluate.py")
    spec = importlib.util.spec_from_file_location("_garmentlab_evaluate_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_fixture_settings():
    settings = (torch.get_num_threads(), torch.backends.mkldnn.enabled,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    yield
    torch.set_num_threads(settings[0])
    torch.backends.mkldnn.enabled = settings[1]
    torch.use_deterministic_algorithms(settings[2], warn_only=settings[3])
    torch.backends.cudnn.benchmark = settings[4]
    torch.backends.cuda.matmul.allow_tf32 = settings[5]
    torch.backends.cudnn.allow_tf32 = settings[6]


def test_recovery_real_finalize_and_resume(tmp_path, monkeypatch, cpu_fixture_settings):
    helpers = _helpers()
    prepared = helpers.synthetic_prepared(tmp_path / "prepared")
    index_path = prepared / "index_manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_hashes = {key: letter * 64 for key, letter in zip(
        ("train_metadata", "assessment_metadata", "train_archive", "assessment_archive"), "abcd")}
    binding = {"source_sha256": source_hashes, "resolution": 224,
               "near_duplicate_distance": 2, "preparation_source_sha256": "e" * 64}
    index["binding"] = binding
    atomic_json(index_path, index)
    ready = {"training_ready": True, "scope": "Synthetic software fixture only",
             "binding": binding, "sources": {key: {"sha256": sha} for key, sha in source_hashes.items()},
             "files": {name: digest(prepared / name) for name in (*recovery.ARRAY_NAMES, "index_manifest.json")}}
    atomic_json(prepared / "DATA_READINESS.json", ready)

    original = tmp_path / "original"
    config = helpers.smoke_config()
    assert training.train_study(prepared, original, config,
                                model_factory=helpers.tiny_factory)["status"] == "ALL_TASKS_COMPLETE"
    study = json.loads((original / "study_manifest.json").read_text(encoding="utf-8"))
    for name in ("index_manifest.json", "DATA_READINESS.json"):
        target = original / "data_evidence" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(prepared / name, target)
    atomic_json(original / "RUN_CLOCK.json", {"active_process_seconds": 1000.0})
    atomic_json(original / "RUN_MANIFEST.json", {"source_fingerprint": "original_software_fixture"})
    atomic_json(original / "RUN_STATUS.json", {"status": "FAILED", "scope": "fixture failure"})
    (original / "LAST_ERROR.txt").write_text("Original fixture failure retained for audit.\n", encoding="utf-8")
    config_path = original / "source/configs/garmentlab_data.json"
    config_path.parent.mkdir(parents=True)
    atomic_json(config_path, {"fixture_only": True})
    bad_cache = original / "evaluation/frozen_seed314/selection/logits.npz"
    bad_cache.parent.mkdir(parents=True)
    bad_cache.write_bytes(b"old invalid cache must never be used by the new finalizer")
    original_archive = export_results(original)
    original_sha = digest(original_archive)
    with zipfile.ZipFile(original_archive) as archive:
        original_payloads = {name: archive.read(name) for name in archive.namelist()}

    rebuilt = tmp_path / "rebuilt"
    rebuilt.mkdir()
    for name in recovery.ARRAY_NAMES:
        shutil.copyfile(prepared / name, rebuilt / name)
    # Realistic metadata byte drift; every record and array remains identical.
    (rebuilt / "index_manifest.json").write_bytes(index_path.read_bytes().replace(b"\r\n", b"\n"))
    rebuilt_ready = dict(ready)
    rebuilt_ready["files"] = {**ready["files"], "index_manifest.json": digest(rebuilt / "index_manifest.json")}
    (rebuilt / "DATA_READINESS.json").write_bytes((json.dumps(rebuilt_ready, sort_keys=True) + "\n").encode())

    patch = tmp_path / "patch"
    (patch / "garmentlab").mkdir(parents=True)
    (patch / "tools").mkdir()
    (patch / "garmentlab/backend.py").write_text("# Synthetic source snapshot\n", encoding="utf-8")
    (patch / "tools/recover_garmentlab.py").write_text("# Synthetic recovery CLI snapshot\n", encoding="utf-8")
    atomic_json(patch / "SOURCE_MANIFEST.json", {"software_fixture": True})

    def no_training(*args, **kwargs):
        pytest.fail("Recovery must never call train_study")

    monkeypatch.setattr(training, "train_study", no_training)
    target = tmp_path / "recovered"
    inferred_assessment = []
    actual_infer = evaluate._infer

    def verify_freeze(model, images, indices, **kwargs):
        if any(int(value) >= 18 for value in indices):
            assert (target / "evaluation/SELECTION_FREEZE.json").is_file()
            inferred_assessment.extend(int(value) for value in indices)
        return actual_infer(model, images, indices, **kwargs)

    monkeypatch.setattr(evaluate, "_infer", verify_freeze)
    arguments = dict(expected_results_sha256=original_sha, source_dir=patch,
                     model_factory=helpers.tiny_factory, smoke=True,
                     prepare_fn=lambda work, cache, data_config: rebuilt,
                     environment_fn=lambda: study["environment"])
    recovered = recovery.recover_results(original_archive, target, tmp_path / "work", tmp_path / "cache", **arguments)
    assert recovered["error"] is None, recovered
    assert recovered["status"]["status"] == "COMPLETE"
    summary = json.loads((target / "FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"] == "SMOKE_ONLY_COMPLETE"
    assert summary["task_count"] == 6 and summary["ready_for_local_service"] is False
    assert summary["gates"]["parity"]["status"] == "pass", summary["gates"]["parity"]
    assert len(summary["assessment"]) == 6 and len(summary["corruptions"]) == 9
    assert not (target / "release_candidate/release.json").exists()
    assert (target / "release_candidate/SMOKE_ONLY.json").is_file()
    assert (target / "EVALUATION_BACKEND.json").is_file()
    assert inferred_assessment and set(inferred_assessment) == set(range(18, 24))
    assert (target / "original_failure/evaluation/frozen_seed314/selection/logits.npz").read_bytes() == bad_cache.read_bytes()
    assert (target / "evaluation/frozen_seed314/selection/logits.npz").read_bytes() != bad_cache.read_bytes()
    mapping = json.loads((target / "ORIGINAL_PAYLOAD_MAP.json").read_text(encoding="utf-8"))["mapping"]
    for name, payload in original_payloads.items():
        assert (target / mapping[name]).read_bytes() == payload, name
    assert digest(original_archive) == original_sha
    with zipfile.ZipFile(recovered["result_zip"]) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("RUN_STATUS.json"))["status"] == "COMPLETE"

    before = {name: (target / name).read_bytes() for name in
              ("PREPARED_RECONSTITUTION.json", "EVALUATION_BACKEND.json", "evaluation/SELECTION_FREEZE.json", "FINAL_SUMMARY.json")}
    inference_count = len(inferred_assessment)
    monkeypatch.setattr(evaluate, "_load_model", lambda *args, **kwargs: pytest.fail("Completed recovery must not rebuild any model"))
    repeated = recovery.recover_results(original_archive, target, tmp_path / "work", tmp_path / "cache", **arguments)
    assert repeated["error"] is None, repeated
    assert repeated["status"]["status"] == "COMPLETE"
    assert len(inferred_assessment) == inference_count
    assert all((target / name).read_bytes() == payload for name, payload in before.items())
