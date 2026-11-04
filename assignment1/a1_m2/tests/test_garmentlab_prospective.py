"""Prospective boundaries: no real photographs, model training or network access."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from garmentlab import prospective as p
from garmentlab.data import DataValidationError, _source_group
from garmentlab.runtime import atomic_json, digest


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 6)

    def forward(self, value):
        return self.head(value.mean((2, 3)))


def tiny_factory(**_):
    return Tiny()


def _png(seed):
    image = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def fixture(tmp_path, *, extra_candidates=0):
    paths = {}
    for role, identifiers in (("train", [1, *range(10, 16 + extra_candidates)]), ("assessment", [100])):
        metadata = {"licenses": [{"id": i, "url": f"https://example.org/license/{i}"} for i in [0, 1, 3, 5, 6]],
                    "images": [], "annotations": [], "categories": []}
        archive = tmp_path / f"{role}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for image_id in identifiers:
                metadata["images"].append({"id": image_id, "file_name": f"{image_id}.png", "width": 32,
                    "height": 32, "license": 0 if image_id in [1, 100] else (3 if image_id % 2 else 5),
                    "original_url": f"https://example.org/images/{image_id}.png"})
                category = p.CLASS_MAP[(image_id - 10) % 6][0] if 10 <= image_id < 16 else 23
                metadata["annotations"].append({"id": image_id, "image_id": image_id,
                                                "category_id": category, "bbox": [0, 0, 32, 32]})
                z.writestr(f"{role}/{image_id}.png", _png(image_id))
        source = tmp_path / f"{role}_metadata.json"
        atomic_json(source, metadata)
        paths[f"{role}_metadata"] = source
        paths[f"{role}_archive"] = archive
    old_group = _source_group("https://example.org/images/1.png")
    index = tmp_path / "index_manifest.json"
    atomic_json(index, {"labels": p.LABELS, "records": [{"image_id": 1, "source_group_id": old_group}],
                        "binding": {"source_sha256": {k: digest(v) for k, v in paths.items()}}})
    duplicates = tmp_path / "DUPLICATE_AUDIT.json"
    atomic_json(duplicates, {"cross_role_components": []})
    paths.update(historical_index=index, duplicate_audit=duplicates)
    study = tmp_path / "study"
    study.mkdir()
    atomic_json(study / "study_v2_manifest.json", {"labels": p.LABELS})
    binding = digest(study / "study_v2_manifest.json")
    descriptors = []
    for candidate in ["C0", "C4"]:
        for seed in p.SEEDS:
            task_id = f"final_{candidate}_seed{seed}"
            folder = study / "tasks" / task_id
            folder.mkdir(parents=True)
            task = {"task_id": task_id, "candidate_id": candidate, "seed": seed, "labels": p.LABELS}
            model = Tiny()
            with torch.no_grad():
                model.head.weight.zero_()
                model.head.bias.fill_(0)
                model.head.bias[0 if candidate == "C0" else 1] = 1
            atomic_json(folder / "task.json", task)
            atomic_json(folder / "history.json", [{"epoch": 1}])
            torch.save({"task": task, "model_state_dict": model.state_dict()}, folder / "best.pt")
            hashes = {name: {"sha256": digest(folder / name)} for name in ["task.json", "history.json", "best.pt"]}
            atomic_json(folder / "COMPLETE.json", {"task_id": task_id, "binding": binding, "files": hashes})
            descriptors.append({"task_id": task_id, "candidate_id": candidate, "seed": seed,
                "task": task, "checkpoint": f"tasks/{task_id}/best.pt", "weights_sha256": hashes["best.pt"]["sha256"]})
    atomic_json(study / "FINAL_SELECTION.json", {"labels": p.LABELS, "study_binding": binding,
        "selection_candidate_id": "C4", "final_tasks": descriptors,
        "data": {"index_manifest.json": digest(index)}})
    atomic_json(study / "study_v2_state.json", {"status": "TRAINING_COMPLETE_CHOICES_FROZEN", "binding": binding,
        "final_selection_sha256": digest(study / "FINAL_SELECTION.json")})
    c4 = next(d for d in descriptors if d["task_id"] == "final_C4_seed314")
    atomic_json(study / "release_candidate" / "release.json", {"labels": p.LABELS,
        "weights": {"sha256": c4["weights_sha256"]}, "calibration": {"temperature": 2., "review_threshold": .9,
            "policy": {"mode": "confidence_threshold"}}})
    protocol = {"source_group_cap": 100, "minimum_components_per_class": 1,
                "near_duplicate_distance": 0, "bootstrap_replicates": 100}
    attestation = {"recorded_exposure_inventory_complete": True, "basis": "Synthetic fixture history only"}
    return paths, study, protocol, attestation


def freeze(out, study, paths, protocol, attestation):
    return p.freeze_prospective(out, study, **paths, protocol=protocol,
                               exposure_attestation=attestation, smoke=True)


def test_history_inventory_missing_fails_before_cohort_access(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    with pytest.raises(DataValidationError, match="historical image-exposure"):
        p.freeze_prospective(tmp_path / "audit", study, **paths, protocol=protocol, smoke=True)
    assert not (tmp_path / "audit" / "PROSPECTIVE_FREEZE.json").exists()


def test_complete_blinded_prep_and_six_model_eval_are_reproducible(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    locked = freeze(out, study, paths, protocol, attestation)
    assert freeze(out, study, paths, protocol, attestation) == locked
    prepared = p.prepare_prospective(out, **paths)
    ready = json.loads((prepared / "COHORT_READY.json").read_text())
    assert ready["crops"] == ready["components"] == 6
    assert ready["model_inference_performed"] is False
    assert not list(out.rglob("*.png")) and not list(out.rglob("*.jpg"))
    first = p.evaluate_prospective(out, study, device="cpu", batch_size=3, model_factory=tiny_factory, smoke=True)
    assert first["status"] == "PROSPECTIVE_AUDIT_COMPLETE" and len(first["models"]) == 6
    # The bound/policy applies only to its original predetermined C4seed314.
    assert [r["task_id"] for r in first["models"] if "frozen_policy" in r] == ["final_C4_seed314"]
    assert p.evaluate_prospective(out, study, device="cpu", batch_size=3,
                                  model_factory=tiny_factory, smoke=True) == first
    assert p.prepare_prospective(out, **paths) == prepared


def test_sample_cap_precedes_target_category_filter_and_no_retry(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path, extra_candidates=8)
    protocol["source_group_cap"] = 3
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    with pytest.raises(DataValidationError, match="class support"):
        p.prepare_prospective(out, **paths)
    selected = json.loads((out / "BLINDED_COHORT_SELECTION.json").read_text())
    metadata = json.loads(paths["train_metadata"].read_text())
    groups = {_source_group(r["original_url"]) for r in metadata["images"] if r["license"] in [3, 5]}
    expected = sorted(groups, key=lambda g: (hashlib.sha256(f"20261029:{g}".encode()).hexdigest(), g))[:3]
    assert set(selected["selected_source_groups"]) == set(expected)
    assert selected["replacement_after_exclusions"] is False
    assert not (out / "prepared").exists()


def test_transitive_duplicate_bridge_in_old_quarantine_drops_new_group():
    # NewA is not directly a neighbor of retainedOld, but droppedOld bridges it.
    sources = [{"source_group_id": "retainedOld", "image_id": 1, "historical": True,
                "decoded_rgb_sha256": "a", "dhash64": "0000000000000000"},
               {"source_group_id": "droppedOld", "image_id": 2, "historical": True,
                "decoded_rgb_sha256": "b", "dhash64": "0000000000000001"},
               {"source_group_id": "newA", "image_id": 3, "historical": False,
                "decoded_rgb_sha256": "c", "dhash64": "0000000000000003"},
               {"source_group_id": "newB", "image_id": 4, "historical": False,
                "decoded_rgb_sha256": "d", "dhash64": "ffffffffffffffff"}]
    dropped, components = p._duplicate_components(sources, 1)
    assert dropped == {"newA"}
    assert components["newA"] == components["retainedOld"]
    assert components["newB"] != components["newA"]


def test_raw_inputs_and_source_cannot_change_after_lock(tmp_path, monkeypatch):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    original = paths["duplicate_audit"].read_bytes()
    paths["duplicate_audit"].write_bytes(original + b"\n")
    with pytest.raises(DataValidationError, match="inputs changed"):
        p.prepare_prospective(out, **paths)
    paths["duplicate_audit"].write_bytes(original)
    monkeypatch.setattr(p, "_sources", lambda: {"changed": "source"})
    with pytest.raises(DataValidationError, match="source changed"):
        p.prepare_prospective(out, **paths)


def test_changed_checkpoint_is_rejected_before_model_construction(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    p.prepare_prospective(out, **paths)
    checkpoint = study / "tasks/final_C4_seed314/best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    called = []
    def factory(**kwargs):
        called.append(kwargs)
        return Tiny()
    with pytest.raises(DataValidationError, match="payload changed"):
        p.evaluate_prospective(out, study, device="cpu", model_factory=factory, smoke=True)
    assert called == []


def test_cache_and_fixture_evidence_cannot_be_promoted_or_rebound(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    prepared = p.prepare_prospective(out, **paths)
    with pytest.raises(DataValidationError, match="fixture provenance"):
        p.evaluate_prospective(out, study, device="cpu")
    path = prepared / "labels.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(DataValidationError, match="payload hash"):
        p.evaluate_prospective(out, study, device="cpu", model_factory=tiny_factory, smoke=True)


def test_quarantine_scope_missing_raw_groups_fails_closed(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    atomic_json(paths["duplicate_audit"], {"cross_role_components": [
        {"quarantined_source_group_ids": ["missing-historical-source"]}]})
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    with pytest.raises(DataValidationError, match="quarantine scope"):
        p.prepare_prospective(out, **paths)
    assert not (out / "prepared").exists()


def test_omitted_required_array_hash_is_rejected_before_forward(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    prepared = p.prepare_prospective(out, **paths)
    ready = json.loads((prepared / "COHORT_READY.json").read_text())
    del ready["files"]["labels.npy"]
    atomic_json(prepared / "COHORT_READY.json", ready)
    labels = np.load(prepared / "labels.npy")
    np.save(prepared / "labels.npy", (labels + 1) % 6)
    called = []
    def factory(**kwargs):
        called.append(kwargs)
        return Tiny()
    with pytest.raises(DataValidationError, match="every required prepared payload"):
        p.evaluate_prospective(out, study, device="cpu", model_factory=factory, smoke=True)
    assert called == []


def test_unbound_and_legacy_policy_schemas_fail_at_freeze_before_inference(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    release = study / "release_candidate/release.json"
    original = json.loads(release.read_text())
    invalid = deepcopy(original)
    del invalid["weights"]
    atomic_json(release, invalid)
    with pytest.raises(DataValidationError, match="explicitly bind"):
        freeze(tmp_path / "unbound", study, paths, protocol, attestation)
    invalid = deepcopy(original)
    del invalid["calibration"]["policy"]
    atomic_json(release, invalid)
    with pytest.raises(DataValidationError, match="Unsupported frozen policy"):
        freeze(tmp_path / "legacy", study, paths, protocol, attestation)


def test_manual_group_only_exclusion_is_fingerprinted_as_historical(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    additional = tmp_path / "additional_exclusions.json"
    group = _source_group("https://example.org/images/10.png")
    atomic_json(additional, {"source_group_ids": [group], "reason": "Recorded manual image view"})
    paths["additional_exclusions"] = additional
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    with pytest.raises(DataValidationError, match="class support"):
        p.prepare_prospective(out, **paths)
    audit = json.loads((out / "BLINDED_DUPLICATE_AUDIT.json").read_text())
    assert any(r["image_id"] == 10 and r["historical"] for r in audit["source_fingerprints"])
    selected = json.loads((out / "BLINDED_COHORT_SELECTION.json").read_text())
    assert group not in selected["selected_source_groups"]


def test_completion_receipt_detects_standalone_result_and_prediction_tamper(tmp_path):
    paths, study, protocol, attestation = fixture(tmp_path)
    out = tmp_path / "audit"
    freeze(out, study, paths, protocol, attestation)
    p.prepare_prospective(out, **paths)
    result = p.evaluate_prospective(out, study, device="cpu", model_factory=tiny_factory, smoke=True)
    assert p.verify_prospective_complete(out) == result
    result_file = out / "PROSPECTIVE_RESULTS.json"
    original = result_file.read_bytes()
    changed = deepcopy(result)
    changed["primary"]["paired_mean_macro_f1_delta"] = .99
    atomic_json(result_file, changed)
    with pytest.raises(DataValidationError, match="completion result/input hash"):
        p.verify_prospective_complete(out)
    result_file.write_bytes(original)
    assert p.verify_prospective_complete(out) == result
    logits_file = out / result["payloads"][1]["path"]
    logits_file.write_bytes(logits_file.read_bytes() + b"tampered prediction")
    with pytest.raises(DataValidationError, match="prediction payload changed"):
        p.verify_prospective_complete(out)
