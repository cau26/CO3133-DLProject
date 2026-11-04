"""Release boundaries and local rollback; synthetic bytes, no training or inference."""
import hashlib
import json
from pathlib import Path

import pytest

from garmentlab.release import activate_release, rollback_release, validate_release


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("policy,threshold", [
    ({"mode": "unknown", "bound_eligible": True}, .8),
    ({"mode": "review_all", "bound_eligible": False}, .8),
    ({"mode": "confidence_threshold", "bound_eligible": False}, .8),
])
def test_explicit_review_policy_rejects_inconsistent_ready_release(tmp_path, policy, threshold):
    root, manifest = make_release(tmp_path)
    manifest["calibration"].update(policy=policy, review_threshold=threshold)
    change_receipt(root, manifest, lambda receipt: receipt.update(calibration=manifest["calibration"]))
    with pytest.raises(ValueError, match="policy|review_all|risk bound"):
        validate_release(root)


def make_release(tmp_path, name="v1", domain="garment_crop_rgb", status="ready"):
    root = tmp_path / name
    root.mkdir()
    weights = root / "weights.pt"
    weights.write_bytes(b"synthetic test weights: " + name.encode())
    real = domain == "garment_crop_rgb"
    manifest = {"format_version": 1, "release_id": name, "domain": domain,
                "architecture": "mobilenet_v3_small" if real else "cnn_c5", "status": status,
                "labels": ["shirt", "trousers"] if real else [f"class_{n}" for n in range(10)],
                "preprocessing": {"height": 224 if real else 28, "width": 224 if real else 28,
                                  "channels": 3 if real else 1, "mode": "RGB" if real else "L"},
                "weights": {"file": weights.name, "sha256": sha(weights)}}
    if real and status == "ready":
        manifest["calibration"] = {"status": "fitted", "temperature": 1.25,
                                   "review_threshold": 0.7, "fit_split_sha256": "a" * 64,
                                   "threshold_selection_split_sha256": "b" * 64}
        receipt = {"format_version": 1, "release_id": name, "weights_sha256": sha(weights),
                   "calibration": manifest["calibration"],
                   "gates": {gate: {"status": "pass", "scope": "synthetic test fixture"}
                             for gate in ("data", "quality", "calibration", "parity")}}
        dump(root / "evaluation.json", receipt)
        manifest["evaluation_receipt"] = {"file": "evaluation.json", "sha256": sha(root / "evaluation.json")}
    dump(root / "release.json", manifest)
    return root, manifest


def change_receipt(root, manifest, mutate):
    path = root / "evaluation.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    dump(path, receipt)
    manifest["evaluation_receipt"]["sha256"] = sha(path)
    dump(root / "release.json", manifest)


def test_ready_release_and_reference_inspection(tmp_path):
    ready, manifest = make_release(tmp_path)
    result = validate_release(ready)
    assert result["manifest"] == manifest
    assert result["weights_path"] == str((ready / "weights.pt").resolve())
    assert result["evaluation_receipt"]["release_id"] == "v1"
    assert result["reference_only"] is False
    reference, _ = make_release(tmp_path, "reference", "fashion_mnist", "reference")
    assert validate_release(reference, require_ready=False)["reference_only"] is True
    with pytest.raises(ValueError, match="reference"):
        activate_release(reference, tmp_path / "registry")
    assert not (tmp_path / "registry" / "active.json").exists()


def test_real_candidate_cannot_activate(tmp_path):
    candidate, _ = make_release(tmp_path, status="candidate")
    assert validate_release(candidate, require_ready=False)["evaluation_receipt"] is None
    with pytest.raises(ValueError, match="candidate"):
        activate_release(candidate, tmp_path / "registry")


def test_ready_flag_alone_is_insufficient_even_in_inspection_mode(tmp_path):
    root, manifest = make_release(tmp_path, status="candidate")
    manifest["status"] = "ready"
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError, match="calibration"):
        validate_release(root, require_ready=False)


@pytest.mark.parametrize("gate", ["data", "quality", "calibration", "parity"])
def test_each_gate_is_required(tmp_path, gate):
    root, manifest = make_release(tmp_path)
    change_receipt(root, manifest, lambda receipt: receipt["gates"].update({gate: "fail"}))
    with pytest.raises(ValueError, match=gate):
        validate_release(root)


def test_receipt_is_bound_to_release_and_weights(tmp_path):
    root, manifest = make_release(tmp_path)
    change_receipt(root, manifest, lambda receipt: receipt.update(release_id="another-release"))
    with pytest.raises(ValueError, match="release_id"):
        validate_release(root)
    change_receipt(root, manifest, lambda receipt: receipt.update(release_id="v1", weights_sha256="0" * 64))
    with pytest.raises(ValueError, match="weights binding"):
        validate_release(root)


@pytest.mark.parametrize("field,value", [("temperature", 0), ("temperature", True),
                                        ("temperature", float("inf")),
                                        ("review_threshold", -0.1), ("review_threshold", True),
                                        ("review_threshold", float("nan")),
                                        ("fit_split_sha256", "missing"),
                                        ("threshold_selection_split_sha256", None)])
def test_ready_calibration_requires_valid_numeric_fields_and_split_hashes(tmp_path, field, value):
    root, manifest = make_release(tmp_path)
    manifest["calibration"][field] = value
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError):
        validate_release(root)


def test_receipt_calibration_must_exactly_match_manifest(tmp_path):
    root, manifest = make_release(tmp_path)
    change_receipt(root, manifest, lambda receipt: receipt["calibration"].update(temperature=2.0))
    with pytest.raises(ValueError, match="calibration does not match"):
        validate_release(root)


def test_receipt_cannot_omit_weight_binding(tmp_path):
    root, manifest = make_release(tmp_path)
    change_receipt(root, manifest, lambda receipt: receipt.pop("weights_sha256"))
    with pytest.raises(ValueError, match="receipt.weights_sha256"):
        validate_release(root)


def test_tampered_weight_and_receipt_bytes_are_rejected(tmp_path):
    root, _ = make_release(tmp_path)
    (root / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights SHA-256"):
        validate_release(root)
    second, _ = make_release(tmp_path, "v2")
    with (second / "evaluation.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="evaluation_receipt SHA-256"):
        validate_release(second)


@pytest.mark.parametrize("path", ["../outside.pt", "sub/../../outside.pt", "..\\outside.pt",
                                  "/absolute.pt", "C:\\absolute.pt", "C:relative.pt",
                                  "weights.pt:stream", "//server/share/weights.pt"])
def test_artifact_traversal_absolute_and_windows_stream_paths_rejected(tmp_path, path):
    root, manifest = make_release(tmp_path)
    manifest["weights"]["file"] = path
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError, match="Artifact"):
        validate_release(root)


def test_symlink_escape_is_rejected_where_symlinks_are_available(tmp_path):
    root, manifest = make_release(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes((root / "weights.pt").read_bytes())
    linked = root / "link.pt"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires additional Windows privileges")
    manifest["weights"]["file"] = linked.name
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError, match="within"):
        validate_release(root)


@pytest.mark.parametrize("field,value", [("labels", ["shirt", "Shirt"]),
                                        ("architecture", "cnn_c5"),
                                        ("format_version", True)])
def test_schema_rejects_ambiguous_labels_architecture_or_version(tmp_path, field, value):
    root, manifest = make_release(tmp_path)
    manifest[field] = value
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError):
        validate_release(root)


def test_domain_preprocessing_is_checked(tmp_path):
    root, manifest = make_release(tmp_path)
    manifest["preprocessing"]["channels"] = 1
    dump(root / "release.json", manifest)
    with pytest.raises(ValueError, match="channels"):
        validate_release(root)


def test_duplicate_manifest_keys_are_not_silently_accepted(tmp_path):
    root, _ = make_release(tmp_path)
    text = (root / "release.json").read_text(encoding="utf-8")
    (root / "release.json").write_text(text.replace('"status": "ready"',
                                                   '"status": "ready", "status": "candidate"'), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        validate_release(root)


def test_activate_and_rollback_record_real_local_transitions(tmp_path):
    first, _ = make_release(tmp_path, "v1")
    second, _ = make_release(tmp_path, "v2")
    registry = tmp_path / "registry"
    initial = activate_release(first, registry)
    assert initial["active"]["release_id"] == "v1" and initial["previous"] is None
    assert len(activate_release(first, registry)["ledger"]) == 1  # Idempotent activation.
    activate_release(second, registry)
    restored = rollback_release(registry)
    assert restored["active"]["release_id"] == "v1"
    assert restored["previous"]["release_id"] == "v2"
    assert [event["action"] for event in restored["ledger"]] == ["activate", "activate", "rollback"]
    assert restored["ledger"][-1]["from"]["release_id"] == "v2"
    stored = json.loads((registry / "active.json").read_text(encoding="utf-8"))
    assert stored["active"] == restored["active"] and stored["ledger"] == restored["ledger"]
    assert not list(registry.glob(".active-*.tmp"))
    assert not (registry / ".release.lock").exists()


def test_rollback_refuses_changed_previous_artifact_without_modifying_pointer(tmp_path):
    first, _ = make_release(tmp_path, "v1")
    second, _ = make_release(tmp_path, "v2")
    registry = tmp_path / "registry"
    activate_release(first, registry)
    activate_release(second, registry)
    before = (registry / "active.json").read_bytes()
    (first / "weights.pt").write_bytes(b"modified previous weights")
    with pytest.raises(ValueError, match="SHA-256"):
        rollback_release(registry)
    assert (registry / "active.json").read_bytes() == before
    assert not (registry / ".release.lock").exists()


def test_rollback_can_recover_from_corrupted_current_release(tmp_path):
    first, _ = make_release(tmp_path, "v1")
    second, _ = make_release(tmp_path, "v2")
    registry = tmp_path / "registry"
    activate_release(first, registry)
    activate_release(second, registry)
    (second / "weights.pt").write_bytes(b"current artifact is unusable")
    restored = rollback_release(registry)
    assert restored["active"]["release_id"] == "v1"
    assert restored["ledger"][-1]["action"] == "rollback"


def test_release_identity_cannot_be_rebound_to_different_directory(tmp_path):
    first, _ = make_release(tmp_path, "v1")
    second, manifest = make_release(tmp_path, "v2")
    manifest["release_id"] = "v1"
    change_receipt(second, manifest, lambda receipt: receipt.update(release_id="v1"))
    registry = tmp_path / "registry"
    activate_release(first, registry)
    with pytest.raises(ValueError, match="cannot be reused"):
        activate_release(second, registry)
    stored = json.loads((registry / "active.json").read_text(encoding="utf-8"))
    assert stored["active"]["directory"] == str(first.resolve())


def test_missing_previous_and_existing_lock_are_clear_failures(tmp_path):
    root, _ = make_release(tmp_path)
    registry = tmp_path / "registry"
    activate_release(root, registry)
    with pytest.raises(ValueError, match="No previous"):
        rollback_release(registry)
    (registry / ".release.lock").write_text("test owner", encoding="utf-8")
    with pytest.raises(RuntimeError, match="busy"):
        activate_release(root, registry)


def test_failed_atomic_replace_leaves_pointer_and_ledger_unchanged(tmp_path, monkeypatch):
    from garmentlab import release
    first, _ = make_release(tmp_path, "v1")
    second, _ = make_release(tmp_path, "v2")
    registry = tmp_path / "registry"
    activate_release(first, registry)
    before = (registry / "active.json").read_bytes()
    def fail_replace(source, destination):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(release.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        activate_release(second, registry)
    assert (registry / "active.json").read_bytes() == before
    assert not list(registry.glob(".active-*.tmp"))
    assert not (registry / ".release.lock").exists()
