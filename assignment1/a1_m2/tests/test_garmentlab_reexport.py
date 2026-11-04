"""Export-only safety and archived numerical-oracle guards; no research fitting."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools import reexport_garmentlab as export


def fixture_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "evidence.txt").write_bytes(b"immutable evidence")
    manifest = {"payloads": [{"path": "evidence.txt", "bytes": 18,
                              "sha256": export.sha256_file(run / "evidence.txt")}]}
    (run / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run, export.sha256_file(run / "EXPORT_MANIFEST.json")


def test_verifies_all_payloads_then_refuses_tampering(tmp_path):
    run, pin = fixture_run(tmp_path)
    assert export.verify_run(run, expected_manifest_sha256=pin)["payload_count"] == 1
    (run / "evidence.txt").write_bytes(b"tampered  evidence")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        export.verify_run(run, expected_manifest_sha256=pin)


def test_refuses_changed_manifest_even_if_payload_manifest_agree(tmp_path):
    run, pin = fixture_run(tmp_path)
    (run / "EXPORT_MANIFEST.json").write_text('{"payloads": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="supplied pin"):
        export.verify_run(run, expected_manifest_sha256=pin)


@pytest.mark.parametrize("name", ["../outside.txt", "C:/outside.txt", "/outside.txt", "dir\\file.txt", "./evidence.txt"])
def test_refuses_unsafe_manifest_paths(tmp_path, name):
    with pytest.raises(ValueError):
        export._member(tmp_path.resolve(), name)


def test_refuses_unmanifested_payload(tmp_path):
    run, pin = fixture_run(tmp_path)
    (run / "unexpected.pt").write_bytes(b"not covered by audit")
    with pytest.raises(ValueError, match="file set differs"):
        export.verify_run(run, expected_manifest_sha256=pin)


def test_refuses_existing_output_without_modifying_it(tmp_path):
    run, prepared, out = [tmp_path / name for name in ("run", "prepared", "out")]
    for folder in (run, prepared, out):
        folder.mkdir()
    marker = out / "keep.txt"
    marker.write_bytes(b"keep this release")
    with pytest.raises(FileExistsError):
        export.assert_new_output(run, prepared, out)
    assert marker.read_bytes() == b"keep this release"


@pytest.mark.parametrize("inside", [True, False])
def test_refuses_output_ancestor_or_descendant_of_input(tmp_path, inside):
    run = tmp_path / "run"
    prepared = tmp_path / "prepared"
    out = run / "child" if inside else tmp_path
    with pytest.raises(ValueError, match="separate"):
        export.assert_new_output(run, prepared, out)


def test_archived_oracle_failure_cannot_be_hidden_by_top1_agreement(tmp_path):
    archived, local = {}, {}
    for batch in (1, 3, 64):
        expected = np.tile(np.asarray([[10., 1., 0.]], dtype=np.float64), (batch, 1))
        archived[f"pytorch_batch_{batch}"] = expected
        local[f"onnx_batch_{batch}"] = expected + (0.00015 if batch == 64 else 0.)
    np.savez(tmp_path / "old.npz", **archived)
    np.savez(tmp_path / "new.npz", **local)
    result = export.compare_archived_oracle(tmp_path / "new.npz", tmp_path / "old.npz", tolerance=1e-4)
    assert result["status"] == "fail"
    assert result["top1_disagreements"] == 0
    assert result["maximum_absolute_error"] > 1e-4


def test_archived_oracle_shape_or_nonfinite_is_not_accepted():
    with pytest.raises(ValueError, match="matching finite"):
        export.compare_logits(np.asarray([[np.nan, 1.]]), np.asarray([[0., 1.]]), tolerance=1e-4)
    with pytest.raises(ValueError, match="matching finite"):
        export.compare_logits(np.zeros((1, 2)), np.zeros((2, 2)), tolerance=1e-4)


def test_png_contract_reconstructs_exact_prepared_tensor_bytes():
    from garmentlab.inference import preprocess_image
    rng = np.random.default_rng(28)
    data = {"images": rng.integers(0, 256, (6, 224, 224, 3), dtype=np.uint8),
            "labels": np.arange(6), "classnames": [f"class_{i}" for i in range(6)]}
    probabilities = np.eye(6) * .9 + (1 - np.eye(6)) * .02
    cases = export._png_cases(data, np.arange(6), probabilities, np.ones(6, bool), .5)
    manifest = {"domain": "garment_crop_rgb", "preprocessing": {"version": "garment_rgb_letterbox_v1",
        "height": 224, "width": 224, "channels": 3, "mode": "RGB", "resize": "letterbox_bilinear",
        "fill": [0, 0, 0], "normalization": "imagenet_v1"}}
    assert len(cases) == 6
    for case in cases:
        tensor, _ = preprocess_image(case["png"], manifest, crop=[0, 0, 224, 224])
        assert export.hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == case["expected_tensor_sha256"]
