"""Release regressions run on ZIP bytes, independent of the working tree."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_package_gate", ROOT / "tools/verify_public_package.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class PublicPackageIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def archive(self, missing=None, tampered=False, unsafe=False):
        payloads = {name: b"test fixture\n" for name in gate.REQUIRED_PATHS if name != missing}
        if unsafe:
            payloads["../outside.py"] = b"escaped\n"
        manifest = {"payloads": [{"path": name, "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}
                                 for name, data in payloads.items()]}
        if tampered:
            payloads[gate.REQUIRED_PATHS[0]] = b"wrong bytes!\n"
        path = self.folder / "publication.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name, data in payloads.items():
                z.writestr(name, data)
            z.writestr(gate.MANIFEST, json.dumps(manifest))
        return path

    def test_complete_payload_inventory_passes(self):
        result = gate.inspect_integrity(self.archive())
        self.assertEqual(result["payloads_verified"], len(gate.REQUIRED_PATHS))

    def test_known_omissions_are_rejected_even_when_manifest_agrees(self):
        for name in gate.REQUIRED_PATHS:
            with self.subTest(missing=name):
                with self.assertRaisesRegex(ValueError, "Missing required publication files"):
                    gate.inspect_integrity(self.archive(missing=name))

    def test_tampered_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Payload (hash|size) differs"):
            gate.inspect_integrity(self.archive(tampered=True))

    def test_parent_traversal_is_rejected_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            gate.inspect_integrity(self.archive(unsafe=True))

    def test_fresh_extraction_cannot_merge_with_a_working_checkout(self):
        existing = self.folder / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(ValueError, "must not exist"):
            gate.extract_fresh(self.archive(), existing)

    def reference_archive(self, missing=None):
        dump = lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8")
        sha = lambda value: hashlib.sha256(value).hexdigest()
        reference = "outputs/historical_fixture/"
        config = {"device": "cuda", "learning_rate": 0.001}
        payloads = {reference + "config.json": dump(config),
                    reference + "linear/validation_predictions.csv": b"id,true,pred\n1,0,0\n"}
        frozen = {"run_id": "fixture", "split_sha256": "split", "dataset_sha256": "data",
                  "files_sha256": {name[len(reference):]: sha(value) for name, value in payloads.items()}}
        frozen["manifest_sha256"] = sha(json.dumps(frozen, sort_keys=True, separators=(",", ":"),
                                                  ensure_ascii=False).encode("utf-8"))
        freeze_bytes = dump(frozen)
        exported = {name[len(reference):]: sha(value) for name, value in payloads.items()}
        exported["freeze_manifest.json"] = sha(freeze_bytes)
        export_bytes = dump(exported)
        payloads[reference + "freeze_manifest.json"] = freeze_bytes
        payloads[reference + "artifact_manifest.json"] = export_bytes
        profile = {"reference_run": reference.rstrip("/"), "base_config": "configs/base.json",
                   "reference_run_id": "fixture", "expected_split_sha256": "split", "expected_dataset_sha256": "data",
                   "reference_freeze_sha256": sha(freeze_bytes),
                   "reference_artifact_manifest_sha256": sha(export_bytes)}
        payloads["configs/a1_upgrade_study.json"] = dump(profile)
        payloads["configs/base.json"] = dump({**config, "device": "auto"})
        if missing == "directory":
            payloads = {name: value for name, value in payloads.items() if not name.startswith(reference)}
        elif missing:
            del payloads[reference + missing]
        path = self.folder / "study_reference.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name, data in payloads.items():
                z.writestr(name, data)
        return path

    def test_complete_historical_reference_passes(self):
        with zipfile.ZipFile(self.reference_archive()) as z:
            result = gate.inspect_study_reference(z)
        self.assertEqual(result["unique_historical_payloads_verified"], 3)

    def test_historical_reference_directory_and_manifest_listed_file_are_required(self):
        for missing in ["directory", "linear/validation_predictions.csv"]:
            with self.subTest(missing=missing):
                with zipfile.ZipFile(self.reference_archive(missing=missing)) as z:
                    with self.assertRaisesRegex(ValueError, "(Missing historical study reference|Historical reference payload is missing)"):
                        gate.inspect_study_reference(z)


if __name__ == "__main__":
    unittest.main()
