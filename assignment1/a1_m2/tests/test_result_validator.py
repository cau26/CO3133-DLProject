"""Synthetic unit checks of artifact review; no training or benchmark results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_a1_results import (Review, check_history, main, object_hash,
                                     read_json, safe_child, score_predictions,
                                     source_hash_match,
                                     test_fingerprint as saved_test_fingerprint, validate)
import hashlib


def temporary():
    # Workspace-local scratch also works under the Windows restricted temp policy.
    return tempfile.TemporaryDirectory(prefix="validator-unit-", dir=Path(__file__).resolve().parent)


class ResultValidatorTests(unittest.TestCase):
    def test_macro_f1_always_includes_ten_classes(self):
        rows = [{"official_train_index": "8", "true": "0", "predicted": "0"},
                {"official_train_index": "9", "true": "1", "predicted": "0"}]
        result = score_predictions(rows, "official_train_index", [8, 9])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 10)
        self.assertEqual(result["confusion"][1][0], 1)

    def test_prediction_ids_duplicates_order_and_label_bounds(self):
        rows = [{"idx": "8", "true": "0", "predicted": "0"},
                {"idx": "8", "true": "1", "predicted": "0"}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_predictions(rows, "idx")
        rows[1]["idx"] = "9"
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            score_predictions(rows, "idx", [9, 8])
        rows[1]["predicted"] = "10"
        with self.assertRaisesRegex(ValueError, "labels"):
            score_predictions(rows, "idx", [8, 9])

    @staticmethod
    def history():
        # Controlled numerical input to test a selection rule, never a saved run.
        rows = [{"epoch": str(epoch), "train_loss": "0.9", "val_loss": str(loss),
                 "train_accuracy": "0.5", "val_accuracy": "0.6",
                 "train_macro_f1": "0.4", "val_macro_f1": "0.5",
                 "train_seconds": "2", "val_seconds": "1"}
                for epoch, loss in [(1, 0.8), (2, 0.7), (3, 0.7)]]
        metrics = {"epochs_completed": 3, "best_epoch": 2, "loss": 0.7,
                   "accuracy": 0.6, "macro_f1": 0.5, "train_seconds": 6,
                   "validation_seconds": 3}
        return rows, metrics, {"epochs": 3}

    def test_history_selects_earliest_minimum_loss(self):
        rows, metrics, config = self.history()
        self.assertEqual(check_history(rows, metrics, config)["selected_epoch"], 2)
        metrics["best_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "earliest"):
            check_history(rows, metrics, config)

    def test_history_rejects_gaps_nonfinite_and_wrong_time_sum(self):
        rows, metrics, config = self.history()
        rows[1]["epoch"] = "4"
        with self.assertRaisesRegex(ValueError, "contiguous"):
            check_history(rows, metrics, config)
        rows, metrics, config = self.history()
        rows[0]["val_loss"] = "nan"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            check_history(rows, metrics, config)
        rows, metrics, config = self.history()
        metrics["train_seconds"] = 99
        with self.assertRaisesRegex(ValueError, "training time"):
            check_history(rows, metrics, config)

    def test_empty_run_pending_and_required_run_fails_without_writes(self):
        with temporary() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            pending = validate(root)
            self.assertEqual(pending["status"], "PENDING")
            required = validate(root, require_complete=True)
            self.assertEqual(required["status"], "FAIL")
            self.assertEqual(list(root.iterdir()), [])

    def test_manifest_cannot_escape_run_or_follow_external_symlink(self):
        with temporary() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            for relative in ("../secret", "/secret", "C:/secret", "..\\secret"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    safe_child(root, relative)
            self.assertEqual(safe_child(root, "linear/metrics.json"), root / "linear/metrics.json")

    def test_manifest_detects_tampered_file(self):
        with temporary() as folder:
            root = Path(folder)
            (root / "data.json").write_text("{}", encoding="utf-8")
            review = Review(root)
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                review.hashes({"data.json": "0" * 64})

    def test_strict_json_and_stable_hash(self):
        self.assertEqual(object_hash({"b": 2, "a": 1}), object_hash({"a": 1, "b": 2}))
        with temporary() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"loss": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_json(path)

    def test_receipt_cannot_modify_run_and_never_overwrites(self):
        with temporary() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = main(["--out", str(root), "--receipt", str(root / "receipt.json")])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "FAIL")
            self.assertEqual(list(root.iterdir()), [])
            receipt = Path(folder) / "receipt.json"
            receipt.write_text("preserve", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--out", str(root), "--receipt", str(receipt)]), 1)
            self.assertEqual(receipt.read_text(encoding="utf-8"), "preserve")

    def test_source_lf_crlf_equivalence_is_explicit(self):
        with temporary() as folder:
            path = Path(folder) / "source.py"
            lf = b"value = 1\nprint(value)\n"
            crlf = lf.replace(b"\n", b"\r\n")
            path.write_bytes(crlf)
            self.assertEqual(source_hash_match(path, hashlib.sha256(crlf).hexdigest())["match"], "BYTE_EXACT")
            normalized = source_hash_match(path, hashlib.sha256(lf).hexdigest())
            self.assertEqual(normalized["match"], "GIT_CHECKOUT_EOL_EQUIVALENT")
            self.assertEqual(normalized["transformation"], "LF")
            path.write_bytes(lf)
            self.assertEqual(source_hash_match(path, hashlib.sha256(crlf).hexdigest())["transformation"], "CRLF")

    def test_source_changes_other_than_newlines_are_rejected(self):
        with temporary() as folder:
            path = Path(folder) / "source.py"
            path.write_bytes(b"value = 2\r\n")
            with self.assertRaisesRegex(ValueError, "beyond LF/CRLF"):
                source_hash_match(path, hashlib.sha256(b"value = 1\n").hexdigest())
            # Output manifests never apply the source newline exception.
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                Review(folder).hashes({"source.py": hashlib.sha256(b"value = 2\n").hexdigest()})

    def test_official_test_fingerprint_format_and_consistency(self):
        started = {"official_test_dataset_sha256": "a" * 64}
        self.assertEqual(saved_test_fingerprint(started, dict(started)), "a" * 64)
        with self.assertRaisesRegex(ValueError, "differs"):
            saved_test_fingerprint(started, {"official_test_dataset_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            saved_test_fingerprint({"official_test_dataset_sha256": "not-a-digest"})


if __name__ == "__main__":
    unittest.main()
