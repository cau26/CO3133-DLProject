"""Tiny synthetic software checks only; these do not measure Fashion-MNIST performance."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import run_a1
from a1.common import json_hash, load_json, save_json, sha256_file, source_metadata
from a1.data import validate_split
from a1.cnn import GlobalAveragePool2d
from a1.finalize import freeze_run, locked_paths, validate_freeze
from a1.train import (MODEL_NAMES, build_model, capture_rng, model_config, restore_rng,
                      plot_results, train_model)
import a1.train as training


def tiny_config():
    config = load_json(Path(__file__).resolve().parents[1] / "configs/a1_final.json")
    config.update(device="cpu", epochs=2, batch_size=2, num_workers=0, cpu_threads=1,
                  inference_warmup_batches=1, inference_measured_batches=1)
    config["per_model"]["cnn"].update(cnn_channels=[4, 8, 8], dropout=0.2)
    return config


class SafetyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_model_output_shapes_and_unknown_name(self):
        config = tiny_config()
        batch = torch.rand(2, 1, 28, 28)
        for name in MODEL_NAMES:
            model = build_model(name, config)
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 10))
            self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            build_model("typo", config)

    def test_cnn_global_mean_matches_pooling_and_has_finite_gradients(self):
        model = build_model("cnn", tiny_config())
        self.assertFalse(any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules()))
        self.assertIsInstance(model.features[-1], GlobalAveragePool2d)
        self.assertEqual(model.features[-1].state_dict(), {})
        features = torch.randn(2, 8, 7, 7, requires_grad=True)
        actual = model.features[-1](features)
        expected = torch.nn.functional.adaptive_avg_pool2d(features.detach(), 1)
        self.assertEqual(tuple(actual.shape), (2, 8, 1, 1))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        torch.testing.assert_close(features.grad, torch.full_like(features, 1 / 49))
        logits = model(torch.rand(2, 1, 28, 28))
        logits.sum().backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters() if parameter.grad is not None))

    def test_model_overrides_keep_shared_split_and_seed(self):
        config = tiny_config()
        self.assertEqual(model_config("transformer", config)["learning_rate"], 0.0005)
        config["per_model"]["transformer"]["seed"] = 999
        with self.assertRaises(ValueError):
            model_config("transformer", config)

    def test_corrupt_split_rejected(self):
        validate_split(np.array([0, 1]), np.array([2, 3]), 4, 2)
        for train, val in [(np.array([0, 1]), np.array([1, 3])),
                           (np.array([0., 1.]), np.array([2, 3])),
                           (np.array([1, 0]), np.array([2, 3]))]:
            with self.assertRaises(ValueError):
                validate_split(train, val, 4, 2)

    def test_config_and_incomplete_output_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            config = tiny_config()
            run_a1.prepare_run(root, config)
            before = sha256_file(root / "config.json")
            changed = copy.deepcopy(config)
            changed["epochs"] += 1
            with self.assertRaises(ValueError):
                run_a1.prepare_run(root, changed)
            self.assertEqual(before, sha256_file(root / "config.json"))
            (root / "cnn").mkdir()
            marker = root / "cnn" / "interrupted.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_model("cnn", {}, config, root, torch.device("cpu"))
            self.assertEqual(marker.read_text(), "preserve")

    def test_rng_round_trip_including_shuffle_generator(self):
        import random
        generator = torch.Generator().manual_seed(123)
        state = capture_rng(generator)
        expected = (random.random(), np.random.rand(), torch.rand(2), torch.randperm(9, generator=generator))
        restore_rng(state, generator)
        actual = (random.random(), np.random.rand(), torch.rand(2), torch.randperm(9, generator=generator))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_resume_matches_uninterrupted_tiny_synthetic_training(self):
        """Interrupt after epoch 1, then prove exact saved-state continuation on CPU."""
        config = tiny_config()
        generator = torch.Generator().manual_seed(7)
        images = torch.rand(8, 1, 28, 28, generator=generator)
        labels = torch.arange(8) % 10
        data = {"train": TensorDataset(images[:4], labels[:4]),
                "validation": TensorDataset(images[4:], labels[4:]),
                "val_ids": np.arange(4, 8),
                "summary": {"classes": [str(i) for i in range(10)],
                            "split_sha256": "SYNTHETIC-SOFTWARE-TEST", "dataset_sha256": "SYNTHETIC-SOFTWARE-TEST"}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            reference, resumed = root / "reference", root / "resumed"
            for output in (reference, resumed):
                run_a1.prepare_run(output, config)
            with patch.object(training, "plot_results", return_value=[]), patch.object(
                    training, "measure_inference", return_value={"per_image_ms_amortized": 0.0}):
                train_model("cnn", data, config, reference, torch.device("cpu"))
                real_epoch, train_calls = training.run_epoch, [0]

                def interrupted(*args, **kwargs):
                    if len(args) > 4 and args[4] is not None:
                        train_calls[0] += 1
                        if train_calls[0] == 2:
                            raise RuntimeError("SIMULATED INTERRUPTION")
                    return real_epoch(*args, **kwargs)

                with patch.object(training, "run_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "SIMULATED INTERRUPTION"):
                        train_model("cnn", data, config, resumed, torch.device("cpu"))
                self.assertFalse((resumed / "cnn" / "metrics.json").exists())
                self.assertTrue((resumed / "cnn" / "last.pt").is_file())
                train_model("cnn", data, config, resumed, torch.device("cpu"), resume=True)
            a = torch.load(reference / "cnn" / "last.pt", weights_only=True)
            b = torch.load(resumed / "cnn" / "last.pt", weights_only=True)
            for key in a["model_state"]:
                self.assertTrue(torch.equal(a["model_state"][key], b["model_state"][key]), key)
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            for first, second in zip(a["history"], b["history"]):
                for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"):
                    self.assertEqual(first[key], second[key], key)
            with self.assertRaises(FileExistsError):
                train_model("cnn", data, config, resumed, torch.device("cpu"), resume=True)

    def test_invalid_freeze_fails_before_test_access(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            save_json(root / "freeze_manifest.json", {"schema_version": 1, "models": {"linear": {}}})
            with self.assertRaises(ValueError):
                validate_freeze(root, root / "freeze_manifest.json")
            with self.assertRaises(ValueError):
                freeze_run(root, confirmed=False)

    def test_all_five_freeze_hashes_are_validated(self):
        """Only placeholder files in an ephemeral software fixture; no model results."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for relative in locked_paths():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("SYNTHETIC SOFTWARE FIXTURE", encoding="utf-8")
            save_json(root / "run_manifest.json", {"run_id": "software-fixture"})
            manifest = {"schema_version": 1, "run_id": "software-fixture",
                        "models": {name: {} for name in MODEL_NAMES},
                        "source_sha256": source_metadata()["source_sha256"],
                        "files_sha256": {relative: sha256_file(root / relative) for relative in locked_paths()}}
            manifest["manifest_sha256"] = json_hash(manifest)
            save_json(root / "freeze_manifest.json", manifest)
            validate_freeze(root, root / "freeze_manifest.json")
            (root / "transformer" / "best.pt").write_text("edited fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen file was changed"):
                validate_freeze(root, root / "freeze_manifest.json")

    def test_plot_results_uses_requested_partition(self):
        images = torch.rand(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 0, 1])
        data = {"validation": TensorDataset(images, labels),
                "test": TensorDataset(images, labels),
                "summary": {"classes": [str(i) for i in range(10)]}}
        true, predicted = np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1])
        history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.1,
                    "train_accuracy": 0.5, "val_accuracy": 0.5}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for partition in ("validation", "test"):
                destination = root / partition
                destination.mkdir()
                plot_results("SYNTHETIC CHECK", history if partition == "validation" else [],
                             true, predicted, data, destination, split=partition)
                self.assertTrue((destination / "examples.png").is_file())
                self.assertTrue((destination / "confusion_matrix.csv").is_file())
                self.assertEqual((destination / "curves.png").exists(), partition == "validation")


if __name__ == "__main__":
    unittest.main()
