"""CPU software checks for the validation-only upgrade; no real experiment training."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset

import run_a1
import a1.train as training
from a1.common import json_hash, load_json, set_seed, sha256_file
from a1.data import make_loaders, prepare_data, resolve_split_seed, split_digest
from a1.gru import RowGRU
from a1.transformer import PatchTransformer
from a1.train import (build_model, evaluation_budgets, model_config, save_budget_snapshot,
                      train_model, validate_budget_snapshot)

ROOT = Path(__file__).resolve().parents[1]


def config():
    value = load_json(ROOT / "configs/a1_final.json")
    value.update(device="cpu", cpu_threads=1, num_workers=0, epochs=3, batch_size=2)
    value["per_model"]["cnn"].update(cnn_channels=[4, 8, 8], dropout=0.2)
    return value


class SplitFixture(Dataset):
    """60,000 tiny dummy images for index/schema tests, never performance evidence."""
    resources = [("fixture", "not-a-real-dataset-checksum")]
    classes = [str(index) for index in range(10)]

    def __init__(self, root, train=True, download=True, transform=None):
        if not train:
            raise AssertionError("These validation-only checks must not open the test set")
        self.data = torch.zeros(60000, 1, 1, dtype=torch.uint8)
        self.targets = torch.arange(60000) // 6000

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index].float(), self.targets[index]


class UpgradeCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_legacy_split_summary_and_bytes_are_unchanged_on_reread(self):
        cfg = config()
        with self.temporary_directory() as directory, patch("a1.data.datasets.FashionMNIST", SplitFixture):
            first = prepare_data("unused", directory, cfg, include_test=False)
            root = Path(directory)
            before = {name: (root / name).read_bytes() for name in ("data_summary.json", "split_indices.npz")}
            again = prepare_data("unused", directory, cfg, include_test=False)
            self.assertEqual(first["summary"], again["summary"])
            self.assertNotIn("split_seed", first["summary"])
            self.assertNotIn("split_metadata_schema_version", first["summary"])
            for name, content in before.items():
                self.assertEqual(content, (root / name).read_bytes())
            with self.assertRaisesRegex(ValueError, "schema differs"):
                prepare_data("unused", directory, {**cfg, "split_seed": 42}, include_test=False)
            self.assertEqual(before["data_summary.json"], (root / "data_summary.json").read_bytes())

    def test_training_seed_changes_do_not_change_explicit_split(self):
        cfg = config()
        with self.temporary_directory() as directory, patch("a1.data.datasets.FashionMNIST", SplitFixture):
            root = Path(directory)
            legacy = prepare_data("unused", root / "legacy", cfg, include_test=False)
            for seed in (42, 43, 44):
                current = prepare_data("unused", root / str(seed), {**cfg, "seed": seed, "split_seed": 42}, include_test=False)
                np.testing.assert_array_equal(legacy["train_ids"], current["train_ids"])
                np.testing.assert_array_equal(legacy["val_ids"], current["val_ids"])
                self.assertEqual(current["summary"]["seed"], seed)
                self.assertEqual(current["summary"]["split_seed"], 42)
                self.assertEqual(current["summary"]["split_metadata_schema_version"], 2)
                self.assertEqual(legacy["summary"]["split_sha256"], current["summary"]["split_sha256"])
            with self.assertRaisesRegex(ValueError, "schema differs"):
                prepare_data("unused", root / "42", cfg, include_test=False)
            with self.assertRaisesRegex(ValueError, "Saved split_seed"):
                prepare_data("unused", root / "42", {**cfg, "split_seed": 43}, include_test=False)
            with self.assertRaisesRegex(ValueError, "training/legacy seed"):
                prepare_data("unused", root / "42", {**cfg, "seed": 43, "split_seed": 42}, include_test=False)

    def test_local_original_labels_produce_exact_original_split_hash(self):
        """Optional offline source-data check: no download, training or test access."""
        from torchvision.datasets import FashionMNIST
        try:
            dataset = FashionMNIST(str(ROOT / "data"), train=True, download=False)
        except RuntimeError:
            self.skipTest("Original Fashion-MNIST training files are not available locally")
        original = load_json(ROOT / "outputs/a1_m2_final_returned_867c74469240/data_summary.json")
        for training_seed in (42, 43, 44):
            split_seed = resolve_split_seed({"seed": training_seed, "split_seed": 42})
            train_ids, val_ids = train_test_split(np.arange(60000), test_size=6000,
                                                 random_state=split_seed, stratify=dataset.targets.numpy())
            self.assertEqual(split_digest(np.sort(train_ids), np.sort(val_ids)), original["split_sha256"])

    def test_seed_types_and_per_model_split_override_rejected(self):
        for key in ("seed", "split_seed"):
            for value in (True, 42.0, -1, 2 ** 32, "42"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_split_seed({"seed": 42, key: value})
        cfg = config()
        cfg["per_model"]["cnn"]["split_seed"] = 43
        with self.assertRaises(ValueError):
            model_config("cnn", cfg)

    def test_loader_reset_uses_training_seed_not_split_seed(self):
        dataset = TensorDataset(torch.arange(40).float().reshape(40, 1), torch.arange(40))
        data = {"train": dataset, "validation": dataset}
        cfg = config()

        def order(value):
            return torch.cat([labels for _, labels in make_loaders(data, value)[0]])

        torch.testing.assert_close(order({**cfg, "split_seed": 42}), order({**cfg, "split_seed": 99}))
        torch.testing.assert_close(order(cfg), order(cfg))
        self.assertFalse(torch.equal(order(cfg), order({**cfg, "seed": 43})))

    def test_gru_columns_are_exact_rows_of_transposed_image(self):
        set_seed(42)
        rows = RowGRU()
        set_seed(42)
        columns = RowGRU(sequence_axis="columns")
        self.assertEqual(sum(parameter.numel() for parameter in rows.parameters()), 61962)
        self.assertEqual(rows.state_dict().keys(), columns.state_dict().keys())
        for key, tensor in rows.state_dict().items():
            self.assertTrue(torch.equal(tensor, columns.state_dict()[key]))
        images = torch.rand(2, 1, 28, 28)
        torch.testing.assert_close(columns(images), rows(images.transpose(-1, -2)), rtol=0, atol=0)
        columns(images).sum().backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in columns.parameters()))
        factory_cfg = config()
        factory_cfg["per_model"]["gru"]["sequence_axis"] = "columns"
        self.assertEqual(build_model("gru", factory_cfg).sequence_axis, "columns")

    def test_transformer_default_and_explicit_position_have_identical_states(self):
        set_seed(42)
        default = PatchTransformer()
        set_seed(42)
        explicit = PatchTransformer(use_position_embedding=True)
        self.assertEqual(sum(parameter.numel() for parameter in default.parameters()), 274954)
        self.assertIn("position", default.state_dict())
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for key, tensor in default.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[key]))
        images = torch.rand(2, 1, 28, 28)
        default.eval()
        explicit.eval()
        torch.testing.assert_close(default(images), explicit(images), rtol=0, atol=0)

    def test_no_position_removes_only_position_and_preserves_shared_initial_weights(self):
        set_seed(42)
        positioned = PatchTransformer()
        set_seed(42)
        unpositioned = PatchTransformer(use_position_embedding=False)
        self.assertIsNone(unpositioned.position)
        self.assertNotIn("position", unpositioned.state_dict())
        self.assertEqual(sum(parameter.numel() for parameter in unpositioned.parameters()), 268682)
        self.assertEqual(sum(parameter.numel() for parameter in positioned.parameters())
                         - sum(parameter.numel() for parameter in unpositioned.parameters()), 49 * 128)
        for key, tensor in unpositioned.state_dict().items():
            self.assertTrue(torch.equal(tensor, positioned.state_dict()[key]), key)
        with torch.no_grad():
            positioned.position.zero_()
        positioned.eval()
        unpositioned.eval()
        images = torch.rand(2, 1, 28, 28)
        torch.testing.assert_close(positioned(images), unpositioned(images), rtol=0, atol=0)
        cfg = config()
        cfg["per_model"]["transformer"]["use_position_embedding"] = False
        restored = build_model("transformer", cfg)
        restored.load_state_dict(unpositioned.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(restored(images), unpositioned(images), rtol=0, atol=0)
        unpositioned.train()
        unpositioned(images).sum().backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in unpositioned.parameters()))

    def test_transformer_without_position_is_invariant_to_patch_order_in_eval(self):
        set_seed(42)
        model = PatchTransformer(use_position_embedding=False).eval()
        images = torch.rand(2, 1, 28, 28)
        patches = images.reshape(2, 1, 7, 4, 7, 4).permute(0, 2, 4, 1, 3, 5).reshape(2, 49, 1, 4, 4)
        shuffled = patches[:, torch.randperm(49)].reshape(2, 7, 7, 1, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(2, 1, 28, 28)
        torch.testing.assert_close(model(images), model(shuffled), atol=1e-6, rtol=1e-5)

    def test_invalid_ablation_options_dimensions_and_shapes_rejected(self):
        for value in ("row", True, None, []):
            with self.subTest(sequence_axis=value), self.assertRaises(ValueError):
                RowGRU(sequence_axis=value)
        for kwargs in ({"hidden_size": True}, {"hidden_size": 0}, {"num_layers": 0}, {"dropout": float("nan")}):
            with self.subTest(gru=kwargs), self.assertRaises(ValueError):
                RowGRU(**kwargs)
        for kwargs in ({"use_position_embedding": 0}, {"use_position_embedding": "false"},
                       {"patch_size": 0}, {"patch_size": 3}, {"nhead": True}, {"d_model": 10},
                       {"num_layers": 0}, {"dim_feedforward": -1}, {"dropout": float("inf")}):
            with self.subTest(transformer=kwargs), self.assertRaises(ValueError):
                PatchTransformer(**kwargs)
        for model in (RowGRU(), PatchTransformer()):
            with self.assertRaises(ValueError):
                model(torch.rand(2, 28, 28))

    def test_budget_configuration_validation(self):
        self.assertEqual(evaluation_budgets(config()), [])
        self.assertEqual(evaluation_budgets({**config(), "evaluation_budgets": [1, 3]}), [1, 3])
        for budgets in ([], [True], [0], [2, 1], [1, 1], [4], (1, 3), "1,3"):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                model_config("cnn", {**config(), "evaluation_budgets": budgets})

    def test_smoke_can_select_only_the_requested_models_and_reject_invalid_selection(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            result = training.smoke_test(config(), root / "linear_only", torch.device("cpu"), models=["linear"])
            self.assertEqual([row["model"] for row in result["models"]], ["linear"])
            self.assertEqual(sorted(path.name for path in (root / "linear_only").glob("*.pt")), ["linear_smoke.pt"])
            for index, models in enumerate(([], ["cnn", "cnn"], ["unknown"], "cnn")):
                output = root / str(index)
                with self.subTest(models=models), self.assertRaises(ValueError):
                    training.smoke_test(config(), output, torch.device("cpu"), models=models)
                self.assertFalse(output.exists())

    def test_budget_snapshot_retries_preserve_bytes_and_reject_changes(self):
        cfg = config()
        binding = {"model_name": "cnn", "config": cfg, "config_sha256": json_hash(cfg),
                   "split_sha256": "SOFTWARE-FIXTURE", "dataset_sha256": "SOFTWARE-FIXTURE",
                   "source_sha256": "SOFTWARE-FIXTURE", "run_id": "SOFTWARE-FIXTURE"}
        history = [{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.5}]
        state = {"weight": torch.tensor([1.0])}
        with self.temporary_directory() as directory:
            root = Path(directory)
            save_budget_snapshot(root, binding, history, 2, state)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            selection = validate_budget_snapshot(root, binding, history, 2)
            self.assertEqual(selection["selected_epoch"], 1)  # earliest tied epoch
            save_budget_snapshot(root, binding, history, 2, state)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            with self.assertRaises(ValueError):
                save_budget_snapshot(root, binding, history, 2, {"weight": torch.tensor([2.0])})
            changed_history = [{"epoch": 1, "val_loss": 0.6}, history[1]]
            with self.assertRaises(ValueError):
                validate_budget_snapshot(root, binding, changed_history, 2)

    def test_tiny_budget_resume_preserves_prefixes_and_matches_reference(self):
        """Only 4 random train and 4 validation tensors in a disposable software test."""
        cfg = {**config(), "seed": 43, "split_seed": 42, "evaluation_budgets": [1, 3]}
        generator = torch.Generator().manual_seed(7)
        images = torch.rand(8, 1, 28, 28, generator=generator)
        labels = torch.arange(8) % 10
        data = {"train": TensorDataset(images[:4], labels[:4]),
                "validation": TensorDataset(images[4:], labels[4:]), "val_ids": np.arange(4, 8),
                "summary": {"classes": [str(index) for index in range(10)],
                            "split_sha256": "SOFTWARE-FIXTURE", "dataset_sha256": "SOFTWARE-FIXTURE"}}
        with self.temporary_directory() as directory:
            root = Path(directory)
            reference, resumed = root / "reference", root / "resumed"
            for output in (reference, resumed):
                run_a1.prepare_run(output, cfg)
            with patch.object(training, "plot_results", return_value=[]), patch.object(
                    training, "measure_inference", return_value={"per_image_ms_amortized": 0.0}):
                train_model("cnn", data, cfg, reference, torch.device("cpu"))
                epoch, calls = training.run_epoch, [0]

                def interrupt(*args, **kwargs):
                    if len(args) > 4 and args[4] is not None:
                        calls[0] += 1
                        if calls[0] == 2:
                            raise RuntimeError("SIMULATED INTERRUPTION AFTER BUDGET 1")
                    return epoch(*args, **kwargs)

                with patch.object(training, "run_epoch", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "SIMULATED INTERRUPTION"):
                        train_model("cnn", data, cfg, resumed, torch.device("cpu"))
                snapshot = resumed / "cnn" / "best_through_1.pt"
                snapshot_bytes = snapshot.read_bytes()
                previous_best_hash = sha256_file(resumed / "cnn" / "best.pt")
                snapshot.unlink()  # only this disposable test fixture
                with self.assertRaisesRegex(FileNotFoundError, "cannot reconstruct"):
                    train_model("cnn", data, cfg, resumed, torch.device("cpu"), resume=True)
                self.assertEqual(previous_best_hash, sha256_file(resumed / "cnn" / "best.pt"))
                snapshot.write_bytes(snapshot_bytes)
                train_model("cnn", data, cfg, resumed, torch.device("cpu"), resume=True)
            self.assertEqual(snapshot_bytes, snapshot.read_bytes())
            a = torch.load(reference / "cnn" / "last.pt", weights_only=True)
            b = torch.load(resumed / "cnn" / "last.pt", weights_only=True)
            for key, tensor in a["model_state"].items():
                self.assertTrue(torch.equal(tensor, b["model_state"][key]), key)
            for budget in (1, 3):
                metrics = load_json(resumed / "cnn" / f"budget{budget}_metrics.json")
                self.assertEqual(metrics["evaluation_split"], "validation")
                self.assertEqual(metrics["prefix_budget"], budget)
                self.assertEqual(metrics["config"]["seed"], 43)
                self.assertEqual(metrics["config"]["split_seed"], 42)
                self.assertFalse(metrics["test_evaluated"])
                expected_epoch = min(b["history"][:budget], key=lambda row: row["val_loss"])["epoch"]
                self.assertEqual(metrics["best_epoch"], expected_epoch)
                self.assertTrue((resumed / "cnn" / f"budget{budget}_validation_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
