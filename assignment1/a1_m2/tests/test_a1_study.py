"""Study orchestration/statistics checks, using mocks and disposable fixtures, never full training."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import run_a1
import a1.study as study
from a1.common import json_hash, load_json, save_csv, save_json, sha256_file
from a1.train import build_model, model_config, save_budget_snapshot
from tools.summarize_a1_study import aggregate_records, compare_arms, mean_sd, paired_effect, summarize

ROOT = Path(__file__).resolve().parents[1]


def fixture_records():
    """Artificial score observations for arithmetic tests, not measured DL results."""
    records = []
    for arm, models in [("baseline", study.MODELS)] + [(a["id"], (a["model"],)) for a in study.ABLATIONS]:
        for model in models:
            for budget in (20, 60):
                for seed in (42, 43, 44):
                    value = .80 + .01 * study.MODELS.index(model) + .001 * (seed - 42)
                    value += .002 if arm != "baseline" else 0
                    records.append({"arm": arm, "model": model, "seed": seed, "budget": budget,
                                    "accuracy": value, "macro_f1": value - .002,
                                    "loss": .5 - budget / 1000, "parameters": 123,
                                    "train_seconds": float(budget), "selected_epoch": budget - 2,
                                    "selected_in_last10": True})
    return records


class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def temporary(self):
        parent = ROOT / ".private-build/study-unit"
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def protocol(self):
        return study.protocol_record(ROOT / "configs/a1_upgrade_study.json")

    def test_matrix_has24_tasks_and_only_one_controlled_change_per_ablation(self):
        protocol = self.protocol()
        runs, tasks = study.build_design(protocol)
        self.assertEqual((len(runs), len(tasks)), (12, 24))
        index = {(r["arm"], r["seed"]): r for r in runs}
        for task in tasks:
            cfg = index[(task["arm"], task["seed"])]["config"]
            self.assertEqual(cfg["seed"], task["seed"])
            self.assertEqual(cfg["split_seed"], 42)
            self.assertEqual(cfg["epochs"], 60)
            self.assertEqual(cfg["evaluation_budgets"], [20, 60])
            self.assertEqual(cfg["evaluation_policy"], study.POLICY)
        for ablation in study.ABLATIONS:
            for seed in (42, 43, 44):
                base = model_config(ablation["model"], index[("baseline", seed)]["config"])
                variant = model_config(ablation["model"], index[(ablation["id"], seed)]["config"])
                base.pop("study_arm")
                variant.pop("study_arm")
                changed = {key for key in base.keys() | variant.keys() if base.get(key) != variant.get(key)}
                self.assertEqual(changed, set(ablation["overrides"]))

    def test_protocol_rejects_test_policy_seed_or_unplanned_factor_changes(self):
        for key, value in (("official_test_already_seen", False), ("training_seeds", [1, 2, 3]),
                           ("split_seed", 43), ("epochs", 80), ("evaluation_budgets", [20, 80]),
                           ("ablations", [])):
            with self.subTest(key=key), self.temporary() as directory:
                protocol = self.protocol()
                protocol[key] = value
                path = Path(directory) / "protocol.json"
                save_json(path, protocol)
                with self.assertRaises(ValueError):
                    study.protocol_record(path)

    def test_prepare_is_idempotent_and_planned_config_tampering_is_rejected(self):
        with self.temporary() as directory:
            root = Path(directory) / "new-study"
            first = study.prepare_study(root, ROOT / "configs/a1_upgrade_study.json")
            before = (root / "study_manifest.json").read_bytes()
            self.assertEqual(first, study.prepare_study(root, ROOT / "configs/a1_upgrade_study.json"))
            self.assertEqual(before, (root / "study_manifest.json").read_bytes())
            path = root / first["runs"][0]["config_relative"]
            changed = load_json(path)
            changed["seed"] = 99
            save_json(path, changed)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                study.validate_study(root)

    def test_plan_cannot_adopt_unbound_runs_or_historical_output(self):
        with self.temporary() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            with self.assertRaisesRegex(ValueError, "Unbound"):
                study.prepare_study(root, ROOT / "configs/a1_upgrade_study.json")
        reference = ROOT / self.protocol()["reference_run"]
        for root in (reference, reference / "nested", reference.parent, ROOT / "outputs/a1_t4"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "separate"):
                study.prepare_study(root, ROOT / "configs/a1_upgrade_study.json")

    def test_reference_hash_tamper_is_detected_without_writing_inputs(self):
        original = study.sha256_file

        def changed(path):
            if str(path).replace("\\", "/").endswith("linear/validation_predictions.csv"):
                return "0" * 64
            return original(path)

        with patch.object(study, "sha256_file", side_effect=changed), self.assertRaisesRegex(ValueError, "exported file changed"):
            study.reference_record(self.protocol())

    def test_relative_paths_cannot_escape_and_partial_task_without_last_is_rejected(self):
        with self.temporary() as directory:
            root = Path(directory)
            for value in ("../x", "/tmp/x", "C:/x", "a\\b"):
                with self.subTest(path=value), self.assertRaises(ValueError):
                    study.relative_child(root, value)
            task = {"run_relative": "runs/test", "model": "cnn", "task_id": "FIXTURE.cnn"}
            (root / "runs/test/cnn").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no resumable"):
                study.task_state(root, task, {})

    def test_training_skips_complete_and_resumes_only_incomplete_tasks(self):
        with self.temporary() as directory:
            root = Path(directory)
            tasks = [{"task_id": f"FIXTURE.{i}", "run_relative": f"runs/fixture{i}",
                      "config_relative": f"configs/fixture{i}.json", "model": "cnn"} for i in range(24)]
            states = {t["task_id"]: "COMPLETE" for t in tasks}
            states[tasks[0]["task_id"]] = "PENDING"
            states[tasks[1]["task_id"]] = "INCOMPLETE"
            manifest = {"tasks": tasks, "protocol": {"study_id": "SOFTWARE_FIXTURE"}, "manifest_sha256": "FIXTURE"}
            calls = []

            def invoke(command):
                calls.append(command)
                number = int(Path(command[command.index("--out") + 1]).name.removeprefix("fixture"))
                states[tasks[number]["task_id"]] = "COMPLETE"

            with patch.object(study, "prepare_study", return_value=manifest), patch.object(study, "register_environment"), \
                    patch.object(study, "task_state", side_effect=lambda out, task, m: states[task["task_id"]]), redirect_stdout(io.StringIO()):
                result = study.execute_study(root, Path("FIXTURE"), Path("FIXTURE"), torch.device("cuda"), invoke=invoke)
            self.assertEqual(result["completed_tasks"], 24)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--resume", calls[0])
            self.assertIn("--resume", calls[1])
            self.assertTrue(all(c[c.index("--stage") + 1] == "train" for c in calls))
            self.assertEqual(load_json(root / "STUDY_COMPLETE.json")["completed_tasks"], 24)

    def test_failed_task_stops_without_marking_study_complete(self):
        with self.temporary() as directory:
            root = Path(directory)
            manifest = {"tasks": [{"task_id": "FIXTURE", "run_relative": "runs/fixture",
                                   "config_relative": "configs/fixture.json", "model": "cnn"}]}
            invoke = unittest.mock.Mock(side_effect=RuntimeError("SIMULATED PROCESS FAILURE"))
            with patch.object(study, "prepare_study", return_value=manifest), patch.object(study, "register_environment"), \
                    patch.object(study, "task_state", return_value="PENDING"), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "SIMULATED"):
                    study.execute_study(root, Path("FIXTURE"), Path("FIXTURE"), torch.device("cuda"), invoke=invoke)
            self.assertEqual(invoke.call_count, 1)
            self.assertFalse((root / "STUDY_COMPLETE.json").exists())

    def test_full_training_requires_A100_and_local_execution_lock_releases(self):
        with self.temporary() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "A100"):
                study.register_environment(root, {}, torch.device("cpu"))
            with study.execution_lock(root):
                with self.assertRaisesRegex(RuntimeError, "already training"):
                    with study.execution_lock(root):
                        self.fail("second process lock acquired")
            with study.execution_lock(root):
                pass

    def test_study_cli_blocks_test_freeze_and_all_before_dataset_access(self):
        with self.temporary() as directory:
            root = Path(directory)
            cfg = study.build_design(self.protocol())[0][0]["config"]
            save_json(root / "config.json", cfg)
            save_json(root / "run_manifest.json", {})
            save_json(root / "data_summary.json", {})
            (root / "split_indices.npz").write_bytes(b"SOFTWARE-FIXTURE")
            with patch.object(run_a1, "prepare_data") as data:
                for stage in ("test", "freeze", "all", "eda"):
                    with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "validation"):
                        run_a1.main(["--stage", stage, "--out", str(root), "--config", str(root / "config.json")])
                data.assert_not_called()

    def test_summary_uses_sample_SD_and_requires_seed_pairs(self):
        stats = mean_sd([1, 2, 3])
        self.assertEqual(stats, {"n": 3, "mean": 2, "sample_sd": 1.0})
        with self.assertRaises(ValueError):
            mean_sd([1])
        with self.assertRaises(ValueError):
            mean_sd([1, float("nan")])
        paired = paired_effect({42: 1, 43: 2, 44: 3}, {42: 2, 43: 4, 44: 6})
        self.assertEqual(paired["mean"], 2)
        self.assertEqual(paired["sample_sd"], 1)
        with self.assertRaises(ValueError):
            paired_effect({42: 1, 43: 2, 44: 3}, {42: 2, 43: 4, 45: 6})

    def test_aggregation_and_budget_pairing_require_complete_nonduplicate_seeds(self):
        records = fixture_records()
        self.assertEqual(len(records), 48)
        self.assertEqual(len(aggregate_records(records)), 16)
        effects, budgets = compare_arms(records)
        self.assertEqual((len(effects), len(budgets)), (6, 5))
        self.assertAlmostEqual(effects[0]["accuracy_delta_mean"], .002)
        self.assertAlmostEqual(effects[0]["accuracy_delta_sample_sd"], 0)
        for corrupt in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                aggregate_records(corrupt)
            with self.assertRaises(ValueError):
                compare_arms(corrupt)
        damaged = copy.deepcopy(records)
        next(r for r in damaged if r["arm"] == "baseline" and r["model"] == "linear" and r["budget"] == 60)["loss"] = 10
        with self.assertRaisesRegex(ValueError, "Minimum"):
            compare_arms(damaged)

    def test_incomplete_study_cannot_emit_complete_analysis_or_fake_results(self):
        with self.temporary() as directory:
            root = Path(directory)
            study.prepare_study(root, ROOT / "configs/a1_upgrade_study.json")
            with self.assertRaisesRegex(ValueError, "Incomplete study: 0/24"):
                summarize(root)
            self.assertFalse((root / "analysis").exists())

    def test_full_summary_contract_and_input_tamper_using_explicit_software_fixture(self):
        """Build synthetic artifact records, not a training experiment or measured model scores."""
        with self.temporary() as directory:
            root = Path(directory) / "software-fixture-no-training"
            protocol = self.protocol()
            protocol["study_id"] = "software_fixture_no_training"
            protocol_path = Path(directory) / "fixture_protocol.json"
            save_json(protocol_path, protocol)
            manifest = study.prepare_study(root, protocol_path)
            reference = ROOT / protocol["reference_run"]
            truth_rows = study.read_csv(reference / "linear/validation_predictions.csv")
            perfect = [{**row, "predicted": row["true"]} for row in truth_rows]
            env = {"device": "cuda", "gpu": "SOFTWARE_FIXTURE_A100_NOT_REAL_HARDWARE",
                   "versions": {"scope": "SOFTWARE_FIXTURE_NO_EXECUTION"},
                   "cuda_runtime": "FIXTURE", "cudnn_version": "FIXTURE"}
            save_json(root / "study_environment.json", env)
            source = manifest["source"]["training_source"]
            history = [{"epoch": epoch, "train_loss": 2 - epoch / 100, "val_loss": 2 - epoch / 100,
                        "train_accuracy": 1.0, "val_accuracy": 1.0,
                        "train_macro_f1": 1.0, "val_macro_f1": 1.0,
                        "learning_rate": .001, "train_seconds": .001, "val_seconds": .001,
                        "gpu_peak_allocated_bytes": 0} for epoch in range(1, 61)]
            for run in manifest["runs"]:
                output = root / run["run_relative"]
                cfg = run["config"]
                identity = "SOFTWARE_FIXTURE_" + run["run_key"]
                save_json(output / "config.json", cfg)
                save_json(output / "run_manifest.json", {"schema_version": 1, "run_id": identity,
                          "config_sha256": json_hash(cfg), "source_sha256": source["source_sha256"], "source": source})
                data = load_json(reference / "data_summary.json")
                data.update(seed=run["seed"], split_seed=42, split_metadata_schema_version=2)
                save_json(output / "data_summary.json", data)
                (output / "split_indices.npz").write_bytes((reference / "split_indices.npz").read_bytes())
                for name in run["models"]:
                    target = output / name
                    target.mkdir()
                    effective = model_config(name, cfg)
                    torch.manual_seed(run["seed"])
                    model = build_model(name, effective)  # Initialization only, no forward/backward/training.
                    states = model.state_dict()
                    count = sum(p.numel() for p in model.parameters())
                    binding = {"model_name": name, "config": effective, "config_sha256": json_hash(effective),
                               "split_sha256": data["split_sha256"], "dataset_sha256": data["dataset_sha256"],
                               "source_sha256": source["source_sha256"], "run_id": identity}
                    torch.save({**binding, "epoch": 60, "val_loss": history[-1]["val_loss"], "model_state": states}, target / "best.pt")
                    save_json(target / "model_config.json", effective)
                    save_csv(target / "history.csv", history)
                    save_csv(target / "validation_predictions.csv", perfect)
                    save_csv(target / "confusion_matrix.csv", [{"true_label": k, **{
                        f"predicted_{j}": 600 if k == j else 0 for j in range(10)}} for k in range(10)])
                    for filename in ("curves.png", "confusion_matrix.png", "examples.png"):
                        (target / filename).write_bytes(b"SOFTWARE_FIXTURE_NOT_A_FIGURE")
                    metrics = {"model": name, "evaluation_split": "validation", "n_eval": 6000,
                               "loss": history[-1]["val_loss"], "accuracy": 1.0, "macro_f1": 1.0,
                               "parameters": count, "best_epoch": 60, "epochs_completed": 60,
                               "checkpoint_rule": "lowest validation loss; first if tied", "checkpoint": "best.pt",
                               "checkpoint_sha256": sha256_file(target / "best.pt"), "train_seconds": .06,
                               "validation_seconds": .06, "fit_wall_seconds": .12, "gpu_peak_allocated_bytes": 0,
                               "run_id": identity, "config": effective, "config_sha256": json_hash(effective),
                               "data": data, "source": source, "environment": env, "test_evaluated": False,
                               "top_errors": [], "inference": {"batch_size": 128, "warmup_batches": 10,
                               "measured_batches": 50, "batch_ms": 1, "per_image_ms_amortized": 1/128}}
                    save_json(target / "metrics.json", metrics)
                    for budget in (20, 60):
                        save_budget_snapshot(target, binding, history, budget, states)
                        selection = load_json(target / f"best_through_{budget}_selection.json")
                        prediction_path = target / f"budget{budget}_validation_predictions.csv"
                        save_csv(prediction_path, perfect)
                        budget_metrics = {**metrics, "prefix_budget": budget, "best_epoch": budget,
                            "loss": history[budget-1]["val_loss"], "train_seconds": budget * .001,
                            "epoch_validation_seconds": budget * .001, "source_sha256": source["source_sha256"],
                            "split_sha256": data["split_sha256"], "dataset_sha256": data["dataset_sha256"],
                            "checkpoint": selection["checkpoint"], "checkpoint_sha256": selection["checkpoint_sha256"],
                            "selection_history_sha256": selection["selection_history_sha256"],
                            "validation_predictions": prediction_path.name, "predictions_sha256": sha256_file(prediction_path)}
                        save_json(target / f"budget{budget}_metrics.json", budget_metrics)
            save_json(root / "STUDY_COMPLETE.json", {"study_id": protocol["study_id"],
                      "manifest_sha256": manifest["manifest_sha256"], "completed_tasks": 24, "evaluation_policy": study.POLICY})
            # The renderer is separate; this test audits all48 records/checkpoint/prediction bindings.
            with patch("tools.summarize_a1_study.render_figures"):
                result = summarize(root)
                self.assertEqual(result["study_id"], "software_fixture_no_training")
                self.assertEqual(result["n_budget_records"], 48)
                self.assertEqual(len(result["paired_ablations"]), 6)
                self.assertTrue((root / "analysis/ANALYSIS_COMPLETE.json").exists())
                self.assertEqual(summarize(root)["n_tasks"], 24)  # idempotent readback
            damaged = root / manifest["runs"][0]["run_relative"] / "linear/budget20_validation_predictions.csv"
            damaged.write_bytes(damaged.read_bytes() + b"TAMPERED-SOFTWARE-FIXTURE\n")
            with self.assertRaisesRegex(ValueError, "input changed"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
