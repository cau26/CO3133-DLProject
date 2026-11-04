"""Notebook orchestration checks with a fake CLI; no GPU, pip, data or training.

The first cell contains platform authentication, package installation and GPU
preflight. It is syntax-checked here, never executed. The remaining cells run
against a tiny fake controller and temporary export files.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/A1_Optimize_A100_v3.ipynb"


def _code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def _training_switch(source, enabled):
    """Represent the student's documented edit without executing other changes."""
    tree = ast.parse(source)
    switches = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "RUN_OPTIMIZATION"
                        for target in node.targets)]
    if len(switches) != 1:
        raise AssertionError("The notebook must expose one explicit training switch")
    switches[0].value = ast.Constant(enabled)
    return compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec")


class FakeController:
    """Model plan/smoke/run/status/export effects, never fitting any model."""

    def __init__(self, outcome="INCOMPLETE", fail_run=False):
        self.outcome = outcome
        self.fail_run = fail_run
        self.stages = []
        self.state = "PLANNED_NOT_TRAINED"
        self.completed_override = None

    def __call__(self, args, cwd=None):
        args = [str(value) for value in args]
        if args[:3] != [sys.executable, "-u", "tools/run_a1_optimization.py"]:
            raise AssertionError("Unexpected command, installation or external process")
        stage = args[args.index("--stage") + 1]
        if stage not in {"plan", "smoke", "run", "status", "export"}:
            raise AssertionError("Unexpected stage, including any official-test evaluation")
        self.stages.append(stage)
        out = Path(args[args.index("--out") + 1])
        if stage == "plan":
            out.mkdir(parents=True, exist_ok=True)
        elif stage == "smoke":
            print("FAKE_SMOKE_OK")
        elif stage == "run":
            if self.fail_run:
                self.state = "INCOMPLETE_INTERRUPTED"
                raise subprocess.CalledProcessError(17, args)
            self.state = self.outcome
        elif stage == "status":
            print("CONTROLLER_STATUS=" + self.state)
        elif stage == "export":
            archive_path = out.parent / (out.name + "_results.zip")
            completed = 61 if self.state == "COMPLETE" else 7
            if self.completed_override is not None:
                completed = self.completed_override
            receipt = {"status": self.state, "completed_tasks": completed,
                       "maximum_tasks": 61, "synthetic_orchestration_only": True}
            (out / "export_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("status.json", json.dumps(receipt))
        return subprocess.CompletedProcess(args, 0)


class OptimizationNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook, self.cells = _code_cells()
        temporary_parent = ROOT / ".private-build"
        temporary_parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="a1opt-notebook-test-", dir=temporary_parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def namespace(self, outcome="INCOMPLETE", fail_run=False):
        controller = FakeController(outcome, fail_run)
        download = Mock(name="files.download")
        namespace = {
            "Path": Path, "sys": sys, "json": json, "PREFLIGHT_OK": True,
            "REPO": self.directory / "source",
            "DEFAULT_OUT": self.directory / "runs" / "fake_run",
            "checked": Mock(name="checked", side_effect=controller),
            "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "files": SimpleNamespace(download=download),
        }
        return namespace, controller, download

    def execute(self, index, namespace, training=False):
        source = _source(self.cells[index])
        compiled = (_training_switch(source, training) if index == 3
                    else compile(source, str(NOTEBOOK), "exec"))
        exec(compiled, namespace)

    def run_after_preflight(self, training=False, outcome="INCOMPLETE"):
        namespace, controller, download = self.namespace(outcome)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for index in range(1, 5):
                self.execute(index, namespace, training)
        return namespace, controller, download, output.getvalue()

    def test_all_cells_parse_and_notebook_contains_no_executed_results(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.cells), 5)
        for cell in self.cells:
            ast.parse(_source(cell))
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_default_training_is_false_and_only_allowed_controller_stages_are_called(self):
        tree = ast.parse(_source(self.cells[3]))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "RUN_OPTIMIZATION"
                                  for target in node.targets))
        self.assertIs(assignment.value.value, False)
        all_stages = []
        forbidden_flags = {"--test", "--official-test", "--eval-test", "--evaluate-test",
                           "test", "official_test", "evaluate_test"}
        for cell in self.cells:
            for node in ast.walk(ast.parse(_source(cell))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(node.value, forbidden_flags)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "run_optimization"):
                    self.assertIsInstance(node.args[0], ast.Constant)
                    all_stages.append(node.args[0].value)
        self.assertEqual(all_stages, ["plan", "smoke", "run", "status", "export"])

    def test_resume_compares_saved_distribution_versions_without_build_suffix_false_alarm(self):
        tree = ast.parse(_source(self.cells[0]))
        conditions = [node for node in tree.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name) and node.test.id == "saved_versions"]
        self.assertEqual(len(conditions), 1)
        conditional = compile(ast.Module(body=conditions, type_ignores=[]), str(NOTEBOOK), "exec")
        saved = {"torch": "2.10.0", "torchvision": "0.25.0"}
        namespace = {
            "saved_versions": saved,
            "torch": SimpleNamespace(__version__="2.10.0+cpu"),
            "torchvision": SimpleNamespace(__version__="0.25.0+cpu"),
            "package_version": Mock(side_effect=saved.__getitem__),
        }
        # A build suffix in the imported module is not an environment change.
        exec(conditional, namespace)
        self.assertEqual([call.args[0] for call in namespace["package_version"].call_args_list],
                         ["torch", "torchvision"])
        # A real distribution-version change must still reject resume.
        for changed_package in ("torch", "torchvision"):
            with self.subTest(changed_package=changed_package):
                installed = dict(saved)
                installed[changed_package] = "99.0.0"
                namespace["package_version"] = Mock(side_effect=installed.__getitem__)
                with self.assertRaises(AssertionError):
                    exec(conditional, namespace)

    def test_training_cell_false_then_true_runs_once(self):
        namespace, controller, _ = self.namespace("COMPLETE")
        with contextlib.redirect_stdout(io.StringIO()):
            self.execute(1, namespace)
            self.execute(2, namespace)
            self.execute(3, namespace, training=False)
            self.assertFalse(namespace["RUN_REQUESTED"])
            self.assertFalse(namespace["TRAINING_RETURNED_OK"])
            self.assertNotIn("run", controller.stages)
            self.execute(3, namespace, training=True)
        self.assertTrue(namespace["RUN_REQUESTED"])
        self.assertTrue(namespace["TRAINING_RETURNED_OK"])
        self.assertEqual(controller.stages.count("run"), 1)

    def test_run_all_false_exports_plan_without_claiming_training_complete(self):
        namespace, controller, download, output = self.run_after_preflight(training=False)
        self.assertEqual(controller.stages, ["plan", "smoke", "status", "export"])
        self.assertIn("SKIPPED", output)
        self.assertIn("CONTROLLER_STATUS=PLANNED_NOT_TRAINED", output)
        self.assertNotIn("CONTROLLER_STATUS=COMPLETE", output)
        self.assertIn("ZIP MỘT PHẦN: 7/61", output)
        self.assertNotIn("HOÀN TẤT: 61/61", output)
        self.assertIn("bước 4 đang False", output)
        self.assertTrue(namespace["RESULT_ZIP"].is_file())
        receipt = json.loads(namespace["EXPORT_RECEIPT"].read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "PLANNED_NOT_TRAINED")
        download.assert_called_once_with(str(namespace["RESULT_ZIP"]))

    def test_run_all_true_exports_both_complete_and_partial_states_honestly(self):
        for outcome in ("COMPLETE", "INCOMPLETE_TIME_BUDGET"):
            with self.subTest(outcome=outcome):
                namespace, controller, download, output = self.run_after_preflight(
                    training=True, outcome=outcome)
                self.assertEqual(controller.stages, ["plan", "smoke", "run", "status", "export"])
                self.assertIn("CONTROLLER_STATUS=" + outcome, output)
                receipt = json.loads(namespace["EXPORT_RECEIPT"].read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], outcome)
                with zipfile.ZipFile(namespace["RESULT_ZIP"]) as archive:
                    self.assertEqual(json.loads(archive.read("status.json"))["status"], outcome)
                if outcome != "COMPLETE":
                    self.assertNotIn("CONTROLLER_STATUS=COMPLETE", output)
                    self.assertIn("ZIP MỘT PHẦN: 7/61", output)
                    self.assertNotIn("HOÀN TẤT: 61/61", output)
                else:
                    self.assertIn("HOÀN TẤT: 61/61", output)
                    self.assertNotIn("ZIP MỘT PHẦN", output)
                self.assertNotIn("SKIPPED", output)
                download.assert_called_once_with(str(namespace["RESULT_ZIP"]))

    def test_failed_training_propagates_and_manual_export_preserves_partial_state(self):
        namespace, controller, download = self.namespace(fail_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.execute(1, namespace)
            self.execute(2, namespace)
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(3, namespace, training=True)
            self.assertFalse(namespace["TRAINING_RETURNED_OK"])
            self.execute(4, namespace)
        receipt = json.loads(namespace["EXPORT_RECEIPT"].read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "INCOMPLETE_INTERRUPTED")
        download.assert_called_once()

    def test_complete_label_requires_both_complete_status_and_all_61_tasks(self):
        for status, count in (("COMPLETE", 60), ("INCOMPLETE_TIME_BUDGET", 61)):
            with self.subTest(status=status, count=count):
                namespace, controller, _ = self.namespace(status)
                controller.completed_override = count
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    for index in range(1, 5):
                        self.execute(index, namespace, training=True)
                self.assertIn(f"ZIP MỘT PHẦN: {count}/61", output.getvalue())
                self.assertNotIn("HOÀN TẤT: 61/61", output.getvalue())

    def test_failed_smoke_guard_prevents_training(self):
        namespace, controller, _ = self.namespace()
        namespace["SMOKE_OK"] = False
        with self.assertRaises(AssertionError):
            self.execute(3, namespace, training=True)
        self.assertEqual(controller.stages, [])


if __name__ == "__main__":
    unittest.main()
