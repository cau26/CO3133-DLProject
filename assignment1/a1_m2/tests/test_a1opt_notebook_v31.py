"""Exercise the corrected notebook using the original fake controller harness."""
import ast
import importlib.util
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "original_notebook_harness", ROOT / "tests/test_a1opt_notebook.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NEW_NOTEBOOK = ROOT / "notebooks/A1_Optimize_A100_v3_1_Fix.ipynb"


class CorrectedNotebookTests(base.OptimizationNotebookTests):
    def setUp(self):
        patched = patch.object(base, "NOTEBOOK", NEW_NOTEBOOK)
        patched.start()
        self.addCleanup(patched.stop)
        super().setUp()

    def test_default_training_is_false_and_only_allowed_controller_stages_are_called(self):
        # The student has explicitly requested a fresh full run; Run all starts it.
        tree = ast.parse(base._source(self.cells[3]))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "RUN_OPTIMIZATION"
                                  for target in node.targets))
        self.assertIs(assignment.value.value, True)
        stages = [node.args[0].value for cell in self.cells
                  for node in ast.walk(ast.parse(base._source(cell)))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "run_optimization"]
        self.assertEqual(stages, ["plan", "smoke", "run", "status", "export"])

    def test_new_source_runtime_and_drive_paths_are_isolated(self):
        setup = base._source(self.cells[0])
        self.assertIn("CO3133_A1_Optimization_v3_1_Source.zip", setup)
        self.assertIn("/content/CO3133-A1-opt3-v3-1", setup)
        self.assertIn("G-M10_A1_Optimization_v3_1", setup)
        self.assertIn("g_m10_a1_opt3_v3_1_run01", setup)
        self.assertNotIn("tự bật training", "".join(base._source(c) for c in self.notebook["cells"]))

