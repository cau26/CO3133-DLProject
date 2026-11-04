"""Seven-day display/config and Run all orchestration, without GPU training."""
import ast
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('seven_day_notebook_harness',ROOT/'tests/test_a1opt_notebook.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NOTEBOOK=ROOT/'notebooks/A1_Optimize_A100_v3_1_7Days.ipynb'


class SevenDayNotebookTests(base.OptimizationNotebookTests):
    def setUp(self):
        patched=patch.object(base,'NOTEBOOK',NOTEBOOK)
        patched.start()
        self.addCleanup(patched.stop)
        super().setUp()

    def namespace(self,*args,**kwargs):
        namespace,controller,download=super().namespace(*args,**kwargs)
        config=namespace['REPO']/'configs/a1_optimization_v3.json'
        config.parent.mkdir(parents=True,exist_ok=True)
        config.write_text(json.dumps({'time_budget_seconds':604800}),encoding='utf-8')
        return namespace,controller,download

    def test_default_training_is_false_and_only_allowed_controller_stages_are_called(self):
        tree=ast.parse(base._source(self.cells[3]))
        switch=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='RUN_OPTIMIZATION' for t in n.targets))
        self.assertIs(switch.value.value,True)
        stages=[n.args[0].value for c in self.cells for n in ast.walk(ast.parse(base._source(c)))
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_optimization']
        self.assertEqual(stages,['plan','smoke','run','status','export'])

    def test_actual_config_budget_is_displayed_and_old_six_hour_label_absent(self):
        _,_,_,output=self.run_after_preflight(training=True,outcome='COMPLETE')
        self.assertIn('7 ngày (604800 giây)',output)
        self.assertNotIn('6 giờ',output)
        self.assertNotIn('6 giờ',''.join(base._source(c) for c in self.notebook['cells']))

    def test_mismatched_runtime_budget_is_rejected_before_plan(self):
        namespace,controller,_=self.namespace()
        config=namespace['REPO']/'configs/a1_optimization_v3.json'
        config.write_text(json.dumps({'time_budget_seconds':21600}),encoding='utf-8')
        with self.assertRaises(AssertionError): self.execute(1,namespace)
        self.assertEqual(controller.stages,[])

    def test_new_paths_are_distinct_from_six_hour_notebook(self):
        setup=base._source(self.cells[0])
        for value in ('CO3133_A1_Optimization_v3_1_7Days_Source.zip',
                      '/content/CO3133-A1-opt3-v3-1-7days',
                      'G-M10_A1_Optimization_v3_1_7days','g_m10_a1_opt3_v3_1_7days_run01'):
            self.assertIn(value,setup)
