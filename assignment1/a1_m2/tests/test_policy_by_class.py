"""Policy-diagnostic contracts use small probability fixtures, never models."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('class_policy_analysis', Path(__file__).resolve().parents[1]/'tools/analyze_policy_by_class.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class ClassPolicyTests(unittest.TestCase):
    def test_threshold_equality_is_accepted_without_rounding_or_tie_breaking(self):
        p=np.array([[.88,.12],[.879999,.120001],[.12,.88]])
        pred,accepted=analysis.policy_decisions(p,np.array([0,0,1]),.88)
        np.testing.assert_array_equal(pred,[0,0,1])
        np.testing.assert_array_equal(accepted,[True,False,True])

    def test_true_and_predicted_groups_preserve_different_denominators(self):
        y=np.array([0,0,0,1,1,1]);pred=np.array([0,1,1,1,0,1]);accepted=np.array([1,1,0,1,0,1],dtype=bool)
        d=analysis.summarize(y,pred,accepted,labels=('shirt','pants'))
        self.assertEqual(d['true_class'][0]['accepted_count'],2)
        self.assertEqual(d['true_class'][0]['accepted_correct'],1)
        self.assertEqual(d['predicted_class'][0]['accepted_count'],1)
        self.assertEqual(d['predicted_class'][0]['accepted_correct'],1)
        self.assertEqual(d['true_class'][0]['accepted_accuracy'],.5)
        self.assertEqual(d['predicted_class'][0]['accepted_accuracy'],1)
        self.assertEqual(sum(r['accepted_count']for r in d['true_class']),d['overall']['accepted_count'])
        self.assertEqual(sum(r['accepted_errors']for r in d['predicted_class']),d['overall']['accepted_errors'])

    def test_empty_accepted_class_has_undefined_accuracy(self):
        d=analysis.summarize(np.array([0,1]),np.array([0,1]),np.array([False,True]),labels=('shirt','pants'))
        self.assertIsNone(d['true_class'][0]['accepted_accuracy'])
        self.assertEqual(d['true_class'][0]['coverage'],0)

    def test_review_routes_errors_and_correct_cases_without_claiming_correction(self):
        d=analysis.summarize(np.array([0,0,1,1]),np.array([1,0,0,1]),np.array([True,False,False,True]),labels=('shirt','pants'))
        self.assertEqual(d['overall']['accepted_errors'],1)
        self.assertEqual(d['overall']['review_errors'],1)
        self.assertEqual(d['overall']['review_correct'],1)
        self.assertEqual(d['overall']['review_error_capture'],.5)

    def test_component_interval_replay_is_seeded_and_accounts_for_group_units(self):
        y=np.array([0,0,1,1]);pred=np.array([0,1,1,1]);accepted=np.array([1,1,0,1],dtype=bool)
        kwargs=dict(labels=('shirt','pants'),replicates=50,seed=123)
        a=analysis.component_intervals(y,pred,accepted,['a','a','b','c'],**kwargs)
        b=analysis.component_intervals(y,pred,accepted,['a','a','b','c'],**kwargs)
        self.assertEqual(a,b)
        self.assertEqual(a['resampling_components'],3)
        self.assertTrue(all(0<=x<=1 for x in a['true_class'][0]['coverage_interval']))

    def test_invalid_probabilities_and_out_of_range_labels_are_rejected(self):
        with self.assertRaises(ValueError):analysis.policy_decisions([[.8,.8]],np.array([0]),.88)
        with self.assertRaises(ValueError):analysis.policy_decisions([[.8,.2]],np.array([2]),.88)


if __name__ == '__main__':
    unittest.main()
