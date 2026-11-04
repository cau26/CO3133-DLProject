import numpy as np
import pytest
from garmentlab.uncertainty_v2 import paired_group_bootstrap


def inputs():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    records = [{"role": "assessment", "source_group_id": f"s{i}", "evaluation_group_id": f"d{i}"} for i in range(6)]
    tasks = [{"task_id": f"{c}-{s}", "candidate_id": c, "seed": s} for c in ["C0", "C1"] for s in [314, 2718, 5772]]
    predictions = {t["task_id"]: (1-y if t["candidate_id"] == "C0" else y.copy()) for t in tasks}
    return y, records, tasks, predictions


def test_paired_components_reproducible_and_perfect_improvement_is_positive():
    y, records, tasks, predictions = inputs()
    first = paired_group_bootstrap(y, predictions, tasks, records, selected_candidate="C1", classes=2)
    assert first == paired_group_bootstrap(y, predictions, tasks, records, selected_candidate="C1", classes=2)
    assert first["paired_mean_macro_f1_delta_interval"][0] > 0
    assert first["resampling_components"] == 6
    same = paired_group_bootstrap(y, predictions, [t for t in tasks if t["candidate_id"] == "C0"], records,
                                  selected_candidate="C0", classes=2)
    assert same["paired_mean_macro_f1_delta_interval"] == [0., 0.]


def test_source_and_duplicate_edges_form_one_resampling_component():
    y, records, tasks, predictions = inputs()
    records[1]["source_group_id"] = records[0]["source_group_id"]
    records[2]["evaluation_group_id"] = records[1]["evaluation_group_id"]
    result = paired_group_bootstrap(y, predictions, tasks, records, selected_candidate="C1", classes=2)
    assert result["resampling_components"] == 4
    records[0]["role"] = "fit"
    with pytest.raises(ValueError, match="assessment-role"):
        paired_group_bootstrap(y, predictions, tasks, records, selected_candidate="C1", classes=2)
