"""Independent numerical and split-contract checks for calibration v2."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from garmentlab.calibration_v2 import (
    binomial_error_upper_bound,
    fit_calibration_v2,
    partition_calibration,
    review_mask,
    select_review_policy,
)


def records(count=100, repeats=2):
    return [{"annotation_id": i * repeats + j, "array_index": i * repeats + j,
             "source_group_id": f"source-{i}", "evaluation_group_id": f"component-{i}",
             "role": "calibration", "partition": "calibration"}
            for i in range(count) for j in range(repeats)]


def test_group_split_and_representatives_are_metadata_only_and_permutation_stable():
    data = records(18, 3)
    # Two otherwise different sources are near-duplicate components.
    for row in data:
        if row["source_group_id"] == "source-1":
            row["evaluation_group_id"] = "component-0"
    partition = partition_calibration(data)
    assert partition["source_groups"] == 18 and partition["components"] == 17
    left, right = set(partition["temperature_indices"]), set(partition["threshold_indices"])
    assert not left & right and left | right == set(range(len(data)))
    for field in ["source_group_id", "evaluation_group_id"]:
        assert {data[i][field] for i in left}.isdisjoint(data[i][field] for i in right)
    reps = partition["threshold_representative_indices"]
    assert len(reps) == partition["threshold_components"]
    assert len({data[i]["evaluation_group_id"] for i in reps}) == len(reps)
    reversed_data = data[::-1]
    reverse = partition_calibration(reversed_data)
    for key in ["metadata_sha256", "temperature_split_sha256", "threshold_representative_split_sha256"]:
        assert partition[key] == reverse[key]
    for key in ["temperature_indices", "threshold_indices", "threshold_representative_indices"]:
        assert {data[i]["annotation_id"] for i in partition[key]} == {reversed_data[i]["annotation_id"] for i in reverse[key]}


def test_source_union_prevents_sources_with_multiple_duplicate_components_crossing():
    data = records(8, 2)
    data[1]["evaluation_group_id"] = "component-1"
    p = partition_calibration(data)
    assert p["components"] == 7
    a, b = p["temperature_indices"], p["threshold_indices"]
    assert {data[i]["source_group_id"] for i in a}.isdisjoint(data[i]["source_group_id"] for i in b)


@pytest.mark.parametrize("bad", [
    [], [{"source_group_id": "x", "annotation_id": 1, "role": "assessment"}] * 2,
    [{"source_group_id": "x", "annotation_id": 1, "role": "calibration"}] * 2,
    [{"source_group_id": "x", "annotation_id": i, "role": "calibration"} for i in range(3)],
])
def test_partition_refuses_noncalibration_duplicate_or_single_component_data(bad):
    with pytest.raises(ValueError):
        partition_calibration(bad)


@pytest.mark.parametrize("cfg", [
    {"temperature_fraction": 0}, {"target_accuracy": 1.1},
    {"family_error_probability": 0}, {"minimum_accepted_groups": True},
    {"threshold_grid": [.5, .5]}, {"threshold_grid": [1, .5]},
    {"threshold_grid": [float("nan")]}, {"temperature_minimum": 2},
])
def test_invalid_configuration_is_rejected(cfg):
    with pytest.raises(ValueError):
        partition_calibration(records(4), cfg)


def test_exact_error_bound_boundary_and_closed_form_zero_errors():
    assert binomial_error_upper_bound(0, 0, alpha=.05) == 1
    assert binomial_error_upper_bound(7, 7, alpha=.05) == 1
    assert binomial_error_upper_bound(0, 100, alpha=.05) == pytest.approx(1 - .05**.01, abs=1e-14)
    for errors in [1, 2, 5, 12, 19]:
        n, alpha = 20, .013
        p = binomial_error_upper_bound(errors, n, alpha=alpha)
        # Independent direct binomial arithmetic instead of the implementation's
        # log-gamma/log-sum-exp path.
        cdf = sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(errors+1))
        assert cdf == pytest.approx(alpha, rel=2e-12, abs=2e-14)
    bounds = [binomial_error_upper_bound(e, 20, alpha=.05) for e in range(21)]
    assert bounds == sorted(bounds)
    assert binomial_error_upper_bound(2, 100, alpha=.001) > binomial_error_upper_bound(2, 100, alpha=.05)


@pytest.mark.parametrize("args", [(-1, 2, .05), (3, 2, .05), (False, 2, .05), (0, 2, 0), (0, 2, float("nan"))])
def test_invalid_binomial_arguments(args):
    with pytest.raises(ValueError):
        binomial_error_upper_bound(args[0], args[1], alpha=args[2])


def test_multiple_threshold_search_uses_corrected_bounds_and_does_not_fallback():
    p = np.tile([.99, .01], (50, 1))
    y = np.zeros(50, dtype=np.int64)
    uncorrected = binomial_error_upper_bound(0, 50, alpha=.05)
    assert uncorrected < .1
    result = select_review_policy(p, y, {"target_accuracy": .9, "threshold_grid": [0, .5, 1]})
    assert result["per_threshold_error_probability"] == pytest.approx(.05/3)
    # Fifty successes at3thresholds do pass a90%target. The default101threshold
    # search does not pass a95%target even with zero observed errors.
    assert result["mode"] == "confidence_threshold"
    strict = select_review_policy(p, y)
    assert strict["mode"] == "review_all" and not strict["eligible"]
    assert strict["empirical_only_diagnostic"]["accepted_accuracy"] == 1
    assert strict["empirical_diagnostic_is_active_policy"] is False


def test_risk_selection_maximizes_eligible_coverage_and_preserves_confidence_ties():
    probabilities = np.r_[np.tile([[.99, .01]], (300, 1)),
                          np.tile([[.6, .4]], (100, 1))]
    targets = np.r_[np.zeros(300, dtype=np.int64), np.ones(100, dtype=np.int64)]
    result = select_review_policy(probabilities, targets, {"threshold_grid": [0, .6, .8, .9, 1]})
    assert result["eligible"]
    assert result["review_threshold"] == .8
    assert result["selected"]["accepted_count"] == 300
    assert result["selected"]["error_upper_bound"] <= .05
    assert result["selected"]["coverage"] == .75


def test_review_all_rejects_even_exactly_one_probability_and_legacy_policy_still_works():
    p = np.array([[1., 0.], [.5, .5]])
    assert review_mask(p, {"policy": {"mode": "review_all"}, "review_threshold": 1}).tolist() == [True, True]
    assert review_mask(p, {"review_threshold": .5}).tolist() == [False, False]
    with pytest.raises(ValueError):
        review_mask(p, {"policy": {"mode": "unknown"}})


def test_fit_uses_only_temperature_partition_labels_and_frozen_representatives():
    data = records(200, 2)
    frozen = partition_calibration(data)
    rng = np.random.default_rng(777)
    logits = rng.normal(size=(len(data), 3)) * 2
    labels = logits.argmax(1).astype(np.int64)
    # Temperature data contain mistakes so the optimum is not just a bound.
    temp = frozen["temperature_indices"]
    labels[temp[::5]] = (labels[temp[::5]] + 1) % 3
    original = fit_calibration_v2(logits, labels, data, partition=frozen)
    changed = labels.copy()
    risk = frozen["threshold_indices"]
    changed[risk] = (changed[risk] + 1) % 3
    other = fit_calibration_v2(logits, changed, data, partition=frozen)
    assert original["binding"]["temperature"] == other["binding"]["temperature"]
    assert original["binding"]["fit_split_sha256"] != original["binding"]["threshold_selection_split_sha256"]
    assert original["risk_control"]["unconditional_accuracy_guarantee"] is False
    assert original["threshold_selection"]["representative_count"] == len(frozen["threshold_representative_indices"])
    json.dumps(original, allow_nan=False)
    tampered = deepcopy(frozen)
    tampered["temperature_indices"] = tampered["temperature_indices"][1:]
    with pytest.raises(ValueError, match="Frozen calibration partition"):
        fit_calibration_v2(logits, labels, data, partition=tampered)


def test_nontarget_representative_rows_do_not_choose_review_threshold():
    data = records(300, 2)
    partition = partition_calibration(data)
    logits = np.tile([8., 0.], (len(data), 1))
    labels = np.zeros(len(data), dtype=np.int64)
    original = fit_calibration_v2(logits, labels, data, partition=partition)
    other_logits = logits.copy()
    ignored = sorted(set(partition["threshold_indices"]) - set(partition["threshold_representative_indices"]))
    other_logits[ignored] = [-100., 100.]
    revised = fit_calibration_v2(other_logits, labels, data, partition=partition)
    assert original["binding"] == revised["binding"]
    assert original["threshold_selection"] == revised["threshold_selection"]
    assert original["threshold_all_crop_diagnostic"] != revised["threshold_all_crop_diagnostic"]


def test_calibration_targets_must_match_bound_metadata():
    data = records(4)
    data[0]["class_index"] = 1
    with pytest.raises(ValueError, match="record class indices"):
        fit_calibration_v2(np.tile([1., 0.], (8, 1)), np.zeros(8, dtype=np.int64), data)
