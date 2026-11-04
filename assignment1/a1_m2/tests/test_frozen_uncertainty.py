"""Statistical edge cases and paired-input safeguards, without model execution."""
import math
from pathlib import Path

import pytest

from tools.analyze_a1_frozen_uncertainty import (
    binomtest, ensure_output_paths, exact_mcnemar_pvalue, holm_adjust,
    paired_counts, read_predictions, validate_predictions, wilson_interval,
)


def test_wilson_edges_remain_bounded_and_non_degenerate():
    lower, upper = wilson_interval(0, 10)
    assert lower == 0 and upper == pytest.approx(0.2775327998628892)
    lower, upper = wilson_interval(10, 10)
    assert lower == pytest.approx(0.7224672001371107) and upper == 1
    assert wilson_interval(0, 1)[1] > 0.5


def test_wilson_known_midpoint_and_complement_symmetry():
    assert wilson_interval(5, 10) == pytest.approx((0.236593090512564, 0.763406909487436))
    low, high = wilson_interval(17, 23)
    reverse_low, reverse_high = wilson_interval(6, 23)
    assert (low, high) == pytest.approx((1 - reverse_high, 1 - reverse_low))


@pytest.mark.parametrize("correct,total,confidence", [(1, 0, .95), (11, 10, .95),
    (-1, 10, .95), (True, 10, .95), (1.0, 10, .95), (1, 10, 1), (1, 10, 0)])
def test_wilson_rejects_invalid_observations(correct, total, confidence):
    with pytest.raises(ValueError):
        wilson_interval(correct, total, confidence)


def test_exact_mcnemar_small_enumerable_cases_and_symmetry():
    assert exact_mcnemar_pvalue(0, 0) == 1
    assert exact_mcnemar_pvalue(0, 1) == 1
    assert exact_mcnemar_pvalue(0, 6) == pytest.approx(1 / 32)
    assert exact_mcnemar_pvalue(1, 5) == pytest.approx(7 / 32)
    assert exact_mcnemar_pvalue(5, 1) == exact_mcnemar_pvalue(1, 5)
    assert exact_mcnemar_pvalue(3, 3) == 1


@pytest.mark.parametrize("b,c", [(794, 303), (468, 383), (458, 413), (453, 388), (0, 1200)])
def test_exact_integer_fallback_agrees_with_scipy(b, c):
    if binomtest is None:
        pytest.skip("SciPy unavailable; exact fallback still tested on enumerated cases")
    assert math.isclose(exact_mcnemar_pvalue(b, c, prefer_scipy=False),
                        exact_mcnemar_pvalue(b, c), rel_tol=1e-10, abs_tol=1e-300)


@pytest.mark.parametrize("b,c", [(-1, 2), (1.5, 2), (True, 2)])
def test_exact_mcnemar_rejects_invalid_discordance_counts(b, c):
    with pytest.raises(ValueError):
        exact_mcnemar_pvalue(b, c)


def test_holm_known_values_ordering_ties_and_clipping():
    raw = {"z": .03, "a": .001, "b": .01, "c": .2}
    assert holm_adjust(raw) == pytest.approx({"z": .06, "a": .004, "b": .03, "c": .2})
    assert holm_adjust({"b": .2, "a": .2, "z": .9}) == pytest.approx({"b": .6, "a": .6, "z": .9})
    assert holm_adjust({"a": .8, "b": .9}) == {"a": 1, "b": 1}
    assert holm_adjust({}) == {}
    assert holm_adjust(dict(reversed(list(raw.items())))) == holm_adjust(raw)


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_holm_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError):
        holm_adjust({"p": p})


def test_paired_counts_preserve_matched_directions():
    left = [(0, 0, 0), (1, 1, 1), (2, 2, 9), (3, 3, 8)]
    right = [(0, 0, 0), (1, 1, 9), (2, 2, 2), (3, 3, 7)]
    result = paired_counts(left, right, 4)
    assert result["both_correct"] == result["reference_correct_other_wrong"] == 1
    assert result["other_correct_reference_wrong"] == result["both_wrong"] == 1
    assert result["correctness_discordant"] == 2
    assert result["reference_minus_other_accuracy_percentage_points"] == 0


def test_alignment_rejects_truth_changes_and_reordered_or_duplicated_ids():
    good = [(0, 0, 0), (1, 1, 1)]
    with pytest.raises(ValueError, match="not aligned"):
        paired_counts(good, [(0, 1, 0), (1, 1, 1)], 2)
    for wrong in [[(1, 1, 1), (0, 0, 0)], [(0, 0, 0), (0, 1, 1)]]:
        with pytest.raises(ValueError, match="IDs/order"):
            paired_counts(good, wrong, 2)


@pytest.mark.parametrize("rows", [[(0, 10, 0)], [(0, 0, 10)], [(0, -1, 0)],
    [(0, 0, False)], [], [(0, 0, 0), (1, 1, 1)]])
def test_prediction_validation_rejects_wrong_counts_and_label_bounds(rows):
    with pytest.raises(ValueError):
        validate_predictions(rows, 1)


def test_csv_rejects_noncanonical_indices_and_missing_columns(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("official_test_index,true,predicted\n0,0,0\n2,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IDs/order"):
        read_predictions(path, 2)
    path.write_text("id,true,predicted\n0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_predictions(path, 1)


def test_derived_outputs_cannot_replace_input_evidence(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        ensure_output_paths(root, [root / "metrics.json", tmp_path / "analysis.md"])
    with pytest.raises(ValueError, match="must differ"):
        ensure_output_paths(root, [tmp_path / "same.json", tmp_path / "same.json"])
    ensure_output_paths(root, [tmp_path / "analysis.json", tmp_path / "analysis.md"])
