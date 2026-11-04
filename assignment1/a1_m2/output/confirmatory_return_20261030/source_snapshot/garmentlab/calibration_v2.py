"""Disjoint calibration and finite-grid selective-risk checks.

The mathematical bound applies to independent representatives of the declared
source/duplicate components. It is not a crop-weighted or deployment guarantee.
Metadata partitioning and representative choice never inspect logits or labels.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .evaluate import fit_temperature
from .probabilities import log_probabilities

DEFAULT_CONFIG = {
    "split_seed": 8401,
    "representative_seed": 8402,
    "temperature_fraction": 0.5,
    "target_accuracy": 0.95,
    "family_error_probability": 0.05,
    "minimum_accepted_groups": 30,
    "threshold_grid": [i / 100 for i in range(101)],
    "temperature_minimum": 0.05,
    "temperature_maximum": 20.0,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _config(value=None):
    result = dict(DEFAULT_CONFIG)
    result["threshold_grid"] = list(DEFAULT_CONFIG["threshold_grid"])
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("Calibration configuration must be an object")
        if set(value) - set(DEFAULT_CONFIG):
            raise ValueError("Unknown calibration configuration keys: " + ", ".join(sorted(set(value) - set(DEFAULT_CONFIG))))
        for key in DEFAULT_CONFIG:
            if key in value:
                result[key] = value[key]
    for key in ("split_seed", "representative_seed"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("temperature_fraction", "target_accuracy", "family_error_probability",
                "temperature_minimum", "temperature_maximum"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < result["temperature_fraction"] < 1:
        raise ValueError("temperature_fraction must be between zero and one")
    if not 0 < result["target_accuracy"] <= 1 or not 0 < result["family_error_probability"] < 1:
        raise ValueError("Invalid target accuracy or family error probability")
    if (type(result["minimum_accepted_groups"]) is not int
            or result["minimum_accepted_groups"] < 1):
        raise ValueError("minimum_accepted_groups must be positive")
    if not 0 < result["temperature_minimum"] <= 1 <= result["temperature_maximum"] or result["temperature_minimum"] >= result["temperature_maximum"]:
        raise ValueError("Temperature bounds must bracket one")
    grid = result["threshold_grid"]
    if (not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 1001
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in grid)):
        raise ValueError("threshold_grid must contain at most 1001 finite values in [0,1]")
    grid = [float(v) for v in grid]
    if grid != sorted(set(grid)):
        raise ValueError("threshold_grid must be strictly increasing")
    result["threshold_grid"] = grid
    return result


def _records(records):
    if not isinstance(records, (list, tuple)) or len(records) < 2:
        raise ValueError("At least two calibration records are required")
    clean, identities = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each calibration record must be an object")
        if record.get("role", record.get("partition")) != "calibration":
            raise ValueError("Only calibration-role records may fit calibration")
        if "role" in record and "partition" in record and record["role"] != record["partition"]:
            raise ValueError("Record role and partition disagree")
        group = record.get("source_group_id")
        if not isinstance(group, str) or not group or len(group) > 4096:
            raise ValueError("Each record requires a bounded source_group_id")
        evaluation = record.get("evaluation_group_id")
        if evaluation is not None and (not isinstance(evaluation, str) or not evaluation or len(evaluation) > 4096):
            raise ValueError("Invalid evaluation_group_id")
        identity = record.get("annotation_id", record.get("array_index"))
        if type(identity) not in (int, str) or identity == "":
            raise ValueError("A stable annotation_id or array_index is required")
        # Include the type: integer1 and string'1' must not alias silently.
        identity_key = _canonical([type(identity).__name__, identity]).decode("utf-8")
        if identity_key in identities:
            raise ValueError("Calibration record identities must be unique")
        identities.add(identity_key)
        clean.append({"identity": identity_key, "source_group_id": group,
                      "evaluation_group_id": evaluation,
                      "record_sha256": _hash(record)})
    return clean


def partition_calibration(records, config=None):
    """Freeze metadata-only group partition before performing model inference.

    Returned indices are positions in ``records``. Union source IDs and available
    near-duplicate component IDs so neither relation crosses the two partitions.
    Exactly one metadata-selected row represents each risk component.
    """
    cfg, metadata = _config(config), _records(records)
    parent = list(range(len(metadata)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        left, right = find(a), find(b)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen = {}
    for i, item in enumerate(metadata):
        for kind in ("source_group_id", "evaluation_group_id"):
            if item[kind] is None:
                continue
            key = (kind, item[kind])
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(metadata)):
        groups.setdefault(find(i), []).append(i)
    components = []
    for indices in groups.values():
        values = sorted(indices, key=lambda i: metadata[i]["identity"])
        component_id = _hash({"source_groups": sorted({metadata[i]["source_group_id"] for i in values}),
                              "evaluation_groups": sorted({metadata[i]["evaluation_group_id"] for i in values if metadata[i]["evaluation_group_id"] is not None})})
        components.append((component_id, values))
    if len(components) < 2:
        raise ValueError("Calibration requires at least two independent metadata components")
    components.sort(key=lambda value: (_hash([cfg["split_seed"], value[0]]), value[0]))
    count = max(1, min(len(components) - 1, math.floor(len(components) * cfg["temperature_fraction"])))
    temperature, threshold = components[:count], components[count:]
    temp_indices = sorted(i for _, rows in temperature for i in rows)
    risk_indices = sorted(i for _, rows in threshold for i in rows)
    representatives = sorted(min(rows, key=lambda i: (_hash([cfg["representative_seed"], metadata[i]["identity"]]), metadata[i]["identity"]))
                             for _, rows in threshold)

    def subset_hash(indices):
        return _hash(sorted((metadata[i] for i in indices), key=lambda item: item["identity"]))

    result = {
        "schema": "garmentlab_disjoint_calibration_partition_v2",
        "config": cfg, "config_sha256": _hash(cfg),
        "metadata_sha256": subset_hash(range(len(metadata))),
        "temperature_indices": temp_indices,
        "threshold_indices": risk_indices,
        "threshold_representative_indices": representatives,
        "temperature_split_sha256": subset_hash(temp_indices),
        "threshold_all_crops_split_sha256": subset_hash(risk_indices),
        "threshold_representative_split_sha256": subset_hash(representatives),
        "samples": len(metadata), "source_groups": len({r["source_group_id"] for r in metadata}),
        "components": len(components), "temperature_components": len(temperature),
        "threshold_components": len(threshold),
        "temperature_component_ids": [identifier for identifier, _ in temperature],
        "threshold_component_ids": [identifier for identifier, _ in threshold],
        "grouping": "union of source_group_id and available evaluation_group_id",
        "representative_rule": "minimum seeded hash of stable record identity, chosen before logits or labels",
        "scope": "Metadata-only split of existing calibration data; no new assessment data and no claim that components are proven independent.",
    }
    result["partition_sha256"] = _hash(result)
    return result


def _binomial_log_cdf(errors, count, probability):
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 0.0 if errors == count else -math.inf
    lp, lq = math.log(probability), math.log1p(-probability)
    base = math.lgamma(count + 1)
    values = [base - math.lgamma(k + 1) - math.lgamma(count - k + 1)
              + k * lp + (count - k) * lq for k in range(errors + 1)]
    largest = max(values)
    return min(0.0, largest + math.log(math.fsum(math.exp(v - largest) for v in values)))


def binomial_error_upper_bound(errors, count, *, alpha):
    """One-sided exact Clopper-Pearson upper error bound, computed without SciPy.

    This counts independent Bernoulli trials; application assumptions remain the
    caller's responsibility. The zero-trial and all-error bounds are one.
    """
    if type(errors) is not int or type(count) is not int or not 0 <= errors <= count:
        raise ValueError("Invalid binomial counts")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if count == 0 or errors == count:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / count)
    left, right, log_alpha = errors / count, 1.0, math.log(alpha)
    for _ in range(72):
        middle = (left + right) / 2
        if _binomial_log_cdf(errors, count, middle) > log_alpha:
            left = middle
        else:
            right = middle
    # Choose the conservative endpoint to avoid understating the computed bound.
    return right


def _probabilities_targets(probabilities, targets):
    p, y = np.asarray(probabilities, dtype=np.float64), np.asarray(targets)
    if (p.ndim != 2 or not len(p) or p.shape[1] < 2 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-7)
            or y.shape != (len(p),) or y.dtype.kind not in "iu"
            or (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError("Invalid normalized probabilities or integer targets")
    return p, y.astype(np.int64, copy=False)


def select_review_policy(probabilities, targets, config=None):
    """Select from a fixed grid using one independent representative per group.

    ``probabilities`` must already contain exactly those metadata-chosen rows.
    Simultaneous bounds use Bonferroni over the whole declared grid, including
    thresholds with too few accepted cases. No empirical fallback is activated.
    """
    cfg = _config(config)
    p, y = _probabilities_targets(probabilities, targets)
    confidence, correct = p.max(1), p.argmax(1) == y
    per_threshold_alpha = cfg["family_error_probability"] / len(cfg["threshold_grid"])
    target_error = 1.0 - cfg["target_accuracy"]
    rows = []
    for threshold in cfg["threshold_grid"]:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        errors = int((~correct[accepted]).sum())
        upper = binomial_error_upper_bound(errors, count, alpha=per_threshold_alpha)
        rows.append({"threshold": threshold, "accepted_count": count, "error_count": errors,
                     "accepted_accuracy": 1 - errors / count if count else None,
                     "coverage": count / len(y), "error_upper_bound": upper,
                     "eligible_under_assumptions": count >= cfg["minimum_accepted_groups"] and upper <= target_error})
    eligible = [r for r in rows if r["eligible_under_assumptions"]]
    chosen = min(eligible, key=lambda row: (-row["accepted_count"], row["threshold"])) if eligible else None
    empirical = [r for r in rows if r["accepted_count"] >= cfg["minimum_accepted_groups"]
                 and r["accepted_accuracy"] >= cfg["target_accuracy"]]
    diagnostic = min(empirical, key=lambda row: (-row["accepted_count"], row["threshold"])) if empirical else None
    mode = "confidence_threshold" if chosen else "review_all"
    return {
        "status": "bound_satisfied_under_assumptions" if chosen else "review_all_no_eligible_threshold",
        "mode": mode, "eligible": bool(chosen), "review_threshold": chosen["threshold"] if chosen else 1.0,
        "selected": chosen,
        "target_accuracy": cfg["target_accuracy"], "target_error": target_error,
        "family_error_probability": cfg["family_error_probability"],
        "per_threshold_error_probability": per_threshold_alpha,
        "minimum_accepted_groups": cfg["minimum_accepted_groups"],
        "representative_count": len(y), "threshold_grid": cfg["threshold_grid"], "grid_results": rows,
        "empirical_only_diagnostic": diagnostic,
        "empirical_diagnostic_is_active_policy": False,
        "no_eligible_rule": "Review every prediction explicitly, including confidence exactly1.0; never lower target or silently activate empirical diagnostic.",
        "method": "fixed-grid one-sided exact binomial error upper bounds with Bonferroni family-wise correction",
        "scope": "Under fixed-model/independent-representative assumptions, simultaneous bounds support adaptive choice among this finite grid. Not a crop-weighted or distribution-shift guarantee.",
    }


def review_mask(probabilities, binding):
    """Return true for every prediction that requires human review."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not len(p):
        raise ValueError("Review probabilities must be a nonempty N x K array")
    _probabilities_targets(p, np.zeros(len(p), dtype=np.int64))
    policy = binding.get("policy", {})
    mode = policy.get("mode", "confidence_threshold")
    if mode == "review_all":
        return np.ones(len(p), dtype=bool)
    if mode != "confidence_threshold":
        raise ValueError("Unknown review policy mode")
    threshold = binding.get("review_threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid review threshold")
    return p.max(1) < threshold


def fit_calibration_v2(logits, targets, records, config=None, *, partition=None):
    """Fit T and threshold on disjoint metadata-frozen calibration components.

    Inputs are ordered calibration rows only. ``partition`` is optional but, when
    supplied, must exactly match the metadata-only partition frozen beforehand.
    Returns JSON-serializable evidence and a release-compatible binding.
    """
    cfg = _config(config)
    expected_partition = partition_calibration(records, cfg)
    if partition is not None and partition != expected_partition:
        raise ValueError("Frozen calibration partition differs from metadata/configuration")
    partition = expected_partition
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if (z.ndim != 2 or len(z) != len(records) or z.shape[1] < 2 or not np.isfinite(z).all()
            or y.shape != (len(z),) or y.dtype.kind not in "iu" or (y < 0).any() or (y >= z.shape[1]).any()):
        raise ValueError("Calibration logits and integer targets must match records")
    for i, record in enumerate(records):
        if "class_index" in record and (type(record["class_index"]) is not int or record["class_index"] != int(y[i])):
            raise ValueError("Calibration targets differ from record class indices")
    temp = partition["temperature_indices"]
    representatives = partition["threshold_representative_indices"]
    fitted = fit_temperature(z[temp], y[temp], minimum=cfg["temperature_minimum"], maximum=cfg["temperature_maximum"])
    probabilities = np.exp(log_probabilities(z, fitted["temperature"]))
    threshold = select_review_policy(probabilities[representatives], y[representatives], cfg)
    binding = {
        "status": "fitted", "temperature": fitted["temperature"],
        "review_threshold": threshold["review_threshold"],
        "fit_split_sha256": partition["temperature_split_sha256"],
        "threshold_selection_split_sha256": partition["threshold_representative_split_sha256"],
        "policy": {
            "version": "disjoint_component_representative_bonferroni_v2",
            "mode": threshold["mode"], "target_accuracy": cfg["target_accuracy"],
            "bound_eligible": threshold["eligible"],
            "family_error_probability": cfg["family_error_probability"],
            "target_bound_satisfied_under_assumptions": threshold["eligible"],
            "unconditional_accuracy_guarantee": False,
            "target_population": "equally weighted metadata-selected representative of an independent source/duplicate component",
            "partition_sha256": partition["partition_sha256"],
            "threshold_grid_sha256": _hash(cfg["threshold_grid"]),
        },
    }
    review = review_mask(probabilities, binding)
    risk_all = np.asarray(partition["threshold_indices"], dtype=np.int64)
    accepted = ~review[risk_all]
    crop_summary = {"samples": len(risk_all), "accepted_count": int(accepted.sum()),
                    "coverage": float(accepted.mean()),
                    "accepted_accuracy": float((probabilities[risk_all].argmax(1)[accepted] == y[risk_all][accepted]).mean()) if accepted.any() else None,
                    "scope": "Descriptive crop-weighted statistics on the same threshold partition; not the statistical target population or an independent test."}
    return {
        "schema": "garmentlab_disjoint_calibration_v2", "binding": binding,
        "temperature_fit": fitted, "threshold_selection": threshold,
        "partition": partition, "threshold_all_crop_diagnostic": crop_summary,
        "risk_control": {
            "target_bound_satisfied_under_assumptions": threshold["eligible"],
            "unconditional_accuracy_guarantee": False,
            "assumptions": [
                "Model and finite threshold grid fixed independently of threshold-selection labels.",
                "Threshold representatives are independent identically distributed draws from the declared component-representative population.",
                "Future inputs follow that same representative population; source-group independence and deployment stationarity are not established by hashes.",
            ],
            "limitations": [
                "One representative per source/duplicate component avoids treating correlated crops as independent Bernoulli trials; it changes the target population from all crops.",
                "The original calibration set was inspected historically. This split improves separation but does not manufacture fresh calibration data or an unconditional guarantee.",
                "If no threshold satisfies the corrected bound, every prediction requires review; study completion is still possible without target certification.",
                "The existing327-case assessment is not used here and remains a previously inspected descriptive/regression set for subsequent model upgrades.",
            ],
        },
    }
