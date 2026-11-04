"""Descriptive paired uncertainty on the already-inspected assessment set."""
from __future__ import annotations

import hashlib
import json
import numpy as np


def paired_group_bootstrap(targets, predictions, tasks, records, *, selected_candidate,
                           classes, replicates=2000, seed=8601):
    """Resample source/duplicate components jointly for fixed trained models.

    The percentile interval is conditional on this empirical regression set and
    the fitted models. It is not a post-selection-adjusted or deployment bound.
    """
    y = np.asarray(targets)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in "iu" or len(records) != len(y)
            or classes < 2 or y.min() < 0 or y.max() >= classes
            or type(replicates) is not int or not 100 <= replicates <= 20000):
        raise ValueError("Invalid grouped-bootstrap inputs")
    parent = list(range(len(y)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for i, record in enumerate(records):
        if record.get("role", record.get("partition")) != "assessment":
            raise ValueError("Grouped uncertainty requires assessment-role metadata")
        for field in ("source_group_id", "evaluation_group_id"):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("Each assessment row requires source and duplicate-component IDs")
            key = field, value
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    roots = [find(i) for i in range(len(y))]
    keys = {value: i for i, value in enumerate(sorted(set(roots)))}
    group = np.asarray([keys[x] for x in roots], dtype=np.int64)
    count = len(keys)
    if count < 2:
        raise ValueError("At least two components are needed for descriptive resampling")
    random = np.random.default_rng(seed)
    weights = random.multinomial(count, np.full(count, 1 / count), size=replicates)
    draws = {}
    by_candidate = {}
    for task in tasks:
        task_id = task["task_id"]
        predicted = np.asarray(predictions[task_id])
        if (predicted.shape != y.shape or predicted.dtype.kind not in "iu"
                or predicted.min() < 0 or predicted.max() >= classes):
            raise ValueError("Predictions must match original target order")
        code = group * classes * classes + y * classes + predicted
        per_group = np.bincount(code, minlength=count * classes * classes).reshape(count, classes * classes)
        matrices = (weights @ per_group).reshape(replicates, classes, classes)
        support = matrices.sum(1) + matrices.sum(2)
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        f1 = np.divide(2 * diagonal, support, out=np.zeros_like(support, dtype=float), where=support != 0).mean(1)
        draws[task_id] = f1
        by_candidate.setdefault(task["candidate_id"], {})[task["seed"]] = task_id
    if "C0" not in by_candidate or selected_candidate not in by_candidate:
        raise ValueError("Both the final anchor and selected recipe must be present")
    seeds = sorted(by_candidate["C0"])
    if len(seeds) != 3 or sorted(by_candidate[selected_candidate]) != seeds:
        raise ValueError("Paired grouped uncertainty requires the same three final training seeds")
    anchor = np.mean([draws[by_candidate["C0"][s]] for s in seeds], axis=0)
    selected = np.mean([draws[by_candidate[selected_candidate][s]] for s in seeds], axis=0)
    delta = selected - anchor
    def interval(values):
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]
    return {"method": "paired source/duplicate-component percentile bootstrap", "replicates": replicates,
        "seed": seed, "crops": len(y), "resampling_components": count, "fixed_training_seeds": seeds,
        "selected_candidate": selected_candidate, "anchor_macro_f1_interval": interval(anchor),
        "selected_macro_f1_interval": interval(selected), "paired_mean_macro_f1_delta_interval": interval(delta),
        "group_assignment_sha256": hashlib.sha256(group.tobytes()).hexdigest(),
        "scope": "Descriptive conditional 95% percentile intervals on the previously inspected assessment. Components, rather than crops, are resampled jointly. Fixed models/seeds are not resampled. No post-selection correction, fresh-test claim or deployment guarantee."}
