"""Prospectively locked audit on project-unused Fashionpedia license cohorts.

This is a same-source, license-defined distribution-shift audit. It cannot turn
the previously inspected assessment into a blind test or establish that ImageNet
pretraining never encountered an image. No fitting, calibration or model choice
is permitted here. Materialized photographs remain in the private prepared cache.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image

from .data import ASSESSMENT_SHA256, DataValidationError, _MetadataReader, _bbox, _image_metadata
from .images import (TRAIN_METADATA_SHA256, _Groups, _BKTree, _read_image,
                     integer_crop, letterbox_rgb, safe_zip_index)
from .runtime import atomic_json, digest, utc_now

ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP = ((0, "shirt_blouse"), (1, "top_tshirt_sweatshirt"), (4, "jacket"),
             (6, "pants"), (8, "skirt"), (10, "dress"))
LABELS = [label for _, label in CLASS_MAP]
SEEDS = (314, 2718, 5772)
DEFAULT_PROTOCOL = {
    "schema": "garmentlab_prospective_license_cohort_v1",
    "candidate_license_ids": [3, 5], "historical_license_ids": [0, 1, 6],
    "source_group_cap": 2000, "sampling_seed": 20261029,
    "near_duplicate_distance": 2, "minimum_components_per_class": 15,
    "bootstrap_replicates": 2000, "bootstrap_seed": 20261030,
    "primary_comparison": "C4 minus C0, mean of three paired fixed-seed Macro-F1 scores",
    "sampling_rule": "SHA256(seed:source_group_id), cap before target-category annotation filtering; no replacement",
    "use": "Noncommercial research only; no photograph redistribution",
    "scope": "Project-unused same-source license-cohort shift, not external-data or IID test evidence",
}
SOURCE_FILES = ("garmentlab/prospective.py", "garmentlab/images.py", "garmentlab/data.py",
                "garmentlab/training.py", "garmentlab/evaluate.py", "garmentlab/evaluate_v2.py",
                "garmentlab/uncertainty_v2.py", "garmentlab/probabilities.py",
                "garmentlab/calibration_v2.py", "garmentlab/backend.py", "a1/cnn.py")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise DataValidationError(f"Immutable prospective receipt differs: {path.name}")
    else:
        atomic_json(path, value)


def _protocol(protocol, smoke):
    p = dict(DEFAULT_PROTOCOL)
    if protocol is not None:
        if set(protocol) - set(p):
            raise ValueError("Unknown prospective protocol field")
        p.update(protocol)
    fixed = ("schema", "candidate_license_ids", "historical_license_ids", "sampling_seed",
             "near_duplicate_distance", "primary_comparison", "sampling_rule", "use", "scope")
    if any(p[k] != DEFAULT_PROTOCOL[k] for k in fixed):
        if not smoke or any(p[k] != DEFAULT_PROTOCOL[k] for k in fixed if k != "near_duplicate_distance"):
            raise ValueError("Prospective cohort identity and selection policy are fixed")
    for key in ("source_group_cap", "minimum_components_per_class", "bootstrap_replicates", "bootstrap_seed"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"Invalid positive integer {key}")
    if not smoke and (p["source_group_cap"] != 2000 or p["minimum_components_per_class"] < 15
                      or p["bootstrap_replicates"] != 2000):
        raise ValueError("Production cohort cap/support/bootstrap cannot be relaxed")
    if not 100 <= p["bootstrap_replicates"] <= 20000 or not 0 <= p["near_duplicate_distance"] <= 2:
        raise ValueError("Invalid bootstrap or duplicate radius")
    return p


def _sources():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def _file_inputs(*, train_metadata, train_archive, assessment_metadata, assessment_archive,
                 historical_index, duplicate_audit, preview_credits=(), additional_exclusions=None):
    paths = {"train_metadata": train_metadata, "train_archive": train_archive,
             "assessment_metadata": assessment_metadata, "assessment_archive": assessment_archive,
             "historical_index": historical_index, "duplicate_audit": duplicate_audit}
    paths.update({f"preview_credits_{i}": p for i, p in enumerate(preview_credits)})
    readiness = Path(historical_index).parent / "DATA_READINESS.json"
    if readiness.is_file():
        paths["historical_readiness"] = readiness
    if additional_exclusions is not None:
        paths["additional_exclusions"] = additional_exclusions
    result = {}
    for key, value in paths.items():
        path = Path(value).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise DataValidationError(f"Prospective input is not an ordinary file: {key}")
        result[key] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def _models(study):
    """Verify six final checkpoints without constructing or forwarding a model."""
    study = Path(study)
    frozen = _json(study / "FINAL_SELECTION.json")
    manifest = _json(study / "study_v2_manifest.json")
    state = _json(study / "study_v2_state.json")
    binding = digest(study / "study_v2_manifest.json")
    if (state.get("status") != "TRAINING_COMPLETE_CHOICES_FROZEN"
            or state.get("binding") != binding or frozen.get("study_binding") != binding
            or state.get("final_selection_sha256") != digest(study / "FINAL_SELECTION.json")
            or frozen.get("labels") != LABELS or manifest.get("labels") != LABELS
            or frozen.get("selection_candidate_id") != "C4"):
        raise DataValidationError("Existing complete C0/C4 study provenance is required")
    tasks = frozen.get("final_tasks", [])
    expected = {(candidate, seed) for candidate in ("C0", "C4") for seed in SEEDS}
    if len(tasks) != 6 or {(t["candidate_id"], t["seed"]) for t in tasks} != expected:
        raise DataValidationError("Prospective comparison requires exactly six paired fixed models")
    result = []
    for item in tasks:
        task_id = f"final_{item['candidate_id']}_seed{item['seed']}"
        folder = study / "tasks" / task_id
        task = _json(folder / "task.json")
        complete = _json(folder / "COMPLETE.json")
        if (item.get("task_id") != task_id or item.get("checkpoint") != f"tasks/{task_id}/best.pt"
                or item.get("task") != task or task.get("labels") != LABELS
                or task.get("candidate_id") != item["candidate_id"] or task.get("seed") != item["seed"]
                or complete.get("task_id") != task_id or complete.get("binding") != binding):
            raise DataValidationError("Final checkpoint task provenance differs")
        files = {name: digest(folder / name) for name in ("best.pt", "task.json", "history.json")}
        if any(complete.get("files", {}).get(name, {}).get("sha256") != sha for name, sha in files.items()):
            raise DataValidationError("A committed final checkpoint payload changed")
        if item.get("weights_sha256") != files["best.pt"]:
            raise DataValidationError("Frozen descriptor/checkpoint hash differs")
        result.append({"task_id": task_id, "candidate_id": item["candidate_id"], "seed": item["seed"],
                       "checkpoint": item["checkpoint"], "weights_sha256": files["best.pt"],
                       "task": task, "committed_files": files,
                       "complete_sha256": digest(folder / "COMPLETE.json")})
    return sorted(result, key=lambda t: t["task_id"]), {
        name: digest(study / name) for name in ("FINAL_SELECTION.json", "study_v2_manifest.json", "study_v2_state.json")}


def freeze_prospective(out_dir, study_dir, *, train_metadata, train_archive, assessment_metadata,
                       assessment_archive, historical_index, duplicate_audit, preview_credits=(),
                       additional_exclusions=None, protocol=None, exposure_attestation=None,
                       policy_files=None, smoke=False):
    """Lock models/source/raw/history/policies before any new crop or score exists.

    exposure_attestation records the available project-history inventory, not a
    claim to know every unrecorded browser view or pretraining photograph.
    policy_files optionally binds pre-existing C0/C4 release JSON files. C4's
    current release manifest is required automatically; no policy is refitted.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = _protocol(protocol, smoke)
    if (not isinstance(exposure_attestation, dict)
            or exposure_attestation.get("recorded_exposure_inventory_complete") is not True
            or not isinstance(exposure_attestation.get("basis"), str)
            or not exposure_attestation["basis"].strip()):
        raise DataValidationError("Recorded historical image-exposure scope evidence is required")
    inputs = _file_inputs(train_metadata=train_metadata, train_archive=train_archive,
        assessment_metadata=assessment_metadata, assessment_archive=assessment_archive,
        historical_index=historical_index, duplicate_audit=duplicate_audit,
        preview_credits=preview_credits, additional_exclusions=additional_exclusions)
    index = _json(historical_index)
    if index.get("labels") != LABELS or not index.get("records"):
        raise DataValidationError("Historical prepared six-class index is required")
    if not isinstance(_json(duplicate_audit).get("cross_role_components"), list):
        raise DataValidationError("Historical duplicate-quarantine evidence is required")
    original_selection = _json(Path(study_dir) / "FINAL_SELECTION.json")
    if original_selection.get("data", {}).get("index_manifest.json") != inputs["historical_index"]["sha256"]:
        raise DataValidationError("Historical image index differs from the models' frozen data binding")
    if not smoke:
        pinned = {"train_metadata": TRAIN_METADATA_SHA256, "assessment_metadata": ASSESSMENT_SHA256}
        pinned.update({key: index.get("binding", {}).get("source_sha256", {}).get(key)
                       for key in ("train_archive", "assessment_archive")})
        if any(not sha or inputs[key]["sha256"] != sha for key, sha in pinned.items()):
            raise DataValidationError("Raw sources differ from the pinned historical Fashionpedia version")
        if not preview_credits:
            raise DataValidationError("Production freshness requires recorded manual-preview credit evidence")
        if "historical_readiness" not in inputs:
            raise DataValidationError("Historical DATA_READINESS.json is required for production freshness")
        readiness = _json(Path(historical_index).parent / "DATA_READINESS.json")
        if (readiness.get("training_ready") is not True
                or readiness.get("binding") != index.get("binding")
                or readiness.get("files", {}).get("index_manifest.json") != inputs["historical_index"]["sha256"]
                or readiness.get("files", {}).get("DUPLICATE_AUDIT.json") != inputs["duplicate_audit"]["sha256"]
                or index.get("license_policy", {}).get("allowed_license_ids") != [0, 1, 6]):
            raise DataValidationError("Historical readiness/index/quarantine provenance differs")
    models, study_files = _models(study_dir)
    policies = dict(policy_files or {})
    policies.setdefault("C4", Path(study_dir) / "release_candidate" / "release.json")
    policy_bindings = {}
    for candidate, path in sorted(policies.items()):
        if candidate not in ("C0", "C4"):
            raise ValueError("Only the frozen C0/C4 policies may be bound")
        document = _json(path)
        policy = document.get("calibration", document.get("binding"))
        if not isinstance(policy, dict):
            raise DataValidationError("Existing release policy must contain calibration/binding")
        if document.get("labels", LABELS) != LABELS:
            raise DataValidationError("Existing policy class order differs")
        if document.get("weights", {}).get("sha256") != next(
                t["weights_sha256"] for t in models if t["task_id"] == f"final_{candidate}_seed314"):
            raise DataValidationError("Existing policy does not explicitly bind the frozen deployment checkpoint")
        temperature, threshold = policy.get("temperature"), policy.get("review_threshold")
        if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0
                or type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1
                or not isinstance(policy.get("policy"), dict)
                or policy["policy"].get("mode") not in ("confidence_threshold", "review_all")):
            raise DataValidationError("Unsupported frozen policy schema; explicit temperature/threshold/mode are required")
        policy_bindings[candidate] = {"file_sha256": digest(path), "calibration": policy,
                                      "deployment_seed": 314}
    identity = {"schema": "garmentlab_prospective_freeze_v1", "protocol": p,
        "inputs": inputs, "models": models, "study_files": study_files,
        "source_sha256": _sources(), "policies": policy_bindings,
        "exposure_attestation": exposure_attestation, "smoke_only": bool(smoke),
        "limitations": "Unused relative to recorded project image exposure; all metadata licenses were previously inventoried. Foundation pretraining and unrecorded views cannot be excluded. Same-source license shift is not an IID or external test."}
    path = out / "PROSPECTIVE_FREEZE.json"
    if path.exists():
        previous = _json(path)
        if {k: v for k, v in previous.items() if k != "frozen_utc"} != identity:
            raise DataValidationError("Prospective freeze changed; preserve this audit rather than refreeze")
        return previous
    if (out / "prepared").exists() or (out / "evaluation").exists():
        raise DataValidationError("Cannot create a prospective lock after crop preparation/evaluation")
    identity["frozen_utc"] = utc_now().isoformat()
    atomic_json(path, identity)
    return identity


def _verify_freeze(out, **inputs):
    frozen = _json(Path(out) / "PROSPECTIVE_FREEZE.json")
    if frozen["source_sha256"] != _sources():
        raise DataValidationError("Prospective source changed after the pre-evaluation freeze")
    if inputs and _file_inputs(**inputs) != frozen["inputs"]:
        raise DataValidationError("Prospective raw/history inputs changed after freeze")
    return frozen


def _metadata_images(path, allowed):
    reader = _MetadataReader(Path(path))
    licenses = {row["id"]: row for row in reader.records("licenses")}
    result = {}
    for image in reader.records("images"):
        record, _ = _image_metadata(image, licenses, set(allowed))
        if record is not None:
            if record["image_id"] in result:
                raise DataValidationError("Duplicate image IDs in source metadata")
            result[record["image_id"]] = record
    return result


def _preview_exclusions(paths, additional):
    ids, groups = set(), set()
    from .data import _http_url, _source_group
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "image_id" not in reader.fieldnames:
                raise DataValidationError("Preview credits must identify every viewed source image")
            for row in reader:
                ids.add(int(row["image_id"]))
                url = _http_url(row.get("original_url"))
                if url:
                    groups.add(_source_group(url))
    if additional is not None:
        value = _json(additional)
        if not isinstance(value, dict) or set(value) - {"image_ids", "source_group_ids", "reason"}:
            raise DataValidationError("Invalid additional historical-exposure exclusions")
        ids.update(int(i) for i in value.get("image_ids", []))
        groups.update(value.get("source_group_ids", []))
    return ids, groups


def _duplicate_components(sources, distance):
    """Old sources always win, including transitive bridges through dropped old rows."""
    groups = _Groups({r["source_group_id"] for r in sources})
    exact, tree = {}, _BKTree()
    for row in sorted(sources, key=lambda r: (r["source_group_id"], r["image_id"])):
        group = row["source_group_id"]
        if row["decoded_rgb_sha256"] in exact:
            groups.join(group, exact[row["decoded_rgb_sha256"]])
        else:
            exact[row["decoded_rgb_sha256"]] = group
        code = int(row["dhash64"], 16)
        for other in tree.neighbors(code, distance):
            groups.join(group, other)
        tree.insert(code, group)
    components = defaultdict(set)
    old = set()
    for row in sources:
        root = groups.find(row["source_group_id"])
        components[root].add(row["source_group_id"])
        if row["historical"]:
            old.add(root)
    assignment = {group: hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()
                  for members in components.values() for group in members}
    dropped = {r["source_group_id"] for r in sources
               if not r["historical"] and groups.find(r["source_group_id"]) in old}
    return dropped, assignment


def prepare_prospective(out_dir, *, train_metadata, train_archive, assessment_metadata,
                        assessment_archive, historical_index, duplicate_audit,
                        preview_credits=(), additional_exclusions=None, stop_check=None):
    """Materialize one metadata-sampled audit, with no image display/model inference.

    All historical eligible raw source images are fingerprinted again, including
    sources dropped by old crop/duplicate gates. This conservative superset also
    excludes allowed-license images which never became training crops.
    """
    out = Path(out_dir)
    kwargs = dict(train_metadata=train_metadata, train_archive=train_archive,
        assessment_metadata=assessment_metadata, assessment_archive=assessment_archive,
        historical_index=historical_index, duplicate_audit=duplicate_audit,
        preview_credits=preview_credits, additional_exclusions=additional_exclusions)
    frozen = _verify_freeze(out, **kwargs)
    binding = digest(out / "PROSPECTIVE_FREEZE.json")
    prepared = out / "prepared"
    if prepared.exists():
        _load_cohort(out)
        return prepared
    def stop():
        if stop_check is not None and stop_check():
            raise TimeoutError("Stopped during blinded prospective preparation; no new metrics exposed")
    p = frozen["protocol"]
    historical = _json(historical_index)
    old_rows = historical["records"]
    used_ids = {r["image_id"] for r in old_rows}
    used_groups = {r["source_group_id"] for r in old_rows}
    preview_ids, preview_groups = _preview_exclusions(preview_credits, additional_exclusions)
    used_ids |= preview_ids
    used_groups |= preview_groups
    quarantine = {g for c in _json(duplicate_audit)["cross_role_components"]
                  for g in c.get("quarantined_source_group_ids", [])}
    old_train = _metadata_images(train_metadata, p["historical_license_ids"])
    old_assessment = _metadata_images(assessment_metadata, p["historical_license_ids"])
    known_ids = set(old_train) | set(old_assessment)
    known_groups = {r["source_group_id"] for r in [*old_train.values(), *old_assessment.values()]}
    if preview_ids - known_ids or preview_groups - known_groups:
        # Additional recorded views may sit outside the old license allowlist.
        # Include their raw pixels as historical duplicate bridges as well as
        # excluding their direct IDs/URLs from the candidate pool.
        all_train_metadata = _metadata_images(train_metadata, [0, 1, 2, 3, 4, 5, 6])
        all_assessment_metadata = _metadata_images(assessment_metadata, [0, 1, 2, 3, 4, 5, 6])
        for source, destination in ((all_train_metadata, old_train), (all_assessment_metadata, old_assessment)):
            destination.update({i: r for i, r in source.items()
                                if i in preview_ids or r["source_group_id"] in preview_groups})
        known_ids = set(old_train) | set(old_assessment)
        known_groups = {r["source_group_id"] for r in [*old_train.values(), *old_assessment.values()]}
        if preview_ids - known_ids or preview_groups - known_groups:
            raise DataValidationError("Additional viewed-source pixels cannot be reconstructed; freshness fails closed")
    old_metadata_groups = {r["source_group_id"] for r in [*old_train.values(), *old_assessment.values()]}
    if not quarantine <= old_metadata_groups:
        raise DataValidationError("Historical quarantine scope cannot be reconstructed from raw sources")
    used_groups |= quarantine | old_metadata_groups
    old_train_ids = set(old_train)
    old_assessment_ids = set(old_assessment)
    candidates = _metadata_images(train_metadata, p["candidate_license_ids"])
    eligible = {i: r for i, r in candidates.items()
                if i not in used_ids and r["source_group_id"] not in used_groups}
    ranked_groups = sorted({r["source_group_id"] for r in eligible.values()},
        key=lambda g: (hashlib.sha256(f"{p['sampling_seed']}:{g}".encode()).hexdigest(), g))
    chosen_groups = set(ranked_groups[:p["source_group_cap"]])
    selected = {i: r for i, r in eligible.items() if r["source_group_id"] in chosen_groups}
    if not selected:
        raise DataValidationError("No project-unused prospective source groups remain")
    # The source-group cap is committed before reading target-category annotations.
    selection = {"freeze_sha256": binding, "candidate_images": len(candidates),
        "eligible_images": len(eligible), "eligible_source_groups": len(ranked_groups),
        "selected_source_groups": sorted(chosen_groups), "selected_image_ids": sorted(selected),
        "sampling_rule": p["sampling_rule"], "replacement_after_exclusions": False}
    _immutable(out / "BLINDED_COHORT_SELECTION.json", selection)
    classes = {category: (index, label) for index, (category, label) in enumerate(CLASS_MAP)}
    crops = defaultdict(list)
    exclusions = Counter()
    for ann in _MetadataReader(Path(train_metadata)).records("annotations"):
        image, target = selected.get(ann["image_id"]), classes.get(ann["category_id"])
        if image is None or target is None:
            continue
        bbox = _bbox(ann.get("bbox"), image["width"], image["height"])
        if bbox is None:
            exclusions["invalid_annotation_bbox"] += 1
            continue
        row = {**image, "annotation_id": ann["id"], "bbox_xywh": bbox,
               "category_id": ann["category_id"], "class_index": target[0], "label": target[1],
               "archive_role": "train", "role": "assessment", "cohort": "prospective_license_shift"}
        try:
            row["crop_xyxy"] = integer_crop(row)
        except ValueError:
            exclusions["invalid_integer_crop"] += 1
            continue
        crops[image["image_id"]].append(row)
    all_train = {**old_train, **selected}
    sources, accepted = [], []
    historical_failures = []
    for archive_path, records, old_ids in ((train_archive, all_train, old_train_ids),
                                           (assessment_archive, old_assessment, old_assessment_ids)):
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_zip_index(archive, {r["file_name"] for r in records.values()})
            for image_id, row in sorted(records.items()):
                stop()
                try:
                    image, fingerprints = _read_image(archive, members[row["file_name"]], row)
                except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                    if image_id in old_ids:
                        historical_failures.append({"image_id": image_id, "reason": str(error)})
                    else:
                        exclusions["new_source_decode_failure"] += 1
                    continue
                sources.append({**row, **fingerprints, "historical": image_id in old_ids})
                if image_id in selected and image_id not in old_ids:
                    accepted.extend({**crop, **fingerprints} for crop in crops[image_id])
    # Fail closed rather than silently losing an old duplicate bridge. Old
    # nonidentity-EXIF images were never decoded into crop arrays historically;
    # the complete exclusion is still recorded, but other old decode failures
    # cannot be distinguished safely from unavailable historical pixels.
    unexpected = [r for r in historical_failures
                  if r["reason"] != "nonidentity_exif_rejected_for_raw_annotation_coordinates"]
    if unexpected:
        atomic_json(out / "HISTORICAL_SCOPE_FAILURE.json", {"freeze_sha256": binding, "failures": unexpected})
        raise DataValidationError("Historical pixel duplicate scope is incomplete; freshness fails closed")
    dropped, components = _duplicate_components(sources, p["near_duplicate_distance"])
    kept = sorted((r for r in accepted if r["source_group_id"] not in dropped),
                  key=lambda r: (r["image_id"], r["annotation_id"]))
    exclusions["historical_duplicate_group_crops"] = len(accepted) - len(kept)
    for i, row in enumerate(kept):
        row["array_index"] = i
        row["evaluation_group_id"] = components[row["source_group_id"]]
    support = {label: {"crops": sum(r["label"] == label for r in kept),
        "components": len({r["evaluation_group_id"] for r in kept if r["label"] == label})} for label in LABELS}
    audit = {"freeze_sha256": binding, "historical_sources_fingerprinted": sum(r["historical"] for r in sources),
        "historical_explicit_decode_exclusions": historical_failures,
        "new_sources_fingerprinted": sum(not r["historical"] for r in sources),
        "historical_quarantine_groups_accounted": len(quarantine),
        "new_groups_dropped_by_old_priority": sorted(dropped), "class_support": support,
        "exclusions": dict(exclusions), "source_fingerprints": sources,
        "limitations": "Exact-pixel and transitive dHash<=2 source components; conservative false positives and undetected duplicates remain possible. No source photographs are exported."}
    _immutable(out / "BLINDED_DUPLICATE_AUDIT.json", audit)
    if any(v["components"] < p["minimum_components_per_class"] for v in support.values()):
        raise DataValidationError("Fixed prospective sample fails class support; no seed/cap retry is permitted")
    with tempfile.TemporaryDirectory(prefix=".prospective-prepared-", dir=out) as temporary:
        build = Path(temporary)
        arrays = np.lib.format.open_memmap(build / "images_uint8.npy", mode="w+", dtype=np.uint8,
                                          shape=(len(kept), 224, 224, 3))
        with zipfile.ZipFile(train_archive) as archive:
            members = safe_zip_index(archive, {r["file_name"] for r in kept})
            last_id, decoded = None, None
            for i, row in enumerate(kept):
                stop()
                if row["image_id"] != last_id:
                    decoded, fingerprints = _read_image(archive, members[row["file_name"]], row)
                    if fingerprints["decoded_rgb_sha256"] != row["decoded_rgb_sha256"]:
                        raise DataValidationError("Prospective source changed between fingerprint and crop")
                    last_id = row["image_id"]
                arrays[i] = letterbox_rgb(decoded, row["crop_xyxy"])
        arrays.flush()
        arrays._mmap.close()
        del arrays
        np.save(build / "labels.npy", np.asarray([r["class_index"] for r in kept], dtype=np.int64))
        atomic_json(build / "index_manifest.json", {"schema": "garmentlab_prospective_cohort_v1",
            "freeze_sha256": binding, "labels": LABELS, "records": kept,
            "scope": p["scope"], "no_image_redistribution": True})
        files = {name: digest(build / name) for name in ("images_uint8.npy", "labels.npy", "index_manifest.json")}
        atomic_json(build / "COHORT_READY.json", {"status": "BLINDED_COHORT_READY", "freeze_sha256": binding,
            "selection_sha256": digest(out / "BLINDED_COHORT_SELECTION.json"),
            "duplicate_audit_sha256": digest(out / "BLINDED_DUPLICATE_AUDIT.json"),
            "files": files, "crops": len(kept), "class_support": support,
            "components": len({r["evaluation_group_id"] for r in kept}), "model_inference_performed": False})
        # Rename the build out of TemporaryDirectory ownership before cleanup.
        build.rename(prepared)
    return prepared


def _load_cohort(out):
    out = Path(out)
    frozen = _verify_freeze(out)
    prepared = out / "prepared"
    ready = _json(prepared / "COHORT_READY.json")
    if (ready.get("status") != "BLINDED_COHORT_READY"
            or ready.get("freeze_sha256") != digest(out / "PROSPECTIVE_FREEZE.json")
            or ready.get("selection_sha256") != digest(out / "BLINDED_COHORT_SELECTION.json")
            or ready.get("duplicate_audit_sha256") != digest(out / "BLINDED_DUPLICATE_AUDIT.json")):
        raise DataValidationError("Prospective cohort readiness/freeze binding changed")
    if set(ready.get("files", {})) != {"images_uint8.npy", "labels.npy", "index_manifest.json"}:
        raise DataValidationError("Prospective readiness must bind every required prepared payload")
    for name, sha in ready["files"].items():
        if name not in ("images_uint8.npy", "labels.npy", "index_manifest.json") or digest(prepared / name) != sha:
            raise DataValidationError("Prospective prepared payload hash differs")
    index = _json(prepared / "index_manifest.json")
    images = np.load(prepared / "images_uint8.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(prepared / "labels.npy", allow_pickle=False)
    if (images.shape != (len(y), 224, 224, 3) or images.dtype != np.uint8
            or y.dtype.kind not in "iu" or len(index["records"]) != len(y)
            or index.get("labels") != LABELS or set(y.tolist()) != set(range(6))
            or any(r["array_index"] != i or r["class_index"] != int(y[i])
                   for i, r in enumerate(index["records"]))):
        raise DataValidationError("Prospective arrays/labels/record order are inconsistent")
    return frozen, ready, images, y, index["records"]


def verify_prospective_complete(out_dir):
    """Verify the committed result and every prediction payload without inference.

    This verifier also works on a photograph-free results export: the original
    prepared-image array need not be redistributed with the statistical record.
    """
    out = Path(out_dir).resolve()
    complete = _json(out / "PROSPECTIVE_COMPLETE.json")
    required = {"schema", "status", "results_sha256", "freeze_sha256", "cohort_ready_sha256",
                "inference_inputs_sha256", "prediction_payloads"}
    if (set(complete) != required or complete.get("schema") != "garmentlab_prospective_completion_v1"
            or complete.get("status") != "PROSPECTIVE_AUDIT_COMPLETE"):
        raise DataValidationError("Prospective completion receipt is incomplete")
    references = {"results_sha256": out / "PROSPECTIVE_RESULTS.json",
                  "freeze_sha256": out / "PROSPECTIVE_FREEZE.json",
                  "cohort_ready_sha256": out / "prepared" / "COHORT_READY.json",
                  "inference_inputs_sha256": out / "evaluation" / "INFERENCE_INPUTS.json"}
    if any(not path.is_file() or digest(path) != complete[key] for key, path in references.items()):
        raise DataValidationError("Prospective completion result/input hash differs")
    frozen = _verify_freeze(out)
    result = _json(references["results_sha256"])
    expected_paths = {f"evaluation/{task['task_id']}/{name}" for task in frozen["models"]
                      for name in ("metrics.json", "logits.npz", "LOGITS_RECEIPT.json")}
    payloads = complete["prediction_payloads"]
    if (not isinstance(payloads, list) or len(payloads) != len(expected_paths)
            or {item.get("path") for item in payloads} != expected_paths
            or payloads != result.get("payloads")
            or result.get("status") != "PROSPECTIVE_AUDIT_COMPLETE"
            or result.get("inference_inputs_sha256") != complete["inference_inputs_sha256"]):
        raise DataValidationError("Prospective completion prediction inventory differs")
    for item in payloads:
        path = (out / item["path"]).resolve()
        if not path.is_relative_to(out) or not path.is_file() or digest(path) != item["sha256"]:
            raise DataValidationError("Prospective completion prediction payload changed")
    return result


def evaluate_prospective(out_dir, study_dir, *, device="cuda", batch_size=128,
                         stop_check=None, model_factory=None, smoke=False):
    """Infer all six frozen checkpoints, then one fixed paired component analysis.

    Reentry only verifies/reuses the same locked predictions; it cannot change a
    model, calibration, threshold, source cohort or primary comparison.
    """
    import torch
    from . import evaluate as metrics
    from .backend import configure_evaluation_backend
    from .evaluate_v2 import acceptance_metrics
    from .training import build_model
    from .uncertainty_v2 import paired_group_bootstrap
    out, study = Path(out_dir), Path(study_dir)
    frozen, ready, images, targets, records = _load_cohort(out)
    if frozen["smoke_only"] != bool(smoke) or (model_factory is not None) != bool(smoke):
        raise DataValidationError("Injected models and fixture provenance require explicit smoke scope")
    models, study_files = _models(study)
    if models != frozen["models"] or study_files != frozen["study_files"]:
        raise DataValidationError("A prospective model changed after the lock")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    backend = configure_evaluation_backend()
    evaluation = out / "evaluation"
    evaluation.mkdir(exist_ok=True)
    identity = {"freeze_sha256": digest(out / "PROSPECTIVE_FREEZE.json"),
        "cohort_ready_sha256": digest(out / "prepared" / "COHORT_READY.json"),
        "device": str(device), "batch_size": batch_size, "backend": backend}
    _immutable(evaluation / "INFERENCE_INPUTS.json", identity)
    prediction_map, rows, payloads = {}, [], []
    indices = np.arange(len(targets))
    for task in models:
        if stop_check is not None and stop_check():
            raise TimeoutError("Prospective inference stopped; only the same locked remaining models may resume")
        folder = evaluation / task["task_id"]
        folder.mkdir(exist_ok=True)
        logits_file, receipt_file = folder / "logits.npz", folder / "LOGITS_RECEIPT.json"
        receipt_identity = {"inference_inputs_sha256": digest(evaluation / "INFERENCE_INPUTS.json"),
                            "weights_sha256": task["weights_sha256"]}
        if receipt_file.exists():
            receipt = _json(receipt_file)
            if ({k: v for k, v in receipt.items() if k != "logits_sha256"} != receipt_identity
                    or receipt["logits_sha256"] != digest(logits_file)):
                raise DataValidationError("Prospective cached prediction binding/hash differs")
            with np.load(logits_file, allow_pickle=False) as values:
                logits = values["logits"]
        else:
            factory = model_factory or build_model
            model = factory(num_classes=6, seed=task["seed"], pretrained=False)
            checkpoint = torch.load(study / task["checkpoint"], map_location="cpu", weights_only=True)
            if checkpoint.get("task") != task["task"]:
                raise DataValidationError("Loaded checkpoint task differs from frozen task")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device).eval()
            logits = metrics._infer(model, images, indices, device=device, batch_size=batch_size, stop_check=stop_check)
            # A crash before the receipt commits leaves an inert NPZ which is
            # replayed from the exact same locked checkpoint on resume.
            metrics._save_npz(logits_file, logits=logits)
            _immutable(receipt_file, {**receipt_identity, "logits_sha256": digest(logits_file)})
            del model
        if logits.shape != (len(targets), 6) or not np.isfinite(logits).all():
            raise DataValidationError("Prospective logits have invalid shape/values")
        raw = metrics.classification_metrics(logits, targets, LABELS)
        row = {"task_id": task["task_id"], "candidate_id": task["candidate_id"],
               "seed": task["seed"], "raw": raw}
        policy = frozen["policies"].get(task["candidate_id"])
        if policy is not None and task["seed"] == policy["deployment_seed"]:
            calibration = policy["calibration"]
            row["frozen_policy"] = acceptance_metrics(logits, targets, calibration)
            row["frozen_policy"]["scope"] = "Descriptive fixed-policy coverage/accuracy on the new license cohort; calibration-population bound does not transfer across this shift."
            row["calibrated"] = metrics.classification_metrics(logits, targets, LABELS,
                                                                temperature=calibration["temperature"])
        _immutable(folder / "metrics.json", row)
        prediction_map[task["task_id"]] = logits.argmax(1).astype(np.int64)
        rows.append(row)
        payloads.extend([folder / "metrics.json", logits_file, receipt_file])
    bootstrap = paired_group_bootstrap(targets, prediction_map, models, records, selected_candidate="C4",
        classes=6, replicates=frozen["protocol"]["bootstrap_replicates"], seed=frozen["protocol"]["bootstrap_seed"])
    bootstrap["scope"] = "Prospectively locked project-unused same-source license-cohort audit. Conditional component-sampling intervals for six fixed models; no seed-population, IID/external-data, multiplicity-adjusted or deployment guarantee."
    means = {c: float(np.mean([r["raw"]["macro_f1"] for r in rows if r["candidate_id"] == c])) for c in ("C0", "C4")}
    result = {"schema": "garmentlab_prospective_results_v1", "status": "PROSPECTIVE_AUDIT_COMPLETE",
        "smoke_only": bool(smoke), "inference_inputs_sha256": digest(evaluation / "INFERENCE_INPUTS.json"),
        "cohort": {"crops": len(targets), "components": ready["components"], "class_support": ready["class_support"]},
        "primary": {"definition": frozen["protocol"]["primary_comparison"], "mean_macro_f1": means,
                    "paired_mean_macro_f1_delta": means["C4"] - means["C0"], "uncertainty": bootstrap},
        "models": rows, "scope": frozen["protocol"]["scope"], "limitations": frozen["limitations"],
        "photographs_redistributed": False,
        "payloads": [{"path": p.relative_to(out).as_posix(), "sha256": digest(p)} for p in payloads]}
    _immutable(out / "PROSPECTIVE_RESULTS.json", result)
    _immutable(out / "PROSPECTIVE_COMPLETE.json", {
        "schema": "garmentlab_prospective_completion_v1", "status": "PROSPECTIVE_AUDIT_COMPLETE",
        "results_sha256": digest(out / "PROSPECTIVE_RESULTS.json"),
        "freeze_sha256": digest(out / "PROSPECTIVE_FREEZE.json"),
        "cohort_ready_sha256": digest(out / "prepared" / "COHORT_READY.json"),
        "inference_inputs_sha256": digest(evaluation / "INFERENCE_INPUTS.json"),
        "prediction_payloads": result["payloads"]})
    verify_prospective_complete(out)
    images._mmap.close()
    return result
