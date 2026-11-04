"""Fashionpedia metadata gates and candidate manifests; never downloads photographs.

Metadata readiness is not training readiness or a certification of image rights.
Official assessment annotations can be audited but cannot create a training manifest.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit


ASSESSMENT_SHA256 = "f831f415edad12d52acde8b138fe7ebd1592b57bf0dcd6a3b63ba8cabd208223"
OFFICIAL_NAMES = {
    "instances_attributes_val2020.json": "official_assessment",
    "instances_attributes_train2020.json": "official_train",
}
PARTITIONS = ("fit", "selection", "calibration", "assessment")
_SMALL_JSON_BYTES = 32 * 1024 * 1024


class DataValidationError(ValueError):
    """The metadata or data-policy configuration is ambiguous or invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DataValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _record(value: Any, section: str) -> dict:
    if not isinstance(value, dict):
        raise DataValidationError(f"Each {section} record must be an object")
    return value


def _reject_json_constant(value: str):
    raise DataValidationError(f"Nonstandard JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


class _MetadataReader:
    """Small fixtures use stdlib; large annotations stream one record at a time."""

    def __init__(self, path: Path):
        self.path = path
        self.document = None
        self.backend = "stdlib_json"
        if path.stat().st_size <= _SMALL_JSON_BYTES:
            self.document = json.loads(path.read_text(encoding="utf-8"),
                                       parse_constant=_reject_json_constant,
                                       object_pairs_hook=_unique_object)
            if not isinstance(self.document, dict):
                raise DataValidationError("Annotation root must be a JSON object")
        else:
            try:
                import ijson
            except ImportError as exc:
                raise RuntimeError("Large metadata requires ijson; install the data dependencies first") from exc
            self.ijson = ijson
            self.backend = "ijson_streaming"

    def records(self, section: str) -> Iterator[dict]:
        if self.document is not None:
            values = self.document.get(section)
            if not isinstance(values, list):
                raise DataValidationError(f"Missing or non-array section: {section}")
            yield from values
            return
        # The source JSON may put licenses/categories after the large annotation array.
        # Separate streaming passes avoid retaining segmentation polygons in memory.
        with self.path.open("rb") as stream:
            any_records = False
            # Native filtering skips irrelevant polygon events without a Python loop.
            for record in self.ijson.items(stream, f"{section}.item"):
                any_records = True
                yield record
            if any_records:
                return
        # Distinguish an empty array from a malformed/missing section. This slower
        # validation path is only needed when no records were found.
        with self.path.open("rb") as stream:
            for prefix, event, _ in self.ijson.parse(stream):
                if prefix == section:
                    if event != "start_array":
                        raise DataValidationError(f"Non-array section: {section}")
                    return
        raise DataValidationError(f"Missing section: {section}")


def _http_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
        if parsed.username or parsed.password or any(c.isspace() for c in value):
            return None
        _ = parsed.port
    except ValueError:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, parsed.query, ""))


def _source_group(url: str) -> str:
    parsed = urlsplit(url)
    # HTTP/HTTPS variants of the same original resource stay together.
    key = urlunsplit(("", parsed.netloc.lower(), parsed.path, parsed.query, ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def validate_data_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise DataValidationError("Data config must be an object")
    classes = config.get("class_map")
    if not isinstance(classes, list) or not 6 <= len(classes) <= 8:
        raise DataValidationError("class_map must declare 6–8 category_id/label objects")
    ids, labels = set(), set()
    normalized = []
    for item in classes:
        _record(item, "class_map")
        category = _integer(item.get("category_id"), "category_id")
        label = item.get("label")
        if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", label):
            raise DataValidationError("Class labels must be stable lowercase identifiers")
        if category in ids or label in labels:
            raise DataValidationError("Duplicate class_map category_id or label")
        ids.add(category)
        labels.add(label)
        normalized.append({"category_id": category, "label": label})
    policy = config.get("license_policy")
    if not isinstance(policy, dict):
        raise DataValidationError("license_policy is required")
    if policy.get("purpose") not in ("public_portfolio_candidate", "public_portfolio_metadata_candidate",
                                      "research_only", "private_demo_candidate"):
        raise DataValidationError("Declare the intended license-policy purpose")
    if not isinstance(policy.get("acknowledgement"), str) or not policy["acknowledgement"].strip():
        raise DataValidationError("license_policy requires an explicit acknowledgement of its limits")
    allowed = policy.get("allowed_license_ids")
    if not isinstance(allowed, list) or not allowed:
        raise DataValidationError("Explicit nonempty allowed_license_ids is required; no permissive default")
    allowed = [_integer(value, "allowed_license_ids item") for value in allowed]
    if len(set(allowed)) != len(allowed):
        raise DataValidationError("Duplicate allowed license ID")
    split = config.get("split")
    if not isinstance(split, dict):
        raise DataValidationError("split configuration is required")
    seed = _integer(split.get("seed"), "split.seed")
    fractions = []
    for role in PARTITIONS[:3]:
        value = split.get(f"{role}_fraction")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise DataValidationError(f"Invalid {role}_fraction")
        fractions.append(float(value))
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
        raise DataValidationError("fit/selection/calibration fractions must sum to one")
    minimum = config.get("min_source_images_per_class")
    if not isinstance(minimum, dict) or set(minimum) != set(PARTITIONS):
        raise DataValidationError("Declare min_source_images_per_class for all four roles")
    minimum = {role: _integer(minimum[role], f"minimum support {role}", minimum=1) for role in PARTITIONS}
    return {"class_map": normalized, "license_policy": {
        "purpose": policy["purpose"], "allowed_license_ids": sorted(allowed),
        "acknowledgement": policy["acknowledgement"]},
        "split": {"seed": seed, **{f"{role}_fraction": value for role, value in zip(PARTITIONS, fractions)}},
        "min_source_images_per_class": minimum}


def _partition(group_id: str, split: dict) -> str:
    digest = hashlib.sha256(f"{split['seed']}:{group_id}".encode("ascii")).digest()
    value = int.from_bytes(digest, "big") / 2**256
    if value < split["fit_fraction"]:
        return "fit"
    if value < split["fit_fraction"] + split["selection_fraction"]:
        return "selection"
    return "calibration"


def _image_metadata(image: dict, licenses: dict, allowed: set[int]) -> tuple[dict | None, list[str]]:
    reasons = []
    width, height = image.get("width"), image.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        reasons.append("invalid_image_dimensions")
    name = image.get("file_name")
    if (not isinstance(name, str) or not name or len(name) > 255
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name in (".", "..")
            or re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", name)
            or Path(name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp")):
        reasons.append("invalid_image_file_name")
    url = _http_url(image.get("original_url"))
    if url is None:
        reasons.append("invalid_original_url")
    license_id = image.get("license")
    if type(license_id) is not int or license_id not in licenses:
        reasons.append("unknown_or_missing_license")
    elif license_id not in allowed:
        reasons.append("license_not_allowed")
    elif not _http_url(licenses[license_id].get("url")):
        reasons.append("license_definition_has_no_valid_url")
    if reasons:
        return None, reasons
    return {"image_id": image["id"], "file_name": name, "width": width, "height": height,
            "original_url": url, "license_id": license_id,
            "source_group_id": _source_group(url)}, []


def _bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float, Decimal)) for item in value):
        return None
    try:
        x, y, w, h = (float(item) for item in value)
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (x, y, w, h)):
        return None
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        return None
    return [x, y, w, h]


def prepare_metadata_candidate(annotation_path: str | Path, config: dict, *, source_role: str,
                               expected_sha256: str | None = None,
                               assessment_annotations: str | Path | None = None,
                               assessment_expected_sha256: str | None = None) -> dict:
    """Audit annotations and return an optional *metadata-only* train candidate.

    Source-group partitioning is hash based, before crops; it does not promise
    stratified or balanced partitions. Support gates report actual group counts.
    """
    if source_role not in ("official_train", "official_assessment"):
        raise DataValidationError("source_role must be official_train or official_assessment")
    if source_role == "official_assessment" and assessment_annotations is not None:
        raise DataValidationError("Assessment audits cannot attach a training/reference partition")
    path = Path(annotation_path)
    normalized = validate_data_config(config)
    source_hash = sha256_file(path)
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) or source_hash != expected_sha256.lower():
            raise DataValidationError("Source annotation SHA-256 does not match")
    known_role = OFFICIAL_NAMES.get(path.name)
    if source_hash == ASSESSMENT_SHA256:
        known_role = "official_assessment"
    if known_role is not None and known_role != source_role:
        raise DataValidationError("Official assessment/train provenance cannot be relabeled")
    assessment_reference = None
    assessment_groups = set()
    if assessment_annotations is not None:
        assessment_result = prepare_metadata_candidate(
            assessment_annotations, config, source_role="official_assessment",
            expected_sha256=assessment_expected_sha256 or ASSESSMENT_SHA256)
        assessment_reference = {
            "source_annotation": assessment_result["audit"]["source_annotation"],
            "metadata_ready": assessment_result["audit"]["readiness"]["metadata_ready"],
            "matching_scope": "all valid normalized original URLs, before image/license/category filtering",
        }
        for raw in _MetadataReader(Path(assessment_annotations)).records("images"):
            url = _http_url(_record(raw, "images").get("original_url"))
            if url is not None:
                assessment_groups.add(_source_group(url))
        if sha256_file(assessment_annotations) != assessment_reference["source_annotation"]["sha256"]:
            raise DataValidationError("Assessment reference changed during preparation")
    reader = _MetadataReader(path)
    licenses, categories = {}, {}
    for section, target in (("licenses", licenses), ("categories", categories)):
        for raw in reader.records(section):
            record = _record(raw, section)
            key = _integer(record.get("id"), f"{section}.id")
            if key in target:
                raise DataValidationError(f"Duplicate {section} ID: {key}")
            if not isinstance(record.get("name"), str) or not record["name"].strip():
                raise DataValidationError(f"Missing {section} name for ID {key}")
            target[key] = record
        if not target:
            raise DataValidationError(f"Empty {section}")
    class_by_id = {item["category_id"]: {**item, "class_index": index}
                   for index, item in enumerate(normalized["class_map"])}
    if not set(class_by_id) <= categories.keys():
        raise DataValidationError("class_map includes category IDs absent from source annotations")
    allowed = set(normalized["license_policy"]["allowed_license_ids"])
    if not allowed <= licenses.keys():
        raise DataValidationError("License allowlist references undefined license IDs")

    all_image_ids, file_names, images = set(), set(), {}
    image_reasons, image_license_counts = Counter(), Counter()
    invalid_images = 0
    quarantined_groups = set()
    quarantined_image_ids = []
    for raw in reader.records("images"):
        image = _record(raw, "images")
        image_id = _integer(image.get("id"), "images.id")
        if image_id in all_image_ids:
            raise DataValidationError(f"Duplicate images ID: {image_id}")
        all_image_ids.add(image_id)
        name = image.get("file_name")
        if isinstance(name, str) and name in file_names:
            raise DataValidationError(f"Duplicate image file_name: {name}")
        if isinstance(name, str):
            file_names.add(name)
        image_license_counts[str(image.get("license", "MISSING"))] += 1
        record, reasons = _image_metadata(image, licenses, allowed)
        if record is not None and record["source_group_id"] in assessment_groups:
            quarantined_groups.add(record["source_group_id"])
            quarantined_image_ids.append(image_id)
            record = None
            reasons = ["source_group_overlaps_assessment"]
        if record is None:
            invalid_images += 1
            image_reasons.update(reasons)
        else:
            # Assign every accepted source image before reading any annotations.
            record["partition"] = ("assessment" if source_role == "official_assessment"
                                   else _partition(record["source_group_id"], normalized["split"]))
            images[image_id] = record
    if not all_image_ids:
        raise DataValidationError("Empty images section")

    annotation_ids, rows, exclusions = set(), [], Counter()
    counts = {role: Counter() for role in PARTITIONS}
    groups = {role: defaultdict(set) for role in PARTITIONS}
    source_images = {role: defaultdict(set) for role in PARTITIONS}
    for raw in reader.records("annotations"):
        annotation = _record(raw, "annotations")
        annotation_id = _integer(annotation.get("id"), "annotations.id")
        if annotation_id in annotation_ids:
            raise DataValidationError(f"Duplicate annotations ID: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _integer(annotation.get("image_id"), "annotations.image_id")
        category_id = _integer(annotation.get("category_id"), "annotations.category_id")
        if image_id not in all_image_ids or category_id not in categories:
            raise DataValidationError("Annotation references an unknown image/category ID")
        image = images.get(image_id)
        if image is None:
            exclusions["image_metadata_or_license_rejected"] += 1
            continue
        box = _bbox(annotation.get("bbox"), image["width"], image["height"])
        if box is None:
            exclusions["invalid_or_out_of_bounds_bbox"] += 1
            continue
        if category_id not in class_by_id:
            exclusions["category_not_selected"] += 1
            continue
        label = class_by_id[category_id]
        role = image["partition"]
        counts[role][label["label"]] += 1
        groups[role][label["label"]].add(image["source_group_id"])
        source_images[role][label["label"]].add(image_id)
        if source_role == "official_train":
            rows.append({**image, "annotation_id": annotation_id, "bbox_xywh": box,
                         "category_id": category_id, "class_index": label["class_index"],
                         "label": label["label"]})

    # Re-read the bytes to detect a file changed between the independent passes.
    if sha256_file(path) != source_hash:
        raise DataValidationError("Annotation source changed during preparation")
    if assessment_reference is not None:
        if sha256_file(assessment_annotations) != assessment_reference["source_annotation"]["sha256"]:
            raise DataValidationError("Assessment reference changed during preparation")
    labels = [item["label"] for item in normalized["class_map"]]
    class_counts = {}
    for role in PARTITIONS:
        class_counts[role] = {label: {
            "crops": counts[role][label], "unique_source_images": len(source_images[role][label]),
            "unique_source_groups": len(groups[role][label])} for label in labels}
    active_roles = ("assessment",) if source_role == "official_assessment" else PARTITIONS[:3]
    failures = []
    if source_role == "official_train" and assessment_reference is None:
        failures.append({"check": "assessment_reference_required_for_candidate_export"})
    if assessment_reference is not None and not assessment_reference["metadata_ready"]:
        failures.append({"check": "assessment_reference_class_support_failed"})
    for role in active_roles:
        for label in labels:
            actual = len(groups[role][label])
            required = normalized["min_source_images_per_class"][role]
            if actual < required:
                failures.append({"check": "minimum_class_source_groups", "partition": role,
                                 "label": label, "actual": actual, "required": required})
    config_hash = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    audit = {
        "schema": "garmentlab_fashionpedia_metadata_audit_v1", "source_role": source_role,
        "source_annotation": {"file_name": path.name, "bytes": path.stat().st_size, "sha256": source_hash},
        "normalized_data_config_sha256": config_hash, "config": normalized,
        "class_map": [{**item, "source_category_name": categories[item["category_id"]]["name"]}
                      for item in normalized["class_map"]],
        "reader": reader.backend,
        "counts": {"images_total": len(all_image_ids), "images_metadata_accepted": len(images),
                   "images_metadata_excluded": invalid_images, "annotations_total": len(annotation_ids),
                   "eligible_crops": sum(sum(value.values()) for value in counts.values()),
                   "image_license_id_counts": dict(sorted(image_license_counts.items())),
                   "by_partition_and_class": class_counts},
        "exclusions": {"images_by_reason_nonexclusive": dict(sorted(image_reasons.items())),
                       "annotations_by_primary_reason": dict(sorted(exclusions.items()))},
        "license_definitions": [licenses[key] for key in sorted(licenses)],
        "grouping": {"key": "SHA256 of normalized original_url; HTTP/HTTPS share a group",
                     "assignment": "seeded hash threshold assigned to source groups before annotations",
                     "stratified": False, "balanced_partitions_claimed": False,
                     "support_gate_unit": "source_group",
                     "assessment_original_url_overlap_audited": assessment_reference is not None,
                     "quarantined_assessment_source_groups": len(quarantined_groups),
                     "quarantined_source_group_ids": sorted(quarantined_groups),
                     "quarantined_train_image_ids": sorted(quarantined_image_ids),
                     "not_verified": ["pixel duplicates", "near duplicates", "product/person identity",
                                      "pixel/near-duplicate train-assessment overlap"]},
        "assessment_reference": assessment_reference,
        "readiness": {"metadata_ready": not failures, "training_ready": False,
                      "failures": failures,
                      "pending": ["image download/decode/dimension/hash checks", "cross-partition duplicate audit",
                                  "taxonomy and crop visual review", "individual image-rights review"],
                      "legal_clearance_claimed": False},
        "assessment_guard": {"metadata_audit_only": source_role == "official_assessment",
                             "training_manifest_export_allowed": source_role == "official_train"},
    }
    manifest = None
    if source_role == "official_train" and not failures:
        rows.sort(key=lambda row: (row["image_id"], row["annotation_id"]))
        manifest = {
            "schema": "garmentlab_candidate_crop_manifest_v1",
            "status": "METADATA_CANDIDATE_ONLY_NOT_TRAINING_READY",
            "source_role": source_role, "source_annotation_sha256": source_hash,
            "normalized_data_config_sha256": config_hash, "class_map": normalized["class_map"],
            "license_policy": normalized["license_policy"], "image_verification": "not_performed",
            "assessment_reference": assessment_reference,
            "records": rows,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    return {"audit": audit, "candidate_manifest": manifest}


def write_preparation_outputs(result: dict, output_dir: str | Path) -> dict[str, str]:
    """Write fresh artifacts only; never overwrite an earlier audit or manifest."""
    manifest = result["candidate_manifest"]
    if manifest is not None:
        if (result["audit"]["source_role"] != "official_train"
                or manifest.get("source_role") != "official_train"
                or not result["audit"]["readiness"]["metadata_ready"]
                or not result["audit"].get("assessment_reference")
                or manifest.get("assessment_reference") != result["audit"]["assessment_reference"]
                or manifest.get("source_annotation_sha256") != result["audit"]["source_annotation"]["sha256"]
                or manifest.get("normalized_data_config_sha256") != result["audit"]["normalized_data_config_sha256"]
                or any(row.get("partition") not in PARTITIONS[:3] for row in manifest["records"])):
            raise DataValidationError("Only ready official-train metadata may export a train candidate")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != manifest.get("manifest_sha256"):
            raise DataValidationError("Candidate manifest changed after preparation")
    output = Path(output_dir)
    names = ["DATA_METADATA_AUDIT.json", "candidate_crop_manifest.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Output contains an earlier audit/manifest; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"audit": output / names[0]}
    if result["candidate_manifest"] is not None:
        artifacts["candidate_manifest"] = output / names[1]
    for key, path in artifacts.items():
        value = result["audit"] if key == "audit" else result["candidate_manifest"]
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return {key: str(path.resolve()) for key, path in artifacts.items()}
