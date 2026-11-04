"""Source-group leakage, malformed metadata and fail-closed license gates."""
from copy import deepcopy
import hashlib
import json

import pytest

import garmentlab.data as data
from garmentlab.data import DataValidationError, write_preparation_outputs


CLASS_IDS = (0, 1, 2, 4, 6, 10)


@pytest.fixture
def config():
    return {"class_map": [{"category_id": key, "label": f"class_{key}"} for key in CLASS_IDS],
            "license_policy": {"purpose": "public_portfolio_candidate", "allowed_license_ids": [0],
                               "acknowledgement": "Metadata allowlist is not legal clearance"},
            "split": {"seed": 20261024, "fit_fraction": .8, "selection_fraction": .1,
                      "calibration_fraction": .1},
            "min_source_images_per_class": {role: 1 for role in data.PARTITIONS}}


@pytest.fixture
def source():
    result = {"licenses": [{"id": 0, "name": "Attribution", "url": "https://creativecommons.org/licenses/by/2.0/"}],
              "categories": [{"id": key, "name": f"Real class {key}"} for key in CLASS_IDS],
              "images": [], "annotations": []}
    for category_index, category in enumerate(CLASS_IDS):
        for index in range(100):
            image_id = category_index * 100 + index
            result["images"].append({"id": image_id, "file_name": f"image_{image_id}.jpg", "width": 100,
                                     "height": 120, "license": 0,
                                     "original_url": f"https://example.org/source/{image_id}.jpg"})
            for sibling in range(2):
                result["annotations"].append({"id": image_id * 2 + sibling, "image_id": image_id,
                                               "category_id": category,
                                               "bbox": [10 + sibling, 20, 30, 40]})
    return result


def save_source(tmp_path, source, name="annotations.json"):
    path = tmp_path / name
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def prepare_metadata_candidate(path, config, **kwargs):
    """Most train tests supply an independently generated assessment fixture."""
    if kwargs.get("source_role") == "official_train" and "assessment_annotations" not in kwargs:
        assessment = source.__wrapped__()
        for image in assessment["images"]:
            image["original_url"] = image["original_url"].replace("/source/", "/independent_assessment/")
        reference = save_source(path.parent, assessment, "assessment_fixture.json")
        kwargs["assessment_annotations"] = reference
        kwargs["assessment_expected_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    return data.prepare_metadata_candidate(path, config, **kwargs)


def test_ready_metadata_is_not_training_ready_and_siblings_never_cross_split(tmp_path, source, config):
    path = save_source(tmp_path, source)
    result = prepare_metadata_candidate(path, config, source_role="official_train")
    audit, manifest = result["audit"], result["candidate_manifest"]
    assert audit["readiness"]["metadata_ready"]
    assert not audit["readiness"]["training_ready"]
    assert not audit["readiness"]["legal_clearance_claimed"]
    assert audit["source_annotation"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(manifest["records"]) == 1200
    partitions = {}
    for row in manifest["records"]:
        partitions.setdefault(row["image_id"], set()).add(row["partition"])
    assert all(len(roles) == 1 for roles in partitions.values())
    assert {next(iter(roles)) for roles in partitions.values()} == {"fit", "selection", "calibration"}
    assert all(row["partition"] != "assessment" for row in manifest["records"])


def test_original_url_aliases_share_group_and_partition(tmp_path, source, config):
    alias = {**source["images"][0], "id": 9999, "file_name": "alias.jpg",
             "original_url": source["images"][0]["original_url"].replace("https:", "http:")}
    source["images"].append(alias)
    source["annotations"].append({"id": 9999, "image_id": 9999, "category_id": 0, "bbox": [0, 0, 20, 20]})
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    rows = [row for row in result["candidate_manifest"]["records"] if row["image_id"] in (0, 9999)]
    assert len({row["source_group_id"] for row in rows}) == 1
    assert len({row["partition"] for row in rows}) == 1


def test_assessment_reference_quarantines_shared_source_before_crops(tmp_path, source, config):
    train_path = save_source(tmp_path, source)
    before = prepare_metadata_candidate(train_path, config, source_role="official_train")
    # The real metadata audit found a calibration/assessment URL collision.
    # Exercise that role specifically, not only a training-gradient collision.
    target_id = next(row["image_id"] for row in before["candidate_manifest"]["records"]
                     if row["partition"] == "calibration")
    target_image = next(image for image in source["images"] if image["id"] == target_id)
    assessment = deepcopy(source)
    for image in assessment["images"]:
        image["original_url"] = image["original_url"].replace("/source/", "/assessment/")
    assessment["images"][0]["original_url"] = target_image["original_url"]
    assessment_path = save_source(tmp_path, assessment, "instances_attributes_val2020.json")
    result = prepare_metadata_candidate(train_path, config, source_role="official_train",
                                        assessment_annotations=assessment_path,
                                        assessment_expected_sha256=hashlib.sha256(assessment_path.read_bytes()).hexdigest())
    assert result["audit"]["grouping"]["assessment_original_url_overlap_audited"]
    assert result["audit"]["grouping"]["quarantined_assessment_source_groups"] == 1
    assert result["audit"]["exclusions"]["images_by_reason_nonexclusive"]["source_group_overlaps_assessment"] == 1
    assert len(result["candidate_manifest"]["records"]) == 1198
    assert all(row["image_id"] != target_id for row in result["candidate_manifest"]["records"])
    assert result["audit"]["grouping"]["quarantined_train_image_ids"] == [target_id]
    assert all(row["partition"] != "assessment" for row in result["candidate_manifest"]["records"])
    assert result["candidate_manifest"]["assessment_reference"]["source_annotation"]["sha256"] == hashlib.sha256(assessment_path.read_bytes()).hexdigest()


def test_missing_assessment_reference_never_exports_unchecked_train_candidate(tmp_path, source, config):
    result = data.prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    assert not result["audit"]["readiness"]["metadata_ready"]
    assert result["candidate_manifest"] is None
    assert {failure["check"] for failure in result["audit"]["readiness"]["failures"]} == {"assessment_reference_required_for_candidate_export"}


def test_assessment_reference_defaults_to_pinned_official_hash(tmp_path, source, config):
    train_path = save_source(tmp_path, source)
    reference = save_source(tmp_path, source, "not_official_val.json")
    with pytest.raises(DataValidationError, match="SHA-256"):
        data.prepare_metadata_candidate(train_path, config, source_role="official_train",
                                        assessment_annotations=reference)


@pytest.mark.parametrize("section", ["images", "annotations", "categories", "licenses"])
def test_duplicate_ids_are_hard_errors(tmp_path, source, config, section):
    source[section].append(deepcopy(source[section][0]))
    with pytest.raises(DataValidationError, match="Duplicate"):
        prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")


@pytest.mark.parametrize("key,value", [("image_id", 999999), ("category_id", 999999), ("id", True)])
def test_unknown_or_boolean_annotation_ids_are_hard_errors(tmp_path, source, config, key, value):
    source["annotations"][0][key] = value
    with pytest.raises(DataValidationError):
        prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")


@pytest.mark.parametrize("box", [[-1, 0, 10, 10], [0, 0, 0, 10], [95, 0, 10, 10],
                                 [0, 119, 1, 2], ["0", 0, 10, 10], [False, 0, 10, 10], [0, 0, 10]])
def test_invalid_bboxes_are_excluded_and_counted(tmp_path, source, config, box):
    source["annotations"][0]["bbox"] = box
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    assert result["audit"]["exclusions"]["annotations_by_primary_reason"]["invalid_or_out_of_bounds_bbox"] == 1
    assert len(result["candidate_manifest"]["records"]) == 1199


def test_nonfinite_bbox_is_never_accepted(tmp_path, source, config):
    path = save_source(tmp_path, source)
    text = path.read_text().replace('"bbox": [10, 20, 30, 40]', '"bbox": [1e309, 20, 30, 40]', 1)
    path.write_text(text)
    result = prepare_metadata_candidate(path, config, source_role="official_train")
    assert result["audit"]["exclusions"]["annotations_by_primary_reason"]["invalid_or_out_of_bounds_bbox"] == 1


@pytest.mark.parametrize("field,value,reason", [("license", 999, "unknown_or_missing_license"),
    ("original_url", "file:///private/secret.jpg", "invalid_original_url"),
    ("file_name", "../escape.jpg", "invalid_image_file_name"),
    ("file_name", "CON.jpg", "invalid_image_file_name"),
    ("width", True, "invalid_image_dimensions")])
def test_invalid_images_are_excluded_fail_closed(tmp_path, source, config, field, value, reason):
    source["images"][0][field] = value
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    assert result["audit"]["exclusions"]["images_by_reason_nonexclusive"][reason] == 1
    assert all(row["image_id"] != 0 for row in result["candidate_manifest"]["records"])


def test_unknown_license_policy_and_empty_license_url_fail_closed(tmp_path, source, config):
    source["licenses"][0]["url"] = ""
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    assert not result["audit"]["readiness"]["metadata_ready"]
    assert result["candidate_manifest"] is None
    config["license_policy"]["allowed_license_ids"] = [999]
    with pytest.raises(DataValidationError, match="undefined license"):
        prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")


def test_official_assessment_has_no_training_manifest_or_training_splits(tmp_path, source, config):
    path = save_source(tmp_path, source, "instances_attributes_val2020.json")
    result = prepare_metadata_candidate(path, config, source_role="official_assessment")
    assert result["candidate_manifest"] is None
    assert result["audit"]["assessment_guard"]["metadata_audit_only"]
    for role in ("fit", "selection", "calibration"):
        assert all(value["crops"] == 0 for value in result["audit"]["counts"]["by_partition_and_class"][role].values())
    paths = write_preparation_outputs(result, tmp_path / "assessment_audit")
    assert set(paths) == {"audit"}
    assert not (tmp_path / "assessment_audit" / "candidate_crop_manifest.json").exists()
    with pytest.raises(DataValidationError, match="cannot be relabeled"):
        prepare_metadata_candidate(path, config, source_role="official_train")


def test_renamed_known_assessment_bytes_cannot_bypass_guard(tmp_path, source, config, monkeypatch):
    path = save_source(tmp_path, source, "renamed.json")
    monkeypatch.setattr(data, "ASSESSMENT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(DataValidationError, match="cannot be relabeled"):
        prepare_metadata_candidate(path, config, source_role="official_train")


def test_writer_independently_blocks_an_assessment_train_manifest_mixup(tmp_path, source, config):
    path = save_source(tmp_path, source)
    assessment = prepare_metadata_candidate(path, config, source_role="official_assessment")
    train = prepare_metadata_candidate(path, config, source_role="official_train")
    assessment["candidate_manifest"] = train["candidate_manifest"]
    with pytest.raises(DataValidationError, match="official-train"):
        write_preparation_outputs(assessment, tmp_path / "mixup")
    assert not (tmp_path / "mixup").exists()


def test_writer_rejects_an_edited_candidate_before_export(tmp_path, source, config):
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    result["candidate_manifest"]["records"][0]["bbox_xywh"][2] = 999
    with pytest.raises(DataValidationError, match="changed after preparation"):
        write_preparation_outputs(result, tmp_path / "edited")
    assert not (tmp_path / "edited").exists()


def test_deterministic_manifest_and_source_hash_binding(tmp_path, source, config):
    path = save_source(tmp_path, source)
    first = prepare_metadata_candidate(path, config, source_role="official_train")
    second = prepare_metadata_candidate(path, config, source_role="official_train")
    assert first == second
    with pytest.raises(DataValidationError, match="SHA-256"):
        prepare_metadata_candidate(path, config, source_role="official_train", expected_sha256="0" * 64)
    path.write_text(path.read_text() + "\n")
    changed = prepare_metadata_candidate(path, config, source_role="official_train")
    assert changed["audit"]["source_annotation"]["sha256"] != first["audit"]["source_annotation"]["sha256"]
    assert changed["candidate_manifest"]["records"] == first["candidate_manifest"]["records"]


def test_insufficient_support_writes_audit_only_and_no_overwrite(tmp_path, source, config):
    config["min_source_images_per_class"] = {role: 1000 for role in data.PARTITIONS}
    result = prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")
    assert result["candidate_manifest"] is None
    assert result["audit"]["readiness"]["failures"]
    output = tmp_path / "not_ready"
    assert set(write_preparation_outputs(result, output)) == {"audit"}
    with pytest.raises(FileExistsError):
        write_preparation_outputs(result, output)


def test_class_map_must_be_real_and_unique(tmp_path, source, config):
    config["class_map"][0]["category_id"] = 9876
    with pytest.raises(DataValidationError, match="absent"):
        prepare_metadata_candidate(save_source(tmp_path, source), config, source_role="official_train")


def test_streaming_matches_stdlib_numeric_and_group_semantics(tmp_path, source, config, monkeypatch):
    pytest.importorskip("ijson")
    path = save_source(tmp_path, source)
    expected = prepare_metadata_candidate(path, config, source_role="official_train")
    monkeypatch.setattr(data, "_SMALL_JSON_BYTES", 0)
    actual = prepare_metadata_candidate(path, config, source_role="official_train")
    assert actual["audit"]["reader"] == "ijson_streaming"
    actual["audit"]["reader"] = expected["audit"]["reader"]
    assert actual == expected


def test_cli_exit_code_and_raw_config_hash(tmp_path, source, config, capsys):
    from tools.prepare_garment_data import main
    annotation_path = save_source(tmp_path, source)
    config["min_source_images_per_class"] = {role: 1000 for role in data.PARTITIONS}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "cli_audit"
    assert main(["--annotations", str(annotation_path), "--config", str(config_path),
                 "--source-role", "official_train", "--out-dir", str(output)]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert not printed["metadata_ready"] and not printed["training_ready"]
    audit = json.loads((output / "DATA_METADATA_AUDIT.json").read_text())
    assert audit["input_config_file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
