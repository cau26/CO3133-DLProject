"""Fashionpedia image preparation with bounded downloads and source-group gates.

No model is trained here. The prepared cache is local; original download objects
can live on Drive. Metadata licenses are filters, not legal clearance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import (ASSESSMENT_SHA256, DataValidationError, _MetadataReader, _bbox,
                   _canonical_bytes, _image_metadata, prepare_metadata_candidate,
                   sha256_file, validate_data_config)


TRAIN_METADATA_SHA256 = "f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006"
ROLE_IDS = {"fit": 0, "selection": 1, "calibration": 2, "assessment": 3}
SOURCES = {
    "train_metadata": ("https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json",
                       "instances_attributes_train2020.json", 542193045, TRAIN_METADATA_SHA256),
    "assessment_metadata": ("https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",
                            "instances_attributes_val2020.json", 14533475, ASSESSMENT_SHA256),
    "train_archive": ("https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
                      "train2020.zip", 3344364592, None),
    "assessment_archive": ("https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
                           "val_test2020.zip", 236499034,
                           "44eb1685364b6ed6bda4595fe0dc07a252e9b923112b97b24ff4d440751a2acc"),
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000


def _write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_cached(url: str, cache_path: str | Path, *, expected_bytes: int,
                    expected_sha256: str | None = None, is_zip: bool = False,
                    total_timeout: float = 7200, request_timeout: float = 60) -> Path:
    """Resume only the same URL/size/ETag; verify complete cache before reuse."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(path.suffix + ".download.json")
    if path.exists():
        digest = sha256_file(path)
        if path.stat().st_size != expected_bytes:
            raise DataValidationError(f"Existing cache has wrong size: {path.name}")
        if expected_sha256 and digest != expected_sha256:
            raise DataValidationError(f"Existing cache hash failed: {path.name}")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("url") != url or receipt.get("sha256") != digest or receipt.get("bytes") != expected_bytes:
                raise DataValidationError(f"Existing download receipt does not match: {path.name}")
            if is_zip and not receipt.get("crc_checked"):
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise DataValidationError("Cached archive CRC failed")
                receipt["crc_checked"] = True
                _write_json(receipt_path, receipt)
            return path
        if is_zip:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise DataValidationError("Cached archive CRC failed")
        _write_json(receipt_path, {"url": url, "bytes": expected_bytes, "sha256": digest,
                                   "crc_checked": bool(is_zip), "pinned_upstream_sha256": expected_sha256})
        return path
    partial = path.with_suffix(path.suffix + ".part")
    state_path = path.with_suffix(path.suffix + ".part.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and (state.get("url") != url or state.get("expected_bytes") != expected_bytes):
        raise DataValidationError("Partial download belongs to another source")
    if partial.exists() and not state:
        raise DataValidationError("Partial download is missing its source receipt")
    deadline = time.monotonic() + total_timeout
    failures = 0
    while time.monotonic() < deadline:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise DataValidationError("Partial download exceeds declared source size")
        if offset == expected_bytes:
            break
        # Appending without an object validator could combine two source versions.
        if offset and not state.get("etag"):
            offset = 0
        headers = {"User-Agent": "GarmentLab-data-preparation/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if state.get("etag"):
                headers["If-Range"] = state["etag"]
        try:
            with urlopen(Request(url, headers=headers), timeout=min(request_timeout, max(1, deadline-time.monotonic()))) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_bytes}"):
                        raise DataValidationError("Server returned an invalid resume range")
                    mode = "ab"
                elif status == 200:
                    mode, offset = "wb", 0
                else:
                    raise DataValidationError(f"Unexpected download HTTP status: {status}")
                etag = response.headers.get("ETag")
                if offset and state.get("etag") and etag and state["etag"] != etag:
                    raise DataValidationError("Source ETag changed during resume")
                state = {"url": url, "expected_bytes": expected_bytes, "etag": etag}
                _write_json(state_path, state)
                with partial.open(mode) as target:
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Download reached total time budget; partial cache retained")
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                        offset += len(block)
                        if offset > expected_bytes:
                            raise DataValidationError("Download exceeded declared size")
        except (HTTPError, URLError, OSError) as exc:
            failures += 1
            if failures >= 5:
                raise RuntimeError("Download interrupted repeatedly; partial cache retained for resume") from exc
            continue
        if partial.stat().st_size < expected_bytes:
            failures += 1
            if failures >= 5:
                raise RuntimeError("Incomplete response; partial cache retained")
    if not partial.exists() or partial.stat().st_size != expected_bytes:
        raise TimeoutError("Download incomplete within total time budget; partial cache retained")
    digest = sha256_file(partial)
    if expected_sha256 and digest != expected_sha256:
        raise DataValidationError("Downloaded source SHA-256 failed; do not edit the expected hash")
    if is_zip:
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise DataValidationError("Downloaded archive CRC failed")
    partial.replace(path)
    _write_json(receipt_path, {"url": url, "bytes": expected_bytes, "sha256": digest,
                               "crc_checked": bool(is_zip), "pinned_upstream_sha256": expected_sha256})
    return path


def safe_zip_index(archive: zipfile.ZipFile, selected_names: set[str], *,
                   max_member_bytes: int = MAX_IMAGE_BYTES) -> dict[str, zipfile.ZipInfo]:
    """Never extract paths; reject ambiguous names and dangerous central entries."""
    seen, selected = set(), {}
    if len(archive.infolist()) > 200_000:
        raise DataValidationError("Archive has too many entries")
    for info in archive.infolist():
        raw_name = info.orig_filename
        path = PurePosixPath(info.filename)
        if ("\\" in raw_name or "\x00" in raw_name or path.is_absolute() or ".." in path.parts
                or any(":" in part for part in path.parts) or not path.parts):
            raise DataValidationError("Unsafe archive member path")
        canonical = str(path).casefold().rstrip("/")
        if canonical in seen:
            raise DataValidationError("Duplicate archive member path")
        seen.add(canonical)
        if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
            raise DataValidationError("Symlink/encrypted ZIP members are unsupported")
        if info.is_dir():
            continue
        if path.name in selected_names:
            if path.name in selected:
                raise DataValidationError("Selected filename is ambiguous across archive directories")
            if not 0 < info.file_size <= max_member_bytes or info.compress_size == 0:
                raise DataValidationError("Selected image exceeds compressed-input limits")
            if info.file_size / info.compress_size > 1000:
                raise DataValidationError("Suspicious selected-image compression ratio")
            selected[path.name] = info
    missing = selected_names - selected.keys()
    if missing:
        raise DataValidationError(f"Archive missing {len(missing)} selected image files")
    return selected


def _read_image(archive: zipfile.ZipFile, info: zipfile.ZipInfo, record: dict) -> tuple[Image.Image, dict]:
    payload = archive.read(info)
    if len(payload) != info.file_size or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("image_byte_limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP") or getattr(image, "n_frames", 1) != 1:
                raise ValueError("unsupported_image_format_or_animation")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("image_pixel_limit")
            if image.getexif().get(274, 1) != 1:
                raise ValueError("nonidentity_exif_rejected_for_raw_annotation_coordinates")
            if image.size != (record["width"], record["height"]):
                raise ValueError("metadata_decoded_dimension_mismatch")
            image.load()
            rgb = image.convert("RGB")
    array = np.asarray(rgb)
    pixel_sha = hashlib.sha256(str(rgb.size).encode("ascii") + array.tobytes()).hexdigest()
    gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    dhash = int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")
    return rgb, {"image_bytes_sha256": hashlib.sha256(payload).hexdigest(),
                 "decoded_rgb_sha256": pixel_sha, "dhash64": f"{dhash:016x}"}


def integer_crop(row: dict) -> list[int]:
    x, y, width, height = row["bbox_xywh"]
    box = [math.floor(x), math.floor(y), math.ceil(x+width), math.ceil(y+height)]
    if not 0 <= box[0] < box[2] <= row["width"] or not 0 <= box[1] < box[3] <= row["height"]:
        raise ValueError("rounded_crop_out_of_bounds")
    if box[2]-box[0] < 8 or box[3]-box[1] < 8:
        raise ValueError("crop_smaller_than_8px")
    return box


def letterbox_rgb(image: Image.Image, crop: list[int], resolution: int = 224) -> np.ndarray:
    """Same PIL contain/centering/fill as garmentlab.inference before normalization."""
    cropped = image.convert("RGB").crop(crop)
    scaled = ImageOps.contain(cropped, (resolution, resolution), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    canvas.paste(scaled, ((resolution-scaled.width)//2, (resolution-scaled.height)//2))
    return np.asarray(canvas, dtype=np.uint8)


class _Groups:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class _BKTree:
    def __init__(self):
        self.root = None

    def insert(self, value, group):
        if self.root is None:
            self.root = [value, group, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[2]:
                node[2][distance] = [value, group, {}]
                return
            node = node[2][distance]

    def neighbors(self, value, threshold):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                yield node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance-threshold <= edge <= distance+threshold)


def duplicate_quarantine(sources: list[dict], near_distance: int = 2) -> tuple[set[str], dict[str, str], dict]:
    """Global connected components; retain highest role, quarantine whole lower groups."""
    if type(near_distance) is not int or not 0 <= near_distance <= 2:
        raise ValueError("near_distance must be 0, 1 or 2")
    groups = _Groups({row["source_group_id"] for row in sources})
    exact, tree = {}, _BKTree()
    exact_links = near_links = 0
    for row in sorted(sources, key=lambda item: (item["source_group_id"], item["image_id"])):
        group, digest = row["source_group_id"], row["decoded_rgb_sha256"]
        if digest in exact:
            groups.join(group, exact[digest])
            exact_links += 1
        else:
            exact[digest] = group
        code = int(row["dhash64"], 16)
        for other in tree.neighbors(code, near_distance):
            groups.join(group, other)
            if other != group:
                near_links += 1
        tree.insert(code, group)
    components = defaultdict(list)
    for row in sources:
        components[groups.find(row["source_group_id"])].append(row)
    quarantine, component_ids, cross = set(), {}, []
    for root, rows in sorted(components.items()):
        component = hashlib.sha256("|".join(sorted({r["source_group_id"] for r in rows})).encode()).hexdigest()
        for row in rows:
            component_ids[row["source_group_id"]] = component
        roles = {row["role"] for row in rows}
        if len(roles) > 1:
            keep = max(roles, key=ROLE_IDS.__getitem__)
            removed = {row["source_group_id"] for row in rows if row["role"] != keep}
            quarantine.update(removed)
            cross.append({"component_id": component, "roles": sorted(roles, key=ROLE_IDS.__getitem__),
                          "kept_role": keep, "quarantined_source_group_ids": sorted(removed)})
    return quarantine, component_ids, {"decoded_exact_links": exact_links, "dhash_neighbor_links": near_links,
        "dhash_hamming_threshold": near_distance, "cross_role_components": cross,
        "quarantined_source_groups": len(quarantine), "priority": ["assessment", "calibration", "selection", "fit"],
        "limitations": "Conservative transitive dHash graph; false positives possible; no complete near-duplicate or identity guarantee"}


def _assessment_rows(path: Path, config: dict, expected_sha256: str) -> tuple[list[dict], dict]:
    audit = prepare_metadata_candidate(path, config, source_role="official_assessment",
                                       expected_sha256=expected_sha256)["audit"]
    normalized = validate_data_config(config)
    reader = _MetadataReader(path)
    licenses = {row["id"]: row for row in reader.records("licenses")}
    classes = {row["category_id"]: {**row, "class_index": index}
               for index, row in enumerate(normalized["class_map"])}
    images = {}
    for image in reader.records("images"):
        record, _ = _image_metadata(image, licenses, set(normalized["license_policy"]["allowed_license_ids"]))
        if record:
            images[image["id"]] = record
    rows = []
    for annotation in reader.records("annotations"):
        image = images.get(annotation["image_id"])
        label = classes.get(annotation["category_id"])
        if image is not None and label is not None:
            box = _bbox(annotation["bbox"], image["width"], image["height"])
            if box:
                rows.append({**image, "partition": "assessment", "annotation_id": annotation["id"],
                             "bbox_xywh": box, "category_id": annotation["category_id"], **label})
    return rows, audit


def _contact_sheet(arrays, rows: list[dict], path: Path, *, limit: int = 36):
    selected = []
    by_class = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row["label"]].append(index)
    for offset in range(max(1, math.ceil(limit/max(1, len(by_class))))):
        for label in sorted(by_class):
            if offset < len(by_class[label]):
                selected.append(by_class[label][offset])
    selected = selected[:limit]
    if not selected:
        return []
    columns, cell_w, cell_h = 6, 180, 220
    canvas = Image.new("RGB", (columns*cell_w, math.ceil(len(selected)/columns)*cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate(selected):
        x, y = position % columns * cell_w, position // columns * cell_h
        thumb = Image.fromarray(np.asarray(arrays[index])).resize((176,176))
        canvas.paste(thumb, (x+2,y+2))
        label = rows[index]["label"]
        draw.text((x+3,y+181), label[:25], fill="black")
        draw.text((x+3,y+196), f"image {rows[index]['image_id']} ann {rows[index]['annotation_id']}", fill="black")
    if selected:
        canvas.save(path, quality=90)
    return [rows[index] for index in selected]


def _credits(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "annotation_id", "label", "original_url", "license_id", "author"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "author_not_provided_in_annotations") for key in writer.fieldnames})


def visual_audit_assessment(annotations: str | Path, archive_path: str | Path, config_path: str | Path,
                            out_dir: str | Path, *, limit: int = 36,
                            expected_metadata_sha256: str = ASSESSMENT_SHA256) -> Path:
    """Visual/decode taxonomy audit only, not a model evaluation or training export."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    rows, audit = _assessment_rows(Path(annotations), config, expected_metadata_sha256)
    sampled = []
    per_class = max(1, math.ceil(limit/len(config["class_map"])))
    selected_count = Counter()
    arrays, exclusions = [], Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_zip_index(archive, {row["file_name"] for row in rows})
        for row in sorted(rows, key=lambda value: value["annotation_id"]):
            if selected_count[row["label"]] >= per_class:
                continue
            try:
                image, _ = _read_image(archive, members[row["file_name"]], row)
                array = letterbox_rgb(image, integer_crop(row))
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                exclusions[str(exc)] += 1
                continue
            arrays.append(array)
            sampled.append(row)
            selected_count[row["label"]] += 1
    shown = _contact_sheet(arrays, sampled, output/"DATA_VISUAL_AUDIT.jpg", limit=limit)
    _credits(output/"PHOTO_CREDITS.csv", shown)
    _write_json(output/"DATA_VISUAL_AUDIT.json", {"scope": "Assessment image decode/taxonomy inspection only; no model scores",
                "source_annotation_sha256": audit["source_annotation"]["sha256"],
                "archive_sha256": sha256_file(archive_path), "sampled_counts": dict(selected_count),
                "exclusions": dict(exclusions), "author_information": "Not supplied by annotations; source URLs preserved",
                "model_inference_performed": False, "legal_clearance_claimed": False})
    return output


def _source_path(key: str, provided, cache: Path, work: Path, download: bool,
                 expected_hashes: dict | None) -> tuple[Path, dict]:
    url, name, expected_bytes, pinned = SOURCES[key]
    expected = (expected_hashes or {}).get(key, pinned)
    if provided is not None:
        source = Path(provided)
        if not source.is_file():
            raise FileNotFoundError(source)
    else:
        source = cache/name
        if not download and not source.exists():
            raise FileNotFoundError(f"Offline cache missing {name}")
        source = download_cached(url, source, expected_bytes=expected_bytes,
                                 expected_sha256=expected, is_zip=key.endswith("archive"))
        target = work/"sources"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise DataValidationError("Local source copy does not match verified cache")
        else:
            temporary = target.with_suffix(target.suffix+".copying")
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != sha256_file(source):
                raise DataValidationError("Local source copy hash failed")
            temporary.replace(target)
        source = target
    digest = sha256_file(source)
    if expected and digest != expected:
        raise DataValidationError(f"Source hash failed: {key}")
    return source, {"url": url, "file_name": source.name, "bytes": source.stat().st_size,
                    "sha256": digest, "pinned_upstream_sha256": expected}


def prepare_dataset(work_dir: str | Path, cache_dir: str | Path, config_path: str | Path, *,
                    train_metadata=None, assessment_metadata=None, train_archive=None,
                    assessment_archive=None, resolution: int = 224, near_duplicate_distance: int = 2,
                    download: bool = True, expected_hashes: dict | None = None) -> Path:
    """Return prepared directory, verifying source/array hashes before cache reuse.

    expected_hashes is for controlled fixtures or a separately frozen data version;
    default production metadata hashes are pinned and must never be hand-edited.
    """
    if type(resolution) is not int or not 32 <= resolution <= 512:
        raise ValueError("resolution must be an integer from 32 through 512")
    work, cache = Path(work_dir), Path(cache_dir)
    work.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    normalized = validate_data_config(config)
    source_paths, provenance = {}, {}
    supplied = {"train_metadata": train_metadata, "assessment_metadata": assessment_metadata,
                "train_archive": train_archive, "assessment_archive": assessment_archive}
    for key in SOURCES:
        print(f"DATA SOURCE: {key}", flush=True)
        source_paths[key], provenance[key] = _source_path(key, supplied[key], cache, work, download, expected_hashes)
    binding = {"source_sha256": {key: value["sha256"] for key,value in provenance.items()},
               "data_config_sha256": hashlib.sha256(_canonical_bytes(normalized)).hexdigest(),
               "resolution": resolution, "near_duplicate_distance": near_duplicate_distance,
               "preparation_source_sha256": sha256_file(__file__)}
    prepared = work/"prepared"
    if prepared.exists():
        receipt = json.loads((prepared/"DATA_READINESS.json").read_text(encoding="utf-8"))
        if receipt.get("binding") != binding or not receipt.get("training_ready"):
            raise DataValidationError("Prepared cache belongs to a different/incomplete data study")
        for name, digest in receipt["files"].items():
            if sha256_file(prepared/name) != digest:
                raise DataValidationError(f"Prepared cache hash failed: {name}")
        return prepared
    train_result = prepare_metadata_candidate(source_paths["train_metadata"], config, source_role="official_train",
        expected_sha256=provenance["train_metadata"]["sha256"],
        assessment_annotations=source_paths["assessment_metadata"],
        assessment_expected_sha256=provenance["assessment_metadata"]["sha256"])
    if train_result["candidate_manifest"] is None:
        raise DataValidationError("Metadata support/reference gates failed before image preparation")
    assessment_rows, assessment_audit = _assessment_rows(source_paths["assessment_metadata"], config,
                                                        provenance["assessment_metadata"]["sha256"])
    all_rows = [{**row, "role": row["partition"], "archive_role": "train"}
                for row in train_result["candidate_manifest"]["records"]]
    all_rows += [{**row, "role": "assessment", "archive_role": "assessment"} for row in assessment_rows]
    grouped_rows = defaultdict(list)
    for row in all_rows:
        grouped_rows[(row["archive_role"], row["image_id"])].append(row)
    archives = {"train": zipfile.ZipFile(source_paths["train_archive"]),
                "assessment": zipfile.ZipFile(source_paths["assessment_archive"])}
    sources, accepted_rows, exclusions = [], [], Counter()
    try:
        members = {role: safe_zip_index(archive, {row["file_name"] for row in all_rows if row["archive_role"] == role})
                   for role,archive in archives.items()}
        for (archive_role,image_id), rows in sorted(grouped_rows.items()):
            first = rows[0]
            try:
                image, fingerprints = _read_image(archives[archive_role], members[archive_role][first["file_name"]], first)
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                exclusions[f"source:{str(exc)}"] += len(rows)
                continue
            valid_rows = []
            for row in rows:
                try:
                    crop = integer_crop(row)
                except ValueError as exc:
                    exclusions[str(exc)] += 1
                    continue
                valid_rows.append({**row, "crop_xyxy": crop, **fingerprints})
            if valid_rows:
                accepted_rows.extend(valid_rows)
                sources.append({**first, **fingerprints})
        quarantine, components, duplicate_audit = duplicate_quarantine(sources, near_duplicate_distance)
        kept = [row for row in accepted_rows if row["source_group_id"] not in quarantine]
        exclusions["cross_role_duplicate_group_crop_quarantine"] = len(accepted_rows)-len(kept)
        kept.sort(key=lambda row: (ROLE_IDS[row["role"]], row["image_id"], row["annotation_id"]))
        counts = {role: {} for role in ROLE_IDS}
        failures = []
        for role in ROLE_IDS:
            for item in normalized["class_map"]:
                rows = [row for row in kept if row["role"] == role and row["label"] == item["label"]]
                support = len({components[row["source_group_id"]] for row in rows})
                counts[role][item["label"]] = {"crops": len(rows), "independent_source_components": support}
                minimum = normalized["min_source_images_per_class"][role]
                if support < minimum:
                    failures.append({"role": role, "label": item["label"], "actual": support, "required": minimum})
        build = work/f"prepared_build_{uuid.uuid4().hex[:10]}"
        build.mkdir()
        _write_json(build/"DUPLICATE_AUDIT.json", duplicate_audit)
        if failures:
            _write_json(build/"DATA_READINESS.json", {"training_ready": False, "failures": failures,
                        "class_support": counts, "exclusions": dict(exclusions), "binding": binding})
            raise DataValidationError(f"Image-level class-support gates failed; inspect {build/'DATA_READINESS.json'}")
        images = np.lib.format.open_memmap(build/"images_uint8.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(kept), resolution, resolution, 3))
        last_key, decoded = None, None
        for index,row in enumerate(kept):
            key = (row["archive_role"], row["image_id"])
            if key != last_key:
                decoded, fingerprints = _read_image(archives[key[0]], members[key[0]][row["file_name"]], row)
                if fingerprints["decoded_rgb_sha256"] != row["decoded_rgb_sha256"]:
                    raise DataValidationError("Source image changed between audit and crop creation")
                last_key = key
            images[index] = letterbox_rgb(decoded, row["crop_xyxy"], resolution)
            row["array_index"] = index
            row["evaluation_group_id"] = components[row["source_group_id"]]
        images.flush()
        np.save(build/"labels.npy", np.asarray([row["class_index"] for row in kept], dtype=np.int64))
        np.save(build/"roles.npy", np.asarray([ROLE_IDS[row["role"]] for row in kept], dtype=np.uint8))
        index_manifest = {"schema": "garmentlab_prepared_images_v1", "labels": [item["label"] for item in normalized["class_map"]],
            "role_ids": ROLE_IDS, "records": kept, "binding": binding,
            "preprocessing": {"version": "garment_rgb_letterbox_v1", "height": resolution, "width": resolution,
                "channels": 3, "mode": "RGB", "fill": [0,0,0], "resize": "letterbox_bilinear", "normalization": "imagenet_v1"},
            "coordinate_policy": "Reject nonidentity EXIF; raw annotation bbox floor-left/top, ceil-right/bottom; same oriented coordinates for identity EXIF",
            "license_policy": normalized["license_policy"]}
        _write_json(build/"index_manifest.json", index_manifest)
        fit_indices = [i for i,row in enumerate(kept) if row["role"] == "fit"]
        preview_arrays = [images[index] for index in fit_indices]
        preview_rows = [kept[index] for index in fit_indices]
        shown = _contact_sheet(preview_arrays, preview_rows, build/"TRAIN_PREVIEW.jpg")
        _credits(build/"TRAIN_PREVIEW_CREDITS.csv", shown)
        _write_json(build/"METADATA_AUDITS.json", {"train": train_result["audit"], "assessment": assessment_audit})
        del preview_arrays
        images._mmap.close()
        del images
        files = {path.name: sha256_file(path) for path in build.iterdir() if path.is_file()}
        _write_json(build/"DATA_READINESS.json", {"schema": "garmentlab_image_readiness_v1", "training_ready": True,
            "scope": "Technical image/split/support checks; metadata license filter is not legal clearance",
            "binding": binding, "sources": provenance, "files": files, "class_support": counts,
            "exclusions": dict(exclusions), "duplicate_audit": duplicate_audit,
            "raw_records": len(all_rows), "prepared_crops": len(kept),
            "legal_clearance_claimed": False, "human_visual_review_attested": False})
        build.rename(prepared)
        return prepared
    finally:
        for archive in archives.values():
            archive.close()
