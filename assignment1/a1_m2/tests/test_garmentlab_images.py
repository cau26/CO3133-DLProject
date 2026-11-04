"""No network/model training: bounded archives, duplicate roles and prepared cache."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

import garmentlab.images as images
from garmentlab.data import DataValidationError


def png(seed, size=(64, 48)):
    array = np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def fixture_dataset(tmp_path):
    class_ids = [0, 1, 4, 6, 8, 10]
    config = {"class_map": [{"category_id": key, "label": f"class_{key}"} for key in class_ids],
        "license_policy": {"purpose": "research_only", "allowed_license_ids": [0],
                           "acknowledgement": "Fixture allowlist is not legal clearance"},
        "split": {"seed": 20261024, "fit_fraction": .8, "selection_fraction": .1, "calibration_fraction": .1},
        "min_source_images_per_class": {role: 1 for role in images.ROLE_IDS}}
    paths = {}
    for role, per_class, offset in [("train", 100, 0), ("assessment", 3, 100000)]:
        data = {"licenses": [{"id":0,"name":"Attribution","url":"https://creativecommons.org/licenses/by/2.0/"}],
                "categories": [{"id":key,"name":f"Class {key}"} for key in class_ids],
                "images": [], "annotations": []}
        archive_path = tmp_path/f"{role}.zip"
        with zipfile.ZipFile(archive_path,"w") as archive:
            for class_index,category in enumerate(class_ids):
                for number in range(per_class):
                    image_id = offset+class_index*100+number
                    name = f"source_{image_id}.png"
                    data["images"].append({"id":image_id,"file_name":name,"width":64,"height":48,
                        "original_url":f"https://example.org/{image_id}.png","license":0})
                    data["annotations"].append({"id":image_id,"image_id":image_id,
                        "category_id":category,"bbox":[0,0,64,48]})
                    archive.writestr(f"{role}/{name}",png(image_id))
        metadata_path = tmp_path/f"{role}_metadata.json"
        metadata_path.write_text(json.dumps(data))
        paths[f"{role}_metadata"] = metadata_path
        paths[f"{role}_archive"] = archive_path
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    return config_path, paths


def prepare_fixture(tmp_path, config, paths):
    return images.prepare_dataset(tmp_path/"work",tmp_path/"cache",config,download=False,resolution=32,
        near_duplicate_distance=0,expected_hashes={key:images.sha256_file(path) for key,path in paths.items()},**paths)


def test_prepared_arrays_bind_hashes_roles_and_cache_reuse(tmp_path):
    config,paths = fixture_dataset(tmp_path)
    output = prepare_fixture(tmp_path,config,paths)
    receipt = json.loads((output/"DATA_READINESS.json").read_text())
    index = json.loads((output/"index_manifest.json").read_text())
    assert receipt["training_ready"] and not receipt["legal_clearance_claimed"]
    array = np.load(output/"images_uint8.npy",mmap_mode="r")
    roles = np.load(output/"roles.npy")
    labels = np.load(output/"labels.npy")
    assert array.shape == (618,32,32,3) and array.dtype == np.uint8
    assert labels.shape == roles.shape == (618,) and set(roles.tolist()) == {0,1,2,3}
    assert len(index["labels"]) == 6 and len(index["records"]) == 618
    assert all(row["array_index"] == number for number,row in enumerate(index["records"]))
    assert all(images.ROLE_IDS[row["role"]] == roles[number] for number,row in enumerate(index["records"]))
    assert (output/"TRAIN_PREVIEW.jpg").exists()
    assert prepare_fixture(tmp_path,config,paths) == output
    array._mmap.close()
    raw = bytearray((output/"labels.npy").read_bytes())
    raw[-1] ^= 1
    (output/"labels.npy").write_bytes(raw)
    with pytest.raises(DataValidationError,match="cache hash failed"):
        prepare_fixture(tmp_path,config,paths)


@pytest.mark.parametrize("bad",["../escape.jpg","/absolute.jpg","C:/drive.jpg","a\\escape.jpg"])
def test_zip_traversal_is_rejected_before_any_member_read(tmp_path,bad):
    path = tmp_path/"bad.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr(bad,b"not-an-image")
    if "\\" in bad:
        # Windows ZipInfo sanitizes separators when writing; reproduce an actual
        # malicious central/local filename rather than testing the writer.
        path.write_bytes(path.read_bytes().replace(bad.replace("\\","/").encode(),bad.encode()))
    with zipfile.ZipFile(path) as archive,pytest.raises(DataValidationError,match="Unsafe"):
        images.safe_zip_index(archive,set())


def test_duplicate_basename_and_oversized_selected_archive_entries_fail(tmp_path):
    path = tmp_path/"ambiguous.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("one/x.png",b"x")
        archive.writestr("two/x.png",b"x")
    with zipfile.ZipFile(path) as archive,pytest.raises(DataValidationError,match="ambiguous"):
        images.safe_zip_index(archive,{"x.png"})
    path = tmp_path/"large.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("x.png",b"x"*20)
    with zipfile.ZipFile(path) as archive,pytest.raises(DataValidationError,match="limits"):
        images.safe_zip_index(archive,{"x.png"},max_member_bytes=10)


def test_global_duplicate_graph_quarantines_whole_lower_priority_groups():
    sources = [
        {"source_group_id":"fit","image_id":1,"role":"fit","decoded_rgb_sha256":"a","dhash64":"0000000000000000"},
        {"source_group_id":"selection","image_id":2,"role":"selection","decoded_rgb_sha256":"b","dhash64":"0000000000000001"},
        {"source_group_id":"calibration","image_id":3,"role":"calibration","decoded_rgb_sha256":"c","dhash64":"0000000000000003"},
        {"source_group_id":"assessment","image_id":4,"role":"assessment","decoded_rgb_sha256":"c","dhash64":"0000000000000007"},
        {"source_group_id":"unique","image_id":5,"role":"fit","decoded_rgb_sha256":"e","dhash64":"ffffffffffffffff"}]
    removed,component,audit = images.duplicate_quarantine(sources,1)
    assert removed == {"fit","selection","calibration"}
    assert len({component[key] for key in removed|{"assessment"}}) == 1
    assert component["unique"] != component["assessment"]
    assert audit["decoded_exact_links"] == 1
    assert audit["cross_role_components"][0]["kept_role"] == "assessment"


def test_raw_bbox_rejects_nonidentity_exif(tmp_path):
    stream = io.BytesIO()
    image = Image.new("RGB",(64,48),"red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(stream,format="JPEG",exif=exif)
    path = tmp_path/"exif.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("x.jpg",stream.getvalue())
    with zipfile.ZipFile(path) as archive,pytest.raises(ValueError,match="nonidentity_exif"):
        images._read_image(archive,archive.getinfo("x.jpg"),{"width":64,"height":48})


def test_integer_crops_and_training_serve_letterbox_parity():
    from garmentlab.inference import preprocess_image
    payload = png(81,size=(60,40))
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    box = images.integer_crop({"bbox_xywh":[.2,1.2,30.1,20.1],"width":60,"height":40})
    assert box == [0,1,31,22]
    prepared = images.letterbox_rgb(image,box,32).astype(np.float32)/255
    manifest = {"domain":"garment_rgb","preprocessing":{"version":"garment_rgb_letterbox_v1",
        "height":32,"width":32,"channels":3,"mode":"RGB","fill":[0,0,0],
        "resize":"letterbox_bilinear","normalization":"imagenet_v1"}}
    tensor,_ = preprocess_image(payload,manifest,box)
    expected = (prepared.transpose(2,0,1)-np.array([.485,.456,.406],dtype=np.float32)[:,None,None]) / np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
    assert np.allclose(tensor.numpy()[0],expected,atol=1e-6)


def test_download_resume_and_cached_byte_hash_gate(tmp_path,monkeypatch):
    target = tmp_path/"object.bin"
    payload = b"0123456789"
    target.with_suffix(".bin.part").write_bytes(payload[:3])
    target.with_suffix(".bin.part.json").write_text(json.dumps({"url":"https://example.org/data",
        "expected_bytes":10,"etag":"etag-v1"}))
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range":"bytes 3-9/10","ETag":"etag-v1"}
    def fetch(request,timeout):
        assert request.get_header("Range") == "bytes=3-"
        return Response(payload[3:])
    monkeypatch.setattr(images,"urlopen",fetch)
    digest = hashlib.sha256(payload).hexdigest()
    assert images.download_cached("https://example.org/data",target,expected_bytes=10,expected_sha256=digest).read_bytes() == payload
    target.write_bytes(b"X"+payload[1:])
    with pytest.raises(DataValidationError,match="hash failed"):
        images.download_cached("https://example.org/data",target,expected_bytes=10,expected_sha256=digest)


def test_failed_post_image_support_never_claims_training_ready(tmp_path):
    config,paths = fixture_dataset(tmp_path)
    parsed = json.loads(config.read_text())
    parsed["min_source_images_per_class"]["assessment"] = 4
    config.write_text(json.dumps(parsed))
    with pytest.raises(DataValidationError,match="Metadata support"):
        prepare_fixture(tmp_path,config,paths)
    assert not (tmp_path/"work"/"prepared").exists()
