"""PASCAL VOC 2007 download, leak-free split verification, Dataset and DataLoader."""
import hashlib
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import CLASS_TO_ID, ID_TO_CLASS, ROOT, VOC_CLASSES, file_digest, save_json


def download_and_extract_voc2007(data_dir, config, manual_archive_path=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    voc_root = data_dir / "VOCdevkit" / "VOC2007"
    archive_path = Path(manual_archive_path) if manual_archive_path else data_dir / config["archive_name"]

    if (voc_root / "JPEGImages").exists() and (voc_root / "Annotations").exists():
        if archive_path.exists():
            actual_md5 = file_digest(archive_path, "md5")
            if actual_md5 != config["archive_md5"]:
                raise ValueError(f"Archive MD5 mismatch: {actual_md5} != {config['archive_md5']}")
        return voc_root, archive_path

    if not archive_path.exists():
        urls = [
            config["archive_url"],
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        ]
        partial = archive_path.with_suffix(".tar.part")
        last_err = None
        for url in urls:
            try:
                print(f"Downloading {url} ...", flush=True)
                req = urllib.request.Request(url, headers={"User-Agent": "CO3133-VOC2007-Pipeline"})
                downloaded, last_report = 0, 0
                with urllib.request.urlopen(req, timeout=90) as resp, partial.open("wb") as out_f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 50 * 1024 * 1024:
                            print(f"  Downloaded {downloaded / (1024 ** 2):.0f} MiB", flush=True)
                            last_report = downloaded
                partial.replace(archive_path)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if partial.exists():
                    partial.unlink()
        if last_err is not None:
            raise RuntimeError(f"Failed to download VOC2007 archive: {last_err}") from last_err

    actual_md5 = file_digest(archive_path, "md5")
    if actual_md5 != config["archive_md5"]:
        raise ValueError(f"MD5 mismatch for {archive_path}: got {actual_md5}, expected {config['archive_md5']}")

    print(f"Extracting {archive_path} ...", flush=True)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if (Path(member.name).is_absolute() or ".." in parts
                    or not parts or parts[0] != "VOCdevkit"
                    or (len(parts) > 1 and parts[1] != "VOC2007")
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsafe or unexpected archive entry: {member.name}")
            target = data_dir / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return voc_root, archive_path


def parse_voc_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            continue
        difficult = int(obj.findtext("difficult", "0"))
        truncated = int(obj.findtext("truncated", "0"))
        xmin = float(obj.findtext("bndbox/xmin")) - 1.0
        ymin = float(obj.findtext("bndbox/ymin")) - 1.0
        xmax = float(obj.findtext("bndbox/xmax"))
        ymax = float(obj.findtext("bndbox/ymax"))
        xmin = max(0.0, min(xmin, float(width - 1)))
        ymin = max(0.0, min(ymin, float(height - 1)))
        xmax = max(xmin + 1.0, min(xmax, float(width)))
        ymax = max(ymin + 1.0, min(ymax, float(height)))
        rel_area = ((xmax - xmin) * (ymax - ymin)) / float(width * height)
        objects.append({
            "class_name": name,
            "label": CLASS_TO_ID[name],
            "difficult": difficult,
            "truncated": truncated,
            "box": [xmin, ymin, xmax, ymax],
            "relative_area": rel_area,
        })
    return width, height, objects


class VOC2007DetectionDataset(Dataset):
    """PyTorch Dataset for VOC2007 object detection using finalized split IDs."""

    def __init__(self, voc_root, image_ids, split="train", augment=False, seed=42):
        self.voc_root = Path(voc_root)
        self.image_ids = list(image_ids)
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = self.voc_root / "JPEGImages" / f"{image_id}.jpg"
        xml_path = self.voc_root / "Annotations" / f"{image_id}.xml"

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        height, width = arr.shape[0], arr.shape[1]
        _, _, objs = parse_voc_xml(xml_path)

        # Training targets exclude difficult=1 boxes; validation retains all boxes for VOC ignore rules
        train_objs = [o for o in objs if o["difficult"] == 0]
        boxes = np.array([o["box"] for o in train_objs], dtype=np.float32).reshape(-1, 4)
        labels = np.array([o["label"] for o in train_objs], dtype=np.int64)

        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 100000 + index)
            if rng.random() < 0.5:
                arr = np.ascontiguousarray(arr[:, ::-1, :])
                if len(boxes) > 0:
                    old_xmin = boxes[:, 0].copy()
                    old_xmax = boxes[:, 2].copy()
                    boxes[:, 0] = float(width) - old_xmax
                    boxes[:, 2] = float(width) - old_xmin

        image_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else np.zeros((0,), dtype=np.float32)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.from_numpy(areas.astype(np.float32)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "voc_image_id": image_id,
            "orig_size": (height, width),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def prepare_voc_data(data_dir, output_dir, config, split_dir=None, manual_archive_path=None):
    voc_root, archive_path = download_and_extract_voc2007(data_dir, config, manual_archive_path)
    split_dir = Path(split_dir) if split_dir else (ROOT / "splits")
    train_ids = [x.strip() for x in (split_dir / "train.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    val_ids = [x.strip() for x in (split_dir / "val.txt").read_text(encoding="utf-8").splitlines() if x.strip()]

    assert len(train_ids) == 4014 and len(val_ids) == 997, f"Expected 4014/997 finalized split, got {len(train_ids)}/{len(val_ids)}"
    assert set(train_ids).isdisjoint(set(val_ids)), "Train and validation image IDs overlap!"
    assert "000863" in train_ids and "008364" in train_ids, "Near-duplicate pair 000863/008364 must both be in train!"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_splits = out / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    (out_splits / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_splits / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    split_bytes = ("\n".join(train_ids) + "\n---\n" + "\n".join(val_ids)).encode("utf-8")
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()

    gt_val = {}
    val_non_diff_count, val_diff_count = 0, 0
    for image_id in val_ids:
        _, _, objs = parse_voc_xml(voc_root / "Annotations" / f"{image_id}.xml")
        boxes = np.array([o["box"] for o in objs], dtype=np.float64).reshape(-1, 4)
        labels = np.array([o["label"] for o in objs], dtype=np.int64)
        difficult = np.array([o["difficult"] for o in objs], dtype=np.int64)
        rel_areas = np.array([o["relative_area"] for o in objs], dtype=np.float64)
        val_non_diff_count += int((difficult == 0).sum())
        val_diff_count += int((difficult == 1).sum())
        gt_val[image_id] = {
            "boxes": boxes,
            "labels": labels,
            "difficult": difficult,
            "relative_areas": rel_areas,
        }

    train_non_diff_count, train_diff_count = 0, 0
    for image_id in train_ids:
        _, _, objs = parse_voc_xml(voc_root / "Annotations" / f"{image_id}.xml")
        for o in objs:
            if o["difficult"] == 0:
                train_non_diff_count += 1
            else:
                train_diff_count += 1

    train_dataset = VOC2007DetectionDataset(voc_root, train_ids, split="train", augment=True, seed=config["seed"])
    val_dataset = VOC2007DetectionDataset(voc_root, val_ids, split="val", augment=False, seed=config["seed"])

    summary = {
        "dataset": "PASCAL VOC 2007",
        "release": config["archive_name"],
        "source": "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/",
        "license_note": "Publisher Database Rights statement and Flickr terms apply; not blanket MIT/CC-BY.",
        "archive_md5": config["archive_md5"],
        "official_trainval_images": len(train_ids) + len(val_ids),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "train_non_difficult_objects": train_non_diff_count,
        "train_difficult_objects_excluded": train_diff_count,
        "val_non_difficult_objects": val_non_diff_count,
        "val_difficult_objects_ignored": val_diff_count,
        "num_foreground_classes": len(VOC_CLASSES),
        "num_classes_with_background": config["num_classes"],
        "classes": VOC_CLASSES,
        "seed": config["seed"],
        "split_method": "Multi-Label Iterative Group Stratified Split + near-duplicate pair (000863/008364) grouped into train",
        "split_sha256": split_sha256,
        "leakage_checks": "0 image ID overlap; 3 exact-duplicate groups and 1 confirmed near-duplicate pair kept within single splits",
        "test_set_evaluated": False,
    }
    save_json(out / "data_summary.json", summary)
    export_batch_preview(train_dataset, out / "eda")

    return {
        "voc_root": voc_root,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "gt_val": gt_val,
        "summary": summary,
    }


def export_batch_preview(train_dataset, eda_out):
    eda_out = Path(eda_out)
    eda_out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))
    info = {
        "batch_size": len(images),
        "image_shapes": [list(img.shape) for img in images],
        "image_dtype": str(images[0].dtype),
        "pixel_min": float(min(img.min().item() for img in images)),
        "pixel_max": float(max(img.max().item() for img in images)),
        "target_box_counts": [int(len(t["boxes"])) for t in targets],
        "example_image_ids": [t["voc_image_id"] for t in targets],
    }
    save_json(eda_out / "batch_info.json", info)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), layout="constrained")
    for ax, img, tgt in zip(axes.flat, images, targets):
        np_img = img.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(np_img, 0.0, 1.0))
        for box, label in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
            xmin, ymin, xmax, ymax = box
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor="#16a34a", linewidth=1.6))
            ax.text(xmin, max(ymin - 3, 8), ID_TO_CLASS[int(label)],
                    color="white", fontsize=7.5,
                    bbox={"facecolor": "#16a34a", "alpha": 0.75, "pad": 1, "edgecolor": "none"})
        ax.set_title(f"ID {tgt['voc_image_id']} ({img.shape[1]}x{img.shape[2]}, {len(tgt['boxes'])} boxes)", fontsize=8.5)
        ax.axis("off")
    fig.suptitle("Training batch after preprocessing & augmentation (float32 [0, 1], difficult=0 boxes)", fontsize=11)
    fig.savefig(eda_out / "batch_preview.png", dpi=150)
    plt.close(fig)


def make_detection_loaders(data, model_cfg, config):
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        data["train_dataset"],
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(config["device"] == "cuda"),
        generator=generator,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        data["val_dataset"],
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(config["device"] == "cuda"),
        collate_fn=detection_collate_fn,
    )
    return train_loader, val_loader
