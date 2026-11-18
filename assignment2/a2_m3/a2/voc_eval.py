"""Official PASCAL VOC 2007 11-point interpolated AP@0.5 evaluator and error taxonomy.

Follows the VOC2007 devkit rules (Everingham et al., IJCV 2010):
- Detections are sorted by descending confidence score per class.
- Ground-truth boxes marked `difficult=1` are ignored: matching a difficult box
  counts neither as True Positive (TP) nor as False Positive (FP), and difficult
  boxes are excluded from the recall denominator (npos).
- Multiple detections matching the same non-difficult ground-truth box count as
  one TP (highest confidence) and subsequent FPs.
- Average Precision (AP) uses 11-point interpolation over recall thresholds
  t in {0.0, 0.1, ..., 1.0}.
"""
import numpy as np

from .common import ID_TO_CLASS, VOC_CLASSES


def compute_iou_matrix(boxes_a, boxes_b):
    """Compute pairwise IoU between two [N, 4] and [M, 4] float box arrays in (xmin, ymin, xmax, ymax)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    ixmin = np.maximum(a[:, None, 0], b[None, :, 0])
    iymin = np.maximum(a[:, None, 1], b[None, :, 1])
    ixmax = np.minimum(a[:, None, 2], b[None, :, 2])
    iymax = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def voc2007_11point_ap(rec, prec):
    """VOC 2007 11-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = rec >= t
        p = float(np.max(prec[mask])) if np.any(mask) else 0.0
        ap += p / 11.0
    return float(ap)


def evaluate_class_voc2007(gt_by_image, detections, class_id, iou_threshold=0.5, size_bucket=None):
    """Evaluate one class under VOC2007 rules.

    size_bucket: None (all objects) or one of {"small", "medium", "large"} based on relative area:
      - small: relative area < 0.02 (bottom ~25% of VOC boxes)
      - medium: 0.02 <= relative area < 0.15
      - large: relative area >= 0.15
    When size_bucket is set, ground-truth boxes outside the bucket are treated as difficult (ignored).
    """
    class_gt = {}
    npos = 0
    for image_id, gt in gt_by_image.items():
        mask = gt["labels"] == class_id
        boxes = gt["boxes"][mask]
        difficult = gt["difficult"][mask].astype(bool).copy()
        if size_bucket is not None and len(boxes) > 0:
            rel_areas = gt["relative_areas"][mask]
            if size_bucket == "small":
                in_bucket = rel_areas < 0.02
            elif size_bucket == "medium":
                in_bucket = (rel_areas >= 0.02) & (rel_areas < 0.15)
            else:
                in_bucket = rel_areas >= 0.15
            difficult = difficult | (~in_bucket)
        npos += int(np.sum(~difficult))
        class_gt[image_id] = {
            "boxes": boxes,
            "difficult": difficult,
            "detected": np.zeros(len(boxes), dtype=bool),
        }

    cls_dets = [d for d in detections if d["label"] == class_id]
    cls_dets.sort(key=lambda d: d["score"], reverse=True)
    nd = len(cls_dets)
    if npos == 0:
        return {"ap": float("nan"), "npos": 0, "nd": nd, "tp": 0, "fp": 0, "recall": 0.0}
    if nd == 0:
        return {"ap": 0.0, "npos": npos, "nd": 0, "tp": 0, "fp": 0, "recall": 0.0}

    tp = np.zeros(nd, dtype=np.float64)
    fp = np.zeros(nd, dtype=np.float64)

    for d_idx, det in enumerate(cls_dets):
        gt_entry = class_gt[det["image_id"]]
        gt_boxes = gt_entry["boxes"]
        if len(gt_boxes) == 0:
            fp[d_idx] = 1.0
            continue
        ious = compute_iou_matrix(det["box"][None, :], gt_boxes)[0]
        jmax = int(np.argmax(ious))
        ovmax = float(ious[jmax])
        if ovmax >= iou_threshold:
            if not gt_entry["difficult"][jmax]:
                if not gt_entry["detected"][jmax]:
                    tp[d_idx] = 1.0
                    gt_entry["detected"][jmax] = True
                else:
                    fp[d_idx] = 1.0
            # If difficult[jmax] is True, neither tp nor fp is incremented (ignored).
        else:
            fp[d_idx] = 1.0

    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    rec = tp_cum / float(npos)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    ap = voc2007_11point_ap(rec, prec)
    return {
        "ap": ap,
        "npos": int(npos),
        "nd": int(nd),
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "recall": float(rec[-1]) if len(rec) else 0.0,
    }


def evaluate_voc2007(gt_by_image, predictions_by_image, iou_threshold=0.5, score_threshold=0.05):
    """Compute VOC2007 mAP@0.5, per-class AP, size-stratified mAP, and error breakdown."""
    detections = []
    for image_id, pred in predictions_by_image.items():
        boxes = np.asarray(pred["boxes"], dtype=np.float64)
        labels = np.asarray(pred["labels"], dtype=np.int64)
        scores = np.asarray(pred["scores"], dtype=np.float64)
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold and 1 <= int(label) <= len(VOC_CLASSES):
                detections.append({
                    "image_id": image_id,
                    "box": box,
                    "label": int(label),
                    "score": float(score),
                })

    per_class = []
    for class_id, class_name in enumerate(VOC_CLASSES, start=1):
        res = evaluate_class_voc2007(gt_by_image, detections, class_id, iou_threshold=iou_threshold)
        small_res = evaluate_class_voc2007(gt_by_image, detections, class_id, iou_threshold=iou_threshold, size_bucket="small")
        med_res = evaluate_class_voc2007(gt_by_image, detections, class_id, iou_threshold=iou_threshold, size_bucket="medium")
        large_res = evaluate_class_voc2007(gt_by_image, detections, class_id, iou_threshold=iou_threshold, size_bucket="large")
        per_class.append({
            "class_id": class_id,
            "class_name": class_name,
            "ap50": res["ap"],
            "npos": res["npos"],
            "num_detections": res["nd"],
            "tp": res["tp"],
            "fp": res["fp"],
            "recall": res["recall"],
            "ap50_small": small_res["ap"],
            "ap50_medium": med_res["ap"],
            "ap50_large": large_res["ap"],
        })

    valid_aps = [r["ap50"] for r in per_class if not np.isnan(r["ap50"])]
    small_aps = [r["ap50_small"] for r in per_class if not np.isnan(r["ap50_small"])]
    med_aps = [r["ap50_medium"] for r in per_class if not np.isnan(r["ap50_medium"])]
    large_aps = [r["ap50_large"] for r in per_class if not np.isnan(r["ap50_large"])]

    error_taxonomy = analyze_detection_errors(gt_by_image, predictions_by_image, iou_threshold=iou_threshold, score_threshold=0.30)

    return {
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "map50_small": float(np.mean(small_aps)) if small_aps else 0.0,
        "map50_medium": float(np.mean(med_aps)) if med_aps else 0.0,
        "map50_large": float(np.mean(large_aps)) if large_aps else 0.0,
        "per_class": per_class,
        "error_taxonomy": error_taxonomy,
    }


def analyze_detection_errors(gt_by_image, predictions_by_image, iou_threshold=0.5, score_threshold=0.30):
    """Categorize predictions (at confidence >= score_threshold) and missed ground-truth objects:
    - correct_detections (TP): IoU >= 0.5 with an unmatched non-difficult GT of the same class
    - duplicate_detections: IoU >= 0.5 with an already-matched GT of the same class
    - wrong_class_detections: IoU >= 0.5 with a GT box of a different class
    - localization_errors: 0.1 <= max IoU < 0.5 with a GT box of the same class
    - background_false_positives: max IoU < 0.1 with any GT box
    - missed_objects (FN): non-difficult GT boxes not matched by any same-class prediction at IoU >= 0.5
    """
    counts = {
        "score_threshold_used": score_threshold,
        "total_non_difficult_gt": 0,
        "correct_detections": 0,
        "missed_objects": 0,
        "missed_small_objects": 0,
        "duplicate_detections": 0,
        "wrong_class_detections": 0,
        "localization_errors": 0,
        "background_false_positives": 0,
    }
    class_confusions = {}
    example_cases = {
        "correct": [],
        "difficult_crowded_or_small_correct": [],
        "missed_object": [],
        "wrong_class": [],
        "localization_or_duplicate": [],
    }

    for image_id in sorted(gt_by_image.keys()):
        gt = gt_by_image[image_id]
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        gt_diff = gt["difficult"].astype(bool)
        gt_rel_area = gt["relative_areas"]
        non_diff_mask = ~gt_diff
        counts["total_non_difficult_gt"] += int(non_diff_mask.sum())

        pred = predictions_by_image.get(image_id, {"boxes": np.zeros((0, 4)), "labels": np.zeros(0), "scores": np.zeros(0)})
        keep = np.asarray(pred["scores"]) >= score_threshold
        p_boxes = np.asarray(pred["boxes"])[keep]
        p_labels = np.asarray(pred["labels"])[keep]
        p_scores = np.asarray(pred["scores"])[keep]

        order = np.argsort(-p_scores)
        p_boxes, p_labels, p_scores = p_boxes[order], p_labels[order], p_scores[order]

        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        ious_all = compute_iou_matrix(p_boxes, gt_boxes)

        img_tp, img_fp_wrong, img_fp_loc, img_fp_dup = 0, 0, 0, 0
        for p_idx in range(len(p_boxes)):
            p_cls = int(p_labels[p_idx])
            if len(gt_boxes) == 0:
                counts["background_false_positives"] += 1
                continue
            ious = ious_all[p_idx]
            same_cls_indices = np.where(gt_labels == p_cls)[0]
            if len(same_cls_indices) > 0:
                best_same_idx = same_cls_indices[int(np.argmax(ious[same_cls_indices]))]
                best_same_iou = float(ious[best_same_idx])
            else:
                best_same_idx, best_same_iou = -1, 0.0

            best_any_idx = int(np.argmax(ious))
            best_any_iou = float(ious[best_any_idx])

            if best_same_iou >= iou_threshold:
                if gt_diff[best_same_idx]:
                    continue
                if not matched_gt[best_same_idx]:
                    matched_gt[best_same_idx] = True
                    counts["correct_detections"] += 1
                    img_tp += 1
                else:
                    counts["duplicate_detections"] += 1
                    img_fp_dup += 1
            elif best_any_iou >= iou_threshold and int(gt_labels[best_any_idx]) != p_cls:
                counts["wrong_class_detections"] += 1
                img_fp_wrong += 1
                true_name = ID_TO_CLASS[int(gt_labels[best_any_idx])]
                pred_name = ID_TO_CLASS[p_cls]
                pair = (true_name, pred_name)
                class_confusions[pair] = class_confusions.get(pair, 0) + 1
            elif best_same_iou >= 0.1:
                counts["localization_errors"] += 1
                img_fp_loc += 1
            else:
                counts["background_false_positives"] += 1

        unmatched_non_diff = non_diff_mask & (~matched_gt)
        missed_cnt = int(unmatched_non_diff.sum())
        counts["missed_objects"] += missed_cnt
        counts["missed_small_objects"] += int((unmatched_non_diff & (gt_rel_area < 0.02)).sum())

        # Record representative image IDs for qualitative visualization
        if img_tp > 0 and missed_cnt == 0 and img_fp_wrong == 0 and img_fp_loc == 0 and len(example_cases["correct"]) < 6:
            example_cases["correct"].append(image_id)
        if img_tp >= 3 and (non_diff_mask.sum() >= 4 or np.any(gt_rel_area[non_diff_mask] < 0.03)) and len(example_cases["difficult_crowded_or_small_correct"]) < 6:
            example_cases["difficult_crowded_or_small_correct"].append(image_id)
        if missed_cnt >= 2 and len(example_cases["missed_object"]) < 6:
            example_cases["missed_object"].append(image_id)
        if img_fp_wrong >= 1 and len(example_cases["wrong_class"]) < 6:
            example_cases["wrong_class"].append(image_id)
        if (img_fp_loc >= 1 or img_fp_dup >= 1) and len(example_cases["localization_or_duplicate"]) < 6:
            example_cases["localization_or_duplicate"].append(image_id)

    top_confusions = [
        {"true_class": k[0], "predicted_class": k[1], "count": v}
        for k, v in sorted(class_confusions.items(), key=lambda x: x[1], reverse=True)[:8]
    ]
    counts["top_class_confusions"] = top_confusions
    counts["example_image_ids"] = example_cases
    return counts


def self_test_voc_evaluator():
    """Unit test checking IoU, 11-point AP, difficult-box ignore, and duplicate-detection FP rules."""
    b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    b2 = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]])
    ious = compute_iou_matrix(b1, b2)
    assert np.isclose(ious[0, 0], 1.0)
    assert np.isclose(ious[0, 1], 50.0 / 150.0)

    gt = {
        "000001": {
            "boxes": np.array([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 140.0, 140.0]]),
            "labels": np.array([1, 1]),
            "difficult": np.array([0, 1]),
            "relative_areas": np.array([0.1, 0.1]),
        }
    }
    # 1st det matches non-difficult box (TP), 2nd matches difficult box (ignored), 3rd duplicates 1st box (FP)
    dets = [
        {"image_id": "000001", "box": np.array([10.0, 10.0, 50.0, 50.0]), "label": 1, "score": 0.95},
        {"image_id": "000001", "box": np.array([100.0, 100.0, 140.0, 140.0]), "label": 1, "score": 0.85},
        {"image_id": "000001", "box": np.array([11.0, 11.0, 50.0, 50.0]), "label": 1, "score": 0.75},
    ]
    res = evaluate_class_voc2007(gt, dets, class_id=1, iou_threshold=0.5)
    assert res["npos"] == 1 and res["tp"] == 1 and res["fp"] == 1 and np.isclose(res["ap"], 1.0)
    return True
