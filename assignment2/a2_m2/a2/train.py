"""Training, A100 pilot profiling, VOC2007 evaluation, latency measurement, and qualitative plots."""
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from .common import ID_TO_CLASS, VOC_CLASSES, environment, load_json, save_csv, save_json, set_seed, source_metadata
from .data import make_detection_loaders
from .models import build_detector, count_parameters
from .voc_eval import evaluate_voc2007, self_test_voc_evaluator


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, optimizer, device, scaler=None, use_amp=False, max_batches=None):
    model.train()
    total_loss = 0.0
    component_sums = {}
    num_steps = 0
    num_images = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        dev_targets = [
            {
                "boxes": t["boxes"].to(device, non_blocking=True),
                "labels": t["labels"].to(device, non_blocking=True),
            }
            for t in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            loss_dict = model(images, dev_targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_val = float(losses.detach().item())
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Non-finite training loss ({loss_val}) at step {step}: {loss_dict}")

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=10.0)
            optimizer.step()

        bsz = len(images)
        total_loss += loss_val * bsz
        num_images += bsz
        num_steps += 1
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + float(v.detach().item()) * bsz

        if max_batches is not None and step >= max_batches:
            break

    avg_components = {k: v / max(num_images, 1) for k, v in sorted(component_sums.items())}
    return {
        "loss": total_loss / max(num_images, 1),
        "steps": num_steps,
        "images": num_images,
        "loss_components": avg_components,
    }


@torch.inference_mode()
def predict_validation(model, val_loader, device):
    model.eval()
    predictions = {}
    for images, targets in val_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for tgt, out in zip(targets, outputs):
            image_id = tgt["voc_image_id"]
            predictions[image_id] = {
                "boxes": out["boxes"].detach().cpu().numpy(),
                "labels": out["labels"].detach().cpu().numpy(),
                "scores": out["scores"].detach().cpu().numpy(),
            }
    return predictions


@torch.inference_mode()
def measure_single_image_latency(model, val_dataset, device, warmup=10, timed_images=100):
    """Measure single-image inference latency (batch_size=1) including model resize and NMS postprocessing."""
    model.eval()
    n_avail = min(len(val_dataset), warmup + timed_images)
    samples = [val_dataset[i][0].to(device) for i in range(n_avail)]
    for i in range(min(warmup, len(samples))):
        model([samples[i]])
    synchronize(device)
    measured = samples[min(warmup, len(samples)):]
    if not measured:
        measured = samples[:min(20, len(samples))]
    start = time.perf_counter()
    for img in measured:
        model([img])
    synchronize(device)
    elapsed = time.perf_counter() - start
    ms_per_image = (elapsed * 1000.0) / len(measured)
    return {
        "batch_size": 1,
        "warmup_images": min(warmup, len(samples)),
        "measured_images": len(measured),
        "mean_latency_ms": float(ms_per_image),
        "fps": float(1000.0 / ms_per_image) if ms_per_image > 0 else 0.0,
        "scope": "batch_size=1 model forward including internal resize/normalization and NMS postprocessing",
    }


def run_pilot(name, data, config, output_dir, device):
    """Run the 200-batch training + 1 full validation pilot described in Proposal Section 3.1."""
    self_test_voc_evaluator()
    set_seed(config["seed"])
    model_cfg = config["models"][name]
    train_loader, val_loader = make_detection_loaders(data, model_cfg, config)
    model = build_detector(name, config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=model_cfg["learning_rate"],
        momentum=model_cfg["momentum"],
        weight_decay=model_cfg["weight_decay"],
    )
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    pilot_batches = min(config.get("pilot_batches", 200), len(train_loader))
    t0 = time.perf_counter()
    train_stats = train_one_epoch(
        model, train_loader, optimizer, device, scaler=scaler, use_amp=use_amp, max_batches=pilot_batches
    )
    synchronize(device)
    train_elapsed = time.perf_counter() - t0
    t_step = train_elapsed / max(train_stats["steps"], 1)

    t1 = time.perf_counter()
    preds = predict_validation(model, val_loader, device)
    synchronize(device)
    val_eval = evaluate_voc2007(
        data["gt_val"], preds, iou_threshold=config["iou_threshold"], score_threshold=config["score_threshold"]
    )
    t_val = time.perf_counter() - t1

    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    )
    n_train = len(data["train_dataset"])
    steps_per_epoch = math.ceil(n_train / model_cfg["batch_size"])
    planned_epochs = model_cfg["epochs"]
    estimated_run_seconds = planned_epochs * (steps_per_epoch * t_step + t_val)

    pilot_record = {
        "model": name,
        "pilot_batches": train_stats["steps"],
        "batch_size": model_cfg["batch_size"],
        "steps_per_full_epoch": steps_per_epoch,
        "measured_t_step_seconds": float(t_step),
        "measured_t_val_seconds": float(t_val),
        "pilot_train_loss": float(train_stats["loss"]),
        "pilot_val_map50": float(val_eval["map50"]),
        "peak_gpu_memory_mb": peak_mem_mb,
        "amp_enabled_and_stable": use_amp,
        "planned_epochs": planned_epochs,
        "estimated_full_run_seconds": float(estimated_run_seconds),
        "estimated_full_run_minutes": float(estimated_run_seconds / 60.0),
        "formula": "T_run = E * (ceil(N_train / B) * t_step + t_val)",
        "environment": environment(device),
    }
    out = Path(output_dir) / "pilots"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"{name}_pilot.json", pilot_record)
    print(
        f"[PILOT {name}] steps={train_stats['steps']} loss={train_stats['loss']:.4f} "
        f"val_mAP@0.5={val_eval['map50']:.4f} t_step={t_step * 1000:.1f}ms t_val={t_val:.2f}s "
        f"peak_VRAM={peak_mem_mb:.1f}MB est_{planned_epochs}ep={estimated_run_seconds / 60.0:.2f}min",
        flush=True,
    )
    return pilot_record


def plot_training_curves(name, history, out_dir):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", color="#2563eb", label="Train total loss")
    axes[0].set(xlabel="Epoch", ylabel="Training Loss", title=f"{name}: Training Loss Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [r["val_map50"] * 100.0 for r in history], marker="s", color="#16a34a", label="Validation mAP@0.5 (%)")
    best_idx = int(np.argmax([r["val_map50"] for r in history]))
    axes[1].axvline(epochs[best_idx], color="#dc2626", linestyle="--", alpha=0.7, label=f"Best epoch ({epochs[best_idx]})")
    axes[1].set(xlabel="Epoch", ylabel="Validation mAP@0.5 (%)", title=f"{name}: Validation VOC2007 mAP@0.5")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(Path(out_dir) / "curves.png", dpi=150)
    plt.close(fig)


def plot_per_class_ap(name, per_class, map50, out_dir):
    sorted_cls = sorted(per_class, key=lambda r: r["ap50"])
    names = [r["class_name"] for r in sorted_cls]
    aps = [r["ap50"] * 100.0 for r in sorted_cls]
    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    bars = ax.barh(names, aps, color="#3b82f6", alpha=0.88)
    ax.axvline(map50 * 100.0, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean AP@0.5 = {map50 * 100.0:.2f}%")
    for bar, val in zip(bars, aps):
        ax.text(min(val + 1.0, 96.0), bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set(xlabel="11-point Interpolated AP@0.5 (%)", title=f"{name}: Per-Class Validation AP@0.5")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(Path(out_dir) / "per_class_ap.png", dpi=150)
    plt.close(fig)


def plot_qualitative_examples(name, voc_root, gt_val, preds, error_taxonomy, out_dir, score_vis=0.35):
    cases = error_taxonomy.get("example_image_ids", {})
    selected = []
    labels_tag = []
    for bucket, tag in [
        ("correct", "Correct detection"),
        ("difficult_crowded_or_small_correct", "Crowded/small scene"),
        ("missed_object", "Missed object (FN)"),
        ("wrong_class", "Wrong class (FP)"),
        ("localization_or_duplicate", "Localization / duplicate"),
    ]:
        for img_id in cases.get(bucket, []):
            if img_id not in selected and len(selected) < 6:
                selected.append(img_id)
                labels_tag.append((img_id, tag))
                break
    for img_id in sorted(gt_val.keys()):
        if len(selected) >= 6:
            break
        if img_id not in selected:
            selected.append(img_id)
            labels_tag.append((img_id, "Validation example"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    for ax, (image_id, tag) in zip(axes.flat, labels_tag):
        img_path = Path(voc_root) / "JPEGImages" / f"{image_id}.jpg"
        with Image.open(img_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        ax.imshow(rgb)
        gt = gt_val[image_id]
        for box, cls_id, diff in zip(gt["boxes"], gt["labels"], gt["difficult"]):
            xmin, ymin, xmax, ymax = box
            style = "--" if diff else "-"
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor="#16a34a", linewidth=1.8, linestyle=style))
            ax.text(xmin, max(ymin - 3, 10), f"GT:{ID_TO_CLASS[int(cls_id)]}",
                    color="white", fontsize=7,
                    bbox={"facecolor": "#16a34a", "alpha": 0.75, "pad": 1, "edgecolor": "none"})
        pred = preds.get(image_id, {"boxes": [], "labels": [], "scores": []})
        for box, cls_id, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < score_vis:
                continue
            xmin, ymin, xmax, ymax = box
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor="#dc2626", linewidth=1.5))
            ax.text(xmin, min(ymax + 10, rgb.shape[0] - 4),
                    f"P:{ID_TO_CLASS[int(cls_id)]} {score:.2f}",
                    color="white", fontsize=7,
                    bbox={"facecolor": "#dc2626", "alpha": 0.75, "pad": 1, "edgecolor": "none"})
        ax.set_title(f"ID {image_id} — {tag}", fontsize=9.5)
        ax.axis("off")
    fig.suptitle(f"{name}: Ground Truth (green) vs. Predictions score>={score_vis} (red)", fontsize=11)
    fig.savefig(Path(out_dir) / "examples.png", dpi=150)
    plt.close(fig)


def train_detector(name, data, config, output_dir, device):
    self_test_voc_evaluator()
    set_seed(config["seed"])
    model_cfg = config["models"][name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_detection_loaders(data, model_cfg, config)
    model = build_detector(name, config).to(device)
    param_counts = count_parameters(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=model_cfg["learning_rate"],
        momentum=model_cfg["momentum"],
        weight_decay=model_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=model_cfg["lr_milestones"], gamma=model_cfg["lr_gamma"]
    )
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    provenance = source_metadata()
    hardware = environment(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_map50 = -1.0
    best_epoch = 0
    synchronize(device)
    overall_start = time.perf_counter()

    for epoch in range(1, model_cfg["epochs"] + 1):
        data["train_dataset"].set_epoch(epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        synchronize(device)
        t_start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, use_amp=use_amp)
        synchronize(device)
        train_seconds = time.perf_counter() - t_start

        scheduler.step()

        v_start = time.perf_counter()
        val_preds = predict_validation(model, val_loader, device)
        synchronize(device)
        val_eval = evaluate_voc2007(
            data["gt_val"], val_preds, iou_threshold=config["iou_threshold"], score_threshold=config["score_threshold"]
        )
        val_seconds = time.perf_counter() - v_start
        val_map50 = float(val_eval["map50"])

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": float(train_stats["loss"]),
            "val_map50": val_map50,
            "val_map50_small": float(val_eval["map50_small"]),
            "val_map50_medium": float(val_eval["map50_medium"]),
            "val_map50_large": float(val_eval["map50_large"]),
            "train_seconds": float(train_seconds),
            "val_seconds": float(val_seconds),
        }
        for comp_k, comp_v in train_stats["loss_components"].items():
            row[f"train_{comp_k}"] = float(comp_v)
        history.append(row)
        save_csv(out / "history.csv", history)

        # Save resumable last.pt checkpoint (Proposal Sec. 3.1)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "model_name": name,
            "config": config,
            "epoch": epoch,
            "val_map50": val_map50,
            "split_sha256": data["summary"]["split_sha256"],
            "source_sha256": provenance["source_sha256"],
        }, out / "last.pt")

        # Strict > keeps earlier checkpoint on ties (Proposal Sec. 2.3)
        if val_map50 > best_map50:
            best_map50 = val_map50
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "model_name": name,
                "config": config,
                "epoch": epoch,
                "val_map50": best_map50,
                "split_sha256": data["summary"]["split_sha256"],
                "source_sha256": provenance["source_sha256"],
            }, out / "best.pt")

        print(
            f"{name:20s} epoch {epoch:02d}/{model_cfg['epochs']:02d} (lr={current_lr:.5f}): "
            f"train_loss={train_stats['loss']:.4f} val_mAP@0.5={val_map50 * 100.0:.2f}% "
            f"(S={val_eval['map50_small'] * 100.0:.1f}% M={val_eval['map50_medium'] * 100.0:.1f}% "
            f"L={val_eval['map50_large'] * 100.0:.1f}%) time={train_seconds:.1f}s+{val_seconds:.1f}s",
            flush=True,
        )

    fit_wall_seconds = time.perf_counter() - overall_start
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    )

    # Reload best.pt to compute final metrics, per-class AP, error taxonomy, and qualitative plots
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_preds = predict_validation(model, val_loader, device)
    best_eval = evaluate_voc2007(
        data["gt_val"], best_preds, iou_threshold=config["iou_threshold"], score_threshold=config["score_threshold"]
    )
    assert abs(best_eval["map50"] - best_map50) < 1e-4, "Reloaded best.pt validation mAP mismatch!"

    save_csv(out / "per_class_ap.csv", best_eval["per_class"])
    save_json(out / "error_taxonomy.json", best_eval["error_taxonomy"])
    plot_training_curves(name, history, out)
    plot_per_class_ap(name, best_eval["per_class"], best_eval["map50"], out)
    plot_qualitative_examples(
        name, data["voc_root"], data["gt_val"], best_preds, best_eval["error_taxonomy"], out
    )
    latency = measure_single_image_latency(model, data["val_dataset"], device)

    # Remove last.pt after successful completion to save archive space while keeping best.pt
    if (out / "last.pt").exists():
        (out / "last.pt").unlink()

    metrics = {
        "model": name,
        "architecture": model_cfg["architecture"],
        "init_weights": model_cfg["init_weights"],
        "freeze_backbone": bool(model_cfg.get("freeze_backbone", False)),
        "evaluation_split": "validation",
        "n_eval_images": len(data["val_ids"]),
        "n_eval_non_difficult_objects": data["summary"]["val_non_difficult_objects"],
        "map50": float(best_eval["map50"]),
        "map50_small": float(best_eval["map50_small"]),
        "map50_medium": float(best_eval["map50_medium"]),
        "map50_large": float(best_eval["map50_large"]),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(history[best_epoch - 1]["train_loss"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "checkpoint_rule": "highest validation VOC2007 11-point mAP@0.5; earlier epoch if tied",
        "checkpoint": "best.pt",
        "epochs_completed": len(history),
        **param_counts,
        "train_seconds": float(sum(r["train_seconds"] for r in history)),
        "validation_seconds": float(sum(r["val_seconds"] for r in history)),
        "fit_wall_seconds": float(fit_wall_seconds),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "amp_used": use_amp,
        "inference": latency,
        "per_class_ap": best_eval["per_class"],
        "error_taxonomy": best_eval["error_taxonomy"],
        "model_config": model_cfg,
        "config": config,
        "data": data["summary"],
        "environment": hardware,
        "source": provenance,
        "test_evaluated": False,
        "human_review_status": "Pending: group must inspect code, rerun and verify before submission",
    }
    save_json(out / "metrics.json", metrics)
    return metrics


def evaluate_checkpoint(name, data, output_dir, device):
    """Reload a saved best.pt checkpoint and verify its validation mAP@0.5 without training."""
    out = Path(output_dir) / name
    saved = load_json(out / "metrics.json")
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    if checkpoint["split_sha256"] != data["summary"]["split_sha256"]:
        raise ValueError("Checkpoint split_sha256 does not match current split!")
    cfg = checkpoint["config"]
    model = build_detector(name, cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, val_loader = make_detection_loaders(data, cfg["models"][name], cfg)
    preds = predict_validation(model, val_loader, device)
    eval_res = evaluate_voc2007(
        data["gt_val"], preds, iou_threshold=cfg["iou_threshold"], score_threshold=cfg["score_threshold"]
    )
    matches = bool(np.isclose(eval_res["map50"], saved["map50"], atol=1e-4, rtol=1e-4))
    verification = {
        "model": name,
        "recomputed_map50": float(eval_res["map50"]),
        "saved_map50": float(saved["map50"]),
        "matches_saved": matches,
        "device": str(device),
        "split_sha256": data["summary"]["split_sha256"],
    }
    save_json(out / "checkpoint_verification.json", verification)
    print(f"{name}: recomputed mAP@0.5={eval_res['map50']:.4f}; matches saved: {matches}", flush=True)
    if not matches:
        raise ValueError(f"Recomputed mAP@0.5 ({eval_res['map50']}) differs from saved ({saved['map50']}).")
    return verification
