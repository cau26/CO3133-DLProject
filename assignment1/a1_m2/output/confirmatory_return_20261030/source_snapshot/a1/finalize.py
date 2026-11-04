"""Explicit five-model freeze followed by a single protected official-test run."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import (environment, json_hash, load_json, protect_json, save_csv,
                     save_json, sha256_file, source_metadata)
from .data import split_digest, validate_split
from .train import (MODEL_NAMES, assert_checkpoint, build_model, measure_inference,
                    model_config, plot_results, run_epoch)


def locked_paths():
    return ["config.json", "run_manifest.json", "split_indices.npz", "data_summary.json",
            *[f"{name}/{file}" for name in MODEL_NAMES
              for file in ("best.pt", "metrics.json", "model_config.json", "history.csv")]]


def freeze_run(output_dir, confirmed=False):
    if not confirmed:
        raise ValueError("Freeze is deliberate: review validation for all five models, then pass --confirm-freeze")
    out = Path(output_dir)
    if (out / "freeze_manifest.json").exists() or (out / "final_test").exists():
        raise FileExistsError("Freeze/test artifacts already exist; preserve them")
    config, run = load_json(out / "config.json"), load_json(out / "run_manifest.json")
    summary, provenance = load_json(out / "data_summary.json"), source_metadata()
    if summary["official_train"] != 60000 or summary["validation"] != config["validation_size"]:
        raise ValueError("Dataset/split sizes do not match the declared Fashion-MNIST protocol")
    with np.load(out / "split_indices.npz", allow_pickle=False) as split:
        train_ids, val_ids = split["train"], split["validation"]
        validate_split(train_ids, val_ids, 60000, config["validation_size"])
        if split_digest(train_ids, val_ids) != summary["split_sha256"]:
            raise ValueError("Saved split hash differs from the data summary")
    if run["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("Source changed since run creation; cannot freeze mixed code")
    models = {}
    for name in MODEL_NAMES:
        metrics = load_json(out / name / "metrics.json")
        cfg = model_config(name, config)
        if load_json(out / name / "model_config.json") != cfg:
            raise ValueError(f"{name}: model_config.json differs from the saved configuration")
        if (metrics["model"] != name or metrics["evaluation_split"] != "validation"
                or metrics["epochs_completed"] != cfg["epochs"] or metrics["run_id"] != run["run_id"]
                or metrics["n_eval"] != summary["validation"]
                or metrics["config"] != cfg or metrics["data"] != summary
                or metrics["source"]["source_sha256"] != provenance["source_sha256"]):
            raise ValueError(f"{name}: incomplete or inconsistent validation run")
        if sha256_file(out / name / "best.pt") != metrics["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint changed since validation completed")
        checkpoint = torch.load(out / name / "best.pt", map_location="cpu", weights_only=True)
        assert_checkpoint(checkpoint, name, cfg, {"summary": summary}, provenance, run["run_id"])
        models[name] = {"best_epoch": metrics["best_epoch"], "config_sha256": json_hash(cfg),
                        "checkpoint_sha256": metrics["checkpoint_sha256"],
                        "selection": "minimum validation loss; earliest epoch wins ties"}
    manifest = {"schema_version": 1, "run_id": run["run_id"],
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "confirmation": "user explicitly invoked freeze with --confirm-freeze",
                "models": models, "split_sha256": summary["split_sha256"],
                "dataset_sha256": summary["dataset_sha256"],
                "source_sha256": provenance["source_sha256"],
                "files_sha256": {path: sha256_file(out / path) for path in locked_paths()},
                "test_policy": "all five frozen checkpoints; no test-driven checkpoint or hyperparameter changes"}
    manifest["manifest_sha256"] = json_hash(manifest)
    protect_json(out / "freeze_manifest.json", manifest)
    return manifest


def validate_freeze(output_dir, manifest_path):
    out = Path(output_dir)
    if Path(manifest_path).resolve() != (out / "freeze_manifest.json").resolve():
        raise ValueError("Use this run's own freeze_manifest.json")
    manifest = load_json(manifest_path)
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != json_hash(content) or manifest.get("schema_version") != 1:
        raise ValueError("Invalid or edited freeze manifest")
    if set(manifest.get("models", {})) != set(MODEL_NAMES):
        raise ValueError("Final testing requires all five models frozen together")
    if set(manifest.get("files_sha256", {})) != set(locked_paths()):
        raise ValueError("Freeze manifest has an incomplete file lock set")
    for path in locked_paths():
        if sha256_file(out / path) != manifest["files_sha256"][path]:
            raise ValueError(f"Frozen file was changed: {path}")
    if source_metadata()["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Current source differs from the frozen training implementation")
    if load_json(out / "run_manifest.json")["run_id"] != manifest["run_id"]:
        raise ValueError("Freeze/run identity mismatch")
    return manifest


def test_frozen_run(data, output_dir, device, manifest_path):
    out = Path(output_dir)
    manifest = validate_freeze(out, manifest_path)
    if data["summary"]["split_sha256"] != manifest["split_sha256"] or data["summary"]["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Prepared dataset differs from the frozen training dataset/split")
    if data.get("test") is None:
        raise ValueError("Official test partition has not been loaded")
    test_dataset_hash = hashlib.sha256(data["test"].data.numpy().tobytes()
                        + data["test"].targets.numpy().astype("<i8").tobytes()).hexdigest()
    destination = out / "final_test"
    destination.mkdir(exist_ok=False)
    save_json(destination / "TEST_STARTED.json", {"run_id": manifest["run_id"],
              "freeze_manifest_sha256": sha256_file(manifest_path),
              "official_test_dataset_sha256": test_dataset_hash,
              "purpose": "official test; directory is reserved and cannot be overwritten"})
    results = []
    for name in MODEL_NAMES:
        model_out = destination / name
        model_out.mkdir(exist_ok=False)
        checkpoint = torch.load(out / name / "best.pt", map_location=device, weights_only=True)
        cfg = checkpoint["config"]
        model = build_model(name, cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        loader = DataLoader(data["test"], batch_size=cfg["batch_size"],
                            num_workers=cfg["num_workers"], shuffle=False)
        stats, true, pred = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
        save_csv(model_out / "test_predictions.csv", [
            {"official_test_index": i, "true": int(t), "predicted": int(p)}
            for i, (t, p) in enumerate(zip(true, pred))])
        errors = plot_results(name, [], true, pred, data, model_out, split="test")
        latency = measure_inference(model, next(iter(loader))[0], device,
                                    cfg.get("inference_warmup_batches", 10), cfg.get("inference_measured_batches", 50))
        metrics = {"model": name, "evaluation_split": "official_test", "n_eval": len(true),
                   **stats, "run_id": manifest["run_id"], "config": cfg,
                   "checkpoint_sha256": sha256_file(out / name / "best.pt"),
                   "freeze_manifest_sha256": sha256_file(manifest_path), "top_errors": errors,
                   "inference": latency, "source": source_metadata(), "environment": environment(device),
                   "split_sha256": manifest["split_sha256"], "test_evaluated": True}
        metrics["official_test_dataset_sha256"] = test_dataset_hash
        protect_json(model_out / "metrics.json", metrics)
        results.append({"model": name, **stats})
    save_csv(destination / "comparison.csv", results)
    protect_json(destination / "RESULTS_COMPLETE.json", {"run_id": manifest["run_id"],
                 "models": list(MODEL_NAMES), "freeze_manifest_sha256": sha256_file(manifest_path)})
    return results
