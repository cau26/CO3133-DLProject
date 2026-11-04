"""A1 experiment CLI. Smoke is safe by default; full training is a deliberate action."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import torch

from a1.common import (environment, json_hash, load_json, protect_json, set_seed,
                       source_metadata)
from a1.data import prepare_data
from a1.eda import run_eda
from a1.finalize import freeze_run, test_frozen_run, validate_freeze
from a1.report import create_report
from a1.train import MODEL_NAMES, evaluate_checkpoint, model_config, smoke_test, train_model


def validate_config(config):
    if config.get("evaluation_policy") not in (None, "validation_only_post_test_study"):
        raise ValueError("Unknown evaluation_policy")
    for name in MODEL_NAMES:
        cfg = model_config(name, config)
        for key in ("epochs", "batch_size", "cpu_threads"):
            if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
                raise ValueError(f"{name}: {key} must be a positive integer")
        if cfg["learning_rate"] <= 0 or cfg.get("weight_decay", 0) < 0 or cfg["num_workers"] < 0:
            raise ValueError(f"{name}: invalid optimizer/worker settings")
        if cfg.get("inference_warmup_batches", 10) < 1 or cfg.get("inference_measured_batches", 50) < 1:
            raise ValueError("Inference timing requires positive warmup and repeat counts")
    if not 10 <= config["validation_size"] <= 59990:
        raise ValueError("validation_size must leave at least 10 samples in each partition")


def resolve_device(requested):
    requested = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    requested = "cpu" if requested == "auto" else requested
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select a GPU runtime or explicitly use --device cpu for smoke only.")
    return torch.device(requested)


def prepare_run(output, config):
    config_path, manifest_path = output / "config.json", output / "run_manifest.json"
    if config_path.exists() != manifest_path.exists():
        raise FileExistsError("Incomplete or historical run metadata. Preserve this output and choose a new --out.")
    if config_path.exists():
        if load_json(config_path) != config:
            raise ValueError("Run configuration is immutable. Restore the original config or choose a new --out.")
        manifest = load_json(manifest_path)
        if manifest["config_sha256"] != json_hash(config) or manifest["source_sha256"] != source_metadata()["source_sha256"]:
            raise ValueError("Run source/config fingerprint changed. Resume requires the original source/config.")
        return manifest
    protected = [output / name for name in MODEL_NAMES] + [output / "split_indices.npz", output / "data_summary.json", output / "smoke_checks.json"]
    if any(path.exists() for path in protected):
        raise FileExistsError("Existing unbound artifacts cannot be adopted into a new run. Choose a new --out.")
    manifest = {"schema_version": 1, "run_id": uuid.uuid4().hex,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": json_hash(config), "source_sha256": source_metadata()["source_sha256"],
                "source": source_metadata(), "environment": environment(torch.device(config["device"]))}
    protect_json(config_path, config)
    protect_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Five A1 models, immutable splits, resumable training, explicit frozen test")
    parser.add_argument("--stage", choices=["smoke", "eda", "train", "evaluate", "freeze", "test", "report", "all"], default="smoke")
    parser.add_argument("--config", default="configs/a1_final.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, help="Fresh-run override only; never changes an existing run")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--resume", action="store_true", help="Resume selected incomplete model(s) from last completed epoch")
    parser.add_argument("--confirm-freeze", action="store_true")
    parser.add_argument("--freeze-manifest", help="Required explicit path for --stage test")
    args = parser.parse_args(argv)
    output = Path(args.out or ("outputs/a1_smoke" if args.stage == "smoke" else "outputs/a1_m2"))
    if len(args.models) != len(set(args.models)):
        raise ValueError("--models cannot contain duplicate names")
    if args.resume and args.stage != "train":
        raise ValueError("--resume is only valid with --stage train")
    if args.stage in ("report", "freeze"):
        if (output / "config.json").is_file() and load_json(output / "config.json").get("evaluation_policy"):
            raise ValueError("This exploratory study is validation-only. Use tools/summarize_a1_study.py; freezing/test is disabled.")
        if args.epochs is not None:
            raise ValueError("Cannot override epochs for reporting/freezing")
        if args.stage == "report":
            create_report(output)
        else:
            manifest = freeze_run(output, args.confirm_freeze)
            print(f"Frozen all five models: {output / 'freeze_manifest.json'}; run {manifest['run_id']}")
        return
    readonly = args.stage in ("evaluate", "test")
    if readonly and not all((output / file).is_file() for file in
                            ("config.json", "run_manifest.json", "split_indices.npz", "data_summary.json")):
        raise FileNotFoundError("Evaluation requires existing complete run configuration and saved split artifacts")
    if readonly and args.epochs is not None:
        raise ValueError("Evaluation always uses saved configuration; --epochs is forbidden")
    config = load_json(output / "config.json" if readonly else args.config)
    study_only = config.get("evaluation_policy") == "validation_only_post_test_study"
    if study_only and args.stage not in ("smoke", "train", "evaluate"):
        raise ValueError("Post-test study permits smoke/train/validation only; official test and all/EDA stages are disabled.")
    if args.epochs is not None:
        config["epochs"] = args.epochs
    device = resolve_device(args.device or config["device"])
    if not readonly:
        config["device"] = str(device)
    validate_config(config)
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    if args.stage == "smoke":
        smoke_test(config, output, device)
        return
    if args.stage == "test":
        if set(args.models) != set(MODEL_NAMES) or not args.freeze_manifest:
            raise ValueError("Final test requires all five models and --freeze-manifest PATH")
        validate_freeze(output, args.freeze_manifest)  # fail before opening the official test partition
        if (output / "final_test").exists():
            raise FileExistsError("Official test output exists, including an interrupted attempt; it cannot be overwritten")
    elif not readonly:
        if args.stage in ("train", "all"):
            if (output / "freeze_manifest.json").exists():
                raise FileExistsError("Frozen run cannot be trained further")
            for name in args.models:
                model_out = output / name
                if (model_out / "metrics.json").exists():
                    raise FileExistsError(f"Completed run cannot be overwritten/resumed: {model_out}")
                if model_out.exists() and not args.resume:
                    raise FileExistsError(f"Existing incomplete model directory: {model_out}; use --resume")
                if args.resume and not (model_out / "last.pt").is_file():
                    raise FileNotFoundError(f"No epoch checkpoint to resume: {model_out / 'last.pt'}")
        prepare_run(output, config)
    else:
        # Read-only evaluation also rejects changed source/config and cannot rewrite run metadata.
        prepare_run(output, config)
    data = prepare_data(args.data, output, config, include_test=args.stage in ("eda", "all", "test"))
    if study_only:
        if data.get("test") is not None:
            raise ValueError("Official test must never be opened by the supplementary study")
        for field in ("split_sha256", "dataset_sha256"):
            if data["summary"][field] != config.get("expected_" + field):
                raise ValueError(f"Study {field} differs from the fixed reference dataset/split")
    if args.stage in ("eda", "all"):
        if (output / "eda").exists():
            raise FileExistsError("EDA output already exists; preserve it and proceed to train")
        run_eda(data, config, output)
    if args.stage in ("train", "all"):
        for name in args.models:
            train_model(name, data, config, output, device, resume=args.resume)
    if args.stage == "evaluate":
        for name in args.models:
            evaluate_checkpoint(name, data, output, device)
    if args.stage == "test":
        test_frozen_run(data, output, device, args.freeze_manifest)
    if args.stage == "all":
        create_report(output)


if __name__ == "__main__":
    main()
