"""Run from the repository root: python run_a1.py --stage all"""
import argparse
import json
from pathlib import Path

import torch

from a1.common import load_json, save_json, set_seed
from a1.data import prepare_data
from a1.eda import run_eda
from a1.report import create_report
from a1.train import evaluate_checkpoint, train_model


def main():
    p = argparse.ArgumentParser(description="CO3133 A1: EDA, Linear, MLP and draft report")
    p.add_argument("--stage", choices=["eda", "train", "evaluate", "report", "all"], default="all")
    p.add_argument("--config", default="configs/a1.json")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="outputs/a1")
    p.add_argument("--models", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"])
    args = p.parse_args()
    if args.stage == "report":
        create_report(args.out)
        return
    config = load_json(Path(args.out) / "config.json" if args.stage == "evaluate" else args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    config["device"] = args.device or config["device"]
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment. Use --device cpu.")
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 10 <= config["validation_size"] <= 59990:
        raise ValueError("validation_size must leave at least 10 samples in both partitions")
    if args.stage == "all" and set(args.models) != {"linear", "mlp"}:
        raise ValueError("--stage all needs both models; use --stage train for one model")
    output = Path(args.out)
    for name in args.models:
        if args.stage in ["train", "all"] and (output / name / "metrics.json").exists():
            raise FileExistsError(f"Existing completed run in {output / name}. "
                                  "Keep it and choose a new --out, for example outputs/a1-run2.")
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    save_json(output / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    data = prepare_data(args.data, output, config)
    if args.stage in ["eda", "all"]:
        run_eda(data, config, output)
    if args.stage in ["train", "all"]:
        for name in args.models:
            train_model(name, data, config, output, torch.device(config["device"]))
    if args.stage == "evaluate":
        for name in args.models:
            evaluate_checkpoint(name, data, output, torch.device(config["device"]))
    if args.stage == "all":
        create_report(output)


if __name__ == "__main__":
    main()  # Required for multiprocessing-safe execution on Windows.
