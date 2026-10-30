"""Run Assignment 2 pipeline from the package root:
  python run_a2.py --stage all --device cuda --out outputs/a2_m2_a100
"""
import argparse
import json
from pathlib import Path

import torch

from a2.common import load_json, save_json, set_seed
from a2.data import prepare_voc_data
from a2.report import create_report
from a2.train import evaluate_checkpoint, run_pilot, train_detector
from a2.voc_eval import self_test_voc_evaluator


def main():
    p = argparse.ArgumentParser(description="CO3133 A2: VOC2007 Object Detection Pipeline (M2 Draft & M3 Controlled Study)")
    p.add_argument("--stage", choices=["eda", "pilot", "train", "evaluate", "report", "all"], default="all")
    p.add_argument("--config", default="configs/a2.json")
    p.add_argument("--data", default="data/a2_voc2007")
    p.add_argument("--splits", default="splits")
    p.add_argument("--archive", default=None, help="Optional path to pre-downloaded VOCtrainval_06-Nov-2007.tar")
    p.add_argument("--out", default="outputs/a2_m2_a100")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["ssdlite320", "fasterrcnn_frozen", "fasterrcnn_finetune"],
        default=["ssdlite320", "fasterrcnn_frozen", "fasterrcnn_finetune"],
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epochs for all selected models")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    args = p.parse_args()

    self_test_voc_evaluator()

    if args.stage == "report":
        create_report(args.out, models=args.models)
        return

    config_path = Path(args.out) / "config.json" if args.stage == "evaluate" and (Path(args.out) / "config.json").exists() else Path(args.config)
    config = load_json(config_path)

    if args.epochs is not None:
        for m in args.models:
            config["models"][m]["epochs"] = args.epochs

    config["device"] = args.device or config.get("device", "auto")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment. Select a GPU runtime or pass --device cpu.")

    output = Path(args.out)
    for name in args.models:
        if args.stage in ["train", "all"] and (output / name / "metrics.json").exists():
            raise FileExistsError(
                f"Existing completed run in {output / name}. Choose a fresh --out directory to preserve recorded evidence."
            )

    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)

    data = prepare_voc_data(args.data, output, config, split_dir=args.splits, manual_archive_path=args.archive)
    device = torch.device(config["device"])

    if args.stage in ["pilot", "all"]:
        for name in args.models:
            run_pilot(name, data, config, output, device)

    if args.stage in ["train", "all"]:
        for name in args.models:
            train_detector(name, data, config, output, device)

    if args.stage == "evaluate":
        for name in args.models:
            evaluate_checkpoint(name, data, output, device)

    if args.stage in ["train", "all"]:
        create_report(output, models=args.models)


if __name__ == "__main__":
    main()
