"""Run Assignment 2 M3 Final pipeline from assignment2/a2_m3/:
  python run_a2.py --stage all_m3 --device cuda --out outputs/a2_m3_a100
"""
import argparse
import json
from pathlib import Path

import torch

from a2.common import load_json, save_json, set_seed
from a2.data import prepare_voc_data, prepare_voc_test_data
from a2.report import create_report, create_test_report
from a2.train import evaluate_checkpoint, evaluate_test_checkpoint, run_pilot, train_detector
from a2.voc_eval import self_test_voc_evaluator


def main():
    p = argparse.ArgumentParser(description="CO3133 A2: VOC2007 Object Detection Pipeline (M3 Final)")
    p.add_argument(
        "--stage",
        choices=["eda", "pilot", "train", "evaluate", "test_evaluate", "report", "all", "all_m3"],
        default="all_m3",
    )
    p.add_argument("--config", default="configs/a2.json")
    p.add_argument("--data", default="data/a2_voc2007")
    p.add_argument("--splits", default="splits")
    p.add_argument("--archive", default=None, help="Optional path to pre-downloaded VOCtrainval_06-Nov-2007.tar")
    p.add_argument("--test-archive", default=None, help="Optional path to pre-downloaded VOCtest_06-Nov-2007.tar")
    p.add_argument("--out", default="outputs/a2_m3_a100")
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
        if all((Path(args.out) / m / "test_metrics.json").exists() for m in args.models):
            create_test_report(args.out, models=args.models)
        return

    config_path = (
        Path(args.out) / "config.json"
        if args.stage in ["evaluate", "test_evaluate"] and (Path(args.out) / "config.json").exists()
        else Path(args.config)
    )
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
        if args.stage in ["train", "all", "all_m3"] and (output / name / "metrics.json").exists():
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

    if args.stage in ["train", "all", "all_m3"]:
        for name in args.models:
            train_detector(name, data, config, output, device)

    if args.stage in ["evaluate", "all_m3"]:
        for name in args.models:
            evaluate_checkpoint(name, data, output, device)

    if args.stage in ["train", "all", "all_m3"]:
        create_report(output, models=args.models)

    if args.stage in ["test_evaluate", "all_m3"]:
        test_data = prepare_voc_test_data(
            args.data,
            output,
            config,
            trainval_data=data,
            manual_test_archive_path=args.test_archive,
        )
        for name in args.models:
            evaluate_test_checkpoint(name, test_data, output, device)
        create_test_report(output, models=args.models)


if __name__ == "__main__":
    main()
