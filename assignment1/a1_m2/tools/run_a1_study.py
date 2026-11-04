"""Student-run A100 supplementary study. Default action only prepares the fixed plan."""
import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a1.study import execute_study, prepare_study, progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "smoke", "train", "status"), default="plan")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/a1_upgrade_study.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-tasks", type=int, help="Deliberately stop after N new/resumed tasks; rerun to continue")
    args = parser.parse_args(argv)
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    if args.stage == "train":
        import torch
        state = execute_study(args.out, args.protocol, args.data, torch.device(args.device), args.max_tasks)
        print(json.dumps({k: v for k, v in state.items() if k != "tasks"}, indent=2))
        return 0
    manifest = prepare_study(args.out, args.protocol)
    if args.stage == "smoke":
        import torch
        from a1.common import set_seed
        from a1.data import prepare_data
        from a1.train import smoke_test
        device = torch.device(args.device)
        if device.type == "cuda" and (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)):
            raise RuntimeError("Select NVIDIA A100; CPU is permitted only for the small smoke check")
        config = manifest["runs"][0]["config"]
        torch.set_num_threads(config["cpu_threads"])
        set_seed(config["seed"])
        data = prepare_data(args.data, args.out / "smoke_data", config, include_test=False)
        if (data["summary"]["split_sha256"] != manifest["protocol"]["expected_split_sha256"]
                or data["summary"]["dataset_sha256"] != manifest["protocol"]["expected_dataset_sha256"]):
            raise ValueError("Smoke data does not match the original fixed dataset/split")
        examples = [data["full_train"][int(i)] for i in data["train_ids"][:8]]
        images = torch.stack([x for x, _ in examples])
        labels = torch.tensor([y for _, y in examples], dtype=torch.long)
        # Eight architecture/config variants, one step each, rather than 24 full runs.
        selected = [r for r in manifest["runs"] if r["seed"] == manifest["protocol"]["training_seeds"][0]]
        smoke_session = args.out / "smoke" / uuid.uuid4().hex
        for run in selected:
            cfg = dict(run["config"])
            cfg["device"] = str(device)
            checks = smoke_test(cfg, smoke_session / run["run_key"], device,
                                images=images, labels=labels, models=run["models"])
            print(f'SMOKE {run["run_key"]}: {len(checks["models"])} model(s) PASS', flush=True)
        print("SMOKE DONE: one-batch checks only; no full training and no official test")
    state = progress(args.out, manifest)
    print(json.dumps({k: v for k, v in state.items() if k != "tasks"}, indent=2))
    if args.stage in ("plan", "status"):
        for task in state["tasks"]:
            print(task["task_id"], task["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
