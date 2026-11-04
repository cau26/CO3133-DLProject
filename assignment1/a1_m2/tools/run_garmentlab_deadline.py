"""Prepare, train, evaluate and export the versioned student-run GarmentLab study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from garmentlab.runtime import RunSession, atomic_json, digest, export_results, utc_now, utc_parse


def source_fingerprint():
    files = sorted([*ROOT.joinpath("garmentlab").glob("*.py"), ROOT / "tools/run_garmentlab_deadline.py"])
    payload = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in files]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bind_run(out, config, data_config):
    value = {"schema": "garmentlab_deadline_run_v1", "config": config,
             "data_config_sha256": digest(data_config), "source_fingerprint": source_fingerprint()}
    path = out / "RUN_MANIFEST.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Source/config changed. Preserve this study and choose a new run directory.")
    atomic_json(path, value)
    snapshot = out / "source_snapshot"
    for item in sorted([*ROOT.joinpath("garmentlab").glob("*.py"),
                        ROOT / "configs/garmentlab_deadline.json", data_config,
                        ROOT / "tools/run_garmentlab_deadline.py"]):
        relative = item.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(item):
            raise ValueError("Saved source snapshot differs from the current source.")
        if not target.exists():
            shutil.copyfile(item, target)
    manifest = ROOT / "SOURCE_MANIFEST.json"
    if manifest.is_file():
        target = snapshot / "SOURCE_MANIFEST.json"
        if target.exists() and digest(target) != digest(manifest):
            raise ValueError("Saved source package identity differs.")
        if not target.exists():
            shutil.copyfile(manifest, target)


def gpu_check(out, smoke_cpu=False):
    import torch
    import torchvision
    from a1.cnn import GlobalAveragePool2d
    torch.set_num_threads(4)
    if not smoke_cpu and (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("This notebook requires an NVIDIA A100 CUDA runtime.")
    device = "cpu" if smoke_cpu else "cuda"
    torch.use_deterministic_algorithms(True)
    model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=6)
    model.avgpool = GlobalAveragePool2d()
    model = model.to(device).train()
    images = torch.zeros(2, 3, 224, 224, device=device)
    targets = torch.tensor([0, 1], device=device)
    loss = torch.nn.functional.cross_entropy(model(images), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise RuntimeError("Device smoke loss is not finite.")
    result = {"status": "PASS_SYNTHETIC_DEVICE_CHECK_ONLY", "device": device,
              "gpu": torch.cuda.get_device_name(0) if not smoke_cpu else None,
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda": torch.version.cuda, "python": sys.version,
              "scope": "One random-weight synthetic forward/backward; no garment model quality or training task."}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "DEVICE_CHECK.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del model, images, targets, loss
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "prepare", "train", "finalize", "all", "export"], default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/content/garmentlab_work"))
    parser.add_argument("--cache", type=Path, default=Path("/content/garmentlab_cache"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/garmentlab_deadline.json")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/garmentlab_data.json")
    parser.add_argument("--smoke-cpu", action="store_true", help="Device check only; never full training.")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "export":
        print("RESULT_ZIP:", export_results(args.out, source_dir=ROOT), flush=True)
        return 0
    if args.stage == "check":
        gpu_check(args.out, args.smoke_cpu)
        return 0
    if args.smoke_cpu:
        parser.error("--smoke-cpu is permitted only with --stage check.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bind_run(args.out, config, args.data_config)
    prepared = args.work / "prepared"
    prepared_pointer = args.out / "PREPARED_POINTER.json"
    status = {"status": "RUNNING", "stage": args.stage, "updated_utc": utc_now().isoformat()}
    exit_code = 0
    acquired = False
    try:
        with RunSession(args.out, total_seconds=config["max_total_process_seconds"], stop_by_utc=config["stop_by_utc"]) as session:
            acquired = True
            atomic_json(args.out / "RUN_STATUS.json", status)
            if args.stage in {"all", "prepare"}:
                if session.should_stop(config["finalization_reserve_seconds"]):
                    raise TimeoutError("Deadline or total budget reservation reached before data preparation.")
                session.stage = "prepare"
                session.tick()
                from garmentlab.images import prepare_dataset
                prepared = Path(prepare_dataset(args.work, args.cache, args.data_config))
                atomic_json(prepared_pointer, {"prepared_dir": str(prepared), "work_dir": str(args.work)})
                for name in ["DATA_READINESS.json", "index_manifest.json", "TRAIN_PREVIEW.jpg", "TRAIN_PREVIEW_CREDITS.csv", "DUPLICATE_AUDIT.json", "METADATA_AUDITS.json"]:
                    source = prepared / name
                    if source.is_file():
                        target = args.out / "data_evidence" / name
                        target.parent.mkdir(exist_ok=True)
                        shutil.copyfile(source, target)
            elif prepared_pointer.is_file():
                prepared = Path(json.loads(prepared_pointer.read_text(encoding="utf-8"))["prepared_dir"])
            if args.stage in {"all", "train"}:
                session.stage = "train"
                session.tick()
                from garmentlab.training import train_study
                trained = train_study(prepared, args.out, config,
                                      stop_check=lambda: session.should_stop(config["finalization_reserve_seconds"]))
                print("TRAIN_STATE:", json.dumps(trained, ensure_ascii=False), flush=True)
                if trained.get("status") != "ALL_TASKS_COMPLETE":
                    status = {"status": "INCOMPLETE", "training": trained, "updated_utc": utc_now().isoformat()}
                    atomic_json(args.out / "RUN_STATUS.json", status)
                    return 0
            if args.stage in {"all", "finalize"}:
                session.stage = "finalize"
                session.tick()
                if session.should_stop(0):
                    raise TimeoutError("Deadline or total process budget reached before finalization.")
                from garmentlab.evaluate import finalize_study
                summary = finalize_study(args.out, prepared, config)
                status = {"status": "COMPLETE", "finalization": summary, "updated_utc": utc_now().isoformat()}
            else:
                status = {"status": "PREPARED" if args.stage == "prepare" else "TRAINING_COMPLETE",
                          "updated_utc": utc_now().isoformat()}
            atomic_json(args.out / "RUN_STATUS.json", status)
    except (Exception, KeyboardInterrupt) as exc:
        if not acquired:
            print("Controller not started:", str(exc), flush=True)
            return 2
        status = {"status": "INCOMPLETE_STOPPED" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "FAILED",
                  "exception_type": type(exc).__name__, "message": str(exc), "updated_utc": utc_now().isoformat()}
        atomic_json(args.out / "RUN_STATUS.json", status)
        (args.out / "LAST_ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        exit_code = 2
    finally:
        if acquired:
            print("RESULT_ZIP:", export_results(args.out, source_dir=ROOT), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
