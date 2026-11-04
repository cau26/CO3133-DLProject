"""Bounded preparation, controlled fitting, frozen evaluation and evidence export."""
from __future__ import annotations

import argparse
import hashlib
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
os.environ.setdefault("MKL_NUM_THREADS", "4")

from garmentlab.runtime import RunSession, atomic_json, digest, export_results, utc_now

EXPECTED_ARRAYS = {
    "images_uint8.npy": "c9ba2ec047594062ab38d7d7d0e5376363c61c86ebf8eb24992f0479c3204b27",
    "labels.npy": "7bd6f836c34ca005b69a962564845b78f6e054de0c44a51db2bda37bf50a3d61",
    "roles.npy": "65227362fe6541996c2a5938126040c0b36249c24e5b1c751ac54fe458137d23",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protect(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable run identity changed: " + path.name)
    else:
        atomic_json(path, value)


def source_files():
    return sorted({*ROOT.joinpath("garmentlab").glob("*.py"), ROOT / "a1/__init__.py",
        ROOT / "a1/cnn.py", ROOT / "tools/run_garmentlab_improve.py",
        ROOT / "tools/run_garmentlab_deadline.py", ROOT / "tools/reexport_garmentlab.py"})


def bind_run(out, config, data_config, *, smoke):
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}
    value = {"schema": "garmentlab_improve_today_run_v2", "config": config,
        "data_config_sha256": digest(data_config), "source_sha256": sources,
        "smoke_only": smoke, "assessment_scope": "Previously inspected v1 assessment; v2 is exploratory/regression evaluation."}
    protect(out / "RUN_MANIFEST_V2.json", value)
    for name, expected in sources.items():
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != expected:
                raise ValueError("Run source snapshot changed: " + name)
        else:
            shutil.copyfile(ROOT / name, target)
    for source, name in [(data_config, "configs/garmentlab_data.json"),
                         (ROOT / "SOURCE_MANIFEST.json", "SOURCE_MANIFEST.json")]:
        source = Path(source)
        if source.is_file():
            target = out / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(source):
                raise ValueError("Packaged source identity changed")
            if not target.exists():
                shutil.copyfile(source, target)
    protect(out / "effective_config.json", config)
    return value


def export_v2(out):
    result = export_results(out)
    destination = Path(out) / "exports/g_m10_garmentlab_improve_today_run01_results.zip"
    result.replace(destination)
    return destination


def validate_data(prepared, out, *, smoke):
    from garmentlab.training import load_prepared
    data = load_prepared(prepared)
    if not smoke:
        for name, expected in EXPECTED_ARRAYS.items():
            if data["bindings"][name] != expected:
                raise ValueError("Original data/role identity differs: " + name)
        if len(data["records"]) != 17827:
            raise ValueError("The declared v2 study preserves the original 17,827 crops")
    for name in ["DATA_READINESS.json", "index_manifest.json", "DUPLICATE_AUDIT.json", "METADATA_AUDITS.json"]:
        source = Path(prepared) / name
        if source.is_file():
            target = out / "data_evidence" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(source):
                raise ValueError("Prepared evidence differs from this run")
            if not target.exists():
                shutil.copyfile(source, target)
    receipt = {"prepared_dir": str(Path(prepared).resolve()), "bindings": data["bindings"],
               "scope": "Same original arrays and four roles; metadata for this reconstruction is recorded separately.",
               "original_arrays_identical": not smoke, "smoke_only": smoke}
    if hasattr(data["images"], "_mmap"):
        data["images"]._mmap.close()
    atomic_json(out / "PREPARED_POINTER_V2.json", receipt)
    return Path(prepared)


def run_workflow(out, work, cache, config_path, data_config, *, stage="all", smoke=False,
                 model_factory=None, prepare_fn=None, check_fn=None):
    """A single-writer controller. Injection is restricted to explicit CPU smoke."""
    if stage not in {"plan", "check", "prepare", "train", "finalize", "all", "export"}:
        raise ValueError("Unknown workflow stage")
    if any(x is not None for x in (model_factory, prepare_fn, check_fn)) and not smoke:
        raise ValueError("Injected dependencies are allowed only in software smoke tests")
    out, work, cache = [Path(p).resolve() for p in (out, work, cache)]
    if (out == work or out.is_relative_to(work) or work.is_relative_to(out)
            or out == cache or out.is_relative_to(cache) or cache.is_relative_to(out)):
        raise ValueError("Output, prepared work and raw cache must use separate directories")
    if out.exists() and not (out / "RUN_MANIFEST_V2.json").exists():
        existing = set(p.name for p in out.iterdir())
        if existing:
            raise ValueError("Output belongs to another run; use the new v2 directory")
    out.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return {"status": read_json(out / "RUN_STATUS.json") if (out / "RUN_STATUS.json").exists() else {},
                "result_zip": str(export_v2(out))}
    from garmentlab.study_v2 import validate_config, plan_study, run_study
    config = validate_config(read_json(config_path))
    bind_run(out, config, Path(data_config), smoke=smoke)
    if stage == "plan":
        plan = {"maximum_fits": len(config["candidates"]) + (config["challenger_count"] + 1) * 2 + 6,
            "epochs_per_fit": config["epochs"], "budget_seconds": config["max_active_seconds"],
            "stop_by_utc": config["stop_by_utc"], "training_reserve_seconds": config["finalization_reserve_seconds"],
            "scope": "Selection-only discovery/confirmation; final paired anchor and selected recipe; no promised metric gain.",
            "assessment": "Original assessment retained; later scores labeled exploratory/regression."}
        atomic_json(out / "PLAN.json", plan)
        return plan
    if stage == "check":
        if check_fn is not None:
            check_fn(out)
        else:
            from tools.run_garmentlab_deadline import gpu_check
            gpu_check(out, smoke_cpu=smoke)
        return {"status": "DEVICE_CHECK_PASSED", "smoke_only": smoke}
    acquired, result_zip = False, None
    status = {"status": "RUNNING", "stage": stage, "smoke_only": smoke}
    try:
        with RunSession(out, total_seconds=config["max_active_seconds"], stop_by_utc=config["stop_by_utc"]) as session:
            acquired = True
            status["updated_utc"] = utc_now().isoformat()
            atomic_json(out / "RUN_STATUS.json", status)
            if session.should_stop(60):
                raise TimeoutError("Controller budget/deadline reached before the next stage")
            prepared = work / "prepared"
            pointer = out / "PREPARED_POINTER_V2.json"
            if stage in {"prepare", "all"}:
                session.stage = "prepare"
                session.tick()
                from garmentlab.recovery import _run_preparer
                prepared = _run_preparer(work, cache, Path(data_config), session, prepare_fn)
                prepared = validate_data(prepared, out, smoke=smoke)
            elif pointer.is_file():
                prepared = Path(read_json(pointer)["prepared_dir"])
                validate_data(prepared, out, smoke=smoke)
            else:
                raise ValueError("Prepare the bound arrays before training or finalization")
            if stage in {"train", "all"}:
                session.stage = "controlled_training"
                session.tick()
                training = run_study(prepared, out, config, smoke=smoke, model_factory=model_factory,
                    stop_check=lambda: session.should_stop(config["finalization_reserve_seconds"]))
                if training["status"] != "TRAINING_COMPLETE_CHOICES_FROZEN":
                    status = {"status": "INCOMPLETE_TRAINING", "training": training,
                              "message": "Partial evidence only. Resume the same source/config/output; no final assessment was opened."}
                else:
                    status = {"status": "TRAINING_COMPLETE", "training": training}
            if stage == "finalize" or (stage == "all" and status["status"] == "TRAINING_COMPLETE"):
                session.stage = "frozen_evaluation"
                session.tick()
                if session.should_stop(60):
                    raise TimeoutError("No evaluation time remains; training checkpoints are preserved")
                from garmentlab.evaluate_v2 import finalize_study_v2
                final = finalize_study_v2(out, prepared, config, smoke=smoke, model_factory=model_factory,
                                         stop_check=lambda: session.should_stop(60))
                status = {"status": "COMPLETE", "finalization": final, "smoke_only": smoke}
            if stage == "prepare":
                status = {"status": "PREPARED", "smoke_only": smoke}
            status["updated_utc"] = utc_now().isoformat()
            atomic_json(out / "RUN_STATUS.json", status)
    except (Exception, KeyboardInterrupt) as error:
        if not acquired:
            raise
        status = {"status": "INCOMPLETE_STOPPED" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "FAILED",
            "stage": stage, "exception_type": type(error).__name__, "message": str(error),
            "smoke_only": smoke, "updated_utc": utc_now().isoformat()}
        atomic_json(out / "RUN_STATUS.json", status)
        (out / "LAST_ERROR_V2.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if acquired:
            result_zip = export_v2(out)
    return {"status": status, "result_zip": str(result_zip) if result_zip else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["plan", "check", "prepare", "train", "finalize", "all", "export"], default="all")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/garmentlab_improve_today.json")
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/garmentlab_data.json")
    args = p.parse_args()
    result = run_workflow(args.out, args.work, args.cache, args.config, args.data_config, stage=args.stage)
    if args.stage == "plan":
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    status = result.get("status", {})
    if isinstance(status, dict):
        print("RUN STATUS:", status.get("status"), flush=True)
        if status.get("message"):
            print(status["message"], flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("result_zip"):
        print("RESULT_ZIP:", result["result_zip"], flush=True)
    if args.stage in {"all", "train", "finalize"} and isinstance(status, dict):
        return 0 if status.get("status") in {"COMPLETE", "TRAINING_COMPLETE"} else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
