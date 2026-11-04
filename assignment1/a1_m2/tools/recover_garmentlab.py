"""Recover a completed GarmentLab result ZIP without repeating training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from garmentlab.recovery import recover_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check","prepare","finalize","export","all"], default="all")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-results-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    result = recover_results(args.results,args.out,args.work,args.cache,
        expected_results_sha256=args.expected_results_sha256,stage=args.stage,source_dir=ROOT)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    print("RESULT_ZIP:",result["result_zip"],flush=True)
    return 0 if result["error"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
