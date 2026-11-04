"""Command-line entry point for the separate matched confirmation study."""
from __future__ import annotations

import argparse
import json

from .study import DEFAULT_SEEDS, FAMILIES, export_results, prepare, run, smoke, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "run", "status", "export"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--publication", default="output/a1_final/data/publication_data.json")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--hours", type=float, default=4.75)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = prepare(args.out, args.publication, args.data_dir, args.device,
                         args.families.split(","), [int(s) for s in args.seeds.split(",")], args.hours * 3600)
        result = {"manifest_sha256": result["manifest_sha256"],
                  "families": result["protocol"]["families"], "seeds": result["protocol"]["seeds"],
                  "trajectories": len(result["protocol"]["families"]) * len(result["protocol"]["seeds"]) * 2,
                  "scope": result["protocol"]["old_validation_scope"]}
    elif args.command == "run":
        result = run(args.out, args.data_dir, args.device, args.max_tasks)
    elif args.command == "smoke":
        result = smoke(args.out, args.data_dir, args.device)
    else:
        result = {"status": status, "export": export_results}[args.command](args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
