"""Local release validation/activation/rollback. Restart serving after a change."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "activate", "rollback"])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("outputs/garmentlab_registry"))
    args = parser.parse_args()
    from garmentlab.release import validate_release, activate_release, rollback_release
    if args.action in {"validate", "activate"} and args.release is None:
        parser.error("--release is required for this action.")
    if args.action == "validate":
        checked = validate_release(args.release, require_ready=True)
        result = {"status": "VALIDATED_RESEARCH_INFERENCE_ARTIFACT",
                  "release_id": checked["manifest"]["release_id"],
                  "manifest_sha256": checked["manifest_sha256"]}
    elif args.action == "activate":
        result = activate_release(args.release, args.registry)
    else:
        result = rollback_release(args.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action != "validate":
        print("Registry updated. Restart the local API to load the selected model; this command does not hot-reload a live process.")


if __name__ == "__main__":
    main()
