"""Audit Fashionpedia metadata; export train candidates only after support gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garmentlab.data import prepare_metadata_candidate, write_preparation_outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-role", choices=("official_train", "official_assessment"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--assessment-annotations", type=Path,
                        help="For official_train: quarantine original-URL groups shared with this assessment metadata")
    parser.add_argument("--expected-assessment-sha256", "--assessment-expected-sha256",
                        dest="assessment_expected_sha256",
                        help="Reference annotation SHA-256; defaults to the pinned Fashionpedia val SHA")
    parser.add_argument("--allow-incomplete-audit", action="store_true",
                        help="Allow exit 0 for an audit failing class support; never emits a train candidate")
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    result = prepare_metadata_candidate(args.annotations, config, source_role=args.source_role,
                                        expected_sha256=args.expected_sha256,
                                        assessment_annotations=args.assessment_annotations,
                                        assessment_expected_sha256=args.assessment_expected_sha256)
    result["audit"]["input_config_file_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    paths = write_preparation_outputs(result, args.out_dir)
    ready = result["audit"]["readiness"]["metadata_ready"]
    print(json.dumps({"metadata_ready": ready, "training_ready": False,
                      "source_role": args.source_role, "outputs": paths,
                      "failed_support_checks": len(result["audit"]["readiness"]["failures"])},
                     ensure_ascii=True))
    return 0 if ready or args.allow_incomplete_audit else 2


if __name__ == "__main__":
    raise SystemExit(main())
