"""Serve the draft locally; a real-photo model is never substituted by a reference."""
from pathlib import Path
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_request_logging(path=None):
    """Enable the API's existing JSON logs without retaining uploaded content."""
    logger = logging.getLogger("garmentlab.requests")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if path is None:
        handler = logging.StreamHandler()
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=2*1024*1024, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    # main is called once by the CLI; keeping handlers explicit also permits an
    # embedding application to configure logging without duplicate JSON lines.
    for previous in list(logger.handlers):
        logger.removeHandler(previous)
        previous.close()
    logger.addHandler(handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-release", type=Path, default=ROOT / "draft/garmentlab/reference_cnn")
    parser.add_argument("--garment-release", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--request-log", type=Path, help="Optional rotating JSON request log; default is stderr")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.threads <= 32:
        parser.error("Invalid port or thread count.")
    import torch
    import uvicorn
    from garmentlab.inference import Predictor
    from garmentlab.api import create_app
    torch.set_num_threads(args.threads)
    reference = Predictor(args.reference_release) if args.reference_release.is_dir() else None
    garment = Predictor(args.garment_release) if args.garment_release else None
    request_logger = configure_request_logging(args.request_log)
    request_logger.info(json.dumps({"event": "service_start", "host": "127.0.0.1", "port": args.port,
        "pytorch_threads": args.threads,
        "garment_release_id": garment.metadata["model_version"] if garment else None,
        "garment_weights_sha256": garment.metadata["weights_sha256"] if garment else None}))
    app = create_app(reference_predictor=reference, garment_predictor=garment)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
