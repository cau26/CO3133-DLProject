"""Read-only integrity review of returned A1 Final artifacts, without model inference.

Usage: python tools/verify_a1_results.py --out RUN_DIR --require-complete --require-test
The optional receipt must be outside RUN_DIR and is created exclusively. Missing work
is PENDING unless a --require-* flag requires it; contradictions always fail. NumPy
and torch checks are optional and explicitly reported when unavailable. This tool
does not reproduce loss/timing, authenticate an experiment, or attest human review.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone

MODELS = ("linear", "mlp", "cnn", "gru", "transformer")
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def source_hash_match(path, expected):
    """Accept only exact bytes or LF/CRLF Git checkout differences for source."""
    data = Path(path).read_bytes()
    raw = hashlib.sha256(data).hexdigest()
    if raw == expected:
        return {"match": "BYTE_EXACT", "local_raw_sha256": raw, "recorded_sha256": expected}
    lf = data.replace(b"\r\n", b"\n")
    for label, candidate in (("LF", lf), ("CRLF", lf.replace(b"\n", b"\r\n"))):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return {"match": "GIT_CHECKOUT_EOL_EQUIVALENT", "transformation": label,
                    "local_raw_sha256": raw, "recorded_sha256": expected}
    raise ValueError(f"Source differs beyond LF/CRLF checkout changes: {path}")


def test_fingerprint(started, metrics=None):
    value = started["official_test_dataset_sha256"]
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("Invalid official_test_dataset_sha256; expected 64 hexadecimal characters")
    if metrics is not None:
        other = metrics["official_test_dataset_sha256"]
        if not isinstance(other, str) or other.lower() != value.lower():
            raise ValueError("Official test dataset fingerprint differs from TEST_STARTED.json")
    return value.lower()


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON literal: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(value, label, lower=0, upper=None):
    if isinstance(value, bool):
        raise ValueError(f"{label}: boolean is not a numeric metric")
    number = float(value)
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise ValueError(f"{label}: invalid/non-finite value {value!r}")
    return number


def integer(value, label):
    number = finite(value, label)
    if not number.is_integer():
        raise ValueError(f"{label}: expected integer")
    return int(number)


def close(actual, expected, label, atol=1e-5, rtol=1e-5):
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
        raise ValueError(f"{label}: saved={expected!r}, recomputed={actual!r}")


def safe_child(root, relative):
    # Manifest paths are input data, never instructions or arbitrary filesystem reads.
    item = PurePosixPath(relative)
    if (not relative or "\\" in relative or ":" in relative or item.is_absolute()
            or any(part in ("..", ".") for part in item.parts)):
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    result = (root / Path(*item.parts)).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes run: {relative!r}")
    return result


def resolved_config(name, config):
    result = dict(config)
    overrides = result.pop("per_model", {}).get(name, {})
    forbidden = {"seed", "split_seed", "evaluation_budgets", "validation_size", "device", "num_workers", "cpu_threads",
                 "group", "members", "per_model"}
    if forbidden.intersection(overrides):
        raise ValueError(f"{name}: forbidden shared config override")
    result.update(overrides)
    return result


def score_predictions(rows, index_key, expected_ids=None):
    if not rows:
        raise ValueError("Prediction CSV is empty")
    ids, true, pred = [], [], []
    confusion = [[0] * 10 for _ in range(10)]
    for row in rows:
        idx = integer(row[index_key], index_key)
        t, p = integer(row["true"], "true"), integer(row["predicted"], "predicted")
        if t > 9 or p > 9:
            raise ValueError("Prediction labels must be in [0,9]")
        ids.append(idx)
        true.append(t)
        pred.append(p)
        confusion[t][p] += 1
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction CSV has duplicate sample IDs")
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError("Prediction IDs/order differ from the saved evaluation split")
    accuracy = sum(confusion[k][k] for k in range(10)) / len(rows)
    f1_values = []
    for k in range(10):
        denominator = sum(confusion[k]) + sum(row[k] for row in confusion)
        f1_values.append(2 * confusion[k][k] / denominator if denominator else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1_values) / 10,
            "n_eval": len(rows), "confusion": confusion, "ids": ids,
            "truth": true}


def check_history(rows, metrics, config):
    if not rows:
        raise ValueError("Completed model has empty history")
    epochs = [integer(row["epoch"], "epoch") for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError("History epochs are not contiguous starting at 1")
    if len(rows) != config["epochs"] or len(rows) != metrics["epochs_completed"]:
        raise ValueError("History/completed/config epoch counts disagree")
    for row in rows:
        for key in ("train_loss", "val_loss", "train_seconds", "val_seconds"):
            finite(row[key], key)
        for key in ("train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"):
            finite(row[key], key, upper=1)
        if "learning_rate" in row:
            finite(row["learning_rate"], "learning_rate")
        if "gpu_peak_allocated_bytes" in row:
            integer(row["gpu_peak_allocated_bytes"], "gpu_peak_allocated_bytes")
    selected = min(rows, key=lambda row: float(row["val_loss"]))  # stable: first tie
    if integer(selected["epoch"], "selected epoch") != metrics["best_epoch"]:
        raise ValueError("best_epoch is not the earliest minimum-validation-loss epoch")
    for key, column in (("loss", "val_loss"), ("accuracy", "val_accuracy"), ("macro_f1", "val_macro_f1")):
        close(float(selected[column]), metrics[key], f"best history {key}")
    close(sum(float(row["train_seconds"]) for row in rows), metrics["train_seconds"], "training time sum")
    close(sum(float(row["val_seconds"]) for row in rows), metrics["validation_seconds"], "validation time sum")
    return {"epochs": len(rows), "selected_epoch": metrics["best_epoch"]}


class Review:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.errors, self.pending, self.checks, self.optional = [], [], [], []
        self.config = self.run = self.summary = self.freeze = None
        self.validation_ids = None
        self.truth_by_split = {}

    def check(self, label, callback):
        try:
            result = callback()
            self.checks.append({"check": label, "status": "PASS", "details": result})
            return result
        except Exception as exc:
            # Keep collecting independent errors, including malformed/untrusted files.
            self.errors.append({"check": label, "message": f"{type(exc).__name__}: {exc}"})
            return None

    def missing(self, label):
        self.pending.append(label)

    def require_files(self, directory, names):
        absent = [name for name in names if not (directory / name).is_file()]
        if absent:
            raise ValueError("Missing completed-run files: " + ", ".join(absent))

    def hashes(self, entries, root=None):
        root = root or self.root
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Hash manifest is empty or malformed")
        for relative, expected in entries.items():
            actual = file_hash(safe_child(root, relative))
            if actual != expected:
                raise ValueError(f"Hash mismatch: {relative}")
        return {"files_verified": len(entries)}

    def source_record(self, source):
        entries = source["files_sha256"]
        combined = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if combined != source["source_sha256"]:
            raise ValueError("Source file hash map does not match source_sha256")
        if source["source_sha256"] != self.run["source_sha256"]:
            raise ValueError("Model source differs from run source")
        return {"source_sha256": combined}

    def source_hashes(self, entries):
        normalized = {}
        for relative, expected in entries.items():
            result = source_hash_match(safe_child(ROOT, relative), expected)
            if result["match"] != "BYTE_EXACT":
                normalized[relative] = result
        return {"files_verified": len(entries), "normalized_checkout_matches": normalized,
                "policy": "Source only: byte-exact or LF/CRLF checkout equivalence; all output/manifest hashes remain byte-exact"}

    def metadata(self):
        self.config = read_json(self.root / "config.json")
        self.run = read_json(self.root / "run_manifest.json")
        self.summary = read_json(self.root / "data_summary.json")
        if self.run["schema_version"] != 1 or not self.run["run_id"]:
            raise ValueError("Unknown or missing run identity/schema")
        if object_hash(self.config) != self.run["config_sha256"]:
            raise ValueError("Run config hash mismatch")
        self.source_record(self.run["source"])
        if self.summary["dataset"] != "Fashion-MNIST" or self.summary["official_train"] != 60000:
            raise ValueError("Unexpected dataset/official training size")
        if (self.summary["seed"] != self.config["seed"]
                or self.summary["validation"] != self.config["validation_size"]
                or self.summary["train"] + self.summary["validation"] != 60000
                or self.summary["test"] != 10000):
            raise ValueError("Dataset summary/config partition counts disagree")
        if "split_seed" in self.config:
            if (self.summary.get("split_metadata_schema_version") != 2
                    or self.summary.get("split_seed") != self.config["split_seed"]):
                raise ValueError("Explicit split seed/schema differs from saved config")
        elif "split_seed" in self.summary or "split_metadata_schema_version" in self.summary:
            raise ValueError("Legacy config cannot adopt a study split schema")
        self.require_files(self.root, ["split_indices.npz"])
        return {"run_id": self.run["run_id"], "config_sha256": object_hash(self.config)}

    def split(self):
        try:
            import numpy as np
        except ImportError:
            self.optional.append({"check": "split array/hash validation", "status": "SKIPPED",
                                  "reason": "NumPy unavailable; install project dependencies and rerun"})
            return {"status": "SKIPPED"}
        with np.load(self.root / "split_indices.npz", allow_pickle=False) as saved:
            train, val = saved["train"], saved["validation"]
        for array in (train, val):
            if array.ndim != 1 or array.dtype.kind not in "iu" or not np.array_equal(array, np.sort(array)):
                raise ValueError("Split arrays must be sorted, one-dimensional integer IDs")
        if (len(train) != self.summary["train"] or len(val) != self.summary["validation"]
                or not np.array_equal(np.sort(np.r_[train, val]), np.arange(60000))):
            raise ValueError("Split has overlapping/duplicate/missing/out-of-range IDs or wrong size")
        digest = hashlib.sha256(train.astype("<i8").tobytes() + val.astype("<i8").tobytes()).hexdigest()
        if digest != self.summary["split_sha256"]:
            raise ValueError("Split hash mismatch")
        self.validation_ids = val.tolist()
        return {"train": len(train), "validation": len(val), "split_sha256": digest}

    def check_metrics(self, name, metrics, official_test=False):
        split = "official_test" if official_test else "validation"
        cfg = resolved_config(name, self.config)
        expected_n = self.summary["test" if official_test else "validation"]
        if (metrics["model"] != name or metrics["evaluation_split"] != split
                or metrics["run_id"] != self.run["run_id"] or metrics["config"] != cfg
                or metrics["n_eval"] != expected_n):
            raise ValueError("Model/split/run/config/sample count binding mismatch")
        for field in ("loss", "accuracy", "macro_f1"):
            finite(metrics[field], field, upper=1 if field != "loss" else None)
        inference = metrics["inference"]
        for key in ("batch_size", "warmup_batches", "measured_batches"):
            if integer(inference[key], key) < 1:
                raise ValueError(f"{key} must be positive")
        expected_timing = {"batch_size": min(expected_n, cfg["batch_size"]),
                           "warmup_batches": cfg.get("inference_warmup_batches", 10),
                           "measured_batches": cfg.get("inference_measured_batches", 50)}
        if any(inference[key] != value for key, value in expected_timing.items()):
            raise ValueError("Inference batch/warmup/repeats differ from saved config")
        finite(inference["batch_ms"], "batch_ms")
        finite(inference["per_image_ms_amortized"], "per_image_ms_amortized")
        close(inference["batch_ms"] / inference["batch_size"],
              inference["per_image_ms_amortized"], "latency amortization", atol=1e-12)
        if metrics["checkpoint_sha256"] != file_hash(self.root / name / "best.pt"):
            raise ValueError("Checkpoint hash differs from metrics")
        self.source_record(metrics["source"])
        if official_test:
            if (metrics["test_evaluated"] is not True or not self.freeze
                    or metrics["split_sha256"] != self.summary["split_sha256"]
                    or metrics["freeze_manifest_sha256"] != file_hash(self.root / "freeze_manifest.json")):
                raise ValueError("Official test is not bound to this frozen run")
        else:
            if (metrics["data"] != self.summary or metrics["config_sha256"] != object_hash(cfg)
                    or metrics["test_evaluated"] is not False
                    or metrics["checkpoint_rule"] != "lowest validation loss; first if tied"):
                raise ValueError("Validation data/config/checkpoint rule mismatch")
            integer(metrics["parameters"], "parameters")
            integer(metrics["best_epoch"], "best_epoch")
            finite(metrics["fit_wall_seconds"], "fit_wall_seconds")
        return cfg

    def predictions(self, name, directory, metrics, official_test=False):
        split = "test" if official_test else "validation"
        rows = read_csv(directory / ("test_predictions.csv" if official_test else "validation_predictions.csv"))
        expected_ids = range(10000) if official_test else self.validation_ids
        scores = score_predictions(rows, "official_test_index" if official_test else "official_train_index", expected_ids)
        if scores["n_eval"] != metrics["n_eval"]:
            raise ValueError("Prediction count differs from n_eval")
        for key in ("accuracy", "macro_f1"):
            close(scores[key], metrics[key], key, atol=1e-12, rtol=1e-12)
        identity = (scores["ids"], scores["truth"])
        if split in self.truth_by_split and self.truth_by_split[split] != identity:
            raise ValueError("Evaluation IDs/ground truths differ between models")
        self.truth_by_split[split] = identity
        cm_rows = read_csv(directory / "confusion_matrix.csv")
        if len(cm_rows) != 10:
            raise ValueError("Confusion matrix CSV must contain ten rows")
        for k, row in enumerate(cm_rows):
            if integer(row["true_label"], "true_label") != k:
                raise ValueError("Confusion matrix row order mismatch")
            if [integer(row[f"predicted_{j}"], "confusion cell") for j in range(10)] != scores["confusion"][k]:
                raise ValueError("Saved confusion matrix differs from predictions")
        expected_errors = [(t, p, scores["confusion"][t][p]) for t in range(10) for p in range(10)
                           if t != p and scores["confusion"][t][p] > 0]
        expected_errors.sort(key=lambda row: row[2], reverse=True)
        expected_errors = expected_errors[:5]
        if len(metrics["top_errors"]) != len(expected_errors):
            raise ValueError("Top errors list is incomplete or has extra rows")
        for error, (expected_t, expected_p, _) in zip(metrics["top_errors"], expected_errors):
            t, p = integer(error["true_label"], "true error label"), integer(error["predicted_label"], "predicted error label")
            if t > 9 or p > 9 or t == p or (t, p) != (expected_t, expected_p):
                raise ValueError("Invalid top error class pair")
            if (error["true_class"] != self.summary["classes"][t]
                    or error["predicted_class"] != self.summary["classes"][p]):
                raise ValueError("Top error class names differ from dataset mapping")
            count = scores["confusion"][t][p]
            if count != error["count"]:
                raise ValueError("Top error count differs from predictions")
            close(count / sum(scores["confusion"][t]), error["rate_within_true_class"], "top error rate", atol=1e-12)
        return {"samples": scores["n_eval"], "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
                "scope": "recomputed from saved class predictions; no checkpoint inference"}

    def checkpoint(self, name, metrics):
        try:
            import torch
        except (ImportError, OSError) as exc:
            self.optional.append({"check": f"{name} checkpoint metadata/parameter count", "status": "SKIPPED",
                                  "reason": f"torch unavailable: {exc}"})
            return {"status": "SKIPPED"}
        checkpoint = torch.load(self.root / name / "best.pt", map_location="cpu", weights_only=True)
        expected = {"model_name": name, "config": metrics["config"], "config_sha256": metrics["config_sha256"],
                    "split_sha256": self.summary["split_sha256"], "dataset_sha256": self.summary["dataset_sha256"],
                    "source_sha256": self.run["source_sha256"], "run_id": self.run["run_id"],
                    "epoch": metrics["best_epoch"]}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint binding mismatch: {key}")
        close(checkpoint["val_loss"], metrics["loss"], "checkpoint val_loss")
        # Import only the canonical architecture module: no training, data download,
        # plotting, sklearn, or forward pass. Match local source before using it.
        source_relative = f"a1/{name}.py"
        source_path = ROOT / source_relative
        source_match = source_hash_match(source_path, metrics["source"]["files_sha256"][source_relative])
        spec = importlib.util.spec_from_file_location(f"a1_result_review_{name}", source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = metrics["config"]
        constructors = {
            "linear": lambda: module.LinearClassifier(),
            "mlp": lambda: module.MLP(cfg.get("hidden_size", 256)),
            "cnn": lambda: module.CustomCNN(cfg.get("cnn_channels", [32, 64, 128]), cfg.get("dropout", 0.2)),
            "gru": lambda: module.RowGRU(cfg.get("gru_hidden_size", 128), cfg.get("gru_num_layers", 1), cfg.get("dropout", 0.0),
                sequence_axis=cfg.get("sequence_axis", "rows")),
            "transformer": lambda: module.PatchTransformer(cfg.get("patch_size", 4), cfg.get("d_model", 128),
                cfg.get("nhead", 4), cfg.get("transformer_layers", 2), cfg.get("dim_feedforward", 256), cfg.get("dropout", 0.1),
                use_position_embedding=cfg.get("use_position_embedding", True)),
        }
        model = constructors[name]()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite weights")
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != metrics["parameters"]:
            raise ValueError(f"Parameter count mismatch: checkpoint model={count}, metrics={metrics['parameters']}")
        return {"parameters": count, "strict_state_load": True, "inference_performed": False,
                "architecture_source_match": source_match}

    def model(self, name):
        directory = self.root / name
        if not (directory / "metrics.json").is_file():
            status = "INCOMPLETE" if directory.exists() else "NOT_RUN"
            self.missing(f"{name}: {status}")
            return {"status": status}
        before = len(self.errors)

        def completed():
            self.require_files(directory, ["metrics.json", "model_config.json", "best.pt", "history.csv",
                "validation_predictions.csv", "curves.png", "confusion_matrix.png", "confusion_matrix.csv", "examples.png"])
            metrics = read_json(directory / "metrics.json")
            cfg = self.check_metrics(name, metrics)
            if read_json(directory / "model_config.json") != cfg:
                raise ValueError("model_config.json differs from effective saved config")
            history = check_history(read_csv(directory / "history.csv"), metrics, cfg)
            predictions = self.predictions(name, directory, metrics)
            self.check(f"{name} checkpoint", lambda: self.checkpoint(name, metrics))
            return {"history": history, "predictions": predictions}

        details = self.check(f"{name} validation artifacts", completed)
        return {"status": "CONSISTENT" if len(self.errors) == before else "FAIL", "details": details}

    def frozen(self):
        path = self.root / "freeze_manifest.json"
        if not path.is_file():
            self.missing("five-model freeze: NOT_CREATED")
            return None
        manifest = read_json(path)
        content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if manifest.get("schema_version") != 1 or manifest.get("manifest_sha256") != object_hash(content):
            raise ValueError("Freeze manifest self-hash/schema mismatch")
        if (set(manifest["models"]) != set(MODELS) or manifest["run_id"] != self.run["run_id"]
                or manifest["source_sha256"] != self.run["source_sha256"]
                or manifest["split_sha256"] != self.summary["split_sha256"]
                or manifest["dataset_sha256"] != self.summary["dataset_sha256"]):
            raise ValueError("Freeze model/run/source/split/dataset identity mismatch")
        expected = {"config.json", "run_manifest.json", "split_indices.npz", "data_summary.json"}
        expected.update(f"{name}/{file}" for name in MODELS
                        for file in ("best.pt", "metrics.json", "model_config.json", "history.csv"))
        if set(manifest["files_sha256"]) != expected:
            raise ValueError("Freeze lock set is incomplete or unexpected")
        self.hashes(manifest["files_sha256"])
        for name in MODELS:
            model, metrics = manifest["models"][name], read_json(self.root / name / "metrics.json")
            if (model["checkpoint_sha256"] != metrics["checkpoint_sha256"]
                    or model["config_sha256"] != metrics["config_sha256"]
                    or model["best_epoch"] != metrics["best_epoch"]):
                raise ValueError(f"Freeze model selection mismatch: {name}")
        self.freeze = manifest
        return {"run_id": manifest["run_id"], "locked_files": len(expected)}

    def test(self):
        destination = self.root / "final_test"
        if not destination.exists():
            self.missing("official test: NOT_RUN")
            return {"status": "NOT_RUN"}
        if not self.freeze:
            raise ValueError("Test artifacts exist without a verified five-model freeze")
        started = read_json(destination / "TEST_STARTED.json")
        freeze_hash = file_hash(self.root / "freeze_manifest.json")
        if started["run_id"] != self.run["run_id"] or started["freeze_manifest_sha256"] != freeze_hash:
            raise ValueError("Test-start record binding mismatch")
        dataset_hash = test_fingerprint(started)
        states = {}
        for name in MODELS:
            directory = destination / name
            if not (directory / "metrics.json").is_file():
                states[name] = "PENDING"
                continue

            def check_one(name=name, directory=directory):
                self.require_files(directory, ["metrics.json", "test_predictions.csv", "confusion_matrix.csv",
                                               "confusion_matrix.png", "examples.png"])
                metrics = read_json(directory / "metrics.json")
                test_fingerprint(started, metrics)
                self.check_metrics(name, metrics, official_test=True)
                return self.predictions(name, directory, metrics, official_test=True)

            states[name] = "CONSISTENT" if self.check(f"{name} official test artifacts", check_one) else "FAIL"
        marker = destination / "RESULTS_COMPLETE.json"
        complete = all(value == "CONSISTENT" for value in states.values())
        if marker.is_file():
            saved = read_json(marker)
            if (not complete or set(saved["models"]) != set(MODELS)
                    or saved["run_id"] != self.run["run_id"] or saved["freeze_manifest_sha256"] != freeze_hash):
                raise ValueError("Test completion marker contradicts artifacts or run identity")
            rows = read_csv(destination / "comparison.csv")
            if len(rows) != 5 or {row["model"] for row in rows} != set(MODELS):
                raise ValueError("Test comparison does not contain exactly five models")
            for row in rows:
                metrics = read_json(destination / row["model"] / "metrics.json")
                for key in ("loss", "accuracy", "macro_f1"):
                    close(finite(row[key], key), metrics[key], f"test comparison {key}", atol=1e-12)
        else:
            complete = False
            self.missing("official test: INCOMPLETE (RESULTS_COMPLETE.json missing)")
        return {"status": "COMPLETE" if complete else "PENDING", "models": states,
                "official_test_dataset_sha256": dataset_hash,
                "fingerprint_scope": "saved test records agree; raw dataset not independently downloaded/rehashed"}


def validate(out, require_complete=False, require_test=False):
    review = Review(out)
    if not review.root.is_dir():
        review.missing("run directory: NOT_FOUND")
    required = ["config.json", "run_manifest.json", "data_summary.json", "split_indices.npz"]
    missing = [name for name in required if not (review.root / name).is_file()]
    for name in missing:
        review.missing(f"{name}: MISSING")
    metadata_ok = not missing and review.check("run metadata", review.metadata) is not None
    models = {name: {"status": "NOT_VERIFIED"} for name in MODELS}
    test = {"status": "NOT_VERIFIED"}
    if metadata_ok:
        review.check("split integrity", review.split)
        review.check("source files in local checkout", lambda: review.source_hashes(review.run["source"]["files_sha256"]))
        models = {name: review.model(name) for name in MODELS}
        review.check("freeze manifest", review.frozen)
        test = review.check("official test", review.test) or {"status": "FAIL"}
    artifact_manifest = review.root / "artifact_manifest.json"
    if artifact_manifest.is_file():
        review.check("export artifact manifest", lambda: review.hashes(read_json(artifact_manifest)))
    else:
        review.optional.append({"check": "export artifact manifest", "status": "NOT_PRESENT"})
    completed = metadata_ok and all(models.get(name, {}).get("status") == "CONSISTENT" for name in MODELS)
    if (require_complete or require_test) and not completed:
        review.errors.append({"check": "required validation completion", "message": "All five consistent completed models are required"})
    if require_test and test.get("status") != "COMPLETE":
        review.errors.append({"check": "required official test completion", "message": "A consistent completed five-model official test is required"})
    status = "FAIL" if review.errors else ("PENDING" if review.pending else "PASS")
    return {"schema_version": 1, "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(review.root), "status": status, "models": models,
            "all_five_validation_consistent": bool(completed), "official_test": test,
            "errors": review.errors, "pending": review.pending, "checks": review.checks,
            "optional_checks": review.optional,
            "limitations": ["This is artifact consistency review, not proof that training was observed or human-approved.",
                "Accuracy/Macro-F1 are recomputed from saved class predictions; their ground truths are cross-checked between models, not against downloaded dataset labels.",
                "No checkpoint inference, independent loss reproduction, training, timing measurement, or image-quality review is performed.",
                "Raw dataset content fingerprints cannot be independently reproduced without the dataset; saved source/hash records are not signed attestations."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--receipt", type=Path, help="Exclusive JSON receipt outside the run directory")
    args = parser.parse_args(argv)
    result = validate(args.out, args.require_complete, args.require_test)
    if args.receipt:
        try:
            receipt, root = args.receipt.resolve(), args.out.resolve()
            if receipt == root or receipt.is_relative_to(root):
                raise ValueError("Receipt must be outside the reviewed run; run artifacts are read-only")
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        except (OSError, ValueError) as exc:
            result["errors"].append({"check": "receipt destination", "message": str(exc)})
            result["status"] = "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
