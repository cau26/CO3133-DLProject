"""Freeze a bounded, balanced random recipe pool before any training occurs.

Optuna is used only to sample realized configurations in memory. Pending ``ask``
trials have no fabricated objective values; this module does not fit any model.
The returned pool, including sampler provenance, belongs in the study manifest.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from numbers import Real

from a1.train import model_config


FAMILIES = ("linear", "mlp", "cnn", "gru", "transformer")
SAMPLER_SEED = 20261021
SCHEDULE_SHUFFLE_SEED = 20261023
MAX_SAMPLING_ATTEMPTS = 100
GENERATION_POLICY = "opt3_balanced_random_fixedshape_v1"
_LR_RANGES = {
    "linear": (3e-4, 3e-3), "mlp": (3e-4, 3e-3), "cnn": (3e-4, 3e-3),
    "gru": (2e-4, 2e-3), "transformer": (1e-4, 1e-3),
}


def _digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scheduled(family: str) -> str:
    return "warmup_cosine" if family == "transformer" else "cosine"


def _set_schedule(config: dict, schedule: str) -> None:
    config["schedule"] = schedule
    config["min_lr_ratio"] = 0.05
    config["warmup_epochs"] = 5 if schedule == "warmup_cosine" else 0


def _resolve(family: str, base_config: dict) -> dict:
    config = model_config(family, base_config)
    # Seeds/splits/device belong to the immutable task/manifest/environment.
    # Keeping inherited historical values here would contradict inner task IDs.
    for field in ("seed", "split_seed", "validation_size", "device"):
        config.pop(field, None)
    config.update(epochs=60, head_dropout=0.0, encoder_init="legacy")
    config.setdefault("batch_size", 128)
    config.setdefault("optimizer", "adam")
    config.setdefault("weight_decay", 0.0)
    config.setdefault("gradient_clip", 1.0)
    if family == "mlp":
        config.setdefault("hidden_size", 256)
    elif family == "cnn":
        config.setdefault("cnn_channels", [32, 64, 128])
        config.setdefault("dropout", 0.2)
    elif family == "gru":
        config.setdefault("gru_hidden_size", 128)
        config.setdefault("gru_num_layers", 1)
        config.setdefault("sequence_axis", "rows")
        config.setdefault("dropout", 0.0)
    elif family == "transformer":
        for key, value in (("patch_size", 4), ("d_model", 128), ("nhead", 4),
                           ("transformer_layers", 2), ("dim_feedforward", 256),
                           ("dropout", 0.1), ("use_position_embedding", True)):
            config.setdefault(key, value)
    # Prefix-budget selection belongs to the old study, not this 60-epoch study.
    config.pop("evaluation_budgets", None)
    _set_schedule(config, "constant")
    return config


def effective_config(family: str, config: dict) -> dict:
    """Canonical active recipe fields; ignore IDs, seeds and inactive options.

    In particular, a one-layer GRU's legacy inter-layer dropout is inactive and
    cannot make an otherwise identical candidate appear to be a new recipe.
    Metadata, loader workers and data split IDs are bound elsewhere in the
    immutable manifest and are not hyperparameters for deduplication.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    result = {"family": family, "optimizer": config.get("optimizer", "adam").lower(),
              "learning_rate": float(config["learning_rate"]),
              "weight_decay": float(config.get("weight_decay", 0.0)),
              "epochs": int(config.get("epochs", 60)),
              "batch_size": int(config.get("batch_size", 128)),
              "gradient_clip": config.get("gradient_clip", 1.0),
              "schedule": config.get("schedule", "constant")}
    if result["schedule"] != "constant":
        result["min_lr_ratio"] = float(config.get("min_lr_ratio", 0.05))
        result["warmup_epochs"] = (int(config.get("warmup_epochs", 5))
                                    if result["schedule"] == "warmup_cosine" else 0)
    if family == "mlp":
        result.update(hidden_size=int(config.get("hidden_size", 256)),
                      head_dropout=float(config.get("head_dropout", 0.0)))
    elif family == "cnn":
        result.update(cnn_channels=list(config.get("cnn_channels", [32, 64, 128])),
                      dropout=float(config.get("dropout", 0.2)))
    elif family == "gru":
        layers = int(config.get("gru_num_layers", 1))
        result.update(gru_hidden_size=int(config.get("gru_hidden_size", 128)),
                      gru_num_layers=layers, sequence_axis=config.get("sequence_axis", "rows"),
                      head_dropout=float(config.get("head_dropout", 0.0)))
        if layers > 1:
            result["dropout"] = float(config.get("dropout", 0.0))
    elif family == "transformer":
        for key, default in (("patch_size", 4), ("d_model", 128), ("nhead", 4),
                              ("transformer_layers", 2), ("dim_feedforward", 256)):
            result[key] = int(config.get(key, default))
        result.update(dropout=float(config.get("dropout", 0.1)),
                      use_position_embedding=config.get("use_position_embedding", True),
                      encoder_init=config.get("encoder_init", "legacy"))
    return result


def effective_config_hash(family: str, config: dict) -> str:
    return _digest(effective_config(family, config))


def generate_candidates(base_config: dict) -> dict[str, list[dict]]:
    """Return exactly 36 recipes with fixed probes and balanced LR schedules.

    A single seeded RandomSampler supplies all random draws in documented family
    order. Rejected *sampling duplicates* are logged before the manifest is
    frozen. A failed training task must never be replaced through this function.
    """
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SAMPLER_SEED),
                               pruner=optuna.pruners.NopPruner(),
                               direction="maximize", storage=None)
    schedule_rng = random.Random(SCHEDULE_SHUFFLE_SEED)
    pool = {}
    for family in FAMILIES:
        baseline = _resolve(family, base_config)
        seen = set()
        candidates = []

        def append(config: dict, kind: str, sampling: dict | None = None) -> None:
            key = effective_config_hash(family, config)
            if key in seen:
                raise ValueError(f"Fixed probes collide for {family}; base configuration is incompatible")
            seen.add(key)
            candidates.append({"candidate_id": f"C{len(candidates)}", "family": family,
                               "config": copy.deepcopy(config), "effective_sha256": key,
                               "provenance": {"kind": kind, "generation_policy": GENERATION_POLICY,
                                              "sampler": "optuna.samplers.RandomSampler",
                                              "sampler_seed": SAMPLER_SEED,
                                              "schedule_shuffle_seed": SCHEDULE_SHUFFLE_SEED,
                                              **(sampling or {})}})

        append(baseline, "original_anchor")
        scheduled_probe = copy.deepcopy(baseline)
        _set_schedule(scheduled_probe, _scheduled(family))
        append(scheduled_probe, "schedule_probe")
        if family != "linear":
            targeted = copy.deepcopy(baseline)
            if family in ("mlp", "gru"):
                targeted["head_dropout"] = 0.2
            elif family == "cnn":
                targeted["weight_decay"] = 1e-3
            else:
                targeted["encoder_init"] = "independent_v1"
            append(targeted, "targeted_probe")

        schedules = (["constant", _scheduled(family)] if family == "linear" else
                     ["constant"] * 2 + [_scheduled(family)] * 3)
        schedule_rng.shuffle(schedules)
        for schedule in schedules:
            rejected = []
            for attempt in range(1, MAX_SAMPLING_ATTEMPTS + 1):
                trial = study.ask()
                config = copy.deepcopy(baseline)
                _set_schedule(config, schedule)
                config["learning_rate"] = trial.suggest_float(
                    f"{family}/learning_rate", *_LR_RANGES[family], log=True)
                config["weight_decay"] = trial.suggest_categorical(
                    f"{family}/weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
                if family in ("mlp", "gru"):
                    config["head_dropout"] = trial.suggest_categorical(
                        f"{family}/head_dropout", [0.0, 0.1, 0.2, 0.3])
                elif family == "cnn":
                    config["dropout"] = trial.suggest_categorical(
                        "cnn/dropout", [0.0, 0.1, 0.2, 0.3])
                elif family == "transformer":
                    config["dropout"] = trial.suggest_categorical(
                        "transformer/dropout", [0.05, 0.1, 0.2, 0.3])
                    config["encoder_init"] = "independent_v1"
                if effective_config_hash(family, config) in seen:
                    rejected.append({"trial_number": trial.number,
                                     "effective_sha256": effective_config_hash(family, config),
                                     "reason": "effective_duplicate_before_manifest_lock"})
                    continue
                append(config, "stratified_random", {
                    "trial_number": trial.number, "sampling_attempts": attempt,
                    "rejected_duplicate_draws": rejected,
                    "objective_observed": False,
                })
                break
            else:
                raise RuntimeError(f"Could not draw a unique {family} recipe after {MAX_SAMPLING_ATTEMPTS} attempts")
        pool[family] = candidates
    return pool


def schedule_lr(base_lr: float, update_index1based: int, total_updates: int,
                warmup_updates: int = 0, min_ratio: float = 0.05) -> float:
    """LR assigned BEFORE optimizer update ``t``; no scheduler state ambiguity.

    Without warmup: first update is ``base_lr``, final update ``base_lr*ratio``.
    With W warmup updates: t=1 is base/W, t=W is base, then cosine reaches the
    minimum at U. A single-update schedule uses base_lr (there is no decay span).
    The controller saves completed updates and resumes at t=completed+1.
    """
    for name, value in (("base_lr", base_lr), ("min_ratio", min_ratio)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if base_lr <= 0 or not 0 <= min_ratio <= 1:
        raise ValueError("base_lr must be positive and min_ratio must be in [0,1]")
    if type(total_updates) is not int or total_updates < 1:
        raise ValueError("total_updates must be a positive integer")
    if type(update_index1based) is not int or not 1 <= update_index1based <= total_updates:
        raise ValueError("update_index1based must be an integer in [1,total_updates]")
    if type(warmup_updates) is not int or not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be an integer in [0,total_updates)")
    t, horizon, warmup = update_index1based, total_updates, warmup_updates
    if warmup and t <= warmup:
        return float(base_lr * t / warmup)
    if horizon == 1:
        return float(base_lr)
    progress = ((t - warmup) / (horizon - warmup) if warmup else
                (t - 1) / (horizon - 1))
    return float(base_lr * (min_ratio + 0.5 * (1 - min_ratio) *
                            (1 + math.cos(math.pi * progress))))
