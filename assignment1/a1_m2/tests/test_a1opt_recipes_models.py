"""CPU recipe/initialization checks; no experiment training or test-set access."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import torch

from a1.train import build_model as legacy_build_model
from a1opt.engine import _schedule_clock
from a1opt.models import build_model
from a1opt.recipes import (FAMILIES, effective_config_hash, generate_candidates,
                           schedule_lr)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/a1_final.json").read_text(encoding="utf-8"))


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls.pool = generate_candidates(config())

    def test_pool_is_deterministic_bounded_balanced_and_unique(self):
        self.assertEqual(self.pool, generate_candidates(config()))
        self.assertEqual(tuple(self.pool), FAMILIES)
        self.assertEqual(sum(map(len, self.pool.values())), 36)
        for family, candidates in self.pool.items():
            expected = 4 if family == "linear" else 8
            self.assertEqual(len(candidates), expected)
            self.assertEqual([x["candidate_id"] for x in candidates],
                             [f"C{i}" for i in range(expected)])
            hashes = {x["effective_sha256"] for x in candidates}
            self.assertEqual(len(hashes), expected)
            self.assertEqual(sum(x["config"]["schedule"] == "constant"
                                 for x in candidates), expected // 2)
            for candidate in candidates:
                cfg = candidate["config"]
                self.assertEqual(cfg["epochs"], 60)
                self.assertNotIn("per_model", cfg)
                self.assertFalse({"seed", "split_seed", "validation_size", "device"}.intersection(cfg))
                self.assertEqual(candidate["effective_sha256"],
                                 effective_config_hash(family, cfg))
                if candidate["provenance"]["kind"] == "stratified_random":
                    self.assertFalse(candidate["provenance"]["objective_observed"])

    def test_fixed_probes_have_only_the_intended_recipe_change(self):
        for family, candidates in self.pool.items():
            baseline, scheduled = [copy.deepcopy(x["config"]) for x in candidates[:2]]
            self.assertEqual(baseline["head_dropout"], 0)
            self.assertEqual(baseline["encoder_init"], "legacy")
            self.assertEqual(baseline["schedule"], "constant")
            for key in ("schedule", "warmup_epochs", "min_lr_ratio"):
                baseline.pop(key)
                scheduled.pop(key)
            self.assertEqual(baseline, scheduled)
        self.assertEqual(self.pool["mlp"][2]["config"]["head_dropout"], .2)
        self.assertEqual(self.pool["gru"][2]["config"]["head_dropout"], .2)
        self.assertEqual(self.pool["cnn"][2]["config"]["weight_decay"], .001)
        self.assertEqual(self.pool["transformer"][2]["config"]["encoder_init"], "independent_v1")

    def test_effective_dedup_ignores_metadata_and_inactive_fields(self):
        original = self.pool["gru"][0]["config"]
        changed = {**original, "seed": 999, "split_seed": 2, "dropout": .9,
                   "min_lr_ratio": .4, "warmup_epochs": 21, "candidate_id": "other"}
        self.assertEqual(effective_config_hash("gru", original),
                         effective_config_hash("gru", changed))
        self.assertNotEqual(effective_config_hash("gru", original),
                            effective_config_hash("gru", {**changed, "head_dropout": .2}))

    def test_duplicate_sampling_is_recorded_before_manifest_freeze(self):
        class Trial:
            def __init__(self, number):
                self.number = number

            def suggest_float(self, *args, **kwargs):
                return .001 if self.number == 0 else .0011 + self.number * 1e-7

            def suggest_categorical(self, name, values):
                return values[0]

        class Study:
            next_number = 0

            def ask(self):
                trial = Trial(self.next_number)
                self.next_number += 1
                return trial

        with patch("optuna.create_study", return_value=Study()) as create:
            pool = generate_candidates(config())
        self.assertIsInstance(create.call_args.kwargs["sampler"], optuna.samplers.RandomSampler)
        self.assertIsInstance(create.call_args.kwargs["pruner"], optuna.pruners.NopPruner)
        self.assertIsNone(create.call_args.kwargs["storage"])
        provenance = pool["linear"][2]["provenance"]
        self.assertEqual(provenance["sampling_attempts"], 2)
        self.assertEqual(len(provenance["rejected_duplicate_draws"]), 1)

    def test_lr_update_boundaries_and_resume_index(self):
        self.assertEqual(schedule_lr(.001, 1, 22800), .001)
        self.assertAlmostEqual(schedule_lr(.001, 22800, 22800), .00005)
        self.assertAlmostEqual(schedule_lr(.001, 1, 22800, 1900), .001 / 1900)
        self.assertEqual(schedule_lr(.001, 1900, 22800, 1900), .001)
        self.assertAlmostEqual(schedule_lr(.001, 22800, 22800, 1900), .00005)
        sequence = [schedule_lr(.001, t, 100, 10) for t in range(1, 101)]
        self.assertTrue(all(a <= b for a, b in zip(sequence[:10], sequence[1:10])))
        self.assertTrue(all(a >= b for a, b in zip(sequence[9:], sequence[10:])))
        self.assertEqual(sequence[27:], [schedule_lr(.001, completed + 1, 100, 10)
                                        for completed in range(27, 100)])
        self.assertEqual(schedule_lr(.001, 1, 1), .001)
        for kwargs in ({"update_index1based": 0, "total_updates": 3},
                       {"update_index1based": 4, "total_updates": 3},
                       {"update_index1based": True, "total_updates": 3},
                       {"update_index1based": 1, "total_updates": 3, "warmup_updates": 3},
                       {"update_index1based": 1, "total_updates": 3, "min_ratio": float("nan")}):
            with self.assertRaises(ValueError):
                schedule_lr(.001, **kwargs)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pool = generate_candidates(config())

    def test_anchors_preserve_legacy_keys_parameters_rng_and_forward(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                cfg = self.pool[family][0]["config"]
                torch.manual_seed(101)
                legacy = legacy_build_model(family, cfg)
                rng = torch.get_rng_state().clone()
                candidate = build_model(family, cfg, 101)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(list(legacy.state_dict()), list(candidate.state_dict()))
                self.assertEqual(sum(x.numel() for x in legacy.parameters()),
                                 sum(x.numel() for x in candidate.parameters()))
                for key, value in legacy.state_dict().items():
                    self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
                images = torch.rand(2, 1, 28, 28)
                legacy.eval()
                candidate.eval()
                with torch.no_grad():
                    torch.testing.assert_close(legacy(images), candidate(images), rtol=0, atol=0)
                candidate.load_state_dict(legacy.state_dict(), strict=True)

    def test_external_head_dropout_changes_training_only_and_no_state_keys(self):
        for family in ("mlp", "gru"):
            with self.subTest(family=family):
                baseline = build_model(family, self.pool[family][0]["config"], 101)
                candidate = build_model(family, self.pool[family][2]["config"], 101)
                self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))
                images = torch.rand(4, 1, 28, 28)
                baseline.eval()
                candidate.eval()
                with torch.no_grad():
                    torch.testing.assert_close(baseline(images), candidate(images), rtol=0, atol=0)
                    baseline.train()
                    candidate.train()
                    self.assertFalse(torch.equal(baseline(images), candidate(images)))
                if family == "gru":
                    self.assertEqual(candidate.gru.dropout, 0.0)

    def test_independent_encoder_only_changes_encoder_and_preserves_global_rng(self):
        cfg = self.pool["transformer"][0]["config"]
        legacy = build_model("transformer", cfg, 101)
        after_legacy = torch.get_rng_state().clone()
        independent = build_model("transformer", {**cfg, "encoder_init": "independent_v1"}, 101)
        self.assertTrue(torch.equal(after_legacy, torch.get_rng_state()))
        for key, value in legacy.state_dict().items():
            if not key.startswith("encoder."):
                self.assertTrue(torch.equal(value, independent.state_dict()[key]), key)
        left, right = independent.encoder.layers
        for suffix in ("self_attn.in_proj_weight", "self_attn.out_proj.weight",
                       "linear1.weight", "linear2.weight"):
            self.assertFalse(torch.equal(dict(left.named_parameters())[suffix],
                                         dict(right.named_parameters())[suffix]), suffix)
        for layer in independent.encoder.layers:
            self.assertTrue(torch.equal(layer.norm1.weight, torch.ones_like(layer.norm1.weight)))
            self.assertTrue(torch.equal(layer.norm2.weight, torch.ones_like(layer.norm2.weight)))
            for name, parameter in layer.named_parameters():
                if name.endswith("bias"):
                    self.assertEqual(torch.count_nonzero(parameter).item(), 0, name)
        again = build_model("transformer", {**cfg, "encoder_init": "independent_v1"}, 101)
        for key, value in independent.state_dict().items():
            self.assertTrue(torch.equal(value, again.state_dict()[key]), key)


class ScheduleClockTests(unittest.TestCase):
    def test_explicit_schedule_fields_control_the_actual_clock(self):
        clock = _schedule_clock({"schedule": "warmup_cosine", "warmup_epochs": 3,
                                 "min_lr_ratio": .1}, 60, 380, .001)
        self.assertEqual(clock, {"schedule": "warmup_cosine", "total_updates": 22800,
                                 "warmup_updates": 1140, "min_ratio": .1, "base_lr": .001})
        self.assertAlmostEqual(schedule_lr(.001, 22800, 22800,
                                          clock["warmup_updates"], clock["min_ratio"]), .0001)
        final = _schedule_clock({"schedule": "warmup_cosine", "warmup_epochs": 5},
                                 60, 422, .0005)
        self.assertEqual(final["total_updates"], 25320)
        self.assertEqual(final["warmup_updates"], 2110)

    def test_schedule_fields_reject_silent_clamping_and_wrong_types(self):
        invalid = ({"schedule": "warmup_cosine", "warmup_epochs": 60},
                   {"schedule": "warmup_cosine", "warmup_epochs": 0},
                   {"schedule": "cosine", "warmup_epochs": 2},
                   {"schedule": "constant", "warmup_epochs": 1},
                   {"schedule": "warmup_cosine", "warmup_epochs": True},
                   {"schedule": "warmup_cosine", "warmup_epochs": 3.0},
                   {"schedule": "cosine", "min_lr_ratio": True},
                   {"schedule": "cosine", "min_lr_ratio": float("nan")},
                   {"schedule": "cosine", "min_lr_ratio": -0.1},
                   {"schedule": "cosine", "min_lr_ratio": 1.1})
        for cfg in invalid:
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                _schedule_clock(cfg, 60, 380, .001)


if __name__ == "__main__":
    unittest.main()
