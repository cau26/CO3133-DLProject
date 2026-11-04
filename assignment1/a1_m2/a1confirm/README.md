# Matched recipe and checkpoint confirmation

This separate study repeats the frozen C0 and OPT3-selected recipes with three
new training seeds (`1009`, `2027`, `3037`) for all five Fashion-MNIST models.
It retains the original 54,000/6,000 training/validation split and the unchanged
60-epoch training engine. Thirty training trajectories produce ninety exported
checkpoint cells.

For each trajectory, minimum validation cross-entropy and maximum validation
Macro-F1 checkpoints are both retained. Their comparison therefore changes the
checkpoint rule without changing the training trajectory. Comparing the two
recipes at a fixed checkpoint rule estimates their combined recipe difference
under this matched protocol. It does not identify separate learning-rate,
dropout, weight-decay or schedule effects. The prespecified primary comparison
uses epoch 60 for both recipes; CE/F1 selection comparisons are secondary.

## Execution

The publication data file supplies existing C0 and selected configurations;
the study does not generate or replace candidates. A new study directory is
required. Full training requires an A100. The default active training budget is
4.75 hours and remains fixed across resumes.

```bash
python -m a1confirm plan --out outputs/a1_matched_confirmation --data-dir data --device cuda
python -m a1confirm smoke --out outputs/a1_matched_confirmation --data-dir data --device cuda
python -m a1confirm run --out outputs/a1_matched_confirmation --data-dir data --device cuda
python -m a1confirm status --out outputs/a1_matched_confirmation
python -m a1confirm export --out outputs/a1_matched_confirmation
```

`plan` also accepts `--publication`, `--families`, `--seeds` and `--hours` before
the manifest is locked. Five fresh seeds can be declared for a separate study;
an existing manifest cannot be edited to add seeds or change the budget.
`run --max-tasks 1` limits one invocation to one newly completed trajectory.

Each epoch is committed atomically. A resumed invocation skips verified
completed trajectories and resumes the last committed epoch of an interrupted
trajectory. Source, environment, recipes, seeds and split fingerprints must
match. The time limit is checked between epochs, so a running epoch and its
finalization can exceed the nominal boundary. Partial exports report
`INCOMPLETE` and their actual completed trajectory count.

## Evidence and interpretation

Each trajectory retains the engine receipt, both selected checkpoints, a
`final_epoch.pt` checkpoint, predictions for all three policies, and
`TRAJECTORY_COMPLETE.json`. The aggregate summary reports paired differences
using only seeds with both recipes complete. Sample SD describes seed
variability on this fixed split and is not a population confidence interval.

The old validation set has already been inspected and selects the CE/F1
checkpoints. Its metrics remain exploratory. Fresh seeds and a new run name do
not create an independent image benchmark. This package does not load the
official test partition, whose historical results have already been seen.

After every trajectory finishes, `EVALUATION_LOCK.json` binds all checkpoint
paths and hashes, the fixed-final primary contrast, and the secondary policies.
An external audit must lock its eligible image cohort, labels, grouping,
preprocessing, metrics and exclusions before inspecting model results. All
prespecified checkpoints must be evaluated without further model selection or
adaptation to that audit. The checkpoint roster alone is not evidence that
such an independent audit has taken place.
