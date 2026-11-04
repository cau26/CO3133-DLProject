# Independent OPT3 statistics and original-label audit

**PASS**. Read-only canonical-ZIP audit; no model execution or new test inference.

Verified **61 tasks / 3,660 epochs / 338,400 prediction rows** against official training-label IDX and exact split IDs.

## Final exploratory old-validation results (n=3; sample SD)

| Model | Candidate | Accuracy (%) | Macro-F1 (%) | Selected epochs |
|---|---|---:|---:|---|
| linear | C1 | 86.8556 ± 0.0096 | 86.8513 ± 0.0239 | [40, 41, 35] |
| mlp | C7 | 91.4389 ± 0.1669 | 91.4483 ± 0.1528 | [56, 47, 58] |
| cnn | C5 | 93.2778 ± 0.0585 | 93.3080 ± 0.0267 | [44, 60, 55] |
| gru | C1 | 91.4000 ± 0.3279 | 91.3669 ± 0.3758 | [37, 28, 34] |
| transformer | C1 | 91.4333 ± 0.1202 | 91.4314 ± 0.0963 | [58, 48, 41] |

## Inner paired selection

Seed101 discovered the challenger; seed202 is the **one fresh confirmation seed**. Both use the same fixed inner-validation partition.

| Model | Challenger | Delta F1 at101 (pp) | Delta F1 at202 (pp) | Mean delta (pp) |
|---|---|---:|---:|---:|
| linear | C1 | +0.001971 | +0.034433 | +0.018202 |
| mlp | C7 | +0.513573 | +0.474332 | +0.493952 |
| cnn | C5 | +0.566773 | +0.093006 | +0.329890 |
| gru | C1 | +0.112975 | +0.123630 | +0.118302 |
| transformer | C1 | +0.581206 | -0.214597 | +0.183305 |

## Claim boundaries

- Final selected checkpoints are evaluated on previously seen 6000-image validation; not new blind test results
- Only one fresh confirmation seed: seed101 also chose each challenger; its delta is subject to selection bias
- Two-seed mean selection is a locked heuristic, not a significance test; Transformer is worse on fresh seed202
- n=3 final sample SD describes training-seed variation on one fixed selected validation set, not generalization uncertainty
- No matched final C0 anchors; historical CE-selected comparison mixes recipe and checkpoint-rule changes
- NLL aggregation validates provided per-image NLL consistency, not independently recomputed logits
- No claim of global hyperparameter optimum or guaranteed grade/job outcome

## Checks completed

- Official training-label IDX header and all four partition identities, counts and set relationships verified
- All 61 predictions CSVs bind every original_id and true label to official training-label IDX and exact phase partition order
- All 61 accuracy and macro-F1 metrics independently recomputed from integer confusion matrices
- All 61 supplied per-example NLL means agree with selected CE within 1e-7; logits were not available for independent NLL recomputation
- All 61 selected F1 epochs recomputed as max macro-F1, then min CE, then earliest exact tie; best CE epoch independently checked
- All 61 histories contain exactly epochs 1..60, finite validation selection metrics and phase-specific update counts
- All five locked discovery challengers independently rebuilt from F1 with candidate-id tie rule within absolute 1e-6
- All five selected recipes independently rebuilt from two matched seeds' mean F1 with 1e-6 anchor preference
- All five published final n=3 means and sample SDs independently rebuilt; no population-SD substitution

The JSON contains all per-task confusion matrices, class F1 values, input hashes, final summaries and descriptive historical differences. Existing return folders, reports, slides and official-test metrics were not modified.
