# OPT3 returned run: independent artifact review

**PASS.**61/61 tasks complete;122 best checkpoints safely loaded on CPU with `weights_only=True`, strict state matching and finite weights. No training or forward inference was run.

Archive SHA-256: `1d2d83edf1dc0608ee25dc1f4a9bc8ad6c1ca61c8762140f3cff924e9f365610`.
Bound runtime source: `98ce5a0c08d90ef4fd46b62b426ee9d4f50f80677fdc11157dfe7d71a41a915e`.
Study manifest: `8ee891e30ad411818f664db31e74f923cf3730a355d4d9687e7234e0f80eee46`.

## Verified evidence

-716 archive files;715 export-manifest hashes;25 exact source snapshots and matching normalized canonical source files.
-36 discovery +10 confirmation +15 final tasks;60epochs each (3,660 task-epochs). Every COMPLETE seals8 artifacts and matches the study/task/split/environment binding.
-Both best-F1 and best-CE checkpoints for every task match the stored task, binding, epoch and validation metrics. Parameter counts and finite tensors match canonical models, with no forward pass.
-Current epoch60 and previous epoch59 transaction pointers/files checked by hash and binding; transaction pickle was not loaded.
-Fixed split54000/6000; inner48600/5400 partitions the original54k only. Raw training dataset SHA-256 and all four saved class-count arrays match.
-Frozen36candidate generation reproduced exactly; canonical read-only status validates challenger/selection locks and61task/summary completion hashes.
-All716 original input files remain byte-identical after the review.

## Saved final exploratory validation summary

These are old6k validation results used for checkpoint selection, not a new official test. Macro-F1 means/sample SD below are the saved summary; the separate statistics audit recomputes the prediction scores and aggregates.

| Family | Recipe | Seeds | Accuracy mean | Macro-F1 mean | Macro-F1 sample SD | Selected epochs |
|---|---|---|---:|---:|---:|---|
| linear | C1 |42,43,44| 86.8556% | 0.868512533 | 0.000238702 | 40,41,35 |
| mlp | C7 |42,43,44| 91.4389% | 0.914483134 | 0.001527710 | 56,47,58 |
| cnn | C5 |42,43,44| 93.2778% | 0.933080472 | 0.000266825 | 44,60,55 |
| gru | C1 |42,43,44| 91.4000% | 0.913668540 | 0.003758156 | 37,28,34 |
| transformer | C1 |42,43,44| 91.4333% | 0.914314472 | 0.000963098 | 58,48,41 |

## Recorded environment and timing

Student environment: NVIDIA A100-SXM4-40GB; torch 2.11.0+cu130; torchvision 0.26.0+cu130; CUDA 13.0; cuDNN 92700; Optuna 4.8.0.
Local structural audit: torch 2.10.0+cpu on CPU. This is a different runtime used only to check saved state compatibility.
Active controller clock: 30029.089649s (8.341414h), below the604800s/7day upper budget.
Summed recorded training loops: 27037.363462s; validation: 2423.788840s; completed fit timers: 29467.814237s.
Attempt ledger: 61 records;0 failures;0 incomplete attempts.

## Limits

- This is artifact integrity/structural review, not authentication of the operator or execution process.
- No training, prediction inference, independent validation loss reproduction or timing remeasurement was performed.
- Both best checkpoint types were loaded with weights_only=True on local torch2.10.0+cpu CPU. That runtime differs from the recorded student A100 training runtime; successful loads establish tensor/schema compatibility, not identical numeric inference.
- Full transaction state files include Python/NumPy RNG pickle; this review did not deserialize them. Both current and previous pointers and transaction file hashes were verified instead.
- Validation prediction Accuracy/MacroF1/selection/aggregate recomputation is also being independently checked by the separate statistics audit; this receipt checks their task/checkpoint/source bindings and recorded summary locks.
- Discovery/confirmation use48600training+5400inner-validation images from the original54k; final fitting uses54k and checkpoint selection on the previously seen6k. No new blind official test is present.
- Three final training seeds measure stochastic training variation on one fixed split; they do not measure split/population uncertainty.
- Historical minimum-CE-selected20epoch records differ from the60epoch maximum-F1 protocol, engine and recipes. Without fresh matched final C0 anchors, before/after differences do not isolate a causal recipe effect.
- The7day cap is an upper bound, not duration consumed. The recorded active controller clock excludes setup/smoke and may omit an abrupt final tail; task fit timers exclude transaction writes/finalization/discarded crash work.
- Checksums detect contradictions and changes, but unsigned saved metadata cannot prove that no unrecorded computation occurred.
