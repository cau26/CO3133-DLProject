# Prospective RGB source-cohort audit

Six fixed C0/C4 models were evaluated after their checkpoints, source code,
class mapping, crop processing, cohort rules and analysis had been locked.
This follow-up uses project-unused Fashionpedia training-source photographs
with license IDs 3/5. It is a same-source license-defined shift, rather than an
external dataset or an IID test. There was no model fitting, calibration or
test-driven model selection during this audit.

## Cohort and exclusion controls

The fixed source-group ranking selected 2,000 groups before category filtering.
Historical image IDs, URL groups, recorded previews, exact-pixel matches and
transitive dHash components were excluded without replacement. The final cohort
contains **2,766 crops and 1,826 source/duplicate components**. Class supports are
182 shirt/blouse, 856 top/T-shirt/sweatshirt, 300 jacket, 476 pants, 257 skirt and
695 dress crops. Historical pre-quarantine sources were retained as duplicate
bridges. The source-scope and decode gates passed before inference.

## Classification results

| Recipe | Training seed | Accuracy (%) | Macro-F1 |
|---|---:|---:|---:|
| C0 | 314 | 80.911 | 0.773627 |
| C0 | 2718 | 80.622 | 0.764349 |
| C0 | 5772 | 81.996 | 0.785505 |
| C4 | 314 | 82.104 | 0.783914 |
| C4 | 2718 | 81.453 | 0.778394 |
| C4 | 5772 | 82.176 | 0.782312 |

Across the three fixed seeds:

| Recipe | Mean accuracy (%) | Mean Macro-F1 | Sample SD of Macro-F1 |
|---|---:|---:|---:|
| C0 | 81.176 | 0.774494 | 0.010605 |
| C4 | 81.911 | 0.781540 | 0.002840 |

C4 minus C0 mean Macro-F1 is **0.007046**.
The paired 2,000-replicate component-bootstrap interval is
**[-0.000360, 0.015124]**. Its inclusion of zero does not establish a
positive population effect. The interval describes component sampling for the
six fixed fitted models; it does not include population variation over training
seeds or establish generalization to arbitrary photographs.

The earlier 327-crop assessment favored C0 (mean F1 0.774722 versus 0.766383).
This new cohort favors C4 descriptively. These are different image populations;
the mixed direction supports reporting both results rather than choosing a
model or discarding a cohort using observed evaluation outcomes.

## Frozen review policy

The existing C4 seed-314 policy, with temperature 5.345757 and confidence
threshold 0.88, accepts 1,506/2,766 crops (54.45% coverage). It correctly classifies
1,445 of the accepted crops (95.95% observed accepted accuracy). The threshold
was not adjusted on this cohort. The earlier calibration risk bound does not
certify this license-defined shift; the new percentage is descriptive.

## Scope and reproducibility

Inference used the six original FP32 checkpoints on local PyTorch 2.10 CPU.
No gradients or optimizer steps were executed. Every model was evaluated on the
same crops and the same paired component-bootstrap draws. All seeds are shown.
Independent replay from saved labels and logits reproduced the classification,
confidence and bootstrap statistics.

The project record supports unused status relative to recorded image exposure,
including four preview-credit lists. It cannot exclude unrecorded views or
ImageNet pretraining exposure. Exact-pixel and dHash checks can miss other
correlated subjects or outfits. No photographs or prepared image arrays are
distributed; source/license metadata and photograph-free statistical evidence
are retained for this noncommercial research audit.

The separate matched Fashion-MNIST experiment is defined in
[the confirmation protocol](CONFIRMATORY_PROTOCOL_20261029.md). Its new training
results are not asserted by this completed RGB audit.
