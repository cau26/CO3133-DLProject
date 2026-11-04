# Matched confirmation and prospective RGB audit

The completed follow-up adds a controlled check of recipe effects and a project-unused RGB image cohort. It strengthens the evidence without establishing a general improvement for every selected model. The prespecified CNN comparison is small and inconsistent across seeds; the RGB interval includes zero. Both observations remain part of the project results.

## Verification and experimental scope

The returned archive contains 30 completed 60-epoch training trajectories and 90 checkpoint cells. An independent replay reconstructed accuracy, Macro-F1 and mean recorded negative log-likelihood from all 540,000 prediction rows. Every prediction file contains the same 6,000 original validation IDs and the corresponding official training labels. All 1,800 epoch-history rows have the expected example and update counts; minimum-CE and maximum-F1 epochs agree with the histories, including the earliest-epoch tie rule. The largest difference between independently reconstructed and reported cell metrics is 2.23e-16. This checks aggregation of stored per-example NLL; without grayscale logits it does not independently recompute their softmax probabilities.

The Fashion-MNIST run used an NVIDIA A100-SXM4-40GB, Python 3.13, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.4.2 and scikit-learn 1.8.0. The active core controller recorded 15,335.76 seconds (4.260 hours); this accounting excludes setup and the initial software smoke test. The run loaded the 60,000-image training partition and preserved the original 54,000/6,000 split. The official test was not evaluated.

A separate read-only CPU replay loaded all 90 unchanged checkpoints under PyTorch 2.10.0+cpu. Across 540,000 cell-image predictions, 11 labels differed from the original A100/PyTorch 2.11 execution; these need not be 11 unique images because policies can reuse the same weights and examples. The maximum per-example NLL difference was 0.042668 and the largest cell mean-NLL difference was 0.00001546. This is a numerical portability limitation, not an exact cross-runtime reproduction. Reported comparisons retain the original A100 predictions. The CNN primary interpretation is unchanged: one positive and two negative seed differences.

## Matched recipe and checkpoint design

Each family has two frozen recipe arms, C0 and the configuration selected by the earlier OPT3 search. Three fresh seeds (1009, 2027, 3037) are paired across arms. The engine, architecture dimensions, training horizon, initialization policy and data-order seed are held fixed within each pair. Recipe execution order alternates across seeds. Separate loader generators keep shuffle order matched when recipes consume different dropout random numbers.

Each trajectory contributes its minimum-validation-CE, maximum-validation-Macro-F1 and fixed-epoch-60 checkpoints. Thus 90 cells represent 30 fits rather than 90 independent fits. The CNN at epoch 60 is the prespecified primary comparison. The other families and both adaptive checkpoint policies are secondary descriptive comparisons. The experiment separates a recipe bundle from the checkpoint rule. It does not identify separate causal contributions of learning rate, dropout, weight decay or scheduling.

### Fixed-epoch results

All values below are Macro-F1. The uncertainty after ± is sample SD over three seeds; the delta SD is computed from the three paired differences. It is not a confidence interval.

| Family | C0, epoch 60 | Selected, epoch 60 | Paired delta | Positive seed pairs |
|---|---:|---:|---:|---:|
| Linear | 0.862804 ± 0.002114 | 0.866184 ± 0.000942 | +0.003380 ± 0.001431 | 3/3 |
| Mlp | 0.902176 ± 0.002360 | 0.911086 ± 0.002060 | +0.008909 ± 0.001150 | 3/3 |
| Cnn | 0.928232 ± 0.000873 | 0.929144 ± 0.002731 | +0.000912 ± 0.003511 | 1/3 |
| Gru | 0.902931 ± 0.002979 | 0.906151 ± 0.002785 | +0.003220 ± 0.003737 | 2/3 |
| Transformer | 0.902279 ± 0.002965 | 0.911743 ± 0.000868 | +0.009464 ± 0.002465 | 3/3 |

The primary CNN difference is +0.000912 F1 (0.0912 F1 percentage points), with paired SD 0.003511. Its seed-level differences are +0.004894, -0.001738 and -0.000419. Only one of three pairs favors C5. These results do not establish a stable recipe gain. The selected arm has a higher between-seed SD at the fixed final epoch (0.002731 versus 0.000873), making a strong robustness claim particularly inappropriate.

MLP and Transformer have larger positive final-epoch differences in all three pairs. Their effects are conditional on this reused validation split and the configurations already selected during earlier development. They are useful secondary replication evidence, rather than fresh test estimates or results of a new architecture search.

### Checkpoint dependence

| Family | C0 CE | Selected CE | C0 F1 | Selected F1 | C0 final | Selected final |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.865037 | 0.864811 | 0.868304 | 0.868065 | 0.862804 | 0.866184 |
| Mlp | 0.902002 | 0.910213 | 0.908665 | 0.913602 | 0.902176 | 0.911086 |
| Cnn | 0.928765 | 0.929400 | 0.929907 | 0.931258 | 0.928232 | 0.929144 |
| Gru | 0.909622 | 0.909587 | 0.912936 | 0.912780 | 0.902931 | 0.906151 |
| Transformer | 0.906300 | 0.905924 | 0.910215 | 0.913260 | 0.902279 | 0.911743 |

The CNN recipe difference becomes +0.001351 at the F1-selected checkpoint and +0.000635 at the CE-selected checkpoint. The former favors C5 in all three seed pairs, but it is a secondary, validation-selected comparison and does not override the weak primary result. The recipe-by-selector interaction is +0.000716 ± 0.001736, indicating that the apparent recipe difference depends partly on the selected checkpoint rule; three pairs are insufficient for a population inference.

For Linear, cosine scheduling improves epoch-60 F1 by +0.003380 while the recipe differences at CE/F1-selected checkpoints are approximately zero (-0.000226/-0.000239). For GRU the corresponding epoch-60 effect is +0.003220 but CE/F1 differences are -0.000035/-0.000156. These patterns are consistent with improved late-trajectory behavior rather than demonstrated superiority of the best available checkpoint. Transformer shows a similar dependence: +0.009464 at epoch 60, -0.000377 under minimum CE and +0.003044 under maximum F1.

F1 selection maximizes the same validation F1 reported in the table. Its score being at least as high as that of the CE-selected checkpoint on those images is expected by construction. This comparison describes the trade-off between validation objectives; it cannot demonstrate better independent generalization. For example, MLP C0 has mean CE 0.274567 at the CE checkpoint, versus 0.321768 at the F1 checkpoint and 0.405531 at epoch 60. Higher F1 and better-calibrated probability loss are distinct objectives.

## Prospective RGB cohort

The RGB audit is a separate six-class task. It freezes six already fitted C0/C4 models, seeds 314/2718/5772, image preprocessing, the seed-314 review policy and component-bootstrap analysis before preparing the new cohort. Candidate photographs come from previously excluded Fashionpedia training license IDs 3/5. The first 2,000 source-URL groups in a fixed SHA256 ranking are committed before target-category filtering; excluded groups are not replaced. Recorded prior image exposure and transitive exact/perceptual-duplicate components touching historical images are excluded.

The final cohort contains 2,766 annotated crops in 1,826 source/duplicate components. Class supports are 182 shirt/blouse, 856 top/T-shirt/sweatshirt, 300 jacket, 476 pants, 257 skirt and 695 dress. These improve support over the earlier 327-crop assessment, but remain an imbalanced, same-source, license-shifted cohort. They do not constitute a new external or IID benchmark. The controls cannot exclude every undetected shared garment/person, foundation-model pretraining exposure or unrecorded prior view.

The completed CPU inference was performed locally before the student Colab run. The returned Colab package reuses its 27 frozen evidence files byte for byte; it contains no additional RGB fitting or independent RGB rerun. An independent replay from the returned six saved-logit payloads reproduced the class counts, NLL, calibration, review decisions and all 2,000 paired component-bootstrap draws.

| Cohort | C0 mean Macro-F1 | C4 mean Macro-F1 | C4 minus C0 | Descriptive paired 95% interval |
|---|---:|---:|---:|---:|
| Earlier assessment: 327 crops, 221 components | 0.774722 | 0.766383 | -0.008339 | [-0.038196, +0.018294] |
| Project-unused cohort: 2,766 crops, 1,826 components | 0.774494 | 0.781540 | +0.007046 | [-0.000360, +0.015124] |

The new point estimate favors C4, but its interval includes zero. The interval jointly resamples source/duplicate components, preserving both recipe predictions and all three fixed models. It describes component-sampling uncertainty conditional on these six fitted models. It neither estimates seed-population uncertainty nor proves equivalence, a positive effect or a deployment guarantee. The opposite signs on the two cohorts show why both are retained; the later cohort does not erase the earlier regression.

### Class-level behavior and frozen review policy

| Class | Crops | C0 mean F1 | C4 mean F1 | Difference |
|---|---:|---:|---:|---:|
| shirt_blouse | 182 | 0.5633 | 0.5659 | +0.0026 |
| top_tshirt_sweatshirt | 856 | 0.8123 | 0.8208 | +0.0085 |
| jacket | 300 | 0.7211 | 0.7395 | +0.0184 |
| pants | 476 | 0.9091 | 0.9185 | +0.0094 |
| skirt | 257 | 0.7840 | 0.7856 | +0.0016 |
| dress | 695 | 0.8572 | 0.8589 | +0.0017 |

The largest mean class-level change is for jacket (+0.0184 F1). Shirt/blouse remains the weakest category, at about 0.566 F1 for C4 despite 182 crops. A high aggregate accuracy should therefore not be read as uniformly strong six-class performance. These class analyses are descriptive and were not prespecified primary hypotheses.

The already frozen C4 seed-314 temperature and threshold accept 1,506 of 2,766 crops (54.4469% coverage); 1,445 accepted predictions are correct (95.9495% accepted accuracy). This conditional accuracy is not the accuracy of the complete dataset: the same model classifies all crops with 82.1041% accuracy and 0.783914 F1. The review policy sends 1,260 crops to review and still makes 61 errors among accepted cases. Its prior calibration risk bound does not automatically carry over to the license-shifted cohort.

## What the follow-up establishes

The project now contains matched recipe/selector evidence and a larger prospectively fixed RGB cohort with traceable exclusions, frozen model identities and grouped uncertainty. The primary CNN result remains inconclusive, while secondary MLP and Transformer final-epoch comparisons are favorable on the reused split. RGB C4 has a modest favorable new-cohort point estimate, with uncertainty that still includes zero. These are research findings, including negative and mixed results, rather than a claim that every upgrade increases accuracy.

The remaining limits are substantive: only three seed pairs, development exposure to Fashion-MNIST validation images, a six-class crop task rather than detection or full-image recognition, one same-source RGB shift and local service validation. The expanded audit does not resolve the previously recorded broad ONNX parity failure or establish production service reliability. The evidence supports controlled experimentation, traceable evaluation and an end-to-end local engineering prototype; production operation and broader generalization remain untested.
