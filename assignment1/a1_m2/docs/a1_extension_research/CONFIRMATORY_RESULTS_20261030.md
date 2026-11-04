# Matched training and prospective image evaluation

CO3133 · Assignment 1 · Group G-M10

A matched study separates changes to the training recipe from changes to checkpoint selection. The predeclared primary comparison, CNN C5 versus C0 at epoch 60, has a small positive mean difference and opposite signs across the three training seeds. The evidence does not establish a consistent CNN recipe improvement. A separate, prospectively locked RGB audit adds image evidence outside recorded project exposure, with a positive C4 point estimate whose grouped uncertainty interval includes zero.

## Questions and experimental design

Five Fashion-MNIST families are each trained with the original C0 anchor and the recipe selected by OPT3, using fresh seeds 1009, 2027 and 3037. Each trajectory runs 60 epochs on the same 54,000 training images. Its 6,000-image validation partition has been used previously in the project. Minimum-validation-CE, maximum-validation-Macro-F1 and fixed-epoch-60 checkpoints are read from each trajectory. The completed run therefore comprises 30 fits, 1,800 training epochs and 90 checkpoint cells; the cells are correlated observations from the same fits.

CNN selected−C0 at fixed epoch 60 is the predeclared primary contrast. Other families, CE/F1 selectors and their interactions are secondary exploratory analyses. Mean ± SD below is the sample standard deviation across three training seeds on one fixed split; it is not a confidence interval over unseen images.

## All recipe and checkpoint cells

| Family / recipe | Minimum CE | Maximum F1 | Fixed epoch 60 |
|---|---|---|---|
| Linear / C0 (anchor) | 0.86504 ± 0.00091 | 0.86830 ± 0.00043 | 0.86280 ± 0.00211 |
| Linear / C1 (selected) | 0.86481 ± 0.00085 | 0.86806 ± 0.00010 | 0.86618 ± 0.00094 |
| MLP / C0 (anchor) | 0.90200 ± 0.00224 | 0.90866 ± 0.00141 | 0.90218 ± 0.00236 |
| MLP / C7 (selected) | 0.91021 ± 0.00369 | 0.91360 ± 0.00192 | 0.91109 ± 0.00206 |
| CNN / C0 (anchor) | 0.92877 ± 0.00187 | 0.92991 ± 0.00062 | 0.92823 ± 0.00087 |
| CNN / C5 (selected) | 0.92940 ± 0.00047 | 0.93126 ± 0.00079 | 0.92914 ± 0.00273 |
| GRU / C0 (anchor) | 0.90962 ± 0.00154 | 0.91294 ± 0.00180 | 0.90293 ± 0.00298 |
| GRU / C1 (selected) | 0.90959 ± 0.00500 | 0.91278 ± 0.00219 | 0.90615 ± 0.00278 |
| Transformer / C0 (anchor) | 0.90630 ± 0.00135 | 0.91022 ± 0.00172 | 0.90228 ± 0.00297 |
| Transformer / C1 (selected) | 0.90592 ± 0.00191 | 0.91326 ± 0.00125 | 0.91174 ± 0.00087 |

## Primary CNN result

CNN C0 records epoch-60 Macro-F1 0.92823, compared with 0.92914 for C5. The paired mean difference is +0.000912, with sample SD 0.003511 across three pairs. Only one pair favors C5. This does not support a claim of a stable recipe gain. C5 remains the highest observed family mean under maximum-F1 selection in this run, but that is a different, secondary comparison and does not establish an architecture-wide or unseen-data advantage.

| Paired seed | C5−C0 epoch-60 Macro-F1 |
|---|---|
| 1009 | +0.004894 |
| 2027 | -0.001738 |
| 3037 | -0.000419 |

## Recipe and checkpoint attribution

| Family | Epoch-60 recipe Δ | Recipe Δ at CE | Recipe Δ at F1 | Epoch-60 positive pairs |
|---|---|---|---|---|
| Linear | +0.00338 | -0.00023 | -0.00024 | 3/3 |
| MLP | +0.00891 | +0.00821 | +0.00494 | 3/3 |
| CNN | +0.00091 | +0.00064 | +0.00135 | 1/3 |
| GRU | +0.00322 | -0.00003 | -0.00016 | 2/3 |
| Transformer | +0.00946 | -0.00038 | +0.00304 | 3/3 |

Recipe effects depend on the checkpoint rule. Linear, GRU and Transformer have negative mean recipe differences at the CE-selected checkpoint despite positive mean differences at fixed epoch 60. For Transformer, the mean changes from −0.00038 at minimum CE to +0.00946 at epoch 60; the selected schedule may preserve late-training performance without improving the best CE checkpoint. MLP shows positive recipe differences under all three rules on these seeds. These observations concern complete recipe bundles; they do not identify separate causal effects of learning rate, decay or dropout.

Maximum validation F1 is at least as high as F1 at any other observed epoch by construction. Its advantage on the same validation set cannot demonstrate better generalization. For selected CNN, changing from minimum CE to maximum F1 adds +0.00186 mean validation F1, while mean CE rises from 0.19376 to 0.19817. Checkpoint choice trades metrics; it is not an independent improvement experiment.

The earlier OPT3 CNN mean, 0.93308 on seeds 42–44, and the new C5 maximum-F1 mean, 0.93126 on seeds 1009/2027/3037, describe different training seeds and recorded runs. Their difference is not a software regression diagnosis, a replacement test score or evidence that the earlier result was invalid. The matched C0 versus selected comparisons within this run are the appropriate recipe evidence.

## Prospective RGB image audit

The RGB protocol, source hashes, six existing C0/C4 checkpoints and cohort rule were frozen before new-image preparation and inference. A fixed hash rank samples source groups from Fashionpedia license IDs 3/5; recorded image/URL exposure and source/duplicate components connected to historical images are excluded. After annotation filtering, the audit contains 2,766 garment crops in 1,826 source/duplicate components. Each component may contain several classes, so class-component counts do not sum to the total. The six fixed models use training seeds 314, 2718 and 5772; their predictions are averaged as metrics, not combined into an ensemble.

| Fixed model recipe | Accuracy (%) | Macro-F1 |
|---|---|---|
| C0 | 81.176 | 0.77449 ± 0.01060 |
| C4 | 81.911 | 0.78154 ± 0.00284 |

C4−C0 mean Macro-F1 is +0.007046. A paired source/duplicate-component percentile bootstrap with 2,000 replicates gives the 95% interval [-0.000360, +0.015124]. The interval includes zero, so the audit does not establish a positive C4 effect. It is conditional on these six fixed models and the component-sampling procedure, not an interval for the population of possible training seeds. The earlier 327-crop assessment gave the opposite point-estimate direction; retaining both results exposes sensitivity to cohort composition instead of replacing an unfavorable table.

## Class errors and selective predictions

| Class | Crops | Components containing class | C0 mean class F1 | C4 mean class F1 |
|---|---|---|---|---|
| shirt blouse | 182 | 175 | 0.56330 | 0.56594 |
| top tshirt sweatshirt | 856 | 811 | 0.81232 | 0.82082 |
| jacket | 300 | 279 | 0.72108 | 0.73949 |
| pants | 476 | 461 | 0.90908 | 0.91846 |
| skirt | 257 | 253 | 0.78398 | 0.78562 |
| dress | 695 | 684 | 0.85720 | 0.85891 |

Shirt/blouse remains the weakest category: mean C4 class F1 is 0.56594, compared with 0.91846 for pants. Jacket has the largest observed class-F1 increase, +0.01841. This is descriptive class analysis across the fixed models, with no class-level multiplicity correction. More crops improve the evidence base but do not remove label ambiguity, class imbalance or source bias.

For the preselected C4 seed314 model, the existing temperature and threshold 0.88 are reused without fitting on the new cohort. The policy accepts 1,506 of 2,766 crops (54.45% coverage); 1,445 are correct (95.95% accepted accuracy). These are selective, descriptive measurements, not accuracy on every input or a transferred 95% risk guarantee. The calibration population differs from this cohort.

## Review policy by class

This post-hoc diagnostic concerns C4 seed314 alone, using the unchanged temperature 5.345757234366385 and acceptance rule max softmax(logits / temperature) ≥ 0.88. It does not average policies over training seeds or fit class-specific thresholds.

### Grouped by true category

| True category | Accepted / total | Correct / accepted | Review | Coverage (%) | Accepted accuracy (%) |
|---|---|---|---|---|---|
| shirt blouse | 33/182 | 22/33 | 149 | 18.13 | 66.67 |
| top tshirt sweatshirt | 431/856 | 406/431 | 425 | 50.35 | 94.20 |
| jacket | 82/300 | 76/82 | 218 | 27.33 | 92.68 |
| pants | 361/476 | 360/361 | 115 | 75.84 | 99.72 |
| skirt | 106/257 | 100/106 | 151 | 41.25 | 94.34 |
| dress | 493/695 | 481/493 | 202 | 70.94 | 97.57 |

The true-class table conditions on the annotated garment category. Shirt/blouse has 33 accepted crops among 182, with 22 correct: 18.13% coverage and 66.67% accepted accuracy. It contributes 11 of the 61 accepted errors despite only 2.19% of the accepted set. These labels are available for offline auditing and cannot determine a routing rule for a new unlabeled crop.

### Grouped by predicted output

| Predicted output | Accepted / predictions | Correct / accepted | Review | Coverage (%) | Accepted precision (%) |
|---|---|---|---|---|---|
| shirt blouse | 24/183 | 22/24 | 159 | 13.11 | 91.67 |
| top tshirt sweatshirt | 428/856 | 406/428 | 428 | 50.00 | 94.86 |
| jacket | 88/284 | 76/88 | 196 | 30.99 | 86.36 |
| pants | 362/484 | 360/362 | 122 | 74.79 | 99.45 |
| skirt | 101/243 | 100/101 | 142 | 41.56 | 99.01 |
| dress | 503/716 | 481/503 | 213 | 70.25 | 95.63 |

The predicted-class table conditions on the emitted label. Accepted jacket outputs have precision 76/88 = 86.36%, while accepted shirt/blouse outputs have precision 22/24 = 91.67%. All 11 accepted errors on true shirt/blouse crops are instead labeled jacket (five), top/T-shirt/sweatshirt (four) or dress (two). Tightening only the predicted shirt/blouse threshold would not directly address those errors. The two table denominators answer different questions and must not be interchanged.

Review contains 434 of the 495 full-cohort classification errors, but also 826 predictions that were already correct. Human review outcomes and costs were not measured. The JSON retains 2,000-replicate pointwise component-bootstrap intervals, conditional on this fixed model; they do not provide simultaneous classwise or future-population certification. The true-label grouping is not a direct test of probability calibration. No policy is modified using these inspected outcomes. A class-aware policy needs separate calibration and an untouched assessment cohort.

## Scope, runtime and reproducibility

The cohort is unused relative to the recorded project exposure and drawn from the same Fashionpedia source under a license-cohort shift. It is neither an external collection nor an IID benchmark. Previously unrecorded image views and ImageNet pretraining exposure cannot be excluded. Once these outcomes are inspected, further RGB tuning needs another untouched audit population. The Fashion-MNIST official test partition was not loaded; fresh training seeds cannot make the old validation images new.

The Fashion-MNIST controller records 4.260 active hours on NVIDIA A100-SXM4-40GB, Python 3.13, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.4.2 and scikit-learn 1.8.0. Setup and smoke checks are excluded from that clock. The prospective RGB inference was completed separately on CPU with PyTorch 2.10.0 and torchvision 0.25.0; its frozen evidence without source photographs was verified and reused by the A100 bundle. It was not a new RGB training or calibration experiment on Colab.

A separate CPU replay loads all 90 locked checkpoints without changing their weights. Compared with the original A100 prediction files, PyTorch 2.10 CPU produces 11 label differences across 540,000 checkpoint–image comparisons; these are not necessarily distinct images. The largest absolute per-cell mean NLL difference is 1.5463e-05, and the largest per-image NLL difference is 0.04267. The original A100 measurements are retained. This check documents numerical variation across hardware and software rather than claiming exact identity; the mixed-sign primary CNN conclusion is unchanged.

AI assisted LaTeX writing and pipeline development.

## Interpretation

The additional study strengthens attribution and reproducibility rather than establishing universal recipe superiority. It fills the missing matched-anchor comparison, reveals checkpoint-dependent conclusions and adds a prospectively locked image audit. The CNN primary contrast remains inconclusive across these seeds, and the RGB effect remains uncertain under grouped sampling. Independent images, additional training seeds and broader operational validation are still required for stronger deployment claims.

## Evidence

- [Run completion and scope](../../output/confirmatory_return_20261030/RESULT_STATUS.json)

- [Recipes, seeds, environment and source identities](../../output/confirmatory_return_20261030/fashion_confirmation/confirmation_manifest.json)

- [All checkpoint aggregates and paired contrasts](../../output/confirmatory_return_20261030/fashion_confirmation/analysis/summary.json)

- [Ninety locked checkpoint identities](../../output/confirmatory_return_20261030/fashion_confirmation/EVALUATION_LOCK.json)

- [Prospective RGB freeze](../../output/confirmatory_return_20261030/rgb_prospective/PROSPECTIVE_FREEZE.json)

- [RGB logits-derived results and component intervals](../../output/confirmatory_return_20261030/rgb_prospective/PROSPECTIVE_RESULTS.json)

- [CPU checkpoint replay and numerical differences](../../output/confirmatory_return_20261030/independent/CHECKPOINT_CPU_REPLAY.json)

- [Independent prediction and trajectory statistics](../../output/confirmatory_return_20261030/independent/INDEPENDENT_STATISTICS.json)

- [Class-conditional policy counts and pointwise uncertainty](../../output/policy_by_class_20261030/POLICY_BY_CLASS.json)

- [Confusion counts among accepted predictions](../../output/policy_by_class_20261030/accepted_confusion.csv)

- [Predeclared protocol](../../docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md)

- [AI usage](../../AI_USAGE.md)
