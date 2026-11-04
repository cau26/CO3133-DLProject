# Matched confirmation and prospective image audit

This is a separately versioned follow-up to the completed project. Historical
measurements, the original official test and existing reports remain unchanged.
The new experiment produces measurements only after execution; software tests
and a locked protocol are not experimental results.

## Recipe and checkpoint factors

Each of the five Fashion-MNIST families has two preselected recipes: its original
C0 and its frozen OPT3-selected configuration. Three fresh training seeds
(1009, 2027, 3037) are paired across the recipes. All 30 trajectories use the
same 54,000/6,000 split, unchanged engine and 60-epoch horizon.

Each trajectory supplies minimum-CE, maximum-Macro-F1 and fixed-epoch-60
checkpoints. This gives 90 checkpoint cells from 30 fits. Recipe differences
under a fixed checkpoint policy and checkpoint-policy differences within a
trajectory are reported separately. The interaction is
`(selected_F1 - C0_F1) - (selected_CE - C0_CE)`.

The principal family is the previously selected CNN. Its primary contrast is
selected recipe minus C0 at fixed epoch 60; the remaining families and CE/F1
factor contrasts are secondary descriptive comparisons. All seeds are retained.
This isolates the combined recipe from checkpoint selection, rather than
separate effects of learning rate, scheduler, dropout and weight decay.

The original validation images have already influenced project development.
Their new scores are matched retrospective evidence, not a fresh blind estimate.
The official test is never loaded. Sample SD over three seeds describes this
fixed split and does not establish population uncertainty or significance.

## Prospective RGB license cohort

Six already fitted C0/C4 checkpoints, their seed identities, the existing C4
seed 314 calibration policy, preprocessing, code and analysis are frozen before
preparing or evaluating the new image cohort. There is no new RGB fitting,
threshold fitting, candidate replacement or seed selection in this follow-up.

The candidate pool consists of official Fashionpedia training photographs with
license IDs 3/5, previously excluded by the project's 0/1/6 image filter. Source
URL groups are ranked by SHA256 with seed 20261029. The first 2,000 groups are
committed before target-category filtering or image validation; exclusions are
not replaced. All eligible six-class annotated crops from those groups are used.

Recorded image IDs, filenames, URL groups, preview lists and quarantines are
excluded. Full historical 0/1/6 sources, including pre-quarantine images, form
the pixel/near-duplicate reference. Transitive exact-pixel/dHash components
touching a historical image remove the new component. Fixed class-support
gates are checked before inference. The process does not display new crops for
subjective selection. Detected duplicates are proxies rather than a guarantee
against every shared person, outfit or undetected near-duplicate.

The primary endpoint is C4 minus C0 mean Macro-F1 over the three paired fixed
seeds. Every seed is reported. A 2,000-replicate paired source/duplicate-component
bootstrap describes image-sampling uncertainty conditional on these six fitted
models. The existing C4 seed 314 review policy is a secondary coverage/accuracy
diagnostic; its earlier calibration bound does not certify this shifted cohort.

This is a prospectively frozen, project-unused, same-source license-cohort
audit. It is neither an external dataset nor an IID test, and does not rule out
foundation-model pretraining exposure or unrecorded prior views. Once its
outcomes are inspected, future model changes need another untouched cohort.

Images and prepared crops remain outside publication/result packages. The new
cohort is for noncommercial research; source/license records are retained.
Metadata screening does not establish individual image-rights clearance.

## Completion and accounting

The Fashion controller has a cumulative active cap of 4.75 hours. The combined
controller has a five-hour soft cap including raw preparation and RGB inference;
dependency installation and initial Drive/source setup are excluded. A running
epoch and graceful shutdown may cross the nominal boundary. Completion requires
30 committed trajectories and a completed six-model RGB audit. Time stops and
disconnects produce explicitly partial evidence and preserve committed work.

The study tests specific weaknesses and may find no improvement. A higher
grade, a positive effect and a future deployment accuracy are not outcomes that
can be guaranteed by its protocol.

## Sources

- [Fashion-MNIST official dataset](https://github.com/zalandoresearch/fashion-mnist).
- [Fashionpedia images, annotations and test metadata](https://github.com/cvdfoundation/fashionpedia).
- [Cawley and Talbot, selection bias in performance evaluation](https://jmlr.org/papers/v11/cawley10a.html).
- [Recht et al., evaluation on newly collected images](https://arxiv.org/abs/1902.10811).
- [Fashionpedia image/annotation terms](https://fashionpedia.github.io/home/data_license.html).
- [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) and [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/).
