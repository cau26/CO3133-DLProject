# CO3133 · Deep Learning and Its Applications

Group **G-M10**, Semester 261 · Ho Chi Minh City University of Technology, VNU-HCM · Faculty of Computer Science and Engineering · Instructor **Lê Thành Sách**.

We compare five PyTorch architecture families on Fashion-MNIST: Linear, MLP, CNN, GRU and Transformer. The project combines a shared data pipeline, recorded checkpoints, experiment control and analysis of accuracy, class errors and computational cost. All five Fashion-MNIST models train without pretrained weights.

**[Website source](index.html) · [Final study](assignment1-final.md) · [Source repository](https://github.com/cau26/CO3133-DLProject)**

This A1 project is stored in `assignment1/a1_m2/`. Run the commands below from that directory.

## Team

| Member | Student ID | Main responsibilities |
|---|---|---|
| [Trần Gia Lâm](https://github.com/n1velo) | 2352670 | Team lead; experimental coordination, Colab training, result analysis and final integration. |
| [Nguyễn Hữu Cầu](https://github.com/cau26) | 2352129 | Data analysis, EDA, MLP and assigned technical sections. |

## Matched follow-up and prospective image audit

The follow-up completes **30 matched 60-epoch fits**, using seeds 1009, 2027 and 3037 with the original C0 and frozen OPT3-selected recipes. Three checkpoint policies yield **90 evaluations**. The predeclared primary CNN epoch-60 difference is **+0.000912 F1**, with paired sample SD **0.003511** and a positive effect in only one of three pairs. This does not establish a stable CNN recipe advantage. MLP and Transformer improve in all three secondary fixed-horizon pairs. All Fashion-MNIST follow-up results remain on previously inspected validation images.

A prospectively locked RGB audit evaluates six fixed models on **2,766 project-unused crops from 1,826 source/duplicate components**. C0/C4 mean F1 is **0.774494/0.781540**. The paired difference **+0.007046** has component-bootstrap interval **[-0.000360, +0.015124]**, leaving the positive population effect unresolved. The preset C4 policy accepts 1,506 crops, with 1,445 correct: **95.95% accepted accuracy at 54.45% coverage**. This same-source license shift is separate from the earlier assessment and from Fashion-MNIST.

[Follow-up results](confirmatory-results.html) · [Full methods and tables](docs/a1_extension_research/CONFIRMATORY_RESULTS_20261030.md) · [Locked protocol](docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md)

## Earlier optimization study

The optimization study completed **61 tasks × 60 epochs**: 36 discovery, 10 confirmation and 15 final tasks. Each selected recipe is fit with seeds **42, 43 and 44**. Checkpoints maximize validation macro-F1, with lower cross-entropy and then the earliest epoch breaking ties.

| Model | Recipe | Accuracy (%) mean ± SD | Macro-F1 mean ± SD | Parameters |
|---|---|---|---|---|
| Linear | C1 | 86.8556 ± 0.0096 | 0.86851 ± 0.00024 | 7,850 |
| MLP | C7 | 91.4389 ± 0.1669 | 0.91448 ± 0.00153 | 203,530 |
| CNN | C5 | 93.2778 ± 0.0585 | 0.93308 ± 0.00027 | 93,962 |
| GRU | C1 | 91.4000 ± 0.3279 | 0.91367 ± 0.00376 | 61,962 |
| Transformer | C1 | 91.4333 ± 0.1202 | 0.91431 ± 0.00096 | 274,954 |

Mean ± sample SD is across three training seeds on the same previously seen 6,000-image validation set. These are checkpoint-selected validation scores, not new official-test results or confidence intervals. CNN C5 has the highest observed mean; the study does not establish a global optimum or universal architecture ranking.

## Original frozen official-test comparison

Before optimization, five 20-epoch models were selected by minimum validation loss and frozen together. They use one training seed and the official 10,000-image test partition.

| Model | Official-test accuracy | Official-test macro-F1 | Parameters |
|---|---|---|---|
| Linear | 84.68% | 0.845902 | 7,850 |
| MLP | 88.74% | 0.886994 | 203,530 |
| CNN | 89.59% | 0.895242 | 93,962 |
| GRU | 89.14% | 0.891450 | 61,962 |
| Transformer | 88.94% | 0.888288 | 274,954 |

This table remains unchanged. Later validation scores use different recipes and checkpoint selection; they cannot be interpreted as official-test gains.

## GarmentLab real-image extension

A separate six-class Fashionpedia extension uses pretrained MobileNetV3-Small. Across three paired seeds, fine-tuning records assessment Macro-F1 0.7747 ± 0.0149, compared with 0.6770 ± 0.0055 for a frozen backbone. Assessment has 327 crops from 221 source groups. The frozen review policy gives 92.54% accepted accuracy at 69.72% coverage, below its empirical 95% calibration target.

[Results page](garmentlab-results.html) · [Reproduction and local inference](docs/garmentlab/REPRODUCIBILITY.md) · [Verified extension data](output/garmentlab_final/data/publication_data.json)

The subsequent 20-fit controlled study selected dropout 0.4 (C4) on fresh confirmation seeds. On the reused assessment, its mean Macro-F1 is 0.7664 ± 0.0226, versus 0.7747 ± 0.0149 for the paired C0 anchor; this does not establish a classification gain. A disjoint calibration procedure gives the preset C4 seed314 95.05% accepted accuracy at 55.66% coverage. The statistical risk bound is conditional on assumptions, not a future-accuracy guarantee. The PyTorch CPU demo passes the expanded numerical and service checks. ONNX passes the original 64-case audit but fails the broader 327-crop 1e-4 logit criterion; it remains experimental and is not the primary serving backend.

[Methods and results](output/pdf/G-M10_A1_Report_Reviewed_20261104.pdf) · [Recorded v2 summary](output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json)

## Read the project

- [Final report PDF](output/pdf/G-M10_A1_Report_Reviewed_20261104.pdf)
- [Editable LaTeX project](output/latex/G-M10_A1_Overleaf_Reviewed_20261104.zip)
- [Slides PowerPoint](output/presentation/G-M10_A1_Slides_Reviewed_20261104.pptx) · [Slides PDF](output/presentation/G-M10_A1_Slides_Reviewed_20261104.pdf)
- [Vietnamese presentation script](docs/a1_submission/VIDEO_SCRIPT_REVIEWED_20261104_VI.md)
- [A100 notebook](notebooks/A1_Optimization_A100.ipynb)
- [Reproduction and checkpoint loading](docs/a1_submission/REPRODUCIBILITY_FINAL.md)
- [Verified publication data](output/a1_final/data/publication_data.json)
- [Matched training and prospective RGB audit](confirmatory-results.html)
- [Matched confirmation notebook](notebooks/CO3133_Confirmatory_A100.ipynb)
- [Completed follow-up methods and full tables](docs/a1_extension_research/CONFIRMATORY_RESULTS_20261030.md)
- [GarmentLab controlled-study notebook and recorded results](notebooks/GarmentLab_A100_ImproveToday.ipynb)
- [GarmentLab methods, reproduction and local service](docs/garmentlab/REPRODUCIBILITY.md)

## Reproduce the optimization study

Use an NVIDIA A100 CUDA runtime with a compatible PyTorch/torchvision pair. The measured environment records PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3 and Optuna 4.8.0. Exact source and environment records accompany the run export.

```bash
python -m pip install -r requirements-a1-optimization.txt
python tools/run_a1_optimization.py --stage plan --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage smoke --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage run --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage export --out outputs/opt3_reproduction
```

The inner split is 48,600/5,400 images drawn only from the original 54,000 training images. Recipe selection compares an anchor and challenger across discovery seed 101 and fresh confirmation seed 202. The final 54,000/6,000 fits use selected recipes and seeds 42–44. No official-test dataset is opened by this runner.

Rerun the same command and output directory to resume committed work. Use a new directory for an independent experiment and restore the recorded source/environment for exact recovery. Completed task hashes and source/split bindings prevent silent replacement. See the [reproducibility guide](docs/a1_submission/REPRODUCIBILITY_FINAL.md) for reconstruction details.

For the original five-model pipeline, see the [reproducibility guide](docs/a1_submission/REPRODUCIBILITY_FINAL.md) and [configs/a1_final.json](configs/a1_final.json). Exact historical reconstruction uses the source snapshot recorded by its export. Evaluation commands report validation; official-test evaluation is a separate frozen protocol.

## Evidence and limits

- [Artifact/checkpoint review](verification/OPT3_RETURN_REVIEW_1d2d83edf1dc.md): 715 manifest hashes and 122 strict CPU checkpoint loads.
- [Independent statistics review](verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.json): 338,400 saved prediction rows, histories, selection and aggregates checked.
- Recorded controller time is 8.34 active hours under a seven-day upper cap; this is not seven days of GPU training.
- Only one fresh seed confirms each challenger; discovery-seed reuse leaves selection bias. Transformer loses to C0 on fresh seed 202 despite winning the declared two-seed mean rule.
- Final training-seed SD does not measure uncertainty over splits or unseen datasets. The 61-fit phase did not include matched final C0 anchors. The later 30-fit follow-up supplies those controls on the reused validation split.

## Verify the completed follow-up

```bash
python tools/verify_confirmatory_evidence.py
```

The [recorded A100 notebook](notebooks/CO3133_Confirmatory_A100.ipynb) uses the [immutable source archive](assets/frozen_sources/CO3133_Confirmatory_Source_20261029.zip). Its code preserves the executed protocol.

This standard-library command verifies the retained payload hashes and all 90 checkpoint identities, then reconstructs the 90 Fashion-MNIST scores from 540,000 prediction rows. It checks all 1,800 history rows and checkpoint decisions without loading model objects or the official test. The compact publication omits optimizer/RNG recovery states; exact resume uses the original complete run directory. [Independent records](output/confirmatory_return_20261030/independent/INDEPENDENT_STATISTICS.json) document the separate official-training-label check and [CPU portability replay](output/confirmatory_return_20261030/independent/CHECKPOINT_CPU_REPLAY.json). CPU/PyTorch 2.10 changes 11 of 540,000 cell-image predicted labels relative to the A100/PyTorch 2.11 export; original A100 measurements remain the reported results.

## CPU software checks

```bash
python -m pip install -r requirements-public-tests.txt
python -m pytest tests -q
```

The CPU profile covers the architecture, study controllers, API and release checks. Historical notebook tests use fake controllers and preserve the archived executable cells. Their required notebooks and the frozen original baseline reference are included. Tests requiring optional raw training data or a separate reference-service release declare their skips explicitly.

The release gate in `tools/verify_public_package.py` checks a supplied publication ZIP against its manifest, frozen source and test fixtures. Its `--software` option runs the complete CPU suite from a fresh extraction and verifies that imports resolve to that extracted source. These checks do not execute the measured A100 study or create a new model-performance result.

## Selective prediction by garment class

The pooled accepted accuracy of 95.95% conceals class-dependent operating risk. With the fixed C4 seed314 policy, true shirt/blouse crops have **18.13% coverage and 66.67% accepted accuracy** (22/33 accepted crops correct). In the separate grouping by emitted labels, accepted jacket outputs have **86.36% precision** (76/88). All eleven accepted true-shirt errors are labeled as other categories, so changing only the predicted-shirt threshold would not address those cases. Both tables concern one fixed model, use different denominators and remain post-hoc diagnostics. No weights, temperature or threshold changed.

[Six-class policy tables](confirmatory-results.html#policy) · [Counts, uncertainty and definitions](output/policy_by_class_20261030/POLICY_BY_CLASS.json) · [Accepted confusion matrix](output/policy_by_class_20261030/accepted_confusion.csv)

Recompute the diagnostic from the saved labels and logits:

```bash
python tools/analyze_policy_by_class.py
```

## Course assignment index

| Assignment | Focus | Materials |
|---|---|---|
| 1 | Image classification and architecture comparison | [Final study](assignment1-final.md) |
| 2 | Large-scale data and specialized tasks | [Existing assignment page](../../assignment2/a2_m1/assignment2.md) |
| 3 | Multimodal deep learning | [Existing assignment page](../../assignment3/a3_m1/assignment3.md) |

## AI usage

AI assisted LaTeX writing and pipeline development. [AI_USAGE.md](AI_USAGE.md).

## Local website preview

```bash
python tools/build_a1_pages.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/`. The static pages use relative URLs and a shared stylesheet, with no installation or frontend build required. Existing M1/A2/A3 source pages remain intact.
