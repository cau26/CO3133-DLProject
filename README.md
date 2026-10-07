# CO3133 - Deep Learning and Its Applications

Course projects by Group G-M10, Semester 261, at Ho Chi Minh City University of Technology (HCMUT), VNU-HCM.

## Course and team

**Faculty:** Faculty of Computer Science and Engineering  
**Instructor:** Lê Thành Sách  
**Group:** G-M10

| Member | Student ID | Role | Responsibilities |
|---|---|---|---|
| [Trần Gia Lâm](https://github.com/n1velo) | 2352670 | Leader | A1 Colab runs and result analysis; A2 method, evaluation and compute planning; report integration and submission coordination. |
| [Nguyễn Hữu Cầu](https://github.com/cau26) | 2352129 | Member | A1 EDA and MLP; A2 dataset analysis, duplicate and split review, and EDA report sections. |

## Assignments

| Assignment | Task | Current status | Details |
|---|---|---|---|
| A1 | Fashion-MNIST classification | Linear and MLP M1 results recorded on Tesla T4 | [Assignment 1](assignment1.md) |
| A2 | PASCAL VOC 2007 object detection | Preliminary EDA available; instructor approval not recorded | [Assignment 2](assignment2.md) |
| A3 | Multimodal learning | Task not selected; page is a template | [Assignment 3](assignment3.md) |

## Assignment 2 - M1 Dataset Proposal

- [Proposal PDF](outputs/a2_eda/20261006T225426_686818Z/G-M10_A2_Proposal.pdf)
- [EDA notebook](notebooks/A2_EDA_VOC2007.ipynb)
- [Recorded EDA run](outputs/a2_eda/20261006T225426_686818Z/)
- [EDA observations and duplicate review](outputs/a2_eda/20261006T225426_686818Z/EDA_NOTES_TO_COMPLETE.md)

**Task:** detect objects in RGB images and predict their class labels, confidence scores and bounding boxes.

**Dataset:** [PASCAL VOC 2007](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/), with 20 object classes. The preliminary EDA uses the official `VOCtrainval_06-Nov-2007.tar` release.

The EDA covers class distribution, objects per image, image dimensions, bounding-box sizes, annotation checks, duplicate candidates and ground-truth examples. No image-level subsampling was applied.

| Recorded split | Images | All annotated objects | Valid non-difficult objects |
|---|---:|---:|---:|
| Trainval pool | 5,011 | 15,662 | 12,608 |
| Proposed training split | 4,013 | 12,449 | 10,014 |
| Proposed validation split | 998 | 3,213 | 2,594 |

The trainval pool includes 3,054 difficult objects. The local training and validation splits are derived from the official trainval pool; they are not the original VOC train/val partition.

**Data usage:** follow the publisher's [Database Rights statement](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/) and the applicable terms for the source images. The dataset is not presented as having a blanket MIT or CC-BY license.

### Split and review status

The preliminary split contains 4,013 training images and 998 validation images. It uses seed 42, multi-label iterative stratification and exact-duplicate grouping within the official trainval pool.

The written review identifies `000863` and `008364` as a true near-duplicate pair. The recorded split still places them in validation and training, respectively. Both images will be grouped into one split before main training. Split statistics and review records will then be updated together.

The review metadata is not yet synchronized: candidate rows are marked `reviewed` in the CSV, while the corresponding review fields in `quality_checks.json` remain `pending`. The official test set was not downloaded or evaluated in this EDA run; train/test overlap checks remain planned.

### Planned experiments

| Configuration | Plan |
|---|---|
| Lightweight baseline | SSDLite320 with a MobileNetV3-Large backbone initialized from ImageNet weights and new VOC detection heads. |
| Pretrained detector | Faster R-CNN ResNet-50-FPN initialized from COCO_V1 detection weights, with its final predictor adapted to the 20 VOC classes and background. |
| Controlled experiment | Compare frozen ResNet-50 features with full fine-tuning, holding the data split, initial predictor, seed and training/evaluation settings fixed. |

The primary metric will be **VOC2007 11-point interpolated mAP at IoU 0.5**. We will also report per-class AP, training time, inference latency and peak GPU memory, together with examples of missed objects, wrong labels, inaccurate boxes and duplicate detections.

The target training environment is **Google Colab with one NVIDIA A100 GPU**. The initial proposal uses 20 epochs per run, with batch sizes of 8 for SSDLite and 2 for Faster R-CNN. The proposed compute budget is 30 GPU-hours, including checks and interrupted runs; it is a planning allowance, not a measured runtime. A pilot will measure runtime and memory use before the main runs; settings may be revised and recorded. No A100 training results or detector performance scores are available yet.

Main experiments will begin after the proposal is **Approved** or **Approved with conditions**, and after the split is finalized.

After approval, Lâm will implement Faster R-CNN and the freezing study, while Cầu will implement the SSDLite baseline. Both members will review the shared evaluator and contribute to result analysis.

### Milestones

| Milestone | Deadline, Vietnam time | Scope |
|---|---|---|
| M1 - Proposal | 7 Oct 2026, 23:59 | Task, dataset, preliminary EDA, split plan, methods, metrics and compute estimate. |
| M2 - Draft | 28 Oct 2026, 23:59 | Baseline results, pretrained model progress and preliminary analysis. |
| M3 - Final | 11 Nov 2026, 23:59 | Completed comparison, controlled study, error analysis and final deliverables. |

### Reproduce the preliminary EDA

Open [the recorded EDA notebook](notebooks/A2_EDA_VOC2007.ipynb) in Google Colab and run its cells from top to bottom. It installs the additional dependencies, downloads the official trainval archive, checks images and annotations, builds the proposed split, and exports tables, figures and run metadata.

If automatic download fails, upload the official archive to Colab and set `MANUAL_ARCHIVE_PATH` in the notebook. The VOC images are not stored in this repository. The notebook currently reproduces the preliminary workflow; the near-duplicate grouping correction is still required before main training.

## Assignment 1 - M1 Draft

- [Assignment page](assignment1.md)
- [PDF report](G-M10_A1_Draft.pdf)
- [Recorded Colab notebook](notebooks/A1_Colab_Run.ipynb)
- [Run provenance and verification](COLAB_RUN.md)
- [Experiment outputs](outputs/a1_t4/)

| Model | Validation accuracy | Validation macro-F1 | Selected epoch |
|---|---:|---:|---:|
| Linear | 86.62% | 0.8644 | 9 |
| MLP | 89.30% | 0.8924 | 9 |

Both checkpoints were selected by lowest validation loss. The test set has not been evaluated. A custom CNN, LSTM/GRU and Transformer are planned for the final milestone.

### Run Assignment 1 on Colab

Select a GPU runtime, then prepare the project:

```python
!git clone https://github.com/cau26/CO3133-DLProject.git
%cd /content/CO3133-DLProject
%pip install -r requirements-a1.txt

import torch
print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), "Select a GPU runtime."
print("GPU:", torch.cuda.get_device_name(0))
```

The recorded run used a Tesla T4. Record the actual GPU and software versions when rerunning, and use a new output directory:

```python
!python run_a1.py --stage all --device cuda --out outputs/a1_t4_rerun
!python run_a1.py --stage evaluate --device cuda --out outputs/a1_t4_rerun
```

Fashion-MNIST is downloaded automatically. Keep `outputs/a1_t4/` as the recorded evidence. Change `outputs/a1_t4_rerun` if that directory already contains a run. If Colab requests a restart after installation, restart and repeat the directory and GPU-check steps.

## AI Usage Disclosure

AI tools supported project planning, implementation assistance, and report drafting and review. Team members are responsible for verifying the code, technical claims and reported results they contribute. See [AI_USAGE.md](AI_USAGE.md) for usage records.

## Submission

Report, notebook and output links document the team's work. Complete the required LMS submission separately; publishing files on GitHub does not submit the assignment to the LMS.
