# CO3133 - Deep Learning and Its Applications

## Course and team

**University:** Ho Chi Minh City University of Technology, VNU-HCM  
**Faculty:** Faculty of Computer Science and Engineering  
**Course:** Deep Learning and Its Applications - CO3133, Semester 261  
**Instructor:** Lê Thành Sách  
**Group:** G-M10

| Member | Student ID | Role | Responsibility | GitHub |
|---|---|---|---|---|
| Trần Gia Lâm | 2352670 | Leader |  Ran the Google Colab Tesla T4 experiments; performed result and error analysis; verified the experimental outputs; integrated and prepared the report | [n1velo](https://github.com/n1velo) |
| Nguyễn Hữu Cầu | 2352129 | Member | Developed and reviewed the EDA and MLP components | [cau26](https://github.com/cau26) |

**Repository:** [cau26/CO3133-DLProject](https://github.com/cau26/CO3133-DLProject)

## Assignments

- [Assignment 1 - M1 Draft](assignment1.md)
- [Assignment 2](assignment2.md)
- [Assignment 3](assignment3.md)

## Assignment 1: current draft

- [Assignment 1 - M1 Draft](assignment1.md)
- [PDF report](G-M10_A1_Draft.pdf)
- [Result table](outputs/a1_t4/comparison.csv)
- [Original Colab notebook](notebooks/A1_Colab_Run.ipynb)
- [Run provenance and verification](COLAB_RUN.md)
- [AI Usage Disclosure](AI_USAGE.md)

| Model | Validation accuracy | Validation macro-F1 | Selected epoch |
|---|---|---|---|
| Linear | 86.62% | 0.8644 | 9 |
| MLP | 89.30% | 0.8924 | 9 |

Both checkpoints were selected by lowest validation loss. The test set has not been evaluated.

## Run on Google Colab with a Tesla T4

The recorded experiment used a Tesla T4, CUDA, Python 3.13.15 and PyTorch 2.8.0+cu128. The original notebook used a source ZIP. The instructions below use the repository checkout. Colab's available Python version can change over time.

### 1. Prepare the source and dependencies

In a new Colab notebook, select a T4 GPU runtime and run:

```python
!git clone https://github.com/cau26/CO3133-DLProject.git
%cd /content/CO3133-DLProject
%pip install -r requirements-a1.txt
```

If the repository directory already exists, reuse it or choose a new clone directory. If Colab requests a restart after installation, restart and repeat the directory and GPU-check cells.

[Commit 5d7430a](https://github.com/cau26/CO3133-DLProject/tree/5d7430a2c5b344ab3bb8af81e7bac7e99f9ab01b) archives the exact source used for the recorded run. Later changes update documentation and reporting artifacts. The model, data and training code is unchanged.

### 2. Check the GPU

```python
import torch
print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), "CUDA is unavailable; select a GPU runtime."
print("GPU:", torch.cuda.get_device_name(0))
```

Check that the device name identifies a Tesla T4 if you want to repeat the experiment on the same GPU model. Record any different device as part of the new run.

### 3. Train and evaluate

```python
!python run_a1.py --stage all --device cuda --out outputs/a1_t4_rerun
```

After training finishes, run:

```python
!python run_a1.py --stage evaluate --device cuda --out outputs/a1_t4_rerun
```

Fashion-MNIST is downloaded automatically into `data/`. The first command runs EDA, trains Linear and MLP, and saves their results. The second reloads the checkpoints for validation evaluation. The M1 pipeline does not evaluate the test set.

Use a fresh output directory for each rerun. Keep `outputs/a1_t4/` as the evidence for the recorded experiment. If `outputs/a1_t4_rerun/` already contains a run, change the directory name in both commands. In a regular terminal, remove the leading `!`.

The recorded T4 configuration is in [outputs/a1_t4/config.json](outputs/a1_t4/config.json). The default `configs/a1.json` selects CPU, so the commands explicitly pass `--device cuda`. The original notebook includes a numba/numpy compatibility warning; this pipeline does not use numba.

## Checkpoints and experiment files

- Linear: [checkpoint](outputs/a1_t4/linear/best.pt), [metrics](outputs/a1_t4/linear/metrics.json), [history](outputs/a1_t4/linear/history.csv), [predictions](outputs/a1_t4/linear/validation_predictions.csv)
- MLP: [checkpoint](outputs/a1_t4/mlp/best.pt), [metrics](outputs/a1_t4/mlp/metrics.json), [history](outputs/a1_t4/mlp/history.csv), [predictions](outputs/a1_t4/mlp/validation_predictions.csv)
- Shared: [split indices](outputs/a1_t4/split_indices.npz), [data summary](outputs/a1_t4/data_summary.json), [configuration](outputs/a1_t4/config.json)

Metrics, checkpoints, predictions, plots and the original notebook retain their recorded contents. The generated `outputs/a1_t4/draft_report.md` has been translated into English. Its numbers still come from the recorded run. See [COLAB_RUN.md](COLAB_RUN.md) for details.

## AI Usage Disclosure

The team used AI to develop ideas, review source code and improve the report. Tools, scope and verification are documented in [AI_USAGE.md](AI_USAGE.md).

## Milestone status

M1 has a working EDA stage, Dataset/DataLoader setup, training and validation loop, and Linear/MLP baselines with T4 results. A custom CNN, LSTM/GRU and Transformer are planned for the final milestone. Publishing this folder does not submit it to the LMS; follow the current M1 submission instructions there.
