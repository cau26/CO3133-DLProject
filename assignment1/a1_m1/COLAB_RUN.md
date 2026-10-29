# Google Colab Tesla T4 run record

## Reported experiment

- Dataset: Fashion-MNIST; seed 42; 54,000 training and 6,000 validation samples; 10 epochs per model.
- Device: Tesla T4, CUDA; Python 3.13.15; PyTorch 2.8.0+cu128.
- Original notebook: [notebooks/A1_Colab_Run.ipynb](notebooks/A1_Colab_Run.ipynb), byte-identical to the supplied Untitled4.ipynb.
- The output directory contains 25 files. Of these, 24 remain byte-identical to the original outputs.zip. The text of draft_report.md has been translated and its AI section updated; the reported values are unchanged.
- Environment metadata was recorded before Linear at 2026-09-22T16:51:30.724936+00:00, and before MLP at 2026-09-22T16:52:45.367547+00:00.

## Source and split provenance

| Item | Value |
|---|---|
| Git commit during training | null; the run used a ZIP without Git metadata |
| Source archive commit created after training | 5d7430a2c5b344ab3bb8af81e7bac7e99f9ab01b |
| Original source SHA-256 | 60f66b45930a48b305abd941d400125bcbe313626fcac45a8ce60d9e23e28c5c |
| Split SHA-256 | 0289c4940126abe338a376ea49eb866788d57cd67fa3ca17fe80139abb8c8fba |

The ten Python files in the archive commit match the checksums recorded in the T4 metrics. That commit was created after training and is not a commit captured by the running notebook.

The current package updates the documentation, `a1/report.py`, and the explanatory pipeline diagram. The model, data, EDA, and training implementations are unchanged from the archived source used for the recorded experiment.

Each model directory contains best.pt, metrics.json, history.csv, validation_predictions.csv and three figures. The summary table is [comparison.csv](outputs/a1_t4/comparison.csv). The supplementary checkpoint review is recorded in [verification/colab_review.json](verification/colab_review.json).

## Reports and recorded evidence

The M1 report is provided as the GitHub Pages source [assignment1.md](assignment1.md) and the formal [PDF report](G-M10_A1_Draft.pdf).

The file outputs/a1_t4/draft_report.md is the pipeline's shorter summary. Its English text was generated from the saved metrics after the original run. The full report contains the additional discussion and error analysis.

Current team-review information is recorded in [AI_USAGE.md](AI_USAGE.md). The human_review_status field in the metrics and notebook is the automatic status saved at run time and is retained as historical metadata. The supplementary checkpoint verification record documents Trần Gia Lâm's CPU inference recheck of the original Colab checkpoints.

The notebook preserves the original execution history, including a cell that printed None after reading incorrect metric keys and the following cell that used the correct keys. It also retains the numba/numpy compatibility warning. The pipeline does not use numba, and the recorded training completed.

## Rerunning safely

Follow the T4 instructions in [README](README.md). Use outputs/a1_t4_rerun or another unused output directory for a new experiment.

To evaluate the original checkpoints on T4, copy the output directory first. From the repository directory in Colab:

```python
from pathlib import Path
from shutil import copytree
original_run = Path("outputs/a1_t4")
check_run = Path("outputs/a1_t4_checkpoint_rerun")
copytree(original_run, check_run)
```

Then run:

```python
!python run_a1.py --stage evaluate --device cuda --out outputs/a1_t4_checkpoint_rerun
```

copytree stops if the destination exists. Choose a fresh name in both cells. Evaluation writes configuration and split metadata, so using a copy preserves the original directory. This step evaluates saved checkpoints on validation; it does not retrain the models or evaluate the test set.
