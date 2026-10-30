# Google Colab NVIDIA A100-SXM4-40GB Run Record (Assignment 2 - M2 Draft)

## 1. Reported Experiment Summary

- **Dataset:** PASCAL VOC 2007 (`VOCtrainval_06-Nov-2007.tar`, MD5 `c52e279531787c972589f7e41ab4ae64`); seed `42`; finalized leak-free stratified split of **`4,014` training images** (`12,451` total boxes: `10,016` non-difficult, `2,435` difficult) and **`997` validation images** (`3,211` total boxes: `2,592` non-difficult, `619` difficult).
- **Hardware Environment:** Google Colab Linux (`Linux-6.6.159+-x86_64-with-glibc2.39`), CPU `Intel(R) Xeon(R) CPU @ 2.20GHz` (`12` logical cores, `4` PyTorch threads), GPU **`NVIDIA A100-SXM4-40GB`** (`42,404,478,976` bytes / `39.49 GB` VRAM), CUDA `13.0`.
- **Software Versions:** Python `3.13.16`, PyTorch `2.11.0+cu130`, Torchvision `0.26.0+cu130`, NumPy `2.1.3`, Pandas `2.2.3`, Matplotlib `3.10.0`, Pillow `11.3.0`, Scikit-learn `1.6.1`.
- **Executed Notebook:** [`notebooks/A2_Colab_Run.ipynb`](notebooks/A2_Colab_Run.ipynb), preserving the full cell-by-cell execution output from the Google Colab A100 session.
- **Output Directory:** [`outputs/a2_m2_a100/`](outputs/a2_m2_a100/) containing `37` verified artifact files (pilots, histories, metrics, per-class AP tables, error taxonomies, checkpoint verifications, and figures).

## 2. Source and Split Provenance

| Provenance Field | Recorded Value |
|---|---|
| `source_sha256` | `82d012d747c5aa0c2a80a0e70757ed84ff164c1ab7970c8d7cfab52bca3aaaa2` |
| `split_sha256` | `c60294fe4e914b27399ff065c9e246dcd3428440307c97cebf2febe139c2bd6c` |
| `train.txt` SHA-256 | `0315233dc9b4eb3e41ef21265b806196b45a16a9f0fd15a1492817fd22083da5` |
| `val.txt` SHA-256 | `69aa5327227d2dd98f6bd3db1b5ca74b1a40cb66cd0084b26f82f7b1cbeb7da7` |
| `test_evaluated` | `false` (Official VOC 2007 test set untouched) |

### Recorded Timestamps (UTC) on NVIDIA A100

| Pipeline Stage | Model / Component | Recorded UTC Timestamp | Main Result |
|---|---|---|---|
| Stage 1: EDA & Batch Check | `eda/batch_info.json` & `data_summary.json` | `2026-10-28T16:40:14Z` | Verified `4,014` train / `997` val images; saved `batch_preview.png` (`B=8`) |
| Stage 2: 200-Step Pilot | `ssdlite320` (`pilots/ssdlite320_pilot.json`) | `2026-10-28T16:41:08Z` | $t_{\text{step}}=110.92\text{ ms}$, $t_{\text{val}}=15.71\text{ s}$, pilot mAP@0.5 $= 8.81\%$, VRAM $= 462.6\text{ MB}$ |
| Stage 2: 200-Step Pilot | `fasterrcnn_frozen` (`pilots/fasterrcnn_frozen_pilot.json`) | `2026-10-28T16:41:48Z` | $t_{\text{step}}=77.55\text{ ms}$, $t_{\text{val}}=22.68\text{ s}$, pilot mAP@0.5 $= 65.02\%$, VRAM $= 1,238.1\text{ MB}$ |
| Stage 2: 200-Step Pilot | `fasterrcnn_finetune` (`pilots/fasterrcnn_finetune_pilot.json`) | `2026-10-28T16:42:27Z` | $t_{\text{step}}=84.98\text{ ms}$, $t_{\text{val}}=20.71\text{ s}$, pilot mAP@0.5 $= 35.81\%$, VRAM $= 3,325.9\text{ MB}$ |
| Stage 3: Main Training | `ssdlite320` (`20` epochs, `B=8`) | `2026-10-28T16:42:27Z` – `17:00:21Z` | Best epoch `18/20`: **Val mAP@0.5 = `51.53%`** (`0.515346`), fit wall `1,074.08 s` |
| Stage 3: Main Training | `fasterrcnn_frozen` (`12` epochs, `B=2`) | `2026-10-28T17:00:53Z` – `17:27:01Z` | Best epoch `12/12`: **Val mAP@0.5 = `83.63%`** (`0.836319`), fit wall `1,567.55 s` |
| Stage 3: Main Training | `fasterrcnn_finetune` (`12` epochs, `B=2`) | `2026-10-28T17:27:41Z` – `18:01:54Z` | Best epoch `10/12`: **Val mAP@0.5 = `75.07%`** (`0.750707`), fit wall `2,053.11 s` |
| Stage 4: Checkpoint Verification | Reload `best.pt` for all 3 models | `2026-10-28T18:02:35Z` – `18:03:54Z` | `matches_saved = true` (`recomputed_map50 == saved_map50` for all 3 models) |

## 3. Checkpoint Verification Summary

During `run_a2.py --stage evaluate`, each model's saved `best.pt` checkpoint was reloaded from disk into a fresh model instance and re-evaluated over the `997` validation images:

| Model | Selected Epoch | Saved `val_map50` | Recomputed `val_map50` | Absolute Difference | Verification Status |
|---|---:|---:|---:|---:|---|
| `ssdlite320` | 18 | `0.5153459966873776` | `0.5153459966873776` | `0.0` | `matches_saved: true` |
| `fasterrcnn_frozen` | 12 | `0.8363193379014581` | `0.8363193379014581` | `0.0` | `matches_saved: true` |
| `fasterrcnn_finetune` | 10 | `0.7507074340210955` | `0.7507074340210955` | `0.0` | `matches_saved: true` |

Note:To keep repository storage within GitHub limits while preserving all quantitative and visual evidence, the `CO3133_A2_Draft_Outputs.zip` archive excludes the multi-hundred-megabyte raw `.pt` weight binaries after `checkpoint_verification.json` has been generated and verified on the A100 instance. Rerunning `run_a2.py --stage train` recreates `best.pt` and `last.pt` in the specified `--out` directory.

## 4. Rerunning Safely on Google Colab

To repeat the experiment without overwriting `outputs/a2_m2_a100/`, specify a fresh output directory such as `outputs/a2_m2_a100_rerun`:

```bash
python run_a2.py --stage all --model all --device cuda --data-root data/VOCdevkit/VOC2007 --out outputs/a2_m2_a100_rerun
```
