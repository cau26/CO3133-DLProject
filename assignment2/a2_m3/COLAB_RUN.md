# Google Colab Run Record (Assignment 2 - M3 Final)

## 1. Environment & Provenance
- **Hardware:** Google Colab `NVIDIA L4` (`22.03 GB` VRAM), CPU `Intel(R) Xeon(R) CPU @ 2.20GHz` (`12` logical cores, `4` PyTorch threads), CUDA `13.0`.
- **Software:** Python `3.13.16`, PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`, NumPy `2.4.6`, Pandas `2.2.3`, Matplotlib `3.10.0`, Pillow `11.3.0`, Scikit-learn `1.6.1`.
- **Recorded UTC Timestamp:** `2026-11-18T15:59:27Z`
- **`source_sha256`:** `ec1ffd3db1b5f01da4d4319c61527148e157e73e6023be295c1fc8febd806d3f`
- **`split_sha256`:** `c60294fe4e914b27399ff065c9e246dcd3428440307c97cebf2febe139c2bd6c`
- **Executed Notebook:** [`notebooks/A2_M3_Colab_Run.ipynb`](notebooks/A2_M3_Colab_Run.ipynb)
- **Checkpoints Backup:** Google Drive (`CO3133_A2_M3_Checkpoints`: `ssdlite320_best.pt`, `fasterrcnn_frozen_best.pt`, `fasterrcnn_finetune_best.pt`).

## 2. Trainval-vs-Test Overlap Audit (`outputs/a2_m3_a100/test_overlap_check.json`)
- `trainval_images`: `5,011` (`4,014` train + `997` val)
- `test_images`: `4,952` (`12,032` non-difficult objects, `2,944` difficult ignored)
- `image_id_overlap_count`: `0`
- `exact_sha256_overlap_count`: `0`
- `near_duplicate_dhash_le_4_count`: `59` (low-variance sky/horizon collisions such as `003183`)
