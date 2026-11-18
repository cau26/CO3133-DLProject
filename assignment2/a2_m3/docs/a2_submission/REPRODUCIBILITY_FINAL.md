# Reproducibility Guide (Assignment 2 - M3 Final)

1. **Colab Notebook:** Open [`notebooks/A2_M3_Colab_Run.ipynb`](../../notebooks/A2_M3_Colab_Run.ipynb) on Google Colab with a GPU runtime (`NVIDIA L4` or `NVIDIA A100`) and run all cells from top to bottom.
2. **CLI Execution (from `assignment2/a2_m3/`):**
   ```bash
   python run_a2.py --stage all_m3 --device cuda --out outputs/a2_m3_a100_rerun
   ```
3. **Checkpoints:** Backed up on Google Drive under `CO3133_A2_M3_Checkpoints` (`ssdlite320_best.pt`, `fasterrcnn_frozen_best.pt`, `fasterrcnn_finetune_best.pt`).
4. **Fingerprints:**
   - `source_sha256`: `ec1ffd3db1b5f01da4d4319c61527148e157e73e6023be295c1fc8febd806d3f`
   - `split_sha256`: `c60294fe4e914b27399ff065c9e246dcd3428440307c97cebf2febe139c2bd6c`
