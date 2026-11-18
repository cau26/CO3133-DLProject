# Assignment 2 - M3 Final (`assignment2/a2_m3/`)

- **Final Report (GitHub Pages):** [`assignment2-final.md`](assignment2-final.md)
- **Final PDF Report (LMS):** [`output/pdf/G-M10_A2_Report.pdf`](output/pdf/G-M10_A2_Report.pdf)
- **Presentation Slides:** [`output/presentation/G-M10_A2_Slides.pdf`](output/presentation/G-M10_A2_Slides.pdf)
- **Presentation Video (YouTube):** [https://youtu.be/OF_lVbQw9zg](https://youtu.be/OF_lVbQw9zg) (Title: `CO3133-Semester-261 - Group G-M10 - Assignment 2`)
- **Vietnamese Video Script:** [`docs/a2_submission/VIDEO_SCRIPT_VI.md`](docs/a2_submission/VIDEO_SCRIPT_VI.md)
- **Executed Colab Notebook:** [`notebooks/A2_M3_Colab_Run.ipynb`](notebooks/A2_M3_Colab_Run.ipynb)
- **Official Test & Val Outputs:** [`outputs/a2_m3_a100/`](outputs/a2_m3_a100/)

## Validation vs. Official VOC 2007 Test Results (NVIDIA L4, `seed = 42`)

| Model | Selected Epoch | Val mAP@0.5 | Official Test mAP@0.5 | Delta (Test − Val) | Test Small AP | Test Med AP | Test Large AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | 17 / 20 | 50.70% | **52.98%** | +2.28 pp | 0.79% | 32.75% | 64.98% |
| `fasterrcnn_frozen` | 11 / 12 | **83.58%** | **83.65%** | +0.07 pp | **45.32%** | **75.15%** | **81.89%** |
| `fasterrcnn_finetune` | 12 / 12 | 75.51% | **76.90%** | +1.40 pp | 32.89% | 63.46% | 76.36% |

## Reproduce
```bash
python run_a2.py --stage all_m3 --device cuda --out outputs/a2_m3_a100_rerun
```
