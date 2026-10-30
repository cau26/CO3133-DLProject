# Assignment 2 - M2 Draft Report Summary

**Group:** G-M10  
**Course:** CO3133 - Deep Learning and Its Applications, Semester-261  
**Instructor:** Lê Thành Sách  
**Members:** Trần Gia Lâm (2352670); Nguyễn Hữu Cầu (2352129)

> Generated automatically from verified run artifacts in `metrics.json`. No test-set results are included.

## 1. Dataset and Finalized Split

- **Dataset:** PASCAL VOC 2007 (`VOCtrainval_06-Nov-2007.tar`, MD5 `c52e279531787c972589f7e41ab4ae64`).
- **Split:** 4,014 training images (10,016 usable non-difficult objects) and 997 validation images (2,592 usable non-difficult objects; 619 difficult objects retained for VOC ignore rules).
- **Split SHA-256:** `c60294fe4e914b27399ff065c9e246dcd3428440307c97cebf2febe139c2bd6c`.
- **Leakage prevention:** 0 image ID overlap; 3 exact-duplicate groups and 1 confirmed near-duplicate pair kept within single splits.

## 2. Quantitative Results (Validation Set)

| Model | Backbone mode | Val mAP@0.5 | Small AP | Med AP | Large AP | Trainable / Total Params | Best Epoch | Train (s) | Latency (ms/img) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | ImageNet backbone | **51.53%** | 2.29% | 34.36% | 61.96% | 4,026,740 / 4,026,740 | 18/20 | 832.3 | 16.62 | 463 |
| `fasterrcnn_frozen` | Frozen ResNet-50 | **83.63%** | 50.08% | 74.51% | 82.24% | 17,941,624 / 41,396,536 | 12/12 | 1320.3 | 29.28 | 1266 |
| `fasterrcnn_finetune` | Full fine-tune | **75.07%** | 42.14% | 62.14% | 73.95% | 41,396,536 / 41,396,536 | 10/12 | 1807.6 | 22.03 | 3613 |

**Hardware environment:** GPU `NVIDIA A100-SXM4-40GB` (39.49 GB VRAM), PyTorch `2.11.0+cu130`, Torchvision `0.26.0+cu130`.

## 3. Per-Class AP@0.5 (%) Breakdown

| Class | Val GT Objects | `ssdlite320` AP@0.5 | `fasterrcnn_frozen` AP@0.5 | `fasterrcnn_finetune` AP@0.5 |
|---|---:|---:|---:|---:|
| aeroplane | 69 | 59.17% | 88.42% | 81.50% |
| bicycle | 84 | 66.00% | 90.24% | 82.44% |
| bird | 98 | 46.59% | 80.95% | 69.44% |
| boat | 67 | 29.93% | 88.93% | 78.63% |
| bottle | 97 | 24.89% | 75.02% | 62.97% |
| bus | 42 | 64.27% | 84.82% | 75.54% |
| car | 262 | 59.68% | 89.04% | 87.98% |
| cat | 75 | 66.31% | 88.27% | 79.33% |
| chair | 166 | 42.36% | 77.11% | 67.56% |
| cow | 55 | 54.80% | 89.02% | 84.41% |
| diningtable | 42 | 55.57% | 78.11% | 70.86% |
| dog | 97 | 58.66% | 85.63% | 72.29% |
| horse | 69 | 67.99% | 92.76% | 85.66% |
| motorbike | 63 | 67.21% | 85.66% | 82.86% |
| person | 942 | 59.83% | 87.92% | 85.23% |
| pottedplant | 140 | 20.28% | 64.55% | 52.85% |
| sheep | 52 | 25.48% | 75.41% | 60.52% |
| sofa | 47 | 51.53% | 77.68% | 64.77% |
| train | 61 | 65.94% | 94.81% | 85.85% |
| tvmonitor | 64 | 44.20% | 78.28% | 70.73% |

## 4. Error Taxonomy (Validation Set, Confidence >= 0.30)

| Model | Correct (TP) | Missed Objects (FN) | Missed Small (<2% area) | Wrong Class | Localization (0.1<=IoU<0.5) | Duplicate Detections | Background FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | 1391 | 1201 | 490 | 263 | 452 | 99 | 235 |
| `fasterrcnn_frozen` | 2389 | 203 | 83 | 205 | 806 | 171 | 546 |
| `fasterrcnn_finetune` | 2180 | 412 | 152 | 314 | 639 | 134 | 436 |
