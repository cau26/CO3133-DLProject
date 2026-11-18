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
| `ssdlite320` | ImageNet backbone | **50.70%** | 1.85% | 34.60% | 63.29% | 4,026,740 / 4,026,740 | 17/20 | 829.4 | 16.22 | 463 |
| `fasterrcnn_frozen` | Frozen ResNet-50 | **83.58%** | 50.10% | 74.36% | 81.71% | 17,941,624 / 41,396,536 | 11/12 | 2425.1 | 31.34 | 1266 |
| `fasterrcnn_finetune` | Full fine-tune | **75.51%** | 42.50% | 62.19% | 75.22% | 41,396,536 / 41,396,536 | 12/12 | 3926.0 | 30.79 | 3612 |

**Hardware environment:** GPU `NVIDIA L4` (22.03 GB VRAM), PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`.

## 3. Per-Class AP@0.5 (%) Breakdown

| Class | Val GT Objects | `ssdlite320` AP@0.5 | `fasterrcnn_frozen` AP@0.5 | `fasterrcnn_finetune` AP@0.5 |
|---|---:|---:|---:|---:|
| aeroplane | 69 | 61.08% | 89.50% | 86.91% |
| bicycle | 84 | 61.37% | 90.05% | 85.93% |
| bird | 98 | 48.06% | 80.85% | 70.04% |
| boat | 67 | 35.65% | 88.48% | 78.63% |
| bottle | 97 | 18.32% | 73.55% | 63.89% |
| bus | 42 | 61.29% | 83.93% | 74.61% |
| car | 262 | 60.17% | 89.19% | 87.92% |
| cat | 75 | 66.31% | 88.12% | 81.20% |
| chair | 166 | 41.84% | 77.94% | 67.35% |
| cow | 55 | 49.15% | 88.72% | 83.89% |
| diningtable | 42 | 54.03% | 78.34% | 69.59% |
| dog | 97 | 58.84% | 85.17% | 71.74% |
| horse | 69 | 65.82% | 92.17% | 85.94% |
| motorbike | 63 | 63.95% | 84.70% | 84.10% |
| person | 942 | 60.55% | 88.03% | 85.46% |
| pottedplant | 140 | 22.75% | 66.46% | 48.59% |
| sheep | 52 | 25.92% | 75.82% | 68.79% |
| sofa | 47 | 52.44% | 76.45% | 66.90% |
| train | 61 | 64.45% | 95.42% | 84.04% |
| tvmonitor | 64 | 41.94% | 78.67% | 64.64% |

## 4. Error Taxonomy (Validation Set, Confidence >= 0.30)

| Model | Correct (TP) | Missed Objects (FN) | Missed Small (<2% area) | Wrong Class | Localization (0.1<=IoU<0.5) | Duplicate Detections | Background FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | 1389 | 1203 | 493 | 262 | 454 | 102 | 216 |
| `fasterrcnn_frozen` | 2380 | 212 | 84 | 213 | 844 | 180 | 567 |
| `fasterrcnn_finetune` | 2146 | 446 | 161 | 272 | 507 | 104 | 329 |
