| Model | Backbone mode | Val mAP@0.5 | Small AP | Med AP | Large AP | Trainable / Total Params | Best Epoch | Train (s) | Latency (ms/img) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | ImageNet backbone | **50.70%** | 1.85% | 34.60% | 63.29% | 4,026,740 / 4,026,740 | 17/20 | 829.4 | 16.22 | 463 |
| `fasterrcnn_frozen` | Frozen ResNet-50 | **83.58%** | 50.10% | 74.36% | 81.71% | 17,941,624 / 41,396,536 | 11/12 | 2425.1 | 31.34 | 1266 |
| `fasterrcnn_finetune` | Full fine-tune | **75.51%** | 42.50% | 62.19% | 75.22% | 41,396,536 / 41,396,536 | 12/12 | 3926.0 | 30.79 | 3612 |
