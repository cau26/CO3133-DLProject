| Model | Backbone mode | Val mAP@0.5 | Small AP | Med AP | Large AP | Trainable / Total Params | Best Epoch | Train (s) | Latency (ms/img) | Peak VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ssdlite320` | ImageNet backbone | **51.53%** | 2.29% | 34.36% | 61.96% | 4,026,740 / 4,026,740 | 18/20 | 832.3 | 16.62 | 463 |
| `fasterrcnn_frozen` | Frozen ResNet-50 | **83.63%** | 50.08% | 74.51% | 82.24% | 17,941,624 / 41,396,536 | 12/12 | 1320.3 | 29.28 | 1266 |
| `fasterrcnn_finetune` | Full fine-tune | **75.07%** | 42.14% | 62.14% | 73.95% | 41,396,536 / 41,396,536 | 10/12 | 1807.6 | 22.03 | 3613 |
