# AI Usage Disclosure

**Course:** CO3133 - Deep Learning and Its Applications  
**Semester:** 261  
**Group:** G-M10

The team used AI to develop ideas, review source code and improve the report. AI did not replace running the experiments or generate the reported experimental results. Team members check the results, interpretations and final content before submission.

## 1. Assignment 1 - M1 Draft

| Item | Record |
|---|---|
| Tool / user | ChatGPT / Trần Gia Lâm |
| Role of AI | Help interpret the requirements, review the experimental design, check implementation consistency and edit the technical report. |
| Scope | Review the EDA, Dataset/DataLoader, training and validation pipeline; check the Linear Classifier and MLP implementation; review the experimental setup; compare metrics, learning curves, confusion matrices and prediction examples; improve the presentation of results. |
| Experiment review | Check that both baselines use the same split, seed, preprocessing and training protocol. Review checkpoint selection and the interpretation of accuracy, macro-F1, parameter counts, training time and inference time. |
| Evidence used | Interpretations are checked against artifacts from the team's actual run: training logs, metrics.json, learning curves, confusion matrices, validation predictions and examples in outputs/a1_t4/. |
| Reproducibility | Review the seed, split, hyperparameters, dependencies, hardware information and Colab commands needed to repeat the experiment. |
| Representative prompts | "Check the current implementation against the M1 requirements"; "Review the experimental setup and identify missing information"; "Check the results and base conclusions on the available evidence; do not invent numbers." These are English summaries of the team's requests. |
| Team verification | Trần Gia Lâm ran the pipeline on Google Colab with an NVIDIA Tesla T4, inspected the outputs and checked the reported values. Nguyễn Hữu Cầu is responsible for reviewing EDA and MLP. |
| Final responsibility | The team is responsible for the source code, experimental results, interpretations and submitted report. |

## 2. Experimental results

The M1 results come from the team's recorded experiment:

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- Device: CUDA
- Random seed: 42
- Training samples: 54,000
- Validation samples: 6,000
- Maximum epochs: 10

The main artifacts are stored in `outputs/a1_t4/`: training histories, metrics, learning curves, confusion matrices, validation predictions and prediction examples.

The group retains responsibility for selecting, verifying and interpreting the results used in the report.
