# Reproducing and inspecting the final study

The repository includes a compact copy of the recorded OPT3 evidence in
[`output/a1_final/evidence/`](../../output/a1_final/evidence/). It contains all
61 task configurations, histories, metrics and prediction CSVs, the study and
split records, and the 15 primary checkpoints from final seeds 42–44. Copies
are byte-identical to the returned archive. Optimizer/RNG resume states,
inner-search weights and auxiliary CE weights remain in the complete archive.

## Recompute the published numbers

From the repository root, with Python 3.10 or newer:

```bash
python tools/verify_a1_final_evidence.py
```

This checks 333 copied payloads, recomputes accuracy and macro-F1 from all
338,400 saved prediction rows, checks saved-NLL aggregation, and reproduces
the final mean and sample SD. It requires no GPU or third-party package.
Official Fashion-MNIST labels and phase-specific sample IDs were checked
separately in the retained [independent review](../../verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.md).

## Load a final checkpoint

Install a compatible PyTorch/torchvision pair and the project requirements.
The following loads the CNN seed-42 checkpoint without training or inference:

```python
import torch
from a1opt.models import build_model

checkpoint = torch.load(
    "output/a1_final/evidence/tasks/final_cnn_C5_seed42/best_f1.pt",
    map_location="cpu", weights_only=True,
)
task = checkpoint["task"]
model = build_model(task["family"], task["config"], task["seed"])
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
print(checkpoint["epoch"], checkpoint["metrics"])
```

All models accept float32 tensors with shape `[B,1,28,28]`, pixel values scaled
to `[0,1]`, and return ten logits. Dropout is disabled in evaluation mode.
Checkpoint filenames identify the family, selected candidate and training seed;
the embedded task retains the full effective configuration.

## Repeat training

The [A100 notebook](../../notebooks/A1_Optimization_A100.ipynb) describes the
pipeline. The included
[`study_manifest.json`](../../output/a1_final/evidence/study_manifest.json)
retains the realized candidate set, configuration, source identity, seeds and
selection rules; [`split_ids.npz`](../../output/a1_final/evidence/split_ids.npz)
retains the sample assignments. The recorded run used A100-SXM4-40GB,
Python 3.13, PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3 and Optuna 4.8.0.
Full dependency records are retained in the evidence folder.

The original 54,000/6,000 training–validation split is stratified with seed 42.
Recipe discovery and confirmation use a separate 48,600/5,400 inner split
within the original training images. Discovery evaluates 36 candidates with
training seed 101. One challenger per model family is shortlisted against its
anchor; the additional ten confirmation fits use seed 202. The declared recipe
decision averages seed 101 and 202, so only seed 202 is fresh confirmation.
Final fitting uses seeds 42, 43 and 44 on all 54,000 training images. The
previously inspected 6,000-image validation partition selects each final
checkpoint by maximum Macro-F1, then minimum cross-entropy and earliest epoch.
Every completed task runs 60 epochs; the last epoch is not necessarily selected.

```bash
python -m pip install -r requirements-a1-optimization.txt
python tools/run_a1_optimization.py --stage plan --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage smoke --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage run --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage export --out outputs/opt3_reproduction
```

Use a new output directory for a new experiment. Resume requires the complete
original run directory, including its transaction states; this compact review
subset is not a resumable training directory. Restore the recorded source,
manifest, environment and split identities for exact recovery. A different
hardware/software setup may change numerical results.

The runner uses only official training images and its derived validation sets.
Final OPT3 results are checkpoint-selected exploratory validation results.
The original frozen official-test scores belong to their original checkpoints.

## Publication sources

The main Fashion-MNIST figures and numerical tables use
[`publication_data.json`](../../output/a1_final/data/publication_data.json).
GarmentLab methods, model identity and reproduction are documented separately
in [its reproducibility guide](../garmentlab/REPRODUCIBILITY.md).

The current report's editable source is
[`output/latex/G-M10_A1_Reviewed_20261104/main.tex`](../../output/latex/G-M10_A1_Reviewed_20261104/main.tex).
It contains the associated figures and section files. From that project
directory, compile twice with XeLaTeX:

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The [self-contained LaTeX archive](../../output/latex/G-M10_A1_Overleaf_Reviewed_20261104.zip)
can also be imported into a LaTeX editor using `main.tex` and XeLaTeX. Standard
TeX fonts and local image files make the project independent of absolute paths
on the authoring machine. The [editable slides](../../output/presentation/G-M10_A1_Slides_Reviewed_20261104.pptx)
and [reading copy](../../output/presentation/G-M10_A1_Slides_Reviewed_20261104.pdf)
are separate presentation artifacts.

Static project pages are generated from the recorded publication data:

```bash
python tools/build_a1_pages.py
python tools/build_a1_pages.py --check
```

These commands generate or verify publication files; they do not execute model
training or create additional evaluation measurements.

## Fixed-policy class diagnostics

The class-conditional acceptance tables are derived from the prospectively
recorded C4 seed314 logits and labels. The temperature and confidence threshold
remain frozen. Recompute them using Python 3.11 or newer and NumPy:

```bash
python tools/analyze_policy_by_class.py --out output/policy_reproduction
```

The reference [JSON](../../output/policy_by_class_20261030/POLICY_BY_CLASS.json)
retains input hashes, counts, definitions and pointwise source-component
bootstrap intervals. The two CSV tables condition separately on true and
predicted labels; their denominators must not be interchanged. This command
does not load a model or tune a policy.
