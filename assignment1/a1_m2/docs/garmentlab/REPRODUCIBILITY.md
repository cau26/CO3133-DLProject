# GarmentLab: reproducibility and local inference

GarmentLab classifies one annotated garment crop into six Fashionpedia categories using ImageNet-initialized MobileNetV3-Small. It is a separate extension of the five-model Fashion-MNIST comparison. The recorded v2 experiment contains 20 fits of 18 epochs each.

## Evidence and model identity

The [recorded summary](../../output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json) links results to the prepared data, fixed selections and evaluation artifacts. The [release manifest](../../output/garmentlab_improve_return_20261028/release_candidate/release.json) identifies `g_m10_garmentlab_controlled_v2-C4-seed314`, temperature 5.345757 and review threshold 0.88.

The compact public evidence retains all 20 histories, task configurations, final selections, logits, predictions and six final checkpoints. Optimizer/RNG recovery states and preliminary-search checkpoints remain in the complete return archive. Its SHA-256 is `feac4f00132c4a8500cf94e7c6ada347be7463855477c79e7b7cb47a5b96266a`. The compact copy is not presented as a byte-complete export.

## Recorded training environment and protocol

The A100 run records Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.4.2 and CUDA 13.0 on an NVIDIA A100-SXM4-40GB. Batch size is 128, precision FP32 and batch-normalization running statistics remain fixed at their ImageNet values.

- Fit/selection/calibration/assessment contain 13,827/1,838/1,835/327 crops. Preparation checks source separation and detected duplicate components.
- Discovery seed 101 evaluates eight declared candidates. C0 and the two shortlisted challengers use fresh confirmation seeds 202 and 303; only those confirmation means choose the recipe.
- Final C0 and C4 fits use seeds 314, 2718 and 5772. The deployment seed 314 and its selection guard are preset.
- Checkpoints maximize selection Macro-F1, then minimize clean CE, then use the earliest epoch. Calibration and assessment do not participate in model choice.
- Disjoint calibration components fit temperature and select the risk threshold. Assessment was inspected previously, so v2 scores are post-test exploratory regression results.

Candidate definitions and seeds are recorded in [`configs/garmentlab_improve_today.json`](../../configs/garmentlab_improve_today.json) and the returned study manifest. Data preparation uses [`configs/garmentlab_data.json`](../../configs/garmentlab_data.json), checks download/cache identities and reconstructs the recorded image, label and role arrays.

## A new reproduction run

The archived configuration contains the absolute stop time of its original execution. A new run requires a separate configuration and output directory. This Python snippet creates a new identity and a supported future UTC stop while retaining the three-hour cumulative controller cap:

```python
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

now = datetime.now(timezone.utc)
stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
config = json.loads(Path("configs/garmentlab_improve_today.json").read_text())
config["study_id"] = "g_m10_garmentlab_reproduce_" + stamp
config["stop_by_utc"] = (now + timedelta(hours=4)).isoformat()
target = Path("configs") / ("garmentlab_reproduction_" + stamp + ".json")
with target.open("x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
run_name = "garmentlab_reproduction_" + stamp
print("python tools/run_garmentlab_improve.py --stage all"
      f" --config {target.as_posix()} --out outputs/{run_name}"
      f" --work /content/{run_name}_work --cache /content/garmentlab_raw_cache")
```

Run the printed command in a compatible A100 environment after installing `requirements-garmentlab-improve-colab.txt`. The controller's `plan` and `check` stages inspect the declared configuration and software/device behavior. `all` prepares data and executes the controlled study; `export` writes complete or explicitly partial evidence. Source, configuration and data bindings distinguish a new reproduction from the archived measurements. Existing immutable artifacts are not overwritten.

## Local PyTorch service

The verified local environment uses Windows, Python 3.12.10, PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu. The CPU lock is a tested version snapshot for that environment, not a cross-platform numerical guarantee.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-garmentlab-cpu-lock.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe tools/run_garmentlab_draft.py --garment-release output/garmentlab_improve_return_20261028/release_candidate --threads 1
```

The application binds to `http://127.0.0.1:8766`. It accepts PNG/JPEG input and one bounded crop, then returns six class probabilities and a review decision. The static GitHub Pages site presents results; the PyTorch application runs separately.

The [service receipt](../../verification/GARMENTLAB_IMPROVE_LOCAL_PYTORCH_SERVICE_20261028.json) verifies archived-case numerical agreement, PNG preprocessing and HTTP behavior. PyTorch is the supported backend for this local release. ONNX passes the original 64-case protocol but exceeds the unchanged `1e-4` logit tolerance in the expanded 327-case check; it remains experimental.

## Interpretation

C4's mean assessment Macro-F1 is 0.76638 versus C0's 0.77472. The paired component-bootstrap interval includes zero, so the controlled tuning does not establish an F1 gain. The fixed policy accepts 182/327 assessment crops, of which 173 are correct: 95.05% accepted accuracy at 55.66% coverage. This is distinct from whole-assessment accuracy and from the conditional representative-level calibration bound.

The [HTTP benchmark](../../verification/GARMENTLAB_IMPROVE_LOCAL_HTTP_20261028.json) is a finite loopback workload on two repeated JPEG/crop cases. It does not establish production capacity, availability or classification quality on new photographs.
