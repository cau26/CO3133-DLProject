"""Controlled regularization study with selection-only choices and epoch recovery.

This module never forwards calibration or assessment images. It preserves v1
source/artifacts and fits every task from ImageNet initialization. The final
selection receipt is frozen before a separate calibration/evaluation stage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn

from . import training as base
from .backend import configure_evaluation_backend
from .runtime import atomic_json, digest, utc_now, utc_parse

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/garmentlab_improve_today.json"
PHASES = ("discovery", "confirmation", "final")
TASK_FILES = ("task.json", "best.pt", "history.json", "epoch_state.pt")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    value = json.loads(json.dumps(config))
    if value.get("schema_version") != 2 or value.get("precision") != "fp32":
        raise ValueError("Study v2 requires schema 2 and FP32")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value.get("study_id", "")):
        raise ValueError("Invalid study_id")
    for name in ("epochs", "batch_size", "challenger_count"):
        if type(value.get(name)) is not int or value[name] < 1:
            raise ValueError(f"Invalid positive integer {name}")
    seeds = [value.get("discovery_seed"), *value.get("confirmation_seeds", []), *value.get("final_seeds", [])]
    if (len(value.get("confirmation_seeds", [])) != 2 or len(value.get("final_seeds", [])) != 3
            or any(type(s) is not int or not 0 <= s < 2**32 - 40000 for s in seeds)
            or len(set(seeds)) != len(seeds) or value.get("deployment_seed") not in value["final_seeds"]):
        raise ValueError("Discovery, two confirmation and three final seeds must be distinct bounded integers")
    for name in ("gradient_clip", "max_active_seconds"):
        if type(value.get(name)) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f"Invalid positive finite {name}")
    for name in ("confirmation_f1_tie_tolerance", "deployment_max_selection_f1_regression", "finalization_reserve_seconds"):
        if type(value.get(name)) not in (int, float) or not math.isfinite(value[name]) or value[name] < 0:
            raise ValueError(f"Invalid nonnegative finite {name}")
    utc_parse(value["stop_by_utc"])
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 3 <= len(candidates) <= 16:
        raise ValueError("Study requires 3 to 16 predeclared candidates")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not re.fullmatch(r"C[0-9]{1,2}", candidate.get("id", "")):
            raise ValueError("Candidate IDs must be bounded C-number identifiers")
        ids.append(candidate["id"])
        for name in ("head_lr", "backbone_lr"):
            if type(candidate.get(name)) not in (int, float) or not math.isfinite(candidate[name]) or candidate[name] <= 0:
                raise ValueError(f"Invalid candidate {name}")
        for name in ("weight_decay", "head_dropout", "label_smoothing", "photo_brightness", "photo_contrast"):
            x = candidate.get(name)
            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                raise ValueError(f"Invalid candidate {name}")
            if name != "weight_decay" and x >= (0.5 if name.startswith("photo_") else 1):
                raise ValueError(f"Candidate {name} exceeds its declared bounded range")
    if len(set(ids)) != len(ids) or ids[0] != "C0" or value["challenger_count"] >= len(ids):
        raise ValueError("C0 must be the unique first anchor and challengers must exclude it")
    anchor = candidates[0]
    expected = {"head_lr": .001, "backbone_lr": .0003, "weight_decay": .0001,
                "head_dropout": .2, "label_smoothing": 0., "photo_brightness": 0., "photo_contrast": 0.}
    if any(anchor[key] != number for key, number in expected.items()):
        raise ValueError("C0 must preserve the original finetune anchor settings")
    return value


def _task(config, candidate_id, phase, seed, labels):
    candidate = next(c for c in config["candidates"] if c["id"] == candidate_id)
    return {"task_id": f"{phase}_{candidate_id}_seed{seed}", "candidate_id": candidate_id,
            "phase": phase, "recipe": "finetune", "seed": seed, "epochs": config["epochs"],
            "labels": labels, "architecture": "mobilenet_v3_small", "candidate": candidate}


def _prepare(prepared_dir, out_dir, config, smoke):
    config = validate_config(config if config is not None else _json(CONFIG))
    prepared = Path(prepared_dir).resolve(strict=True)
    out = Path(out_dir).resolve()
    if out == prepared or out.is_relative_to(prepared) or prepared.is_relative_to(out):
        raise ValueError("Study output must be separate from prepared inputs")
    data = base.load_prepared(prepared)
    out.mkdir(parents=True, exist_ok=True)
    configure_evaluation_backend()
    names = ("garmentlab/study_v2.py", "garmentlab/training.py", "garmentlab/backend.py", "a1/cnn.py")
    source = {name: digest(ROOT / name) for name in names}
    manifest = {"schema": "garmentlab_controlled_study_v2", "config": config,
                "labels": data["classnames"], "data": data["bindings"], "source_sha256": source,
                "environment": base._environment(config), "smoke_only": bool(smoke),
                "maximum_tasks": len(config["candidates"]) + (config["challenger_count"]+1)*2 + 6,
                "checkpoint_rule": "max selection Macro-F1, then minimum unsmoothed selection CE, then earliest epoch",
                "candidate_rule": "Discovery selects challengers; only the two fresh confirmation seeds choose the final candidate. Anchor preferred within the preset F1 tolerance; otherwise lower CE and candidate ID break ties.",
                "deployment_rule": "Preset seed; selection-only guard falls back to paired final C0 if selected seed regresses beyond the preset F1 tolerance. No assessment/calibration choice.",
                "assessment_scope": "Assessment was inspected in v1. Future assessment scores are exploratory/regression evidence, not another blind test.",
                "weight_source": "injected_software_smoke_only" if smoke else "MobileNet_V3_Small_Weights.IMAGENET1K_V1"}
    base._protect_json(out / "study_v2_manifest.json", manifest)
    for name in names:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != source[name]:
            raise ValueError("Study source snapshot changed")
        if not target.exists():
            target.write_bytes((ROOT / name).read_bytes())
    return data, out, config, manifest, digest(out / "study_v2_manifest.json")


def plan_study(prepared_dir, out_dir, config=None, *, smoke=False):
    """Freeze source/config/data/environment before any candidate training."""
    _, out, _, manifest, binding = _prepare(prepared_dir, out_dir, config, smoke)
    return {"status": "PLAN_FROZEN", "binding": binding, "manifest": manifest,
            "manifest_path": str(out / "study_v2_manifest.json")}


class _Clock:
    def __init__(self, out, config, external):
        self.path = out / "TRAINING_CLOCK.json"
        self.config, self.external = config, external
        self.previous = _json(self.path)["active_seconds"] if self.path.exists() else 0.
        self.started, self.last = time.monotonic(), 0.

    @property
    def elapsed(self):
        return self.previous + time.monotonic() - self.started

    def tick(self, force=False):
        now = self.elapsed
        if force or now-self.last >= 10:
            atomic_json(self.path, {"active_seconds": now, "budget_seconds": self.config["max_active_seconds"],
                "updated_utc": utc_now().isoformat(), "scope": "Cumulative training-controller time; abrupt uncommitted tail may be unrecorded. Parent controller enforces the complete run budget."})
            self.last = now

    def stopped(self):
        self.tick()
        return (self.elapsed >= self.config["max_active_seconds"]
            or (utc_parse(self.config["stop_by_utc"])-utc_now()).total_seconds() <= self.config["finalization_reserve_seconds"]
            or (self.external is not None and self.external()))


def _photometric(batch, candidate, generator):
    """Mild per-image brightness/contrast, sampled separately from paired flips."""
    bright, contrast = candidate["photo_brightness"], candidate["photo_contrast"]
    if not bright and not contrast:
        return batch
    mean = batch.new_tensor(base.MEAN)[None, :, None, None]
    std = batch.new_tensor(base.STD)[None, :, None, None]
    rgb = batch * std + mean
    factors = torch.rand((len(batch), 2), generator=generator).to(batch.device)
    b = (1 + (factors[:, 0] * 2 - 1) * bright)[:, None, None, None]
    c = (1 + (factors[:, 1] * 2 - 1) * contrast)[:, None, None, None]
    center = rgb.mean(dim=(1, 2, 3), keepdim=True)
    rgb = ((rgb * b - center) * c + center).clamp(0, 1)
    return (rgb-mean) / std


def _epoch(model, data, indices, config, candidate, *, optimizer=None, sampler=None,
           augmentation=None, photometric=None, stop_check=None):
    training = optimizer is not None
    ids = np.asarray(indices, dtype=np.int64)
    required_role = 0 if training else 1
    if not len(ids) or np.any(data["roles"][ids] != required_role):
        raise ValueError("Study training/selection cannot forward any calibration or assessment role")
    if training:
        base.set_training_mode(model, "finetune")
        ids = ids[torch.randperm(len(ids), generator=sampler).numpy()]
    else:
        model.eval()
    ce_sum, objective_sum, truth, predictions = 0., 0., [], []
    with torch.enable_grad() if training else torch.inference_mode():
        for start in range(0, len(ids), config["batch_size"]):
            if stop_check is not None and stop_check():
                raise base.TrainingStopped("Budget/deadline reached during an uncommitted epoch")
            rows = ids[start:start+config["batch_size"]]
            flips = torch.rand(len(rows), generator=augmentation).numpy() < .5 if training else np.zeros(len(rows), bool)
            batch = base.prepare_batch(data["images"], rows, config["device"], flips)
            if training:
                batch = _photometric(batch, candidate, photometric)
            logits = model(batch)
            targets = torch.as_tensor(data["labels"][rows], dtype=torch.long, device=config["device"])
            ce = nn.functional.cross_entropy(logits, targets)
            objective = (nn.functional.cross_entropy(logits, targets, label_smoothing=candidate["label_smoothing"])
                         if training and candidate["label_smoothing"] else ce)
            if not torch.isfinite(logits).all() or not torch.isfinite(ce) or not torch.isfinite(objective):
                raise ValueError("Nonfinite training/selection output")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            ce_sum += float(ce.detach()) * len(rows)
            objective_sum += float(objective.detach()) * len(rows)
            truth.extend(targets.cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    result = base.classification_metrics(truth, predictions, ce_sum, len(data["classnames"]))
    result["objective_loss"] = objective_sum / len(ids)
    result["loss_definition"] = "Unsmoothed CE; objective_loss separately records the training regularization objective"
    return result


def _complete(folder, task, binding):
    if not (folder / "COMPLETE.json").exists():
        return False
    marker = _json(folder / "COMPLETE.json")
    if marker.get("binding") != binding or marker.get("task_id") != task["task_id"] or _json(folder/"task.json") != task:
        raise ValueError("Completed v2 task identity differs")
    for name in TASK_FILES:
        if marker["files"].get(name, {}).get("sha256") != digest(folder/name):
            raise ValueError(f"Completed v2 task artifact changed: {task['task_id']}/{name}")
    return True


def _descriptor(out, task, binding):
    folder = out / "tasks" / task["task_id"]
    if not _complete(folder, task, binding):
        raise ValueError("Cannot use an incomplete task for selection")
    marker = _json(folder/"COMPLETE.json")
    return {"task_id": task["task_id"], "candidate_id": task["candidate_id"], "recipe": "finetune",
        "phase": task["phase"], "seed": task["seed"], "binding": binding, "task": task,
        "checkpoint": f"tasks/{task['task_id']}/best.pt", "weights_sha256": marker["files"]["best.pt"]["sha256"],
        "selection": marker["selection"], "best_epoch": marker["best_epoch"],
        "history": _json(folder/"history.json"), "fit_seconds": marker["fit_seconds"]}


def _train_one(data, out, task, config, binding, clock, factory):
    folder = out / "tasks" / task["task_id"]
    folder.mkdir(parents=True, exist_ok=True)
    base._protect_json(folder/"task.json", task)
    if _complete(folder, task, binding):
        print(f"TASK REUSE {task['task_id']}: committed artifacts verified; no new fitting",flush=True)
        return
    if clock.stopped():
        raise base.TrainingStopped("No new task begins after the reserved stop")
    print(f"TASK START phase={task['phase']} id={task['task_id']} epochs={config['epochs']}",flush=True)
    candidate = task["candidate"]
    model = factory(num_classes=len(data["classnames"]), seed=task["seed"], pretrained=True).to(config["device"])
    if not isinstance(model.classifier[2], nn.Dropout):
        raise ValueError("Declared architecture requires the explicit classifier dropout layer")
    model.classifier[2].p = candidate["head_dropout"]
    initial_head = base._tensor_digest(model.classifier[-1].state_dict())
    encoder = {k:v for k,v in model.state_dict().items() if not k.startswith("classifier.3.")}
    initial_encoder = base._tensor_digest(encoder)
    for marker in (out/"tasks").glob("*/COMPLETE.json"):
        previous = _json(marker)
        if previous["initial_encoder_sha256"] != initial_encoder:
            raise ValueError("Pretrained encoder initialization differs across candidates/seeds")
        if previous["seed"] == task["seed"] and previous["initial_head_sha256"] != initial_head:
            raise ValueError("Paired candidate head initialization differs")
    settings = {**config, **candidate}
    optimizer, scheduler = base.configure_optimizer(model, "finetune", settings)
    sampler = torch.Generator().manual_seed(task["seed"]+10000)
    augmentation = torch.Generator().manual_seed(task["seed"]+20000)
    photometric = torch.Generator().manual_seed(task["seed"]+30000)
    random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(task["seed"])
    history, best, best_checkpoint, next_epoch, seconds = [], None, None, 1, 0.
    state_path = folder/"epoch_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if (state.get("binding") != binding or state.get("task") != task or state.get("initial_head_sha256") != initial_head
                or state.get("initial_encoder_sha256") != initial_encoder):
            raise ValueError("Resume task/source/data/config/initialization binding changed")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        for parameter, values in optimizer.state.items():
            for name,value in values.items():
                if isinstance(value,torch.Tensor) and name != "step":values[name]=value.to(parameter.device)
        scheduler.load_state_dict(state["scheduler"])
        base._restore_rng(state["rng"],sampler,augmentation)
        photometric.set_state(state["photometric_rng"])
        history,best,best_checkpoint = state["history"],state["best"],state["best_checkpoint"]
        seconds,next_epoch=state["fit_seconds"],state["epoch"]+1
        print(f"TASK RESUME {task['task_id']}: last committed epoch={state['epoch']}; next={next_epoch}",flush=True)
        atomic_json(folder/"history.json",history)
        if best_checkpoint is not None:base._atomic_torch(folder/"best.pt",best_checkpoint)
    fit, selection = np.flatnonzero(data["roles"]==0),np.flatnonzero(data["roles"]==1)
    for epoch in range(next_epoch,config["epochs"]+1):
        if clock.stopped():raise base.TrainingStopped("Epoch checkpoint preserved at the deadline/budget reservation")
        start=time.monotonic();rates=[g["lr"] for g in optimizer.param_groups]
        train=_epoch(model,data,fit,config,candidate,optimizer=optimizer,sampler=sampler,
            augmentation=augmentation,photometric=photometric,stop_check=clock.stopped)
        score=_epoch(model,data,selection,config,candidate,stop_check=clock.stopped)
        elapsed=time.monotonic()-start;seconds+=elapsed
        history.append({"epoch":epoch,"train":train,"selection":score,"lr":rates,"wall_seconds":elapsed})
        if (best is None or score["macro_f1"]>best["selection"]["macro_f1"]
                or (score["macro_f1"]==best["selection"]["macro_f1"] and score["loss"]<best["selection"]["loss"])):
            best={"epoch":epoch,"selection":score}
            best_checkpoint={"model_state_dict":base._state_cpu(model),"epoch":epoch,"selection":score,
                "binding":binding,"task":task,"initial_head_sha256":initial_head,"initial_encoder_sha256":initial_encoder}
        scheduler.step()
        state={"model_state_dict":base._state_cpu(model),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "rng":base._rng(sampler,augmentation),"photometric_rng":photometric.get_state(),"epoch":epoch,"history":history,
            "best":best,"best_checkpoint":best_checkpoint,"fit_seconds":seconds,"binding":binding,"task":task,
            "initial_head_sha256":initial_head,"initial_encoder_sha256":initial_encoder}
        base._atomic_torch(state_path,state)
        base._atomic_torch(folder/"best.pt",best_checkpoint)
        atomic_json(folder/"history.json",history)
        clock.tick(True)
        print(f"{task['task_id']} epoch {epoch}/{config['epochs']} trainCE={train['loss']:.4f} objective={train['objective_loss']:.4f} selectionF1={score['macro_f1']:.4f} selectionCE={score['loss']:.4f}",flush=True)
    atomic_json(folder/"COMPLETE.json",{"task_id":task["task_id"],"binding":binding,"seed":task["seed"],
        "epochs":config["epochs"],"best_epoch":best["epoch"],"selection":best["selection"],"fit_seconds":seconds,
        "initial_head_sha256":initial_head,"initial_encoder_sha256":initial_encoder,
        "files":{name:{"sha256":digest(folder/name)} for name in TASK_FILES}})
    print(f"TASK COMPLETE {task['task_id']} selected_epoch={best['epoch']} selectionF1={best['selection']['macro_f1']:.6f} cleanCE={best['selection']['loss']:.6f} fit_seconds={seconds:.1f}",flush=True)
    del model,optimizer,scheduler
    if str(config["device"]).startswith("cuda"):torch.cuda.empty_cache()


def select_confirmation(rows, config):
    """Choose only from fresh-confirmation means, never discovery score reuse."""
    means={}
    for candidate_id in sorted({r["candidate_id"] for r in rows}):
        items=[r for r in rows if r["candidate_id"]==candidate_id]
        if sorted(r["seed"] for r in items)!=sorted(config["confirmation_seeds"]):
            raise ValueError("Each confirmation candidate requires both fresh seeds exactly once")
        means[candidate_id]={"macro_f1":float(np.mean([r["selection"]["macro_f1"] for r in items])),
                             "loss":float(np.mean([r["selection"]["loss"] for r in items]))}
    if "C0" not in means:raise ValueError("Confirmation requires the anchor")
    maximum=max(v["macro_f1"] for v in means.values())
    tied=[k for k,v in means.items() if v["macro_f1"]>=maximum-config["confirmation_f1_tie_tolerance"]]
    winner="C0" if "C0" in tied else min(tied,key=lambda k:(means[k]["loss"],k))
    return winner,means


def _freeze_final(out, config, manifest, binding, selected_id, confirmation, final_tasks):
    selected=next(t for t in final_tasks if t["candidate_id"]==selected_id and t["seed"]==config["deployment_seed"])
    anchor=next(t for t in final_tasks if t["candidate_id"]=="C0" and t["seed"]==config["deployment_seed"])
    regression=selected["selection"]["macro_f1"]-anchor["selection"]["macro_f1"]
    fallback=regression < -config["deployment_max_selection_f1_regression"]
    deploy=anchor if fallback else selected
    frozen={"schema":"garmentlab_v2_final_selection_v1","study_binding":binding,
        "data":manifest["data"],"labels":manifest["labels"],"smoke_only":manifest["smoke_only"],
        "selection_candidate_id":selected_id,"deployment_candidate_id":deploy["candidate_id"],
        "deployment_task_id":deploy["task_id"],"deployment_seed":config["deployment_seed"],
        "confirmation":confirmation,"final_tasks":final_tasks,
        "deployment_guard":{"maximum_selection_f1_regression":config["deployment_max_selection_f1_regression"],
            "selected_minus_anchor_f1":regression,"fallback_to_anchor":fallback,
            "scope":"Preset seed selection-only regression guard; no assessment or calibration inference"},
        "checkpoint_rule":manifest["checkpoint_rule"],"assessment_scope":manifest["assessment_scope"],
        "choice_scope":"Frozen after selection-only fits and before any calibration, threshold or assessment inference"}
    base._protect_json(out/"FINAL_SELECTION.json",frozen)
    return frozen


def run_study(prepared_dir, out_dir, config=None, *, stop_check=None, model_factory=None, smoke=False):
    """Fit at most 20 tasks, preserving epoch transactions on an incomplete stop."""
    if (model_factory is not None) != bool(smoke):
        raise ValueError("Injected models require explicit smoke=True; real runs cannot claim smoke quality")
    data,out,config,manifest,binding=_prepare(prepared_dir,out_dir,config,smoke)
    cuda=str(config["device"]).startswith("cuda")
    if not smoke and (not cuda or not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(torch.device(config["device"]))):
        raise RuntimeError("Real study v2 training requires an available A100")
    clock=_Clock(out,config,stop_check);factory=model_factory or base.build_model
    state={"status":"INCOMPLETE","binding":binding,"completed_tasks":[],"completed":0,
        "planned_maximum":manifest["maximum_tasks"],"planned":None,"smoke_only":smoke,"phase":"discovery"}
    def perform(task):
        _train_one(data,out,task,config,binding,clock,factory)
        row=_descriptor(out,task,binding)
        state["completed_tasks"].append(task["task_id"]);state["completed"]+=1
        atomic_json(out/"study_v2_state.json",state)
        return row
    try:
        print(f"PHASE discovery: {len(config['candidates'])} predeclared candidates; maximum study fits={manifest['maximum_tasks']}",flush=True)
        discovery=[perform(_task(config,c["id"],"discovery",config["discovery_seed"],data["classnames"])) for c in config["candidates"]]
        ranked=sorted((r for r in discovery if r["candidate_id"]!="C0"),key=lambda r:(-r["selection"]["macro_f1"],r["selection"]["loss"],r["candidate_id"]))
        challengers=[r["candidate_id"] for r in ranked[:config["challenger_count"]]]
        base._protect_json(out/"DISCOVERY_SELECTION.json",{"binding":binding,"challengers":challengers,
            "selection_rule":"max selection F1, then lower clean CE, then candidate ID; C0 excluded", "rows":discovery})
        state["phase"]="confirmation"
        print(f"PHASE confirmation: anchor and challengers {challengers}; fresh seeds={config['confirmation_seeds']}",flush=True)
        rows=[perform(_task(config,c,"confirmation",seed,data["classnames"])) for c in ["C0",*challengers] for seed in config["confirmation_seeds"]]
        winner,means=select_confirmation(rows,config)
        confirmed={"binding":binding,"selected_candidate_id":winner,"means":means,"rows":rows,
            "selection_uses_discovery_scores":False,"tie_tolerance":config["confirmation_f1_tie_tolerance"]}
        base._protect_json(out/"CONFIRMATION_SELECTION.json",confirmed)
        state["phase"]="final"
        final_ids=["C0"] if winner=="C0" else ["C0",winner]
        state["planned"]=len(config["candidates"])+(config["challenger_count"]+1)*2+len(final_ids)*len(config["final_seeds"])
        print(f"PHASE final: confirmation winner={winner}; paired candidates={final_ids}; actual study fits={state['planned']}",flush=True)
        finals=[perform(_task(config,c,"final",seed,data["classnames"])) for c in final_ids for seed in config["final_seeds"]]
        frozen=_freeze_final(out,config,manifest,binding,winner,confirmed,finals)
        state.update(status="TRAINING_COMPLETE_CHOICES_FROZEN",phase="choices_frozen",
            selected_candidate_id=winner,deployment_task_id=frozen["deployment_task_id"],final_selection_sha256=digest(out/"FINAL_SELECTION.json"))
        print(f"CHOICES FROZEN completed={state['completed']}/{state['planned']} deployment={frozen['deployment_task_id']} fallback_to_anchor={frozen['deployment_guard']['fallback_to_anchor']}",flush=True)
    except base.TrainingStopped as error:
        state["stop_reason"]=str(error)
    except KeyboardInterrupt:
        state["stop_reason"]="Interrupted; replay resumes the last committed epoch transaction"
    except Exception as error:
        state.update(status="FAILED",exception_type=type(error).__name__,message=str(error))
        raise
    finally:
        clock.tick(True);state["training_active_seconds"]=clock.elapsed
        atomic_json(out/"study_v2_state.json",state)
    return state


def verify_final_tasks(out_dir):
    """Verify committed final descriptors; return frozen choices and task records.

    ``checkpoint`` paths in records are relative to ``out_dir``. Callers must not
    infer choices from assessment. The helper does no model construction/forward.
    """
    out=Path(out_dir).resolve(strict=True)
    manifest=_json(out/"study_v2_manifest.json");config=validate_config(manifest["config"])
    binding=digest(out/"study_v2_manifest.json")
    frozen=_json(out/"FINAL_SELECTION.json");state=_json(out/"study_v2_state.json")
    if (state.get("status")!="TRAINING_COMPLETE_CHOICES_FROZEN" or state.get("binding")!=binding
            or state.get("final_selection_sha256")!=digest(out/"FINAL_SELECTION.json")
            or frozen.get("study_binding")!=binding):
        raise ValueError("Final selection is not bound to a complete study")
    confirmation=_json(out/"CONFIRMATION_SELECTION.json")
    discovery=_json(out/"DISCOVERY_SELECTION.json")
    discoveries=[_descriptor(out,_task(config,c["id"],"discovery",config["discovery_seed"],manifest["labels"]),binding)
                 for c in config["candidates"]]
    ranked=sorted((r for r in discoveries if r["candidate_id"]!="C0"),
                  key=lambda r:(-r["selection"]["macro_f1"],r["selection"]["loss"],r["candidate_id"]))
    challengers=[r["candidate_id"] for r in ranked[:config["challenger_count"]]]
    if discovery.get("binding")!=binding or discovery.get("rows")!=discoveries or discovery.get("challengers")!=challengers:
        raise ValueError("Discovery shortlist differs from committed selection-only evidence")
    rows=[]
    for item in confirmation["rows"]:
        task=_task(config,item["candidate_id"],"confirmation",item["seed"],manifest["labels"])
        verified=_descriptor(out,task,binding)
        if verified!=item:raise ValueError("Confirmation descriptor differs from committed task")
        rows.append(verified)
    expected_pairs={(c,s) for c in ["C0",*challengers] for s in config["confirmation_seeds"]}
    if len(rows)!=len(expected_pairs) or {(r["candidate_id"],r["seed"]) for r in rows}!=expected_pairs:
        raise ValueError("Confirmation candidates differ from the discovery shortlist plus anchor")
    winner,means=select_confirmation(rows,config)
    if (winner!=frozen["selection_candidate_id"] or winner!=confirmation.get("selected_candidate_id")
            or means!=confirmation["means"] or confirmation!=frozen["confirmation"]):
        raise ValueError("Frozen choice differs from fresh-confirmation-only selection")
    finals=[]
    for candidate_id in (["C0"] if winner=="C0" else ["C0",winner]):
        for seed in config["final_seeds"]:
            finals.append(_descriptor(out,_task(config,candidate_id,"final",seed,manifest["labels"]),binding))
    if finals!=frozen["final_tasks"]:raise ValueError("Final task descriptors differ from committed evidence")
    expected=_freeze_final(out,config,manifest,binding,winner,confirmation,finals)
    if expected!=frozen:raise ValueError("Final deployment guard changed")
    return frozen,finals
