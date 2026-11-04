"""Tiny, synthetic controlled-study checks; no experimental quality claim."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from garmentlab import study_v2 as study
from garmentlab import training as base
from garmentlab.runtime import digest
from test_garmentlab_training import make_prepared, tiny_factory


def settings():
    config=json.loads(study.CONFIG.read_text(encoding="utf-8"))
    config.update(device="cpu",epochs=2,batch_size=2,stop_by_utc="2099-01-01T00:00:00Z")
    config["candidates"]=[c for c in config["candidates"] if c["id"] in ("C0","C1","C3")]
    return config


@pytest.fixture(autouse=True)
def cpu_backend():
    threads,mkldnn=torch.get_num_threads(),torch.backends.mkldnn.enabled
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    yield
    torch.set_num_threads(threads);torch.backends.mkldnn.enabled=mkldnn


def test_declared_probes_and_anchor_are_controlled():
    cfg=study.validate_config(json.loads(study.CONFIG.read_text(encoding="utf-8")))
    anchor=cfg["candidates"][0]
    expected={"C1":{"backbone_lr"},"C2":{"weight_decay"},"C3":{"label_smoothing"},
              "C4":{"head_dropout"},"C5":{"photo_brightness","photo_contrast"}}
    for candidate in cfg["candidates"][1:6]:
        assert {k for k in anchor if k not in {"id","name"} and anchor[k]!=candidate[k]}==expected[candidate["id"]]
    assert cfg["discovery_seed"] not in cfg["confirmation_seeds"]
    assert cfg["deployment_seed"]==314 and cfg["epochs"]==18
    wrong=copy.deepcopy(cfg);wrong["candidates"][0]["head_lr"]*=2
    with pytest.raises(ValueError,match="original finetune anchor"):study.validate_config(wrong)


def test_confirmation_ignores_discovery_and_prefers_anchor_within_declared_tie():
    cfg=settings()
    rows=[{"candidate_id":c,"seed":s,"selection":{"macro_f1":f,"loss":l}}
          for c,f,l in [("C0",.8,.5),("C1",.8000005,.1),("C3",.79,.01)] for s in cfg["confirmation_seeds"]]
    winner,means=study.select_confirmation(rows,cfg)
    assert winner=="C0"
    for row in rows:
        if row["candidate_id"]=="C1":row["selection"]["macro_f1"]=.82
    assert study.select_confirmation(rows,cfg)[0]=="C1"
    with pytest.raises(ValueError,match="both fresh seeds"):study.select_confirmation(rows[:-1],cfg)


def test_selection_ce_is_unsmoothed_and_never_augmented(tmp_path,monkeypatch):
    data=base.load_prepared(make_prepared(tmp_path/"prepared"));cfg=settings()
    candidate=next(c for c in cfg["candidates"] if c["id"]=="C3")
    model=tiny_factory().eval();ids=np.flatnonzero(data["roles"]==1)
    def no_photo(*args):raise AssertionError("Selection must never use training photometry")
    monkeypatch.setattr(study,"_photometric",no_photo)
    result=study._epoch(model,data,ids,cfg,candidate)
    with torch.inference_mode():
        logits=model(base.prepare_batch(data["images"],ids))
        expected=torch.nn.functional.cross_entropy(logits,torch.from_numpy(data["labels"][ids]))
    assert result["loss"]==pytest.approx(float(expected),abs=2e-7)
    assert result["objective_loss"]==result["loss"]
    with pytest.raises(ValueError,match="calibration or assessment"):
        study._epoch(model,data,np.flatnonzero(data["roles"]==3),cfg,candidate)


def test_train_records_smoothed_objective_separate_from_clean_ce(tmp_path):
    data=base.load_prepared(make_prepared(tmp_path/"prepared"));cfg=settings()
    candidate=next(c for c in cfg["candidates"] if c["id"]=="C3")
    model=tiny_factory();optimizer,_=base.configure_optimizer(model,"finetune",{**cfg,**candidate})
    result=study._epoch(model,data,np.flatnonzero(data["roles"]==0),cfg,candidate,optimizer=optimizer,
        sampler=torch.Generator().manual_seed(1),augmentation=torch.Generator().manual_seed(2),photometric=torch.Generator().manual_seed(3))
    assert abs(result["objective_loss"]-result["loss"])>1e-7


def test_photometry_is_deterministic_bounded_and_does_not_consume_flip_generator(tmp_path):
    cfg=json.loads(study.CONFIG.read_text(encoding="utf-8"));candidate=cfg["candidates"][5]
    batch=torch.randn(3,3,224,224)
    one=study._photometric(batch,candidate,torch.Generator().manual_seed(7))
    two=study._photometric(batch,candidate,torch.Generator().manual_seed(7))
    assert torch.equal(one,two)
    mean=one.new_tensor(base.MEAN)[None,:,None,None];std=one.new_tensor(base.STD)[None,:,None,None]
    rgb=one*std+mean
    assert rgb.min()>=-1e-7 and rgb.max()<=1+1e-7
    assert not torch.equal(one,batch)
    assert study._photometric(batch,cfg["candidates"][0],torch.Generator()) is batch


def test_whole_tiny_study_freezes_fresh_confirmation_final_pairing_and_no_eval_roles(tmp_path,monkeypatch):
    prepared=make_prepared(tmp_path/"prepared");out=tmp_path/"study";cfg=settings()
    original=base.prepare_batch;seen=[]
    def restricted(images,indices,*args,**kwargs):
        seen.extend(np.asarray(indices).tolist())
        assert max(indices)<12
        return original(images,indices,*args,**kwargs)
    monkeypatch.setattr(base,"prepare_batch",restricted)
    result=study.run_study(prepared,out,cfg,model_factory=tiny_factory,smoke=True)
    assert result["status"]=="TRAINING_COMPLETE_CHOICES_FROZEN" and max(seen)<12
    frozen,tasks=study.verify_final_tasks(out)
    assert frozen["smoke_only"] is True and frozen["deployment_seed"]==314
    assert all(t["phase"]=="final" for t in tasks)
    assert set(t["seed"] for t in tasks)==set(cfg["final_seeds"])
    assert frozen["confirmation"]["selection_uses_discovery_scores"] is False
    assert {r["seed"] for r in frozen["confirmation"]["rows"]}==set(cfg["confirmation_seeds"])
    heads={};encoders=set()
    for marker in (out/"tasks").glob("*/COMPLETE.json"):
        value=json.loads(marker.read_text());heads.setdefault(value["seed"],set()).add(value["initial_head_sha256"])
        encoders.add(value["initial_encoder_sha256"])
        state=torch.load(marker.parent/"epoch_state.pt",weights_only=True)
        assert state["epoch"]==2 and int(state["model_state_dict"]["features.1.num_batches_tracked"])==0
    assert len(encoders)==1 and all(len(h)==1 for h in heads.values())
    def no_model(**kwargs):raise AssertionError("Completed study must not construct another model")
    reused=study.run_study(prepared,out,cfg,model_factory=no_model,smoke=True,stop_check=lambda:True)
    assert reused["status"]=="TRAINING_COMPLETE_CHOICES_FROZEN"


def test_epoch_resume_is_exact_with_smoothed_augmented_candidate(tmp_path):
    prepared=make_prepared(tmp_path/"prepared");cfg=settings()
    # Exercise all RNG streams on C1 while C0 stays its immutable anchor.
    cfg["candidates"][1].update(label_smoothing=.05,photo_brightness=.1,photo_contrast=.1,head_dropout=.4)
    whole,resumed=tmp_path/"whole",tmp_path/"resumed"
    study.run_study(prepared,whole,cfg,model_factory=tiny_factory,smoke=True)
    target=resumed/"tasks/discovery_C1_seed101/epoch_state.pt"
    partial=study.run_study(prepared,resumed,cfg,model_factory=tiny_factory,smoke=True,stop_check=target.exists)
    assert partial["status"]=="INCOMPLETE" and target.exists()
    assert not target.with_name("COMPLETE.json").exists()
    final=study.run_study(prepared,resumed,cfg,model_factory=tiny_factory,smoke=True)
    assert final["status"]=="TRAINING_COMPLETE_CHOICES_FROZEN"
    a=torch.load(whole/"tasks/discovery_C1_seed101/epoch_state.pt",weights_only=True)
    b=torch.load(target,weights_only=True)
    for k in a["model_state_dict"]:assert torch.equal(a["model_state_dict"][k],b["model_state_dict"][k])
    for x,y in zip(a["history"],b["history"]):
        assert x["train"]==y["train"] and x["selection"]==y["selection"] and x["lr"]==y["lr"]


def test_tampered_completed_task_and_changed_config_fail_closed(tmp_path):
    prepared=make_prepared(tmp_path/"prepared");out=tmp_path/"out";cfg=settings()
    study.run_study(prepared,out,cfg,model_factory=tiny_factory,smoke=True)
    wrong=copy.deepcopy(cfg);wrong["candidates"][1]["head_dropout"]=.3
    with pytest.raises(ValueError,match="binding differs"):
        study.run_study(prepared,out,wrong,model_factory=tiny_factory,smoke=True)
    with (out/"tasks/final_C0_seed314/best.pt").open("ab") as f:f.write(b"changed")
    with pytest.raises(ValueError,match="artifact changed"):study.verify_final_tasks(out)


def test_original_output_and_prepared_directory_cannot_overlap(tmp_path):
    prepared=make_prepared(tmp_path/"prepared")
    with pytest.raises(ValueError,match="separate"):
        study.plan_study(prepared,prepared/"study",settings(),smoke=True)


def test_injected_model_cannot_claim_real_training(tmp_path):
    prepared=make_prepared(tmp_path/"prepared")
    with pytest.raises(ValueError,match="explicit smoke"):
        study.run_study(prepared,tmp_path/"out",settings(),model_factory=tiny_factory)


def test_preset_final_guard_falls_back_before_calibration(tmp_path):
    cfg=settings();out=tmp_path/"out";out.mkdir()
    tasks=[{"task_id":f"final_{c}_seed{s}","candidate_id":c,"seed":s,"selection":{"macro_f1":.8 if c=="C0" else .78}}
           for c in ("C0","C1") for s in cfg["final_seeds"]]
    manifest={"data":{},"labels":["a","b"],"smoke_only":True,"checkpoint_rule":"F1 then CE","assessment_scope":"exploratory"}
    frozen=study._freeze_final(out,cfg,manifest,"binding","C1",{},tasks)
    assert frozen["deployment_candidate_id"]=="C0" and frozen["deployment_guard"]["fallback_to_anchor"]
    assert frozen["selection_candidate_id"]=="C1"  # Preserve the actual confirmation outcome.


def test_anchor_matches_original_training_numerically_on_tiny_fixture(tmp_path):
    prepared=make_prepared(tmp_path/"prepared");cfg=settings()
    old=json.loads((Path(base.__file__).resolve().parents[1]/"configs/garmentlab_deadline.json").read_text())
    old.update(device="cpu",epochs=cfg["epochs"],batch_size=cfg["batch_size"],feature_batch_size=3,
               seeds=[cfg["discovery_seed"]],stop_by_utc=cfg["stop_by_utc"])
    original=tmp_path/"original"
    assert base.train_study(prepared,original,old,model_factory=tiny_factory)["status"]=="ALL_TASKS_COMPLETE"
    data,out,cfg,manifest,binding=study._prepare(prepared,tmp_path/"v2",cfg,True)
    task=study._task(cfg,"C0","discovery",cfg["discovery_seed"],data["classnames"])
    study._train_one(data,out,task,cfg,binding,study._Clock(out,cfg,None),tiny_factory)
    a=torch.load(original/f"tasks/finetune_seed{cfg['discovery_seed']}/epoch_state.pt",weights_only=True)
    b=torch.load(out/f"tasks/{task['task_id']}/epoch_state.pt",weights_only=True)
    for key in a["model_state_dict"]:assert torch.equal(a["model_state_dict"][key],b["model_state_dict"][key])
    for x,y in zip(a["history"],b["history"]):
        for name in ["loss","accuracy","macro_f1","confusion"]:
            assert x["train"][name]==y["train"][name]
            assert x["selection"][name]==y["selection"][name]


def test_resume_replays_authoritative_state_after_auxiliary_write_crash(tmp_path,monkeypatch):
    prepared=make_prepared(tmp_path/"prepared");cfg=settings()
    whole,interrupted=tmp_path/"whole",tmp_path/"interrupted"
    for out in [whole,interrupted]:
        data,folder,normalized,manifest,binding=study._prepare(prepared,out,cfg,True)
        task=study._task(normalized,"C0","discovery",normalized["discovery_seed"],data["classnames"])
        clock=study._Clock(folder,normalized,None)
        if out==whole:study._train_one(data,folder,task,normalized,binding,clock,tiny_factory)
        else:
            original=base._atomic_torch
            def crash_auxiliary(path,value):
                if Path(path).name=="best.pt":raise RuntimeError("simulated auxiliary write interruption")
                return original(path,value)
            with monkeypatch.context() as patch:
                patch.setattr(base,"_atomic_torch",crash_auxiliary)
                with pytest.raises(RuntimeError,match="simulated auxiliary"):
                    study._train_one(data,folder,task,normalized,binding,clock,tiny_factory)
            state=folder/f"tasks/{task['task_id']}/epoch_state.pt"
            assert state.is_file() and not state.with_name("best.pt").exists()
            study._train_one(data,folder,task,normalized,binding,clock,tiny_factory)
    left=torch.load(whole/f"tasks/{task['task_id']}/epoch_state.pt",weights_only=True)
    right=torch.load(interrupted/f"tasks/{task['task_id']}/epoch_state.pt",weights_only=True)
    for key in left["model_state_dict"]:assert torch.equal(left["model_state_dict"][key],right["model_state_dict"][key])
