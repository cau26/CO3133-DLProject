"""Recovery orchestration fixtures; no model fitting or real assessment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from garmentlab import recovery
from garmentlab.runtime import digest


def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True)+"\n").encode()


@pytest.fixture
def scenario(tmp_path):
    original=tmp_path/"original_prepared"
    rebuilt=tmp_path/"rebuilt"
    original.mkdir()
    rebuilt.mkdir()
    rng=np.random.default_rng(19)
    np.save(original/"images_uint8.npy",rng.integers(0,256,size=(24,224,224,3),dtype=np.uint8))
    np.save(original/"labels.npy",np.tile(np.arange(6,dtype=np.int64),4))
    roles=np.repeat(np.arange(4,dtype=np.uint8),6)
    np.save(original/"roles.npy",roles)
    labels=[f"class_{i}" for i in range(6)]
    binding={"source_sha256":{key:letter*64 for key,letter in zip(
        ("train_metadata","assessment_metadata","train_archive","assessment_archive"),"abcd")},
        "resolution":224,"near_duplicate_distance":2,"preparation_source_sha256":"e"*64}
    records=[{"array_index":i,"role":tuple(recovery_role for recovery_role in ("fit","selection","calibration","assessment"))[int(role)],
              "source_group_id":f"source_{i}","evaluation_group_id":f"component_{i}",
              "image_id":i,"annotation_id":i,"class_index":i%6,"label":labels[i%6],"crop_xyxy":[0,0,224,224]}
             for i,role in enumerate(roles)]
    index={"labels":labels,"records":records,"binding":binding}
    (original/"index_manifest.json").write_bytes(encoded(index))
    ready={"training_ready":True,"binding":binding,
           "sources":{key:{"sha256":value} for key,value in binding["source_sha256"].items()},
           "files":{name:digest(original/name) for name in (*recovery.ARRAY_NAMES,"index_manifest.json")}}
    (original/"DATA_READINESS.json").write_bytes(encoded(ready))
    for name in recovery.ARRAY_NAMES:
        (rebuilt/name).write_bytes((original/name).read_bytes())
    # Simulate valid Linux/Windows JSON byte drift while retaining every record.
    (rebuilt/"index_manifest.json").write_bytes(encoded(index).replace(b"\n",b"\r\n"))
    local_ready=deepcopy(ready)
    local_ready["files"]["index_manifest.json"]=digest(rebuilt/"index_manifest.json")
    (rebuilt/"DATA_READINESS.json").write_bytes(encoded(local_ready).replace(b"\n",b"\r\n"))
    environment={"torch":"fixture","torchvision":"fixture","gpu":None,"device":"cpu"}
    config={"recipes":["frozen","finetune"],"seeds":[314],"epochs":2,
            "max_total_process_seconds":54000,"stop_by_utc":"2099-01-01T00:00:00Z"}
    tasks=[{"task_id":f"{recipe}_seed314","recipe":recipe,"seed":314,"epochs":2,"labels":labels}
           for recipe in config["recipes"]]
    study={"config":config,"tasks":tasks,"environment":environment,"labels":labels,
           "data":{name:digest(original/name) for name in (*recovery.ARRAY_NAMES,"index_manifest.json","DATA_READINESS.json")}}
    payloads={"study_manifest.json":encoded(study),"study_state.json":encoded({"status":"ALL_TASKS_COMPLETE",
        "completed":2,"planned":2,"binding":"training_binding"}),
        "RUN_MANIFEST.json":encoded({"source_fingerprint":"original_source"}),
        "RUN_CLOCK.json":encoded({"active_process_seconds":1559.37349858}),
        "RUN_STATUS.json":encoded({"status":"FAILED","original":True}),"LAST_ERROR.txt":b"original failure\n",
        "evaluation/frozen_seed314/selection/logits.npz":b"bad original cache must never be reused",
        "source/configs/garmentlab_data.json":encoded({"fixture_only":True}),
        "source/SOURCE_MANIFEST.json":encoded({"original_source":True}),
        "source_snapshot/garmentlab/training.py":b"raise RuntimeError('Archived source is evidence only; never import')\n",
        "data_evidence/index_manifest.json":(original/"index_manifest.json").read_bytes(),
        "data_evidence/DATA_READINESS.json":(original/"DATA_READINESS.json").read_bytes()}
    for task in tasks:
        prefix="tasks/"+task["task_id"]+"/"
        history=[{"epoch":1,"selection":{"macro_f1":.6,"loss":1.}},
                 {"epoch":2,"selection":{"macro_f1":.7,"loss":.9}}]
        for name,data in {"best.pt":b"fixture best "+task["task_id"].encode(),
                          "epoch_state.pt":b"fixture resume state", "task.json":encoded(task),
                          "history.json":encoded(history)}.items():
            payloads[prefix+name]=data
        complete={"task_id":task["task_id"],"binding":"training_binding","epochs":2,"best_epoch":2,
                  "selection":history[-1]["selection"],
                  "files":{name:{"sha256":hashlib.sha256(payloads[prefix+name]).hexdigest()}
                           for name in ("best.pt","epoch_state.pt","task.json","history.json")}}
        payloads[prefix+"COMPLETE.json"]=encoded(complete)
    archive=tmp_path/"original_results.zip"
    write_archive(archive,payloads)
    patch=tmp_path/"patch"
    (patch/"garmentlab").mkdir(parents=True)
    (patch/"tools").mkdir()
    (patch/"garmentlab/backend.py").write_text("# fixed backend fixture\n")
    (patch/"tools/recover_garmentlab.py").write_text("# recovery CLI fixture\n")
    (patch/"SOURCE_MANIFEST.json").write_text('{"current_package":"fixture"}\n')
    return {"archive":archive,"sha":digest(archive),"payloads":payloads,"rebuilt":rebuilt,
            "patch":patch,"environment":environment,"study":study,"tmp":tmp_path}


def write_archive(path,payloads):
    manifest={"payloads":[{"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
                          for name,data in sorted(payloads.items())]}
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in payloads.items():
            archive.writestr(name,data)
        archive.writestr("EXPORT_MANIFEST.json",encoded(manifest))


def run(scenario,*,stage="all",finalizer=None,out=None,**kwargs):
    return recovery.recover_results(scenario["archive"],out or scenario["tmp"]/"recovery",
        scenario["tmp"]/"work",scenario["tmp"]/"cache",expected_results_sha256=scenario["sha"],
        stage=stage,source_dir=scenario["patch"],smoke=True,
        prepare_fn=lambda work,cache,config:scenario["rebuilt"],
        finalize_fn=finalizer,environment_fn=lambda:scenario["environment"],**kwargs)


def fake_finalize(out,prepared,config,*,smoke,stop_check,model_factory=None):
    assert smoke and not stop_check()
    assert not (out/"evaluation/frozen_seed314/selection/logits.npz").exists()
    assert (out/"original_failure/evaluation/frozen_seed314/selection/logits.npz").exists()
    final=out/"FINAL_SUMMARY.json"
    if final.exists():
        return json.loads(final.read_text())
    result={"smoke_only":True,"release_status":"candidate_smoke_only"}
    final.write_bytes(encoded(result))
    (out/"evaluation").mkdir(exist_ok=True)
    (out/"evaluation/SELECTION_FREEZE.json").write_bytes(encoded({"fixed_fixture_choice":True}))
    return result


def test_recovery_preserves_every_original_artifact_and_excludes_failed_live_cache(scenario):
    result=run(scenario,finalizer=fake_finalize,model_factory=lambda **kwargs:None)
    assert result["status"]["status"]=="COMPLETE" and result["status"]["smoke_only"]
    out=Path(result["out_dir"])
    mapping=json.loads((out/"ORIGINAL_PAYLOAD_MAP.json").read_text())["mapping"]
    for name,data in scenario["payloads"].items():
        assert (out/mapping[name]).read_bytes()==data
    assert digest(scenario["archive"])==scenario["sha"]
    assert Path(result["result_zip"]).name=="g_m10_garmentlab_recovery_run01_results.zip"
    assert (out/"recovery_source_snapshot/SOURCE_MANIFEST.json").read_bytes()==(scenario["patch"]/"SOURCE_MANIFEST.json").read_bytes()
    from garmentlab.training import load_prepared
    loaded=load_prepared(result["prepared_view"])
    assert loaded["bindings"]==scenario["study"]["data"]
    loaded["images"]._mmap.close()
    original_receipt=(out/"PREPARED_RECONSTITUTION.json").read_bytes()
    freeze=(out/"evaluation/SELECTION_FREEZE.json").read_bytes()
    resumed=run(scenario,finalizer=fake_finalize,model_factory=lambda **kwargs:None)
    assert resumed["status"]["status"]=="COMPLETE"
    assert (out/"PREPARED_RECONSTITUTION.json").read_bytes()==original_receipt
    assert (out/"evaluation/SELECTION_FREEZE.json").read_bytes()==freeze


def test_array_mismatch_exports_failure_with_all_checkpoints_preserved(scenario):
    path=scenario["rebuilt"]/"labels.npy"
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    result=run(scenario,finalizer=lambda *args,**kwargs:pytest.fail("must not finalize mismatching arrays"))
    assert result["status"]["status"]=="FAILED" and "Rebuilt array" in result["error"]
    with zipfile.ZipFile(result["result_zip"]) as archive:
        assert archive.read("tasks/frozen_seed314/best.pt")==scenario["payloads"]["tasks/frozen_seed314/best.pt"]
        assert archive.read("tasks/finetune_seed314/best.pt")==scenario["payloads"]["tasks/finetune_seed314/best.pt"]
        assert json.loads(archive.read("RUN_STATUS.json"))["status"]=="FAILED"


@pytest.mark.parametrize("mutation",["role","source_group_id","crop_xyxy","class_order"])
def test_semantic_data_changes_cannot_be_papered_over_by_original_metadata(scenario,mutation):
    path=scenario["rebuilt"]/"index_manifest.json"
    index=json.loads(path.read_text())
    if mutation=="class_order":
        index["labels"].reverse()
    else:
        index["records"][0][mutation]={"role":"assessment","source_group_id":"different","crop_xyxy":[1,1,223,223]}[mutation]
    path.write_bytes(encoded(index))
    ready_path=scenario["rebuilt"]/"DATA_READINESS.json"
    ready=json.loads(ready_path.read_text());ready["files"]["index_manifest.json"]=digest(path);ready_path.write_bytes(encoded(ready))
    result=run(scenario,finalizer=fake_finalize)
    assert result["status"]["status"]=="FAILED" and "records/class order/roles" in result["error"]
    assert not (scenario["tmp"]/"work/prepared_view").exists()


def test_pinned_archive_hash_and_unrelated_output_are_never_overwritten(scenario):
    with pytest.raises(ValueError,match="ZIP hash"):
        recovery.inspect_results(scenario["archive"],"0"*64)
    out=scenario["tmp"]/"old_run";out.mkdir();marker=out/"study_manifest.json";marker.write_text("old")
    with pytest.raises(ValueError,match="overwrite"):
        run(scenario,out=out,stage="check")
    assert marker.read_text()=="old" and not (out/"RECOVERY_MANIFEST.json").exists()


def test_malicious_zip_paths_fail_before_copying(scenario):
    bad=scenario["tmp"]/"bad.zip"
    payloads={**scenario["payloads"],"../escape.txt":b"bad"}
    write_archive(bad,payloads)
    with pytest.raises(ValueError,match="Unsafe"):
        recovery.inspect_results(bad,digest(bad))


def test_symlink_payload_and_manifest_hash_corruption_fail(scenario):
    bad=scenario["tmp"]/"symlink.zip"
    with zipfile.ZipFile(scenario["archive"]) as original,zipfile.ZipFile(bad,"w") as altered:
        for info in original.infolist():
            if info.filename=="tasks/frozen_seed314/best.pt":
                info.create_system=3;info.external_attr=(0o120777<<16)
            altered.writestr(info,original.read(info.filename))
    with pytest.raises(ValueError,match="symlink"):
        recovery.inspect_results(bad,digest(bad))
    changed=scenario["tmp"]/"corrupt.zip"
    with zipfile.ZipFile(scenario["archive"]) as original,zipfile.ZipFile(changed,"w") as altered:
        for info in original.infolist():
            data=original.read(info.filename)
            if info.filename=="tasks/frozen_seed314/best.pt":
                data=b"X"*len(data)
            altered.writestr(info,data)
    with pytest.raises(ValueError,match="payload hash"):
        recovery.inspect_results(changed,digest(changed))


def test_resume_refuses_changed_recovery_code_without_replacing_checkpoint_evidence(scenario):
    first=run(scenario,stage="check")
    out=Path(first["out_dir"]);checkpoint=(out/"tasks/frozen_seed314/best.pt").read_bytes()
    (scenario["patch"]/"garmentlab/backend.py").write_text("# different patch\n")
    second=run(scenario,stage="check")
    assert second["status"]["status"]=="FAILED" and "identity changed" in second["error"]
    assert (out/"tasks/frozen_seed314/best.pt").read_bytes()==checkpoint


def test_original_budget_is_subtracted_and_recovery_is_capped_to_two_hours(scenario):
    checked=recovery.inspect_results(scenario["archive"],scenario["sha"])
    assert checked["original_process_seconds"]==pytest.approx(1559.37349858)
    assert checked["recovery_budget_seconds"]==7200
    payloads=deepcopy(scenario["payloads"])
    payloads["RUN_CLOCK.json"]=encoded({"active_process_seconds":53500.})
    later=scenario["tmp"]/"nearly_spent.zip";write_archive(later,payloads)
    assert recovery.inspect_results(later,digest(later))["recovery_budget_seconds"]==500


def test_export_still_works_after_recovery_clock_budget_is_exhausted(scenario):
    first=run(scenario,stage="check")
    out=Path(first["out_dir"])
    clock=json.loads((out/"RUN_CLOCK.json").read_text());clock["active_process_seconds"]=7201
    (out/"RUN_CLOCK.json").write_bytes(encoded(clock))
    exported=run(scenario,stage="export")
    assert exported["error"] is None and Path(exported["result_zip"]).exists()


def test_real_finalization_environment_is_enforced_and_injection_cannot_escape_smoke():
    expected={"torch":"2.11.0+cu130","torchvision":"0.26.0+cu130","gpu":"NVIDIA A100-SXM4-40GB"}
    observed={"torch":"2.10.0+cpu","torchvision":"0.25.0+cpu","gpu":None,"device":"cpu"}
    with pytest.raises(RuntimeError,match="same A100"):
        recovery._check_environment(expected,observed,enforce=True,smoke=False)
    assert not recovery._check_environment(expected,observed,enforce=False,smoke=False)["eligible_for_real_finalization"]


def test_module_never_invokes_training_entry_point():
    import ast
    tree=ast.parse(Path(recovery.__file__).read_text())
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="train_study"
                   for node in ast.walk(tree))
