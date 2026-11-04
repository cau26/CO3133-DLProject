"""Controller lifecycle checks with fake completed jobs; no model training."""
import copy
import io
import time
from collections import Counter

import numpy as np
import pytest
import torch

from a1opt import study
from a1opt.storage import atomic_bytes, atomic_json, json_hash, load_json, sha256_file


@pytest.fixture
def stub_study(tmp_path, monkeypatch):
    out=tmp_path/'study';out.mkdir()
    protocol=load_json(study.ROOT/'configs/a1_optimization_v3.json')
    # Keep the original six-hour regression independent of the delivery budget.
    protocol['time_budget_seconds']=21600
    candidates={family:[{'candidate_id':f'C{i}','family':family,'config':{'stub':i}}
                        for i in range(4 if family=='linear' else 8)] for family in study.FAMILIES}
    stream=io.BytesIO()
    np.savez(stream,original_train=np.arange(20),old_validation=np.arange(20,30),
             inner_train=np.arange(15),inner_validation=np.arange(15,20))
    atomic_bytes(out/'split_ids.npz',stream.getvalue())
    manifest={'schema_version':3,'protocol':protocol,'source':{'files_sha256':{},'sha256':'stub'},
              'candidates':candidates,'split_file_sha256':sha256_file(out/'split_ids.npz'),
              'maximum_tasks':61}
    manifest['manifest_sha256']=json_hash(manifest)
    atomic_json(out/'study_manifest.json',manifest)
    atomic_json(out/'run_environment.json',{'test':'stub'})
    atomic_json(out/'clock.json',{'budget_seconds':21600,'active_run_seconds':0.0})
    monkeypatch.setattr(study,'_prepare',lambda *args:manifest)
    monkeypatch.setattr(study,'load_training_data',lambda *args:(None,np.arange(30)%10))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'NVIDIA A100-SXM4-40GB')
    calls=[]
    def fake_train(folder,task,train_dataset,val_dataset,train_ids,val_ids,binding,device,**kwargs):
        calls.append(task)
        kwargs['heartbeat']()
        folder.mkdir(parents=True,exist_ok=True)
        candidate=int(task['candidate_id'][1:])
        metrics={'best_f1_metrics':{'macro_f1':.7+candidate*.001+(task['seed']%7)*.00001,
                                    'accuracy':.72,'loss':.3},
                 'best_f1_epoch':22,'parameters':100}
        atomic_json(folder/'metrics.json',metrics)
        complete={'status':'COMPLETE','binding_sha256':json_hash(binding),'task_sha256':json_hash(task),
                  'metrics':metrics,'artifacts_sha256':{'metrics.json':sha256_file(folder/'metrics.json')}}
        atomic_json(folder/'COMPLETE.json',complete,immutable=True)
        return complete
    monkeypatch.setattr(study,'train_task',fake_train)
    return out,manifest,calls,fake_train


def test_all61_jobs_resume_across_stage_boundaries(stub_study):
    out,manifest,calls,_=stub_study
    assert study.execute(out,'unused','unused','cuda',max_tasks=1)['completed_tasks']==1
    second=study.execute(out,'unused','unused','cuda',max_tasks=35)
    assert second['completed_tasks']==36 and second['known_tasks']==46
    third=study.execute(out,'unused','unused','cuda',max_tasks=10)
    assert third['completed_tasks']==46 and third['known_tasks']==61
    final=study.execute(out,'unused','unused','cuda',max_tasks=15)
    assert final['status']=='COMPLETE' and final['completed_tasks']==61
    assert Counter(task['phase'] for task in calls)=={'discovery':36,'confirmation':10,'final':15}
    assert len({task['task_id'] for task in calls})==61
    assert all(task['seed'] in (101,202,42,43,44) for task in calls)
    assert study.execute(out,'unused','unused','cuda')['status']=='COMPLETE'
    assert len(calls)==61
    assert load_json(out/'ACTIVE_OWNER.json')['active'] is False


def test_summary_and_selection_corruption_never_report_complete(stub_study):
    out,manifest,_,_=stub_study
    study.execute(out,'unused','unused','cuda')
    (out/'analysis/summary.csv').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='completion marker'):
        study.status(out)
    selection=load_json(out/'selection.json')
    selection['selected']['cnn']['candidate']=manifest['candidates']['cnn'][0]
    atomic_json(out/'selection.json',selection)
    with pytest.raises(ValueError,match='Locked selection differs'):
        study.status(out)


def test_failed_attempt_preserved_and_original_exception_survives(stub_study,monkeypatch):
    out,_,_,_=stub_study
    def fail(*args,**kwargs):
        kwargs['heartbeat']()
        raise OSError('fake GPU/Drive interruption')
    monkeypatch.setattr(study,'train_task',fail)
    with pytest.raises(OSError,match='fake GPU/Drive interruption'):
        study.execute(out,'unused','unused','cuda')
    attempts=load_json(out/'attempts.json')
    assert len(attempts)==1 and attempts[0]['completed'] is False
    assert attempts[0]['failure']['type']=='OSError'
    assert load_json(out/'ACTIVE_OWNER.json')['active'] is False
    assert not (out/'STUDY_COMPLETE.json').exists()


def test_exclusive_lease_and_stale_token_cannot_write(tmp_path):
    out=tmp_path/'lease'
    with study.StudyLease(out,accrue=False) as owner:
        with pytest.raises(RuntimeError,match='Another notebook owns'):
            with study.StudyLease(out,accrue=False): pass
        record=load_json(out/'ACTIVE_OWNER.json')
        record['token']='other-owner'
        atomic_json(out/'ACTIVE_OWNER.json',record)
        with pytest.raises(RuntimeError,match='ownership changed'):
            owner.heartbeat()
        # Restore ownership so the normal context exit can finish cleanly.
        atomic_json(out/'ACTIVE_OWNER.json',owner.record)
    assert not (out/'OWNER_TRANSACTION.lock').exists()


def test_protocol_and_max_tasks_reject_silent_design_changes(stub_study):
    out,manifest,_,_=stub_study
    for key,value in [('sampler_seed',1),('epochs',20),('time_budget_seconds',604801),('final_seeds',[42,43])]:
        protocol=copy.deepcopy(manifest['protocol']);protocol[key]=value
        with pytest.raises(ValueError): study._validate_protocol(protocol)
    for value in (0,-1,True,1.5):
        with pytest.raises(ValueError,match='max_tasks'):
            study.execute(out,'unused','unused','cuda',max_tasks=value)


def test_legacy_output_is_rejected_before_any_ownership_write(tmp_path):
    for marker in ('run_manifest.json','freeze_manifest.json','student_run_export.json'):
        out=tmp_path/marker.replace('.json','');out.mkdir()
        atomic_json(out/marker,{'historical':'preserve'})
        before={p.name:p.read_bytes() for p in out.iterdir()}
        with pytest.raises(ValueError,match='historical experiment'):
            with study.StudyLease(out,accrue=False): pass
        assert {p.name:p.read_bytes() for p in out.iterdir()}==before


def test_interrupted_prepare_commits_clock_before_manifest(tmp_path,monkeypatch):
    protocol=load_json(study.ROOT/'configs/a1_optimization_v3.json')
    base=load_json(study.ROOT/'configs/a1_final.json')
    root=tmp_path/'source';reference=root/protocol['reference'];reference.mkdir(parents=True)
    (root/'configs').mkdir()
    atomic_json(root/protocol['base_config'],base)
    np.savez(reference/'split_indices.npz',train=np.arange(54000),validation=np.arange(54000,60000))
    protocol_path=tmp_path/'protocol.json';atomic_json(protocol_path,protocol)
    monkeypatch.setattr(study,'ROOT',root)
    monkeypatch.setattr(study,'source_identity',lambda:{'sha256':'software-test','files_sha256':{}})
    monkeypatch.setattr(study,'runtime_identity',lambda device:{'software-test':True})
    monkeypatch.setattr(study,'load_training_data',lambda *args:(None,np.arange(60000)%10))
    monkeypatch.setattr(study,'split_hash',lambda *args:protocol['expected_original_split_sha256'])
    monkeypatch.setattr(study,'partition_inner',lambda train,val,labels,seed:(train[:48600],train[48600:]))
    monkeypatch.setattr(study,'generate_candidates',lambda base:{f:[{'candidate_id':'C0'}] for f in study.FAMILIES})
    real_atomic=study.atomic_json;attempts=[]
    def interrupted(path,value,**kwargs):
        if path.name=='study_manifest.json' and not attempts:
            attempts.append(1);raise OSError('injected manifest write interruption')
        return real_atomic(path,value,**kwargs)
    monkeypatch.setattr(study,'atomic_json',interrupted)
    out=tmp_path/'prepared'
    with pytest.raises(OSError,match='injected manifest'):
        study.prepare(out,protocol_path,'unused','cpu')
    assert (out/'clock.json').exists() and not (out/'study_manifest.json').exists()
    assert load_json(out/'ACTIVE_OWNER.json')['active'] is False
    manifest=study.prepare(out,protocol_path,'unused','cpu')
    assert (out/'study_manifest.json').exists()
    assert load_json(out/'clock.json')['active_run_seconds']==0
    assert manifest['manifest_sha256']==json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})


def test_export_is_verified_and_rejects_live_owner(stub_study,monkeypatch):
    out,_,_,_=stub_study
    study.execute(out,'unused','unused','cuda',max_tasks=1)
    monkeypatch.setattr(study.subprocess,'check_output',lambda *args,**kwargs:'test-only==0\n')
    with study.StudyLease(out,accrue=False):
        with pytest.raises(RuntimeError,match='Another notebook owns'):
            study.export_results(out)
    receipt=study.export_results(out)
    assert receipt['status']=='INCOMPLETE' and receipt['completed_tasks']==1
    assert receipt['official_test_evaluated'] is False
    from pathlib import Path
    assert sha256_file(Path(receipt['archive']))==receipt['sha256']
    hashes=load_json(out/'artifact_manifest.json')
    assert 'ACTIVE_OWNER.json' not in hashes and 'OWNER_TRANSACTION.lock' not in hashes
    assert all(sha256_file(out/name)==expected for name,expected in hashes.items())
    assert not (out/'OWNER_TRANSACTION.lock').exists()


def _verified_stub_results(out,manifest):
    return {task['task_id']:result for task in study._all_known_tasks(out,manifest)
            if (result:=study.completed(out,task,manifest)) is not None}


def test_real_first_job_timing_warns_but_does_not_stop_or_repeat(stub_study,capsys):
    """Regression for the student's 402-second first job, with no GPU work."""
    out,manifest,calls,_=stub_study
    study.execute(out,'unused','unused','cuda',max_tasks=1)
    first_id='discovery_linear_C0_seed101'
    first_marker=out/'tasks'/first_id/'COMPLETE.json'
    original_marker=first_marker.read_bytes()
    original_manifest=(out/'study_manifest.json').read_bytes()
    atomic_json(out/'attempts.json',[{'task_id':first_id,'family':'linear',
                                    'wall_seconds':402.009012491,'completed':True,'failure':None}])
    atomic_json(out/'clock.json',{'budget_seconds':21600,'active_run_seconds':402.17865572799997})
    results=_verified_stub_results(out,manifest)
    old_forecast=study._full_remaining_cost(manifest,results,out)
    new_reserve=study._remaining_cost(manifest,results,out)
    reported_seconds_left=21197.900937221
    assert old_forecast==pytest.approx(21578.7553199136)
    assert new_reserve==pytest.approx(10041.9528899352)
    assert new_reserve<reported_seconds_left<old_forecast
    current=study.execute(out,'unused','unused','cuda',max_tasks=1)
    assert current['completed_tasks']==2
    assert [task['task_id'] for task in calls]==[first_id,'discovery_linear_C1_seed101']
    assert first_marker.read_bytes()==original_marker
    assert (out/'study_manifest.json').read_bytes()==original_manifest
    clock=load_json(out/'clock.json')
    assert clock['active_run_seconds']>=402.17865572799997
    assert clock['budget_seconds']==21600
    assert 'TIME FORECAST WARNING' in capsys.readouterr().out
    assert not (out/'STOP_REASON.json').exists()


def test_discovery_reserves_next_task_plus_all25_downstream_jobs(stub_study):
    out,manifest,_,_=stub_study
    rates=study.OLD_FIT_SECONDS
    expected=(rates['linear']+5*sum(rates.values()))*1.2+540
    assert study._remaining_cost(manifest,{},out)==pytest.approx(expected)
    assert study._remaining_cost(manifest,{},out)<study._full_remaining_cost(manifest,{},out)
    assert manifest['maximum_tasks']==61
    assert sum(map(len,manifest['candidates'].values()))==36


def test_confirmation_and_final_reserve_every_remaining_seed(stub_study):
    out,manifest,_,_=stub_study
    rates=study.OLD_FIT_SECONDS
    study.execute(out,'unused','unused','cuda',max_tasks=36)
    results=_verified_stub_results(out,manifest)
    assert len(results)==36
    assert study._remaining_cost(manifest,results,out)==pytest.approx(5*sum(rates.values())*1.2+540)
    assert study._remaining_cost(manifest,results,out)==pytest.approx(study._full_remaining_cost(manifest,results,out))
    study.execute(out,'unused','unused','cuda',max_tasks=1)
    results=_verified_stub_results(out,manifest)
    assert len(results)==37
    assert study._remaining_cost(manifest,results,out)==pytest.approx((5*sum(rates.values())-rates['linear'])*1.2+540)
    study.execute(out,'unused','unused','cuda',max_tasks=9)
    results=_verified_stub_results(out,manifest)
    assert len(results)==46
    assert study._remaining_cost(manifest,results,out)==pytest.approx(3*sum(rates.values())*1.2+540)
    study.execute(out,'unused','unused','cuda',max_tasks=1)
    results=_verified_stub_results(out,manifest)
    assert len(results)==47
    assert study._remaining_cost(manifest,results,out)==pytest.approx((3*sum(rates.values())-rates['linear'])*1.2+540)


def test_too_little_time_still_stops_without_training_or_extending_cap(stub_study):
    out,manifest,calls,_=stub_study
    reserve=study._remaining_cost(manifest,{},out)
    atomic_json(out/'clock.json',{'budget_seconds':21600,'active_run_seconds':21600-reserve+1})
    current=study.execute(out,'unused','unused','cuda')
    assert current['status']=='INCOMPLETE' and current['completed_tasks']==0
    assert calls==[]
    reason=load_json(out/'STOP_REASON.json')
    assert reason['status']=='INCOMPLETE_TIME_RESERVATION'
    assert reason['projected_remaining_seconds']>reason['seconds_left']
    assert reason['reservation_policy']=='next_discovery_plus_all_confirmation_and_final'
    clock=load_json(out/'clock.json')
    assert clock['budget_seconds']==21600
    assert clock['active_run_seconds']>=21600-reserve+1


@pytest.mark.parametrize('budget',[21600,604800])
def test_explicit_six_hour_and_seven_day_budgets_are_valid(budget):
    protocol=load_json(study.ROOT/'configs/a1_optimization_v3.json')
    protocol['time_budget_seconds']=budget
    study._validate_protocol(protocol)


@pytest.mark.parametrize('budget',[0,-1,604801,float('nan'),float('inf'),float('-inf'),True,'604800',None])
def test_budget_rejects_invalid_values_without_silent_extension(budget):
    protocol=load_json(study.ROOT/'configs/a1_optimization_v3.json')
    protocol['time_budget_seconds']=budget
    with pytest.raises(ValueError,match='time_budget_seconds'):
        study._validate_protocol(protocol)


def test_seven_day_warning_reports_configured_cap(stub_study,capsys):
    out,manifest,calls,_=stub_study
    manifest['protocol']['time_budget_seconds']=604800
    manifest['manifest_sha256']=json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    atomic_json(out/'study_manifest.json',manifest)
    atomic_json(out/'clock.json',{'budget_seconds':604800,'active_run_seconds':590000.0})
    current=study.execute(out,'unused','unused','cuda',max_tasks=1)
    assert current['completed_tasks']==1 and len(calls)==1
    output=capsys.readouterr().out
    assert 'configured 168-hour cap unchanged' in output
    assert 'six-hour cap' not in output
    assert load_json(out/'clock.json')['budget_seconds']==604800


def test_all61_synthetic_jobs_complete_with_seven_day_cap(stub_study):
    """Exercise the full controller, not real training or a runtime guarantee."""
    out,manifest,calls,_=stub_study
    manifest['protocol']['time_budget_seconds']=604800
    manifest['manifest_sha256']=json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'})
    atomic_json(out/'study_manifest.json',manifest)
    atomic_json(out/'clock.json',{'budget_seconds':604800,'active_run_seconds':0.0})
    current=study.execute(out,'unused','unused','cuda')
    assert current['status']=='COMPLETE' and current['completed_tasks']==61
    assert Counter(task['phase'] for task in calls)=={'discovery':36,'confirmation':10,'final':15}
    assert len({task['task_id'] for task in calls})==61
    assert load_json(out/'clock.json')['budget_seconds']==604800
    assert study.execute(out,'unused','unused','cuda')['status']=='COMPLETE'
    assert len(calls)==61
