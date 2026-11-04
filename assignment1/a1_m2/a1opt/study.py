"""Immutable candidate study and student-run controller; no official-test API."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch

from .engine import train_task
from .models import build_model
from .recipes import FAMILIES, generate_candidates
from .storage import atomic_bytes, atomic_json, atomic_torch_save, json_hash, load_json, sha256_file, verify_complete

ROOT = Path(__file__).resolve().parents[1]
OLD_FIT_SECONDS = dict(linear=243.475, mlp=240.160, cnn=242.769, gru=252.799, transformer=365.520)


def source_identity():
    files = sorted({*ROOT.glob('a1/*.py'), *ROOT.glob('a1opt/*.py'),
                    ROOT/'tools/run_a1_optimization.py', ROOT/'configs/a1_final.json',
                    ROOT/'configs/a1_optimization_v3.json', ROOT/'requirements-a1-optimization.txt',
                    *ROOT.glob('assets/a1_optimization_reference/*')})
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')
              if p.suffix in ('.py', '.json', '.txt') else p.read_bytes()).hexdigest() for p in files}
    return {'files_sha256': hashes, 'sha256': json_hash(hashes)}


def runtime_identity(device):
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'numpy', 'scikit-learn', 'optuna', 'matplotlib')}
    if versions['numpy'] != np.__version__:
        raise RuntimeError('Loaded NumPy differs from installed version. Restart the Colab runtime, then Run all.')
    if versions['optuna'] != '4.8.0':
        raise RuntimeError('This frozen candidate-generation protocol requires Optuna 4.8.0.')
    return {'versions': versions, 'python_major_minor': list(sys.version_info[:2]),
            'device_type': str(device), 'gpu': torch.cuda.get_device_name(0) if str(device) == 'cuda' else None,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}


def split_hash(train, validation):
    return hashlib.sha256(np.asarray(train, dtype='<i8').tobytes()
                          + np.asarray(validation, dtype='<i8').tobytes()).hexdigest()


def partition_inner(original_train, original_validation, labels, seed):
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(np.asarray(original_train), test_size=5400,
                                  random_state=seed, stratify=labels[original_train])
    train, val = np.sort(train), np.sort(val)
    if (len(train), len(val)) != (48600, 5400):
        raise ValueError('Inner split sizes differ.')
    if not np.array_equal(np.sort(np.r_[train, val]), np.sort(original_train)):
        raise ValueError('Inner split must partition original train IDs exactly.')
    if np.intersect1d(np.r_[train, val], original_validation).size:
        raise ValueError('Inner IDs overlap old validation.')
    return train, val


def load_training_data(data_dir, expected_hash):
    from torchvision import datasets, transforms
    datasets.FashionMNIST.mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']
    dataset = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transforms.ToTensor())
    labels = np.asarray(dataset.targets, dtype=np.int64)
    if len(dataset) != 60000 or tuple(dataset.data.shape) != (60000, 28, 28):
        raise ValueError('Unexpected official training data shape.')
    digest = hashlib.sha256(dataset.data.numpy().tobytes() + labels.astype('<i8').tobytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError('Training dataset fingerprint differs from the verified original dataset.')
    return dataset, labels


def _validate_protocol(protocol):
    fixed = {'schema_version':3, 'study_type':'post_test_exploratory_validation_only',
             'epochs':60, 'batch_size':128, 'num_workers':0, 'sampler_seed':20261021,
             'inner_split_seed':20261022, 'schedule_shuffle_seed':20261023,
             'discovery_seed':101, 'confirmation_seed':202, 'final_seeds':[42,43,44],
             'primary_metric':'macro_f1',
             'checkpoint_rule':'max_macro_f1_then_min_ce_then_earliest_epoch',
             'final_training_policy':'full54k_old6k_validation_selected_60_epochs',
             'recipe_f1_tie_tolerance':1e-6, 'forecast_buffer_fraction':.2,
             'budget_policy':'next_discovery_and_downstream_reservation_v1',
             'finalization_reserve_seconds':540,
             'base_config':'configs/a1_final.json',
             'reference':'assets/a1_optimization_reference',
             'expected_dataset_sha256':'1f243a60b4b748a44c48b9a6f6907be2a08bee3e4e226f0e86fbde8e147b745a',
             'expected_original_split_sha256':'0289c4940126abe338a376ea49eb866788d57cd67fa3ca17fe80139abb8c8fba'}
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            raise ValueError(f'Frozen V3 protocol differs for {key}: expected {expected!r}.')
    budget = protocol.get('time_budget_seconds')
    if isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or not 0 < budget <= 604800:
        raise ValueError('time_budget_seconds must be positive and at most 604800 (7 days); no automatic budget extension.')
    if type(protocol.get('cpu_threads')) is not int or protocol['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be a positive integer.')


def prepare(out, protocol_path, data_dir, device):
    """Prepare under ownership too: concurrent Run-all starts cannot race a manifest."""
    with StudyLease(out, accrue=False):
        return _prepare(out, protocol_path, data_dir, device)


def _prepare(out, protocol_path, data_dir, device):
    out, protocol_path = Path(out), Path(protocol_path)
    protocol = load_json(protocol_path)
    _validate_protocol(protocol)
    source = source_identity()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'study_manifest.json'
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest['source'] != source or manifest['protocol'] != protocol:
            raise ValueError('Study source/protocol changed. Restore the original source and config to resume.')
        if manifest['manifest_sha256'] != json_hash({k:v for k,v in manifest.items() if k != 'manifest_sha256'}):
            raise ValueError('Study manifest checksum differs.')
        if runtime_identity(device) != load_json(out/'run_environment.json'):
            raise ValueError('Resume environment changed. Restore saved library/CUDA/GPU versions; do not mix runs.')
        _verify_splits(out, manifest)
        clock=load_json(out/'clock.json')
        if clock.get('budget_seconds') != protocol['time_budget_seconds'] or not math.isfinite(clock.get('active_run_seconds',math.nan)) or clock['active_run_seconds'] < 0:
            raise ValueError('Study clock missing/invalid or budget changed; do not reset elapsed time.')
        return manifest
    if any((out/name).exists() for name in ('tasks', 'selection.json', 'STUDY_COMPLETE.json')):
        raise FileExistsError('Existing unbound study files; preserve this folder and choose a new run name.')
    atomic_json(out/'preparation_binding.json', {'protocol':protocol, 'source':source,
                                               'environment':runtime_identity(device)}, immutable=True)
    dataset, labels = load_training_data(data_dir, protocol['expected_dataset_sha256'])
    reference = ROOT/protocol['reference']
    with np.load(reference/'split_indices.npz', allow_pickle=False) as saved:
        original_train, original_val = saved['train'].copy(), saved['validation'].copy()
    if (len(original_train), len(original_val)) != (54000, 6000):
        raise ValueError('Original split counts differ.')
    if not np.array_equal(np.sort(np.r_[original_train, original_val]), np.arange(60000)):
        raise ValueError('Original split has missing, duplicate or overlapping IDs.')
    if split_hash(original_train, original_val) != protocol['expected_original_split_sha256']:
        raise ValueError('Original split hash differs.')
    inner_train, inner_val = partition_inner(original_train, original_val, labels, protocol['inner_split_seed'])
    stream = io.BytesIO()
    np.savez_compressed(stream, original_train=original_train, old_validation=original_val,
                        inner_train=inner_train, inner_validation=inner_val)
    if (out/'split_ids.npz').exists() and (out/'split_ids.npz').read_bytes() != stream.getvalue():
        # NumPy ZIP timestamps can differ; compare the actual arrays after an interrupted plan.
        with np.load(out/'split_ids.npz', allow_pickle=False) as previous:
            for key, value in dict(original_train=original_train, old_validation=original_val,
                                   inner_train=inner_train, inner_validation=inner_val).items():
                if not np.array_equal(previous[key], value):
                    raise ValueError('Existing partially prepared split differs.')
    else:
        atomic_bytes(out/'split_ids.npz', stream.getvalue())
    base = load_json(ROOT/protocol['base_config'])
    base.update(epochs=protocol['epochs'], batch_size=protocol['batch_size'], num_workers=protocol['num_workers'],
                cpu_threads=protocol['cpu_threads'])
    candidates = generate_candidates(base)
    manifest = {'schema_version':3, 'protocol':protocol, 'source':source,
                'candidates':candidates, 'split_file_sha256':sha256_file(out/'split_ids.npz'),
                'inner_split_sha256':split_hash(inner_train, inner_val),
                'class_counts':{k:np.bincount(labels[ids], minlength=10).tolist() for k,ids in
                                dict(original_train=original_train, old_validation=original_val,
                                     inner_train=inner_train, inner_validation=inner_val).items()},
                'maximum_tasks':61, 'primary_metric':'macro_f1',
                'old_validation_role':'Final checkpoint selection; previously seen exploratory data',
                'official_test_policy':'No official-test dataset/model evaluation in this study'}
    manifest['manifest_sha256'] = json_hash(manifest)
    atomic_json(out/'run_environment.json', runtime_identity(device), immutable=True)
    atomic_json(out/'clock.json', {'active_run_seconds':0.0, 'budget_seconds':protocol['time_budget_seconds'],
                                  'scope':'Controller process monotonic heartbeat; setup/smoke excluded; abrupt final tail may be unrecorded'}, immutable=True)
    # Manifest is the preparation commit marker; all prerequisites exist first.
    atomic_json(manifest_path, manifest, immutable=True)
    return manifest


def _verify_splits(out, manifest):
    if sha256_file(Path(out)/'split_ids.npz') != manifest['split_file_sha256']:
        raise ValueError('Saved split file changed.')


def _task(manifest, family, candidate, seed, phase):
    cfg = dict(candidate['config'])
    return {'task_id':f'{phase}_{family}_{candidate["candidate_id"]}_seed{seed}',
            'family':family, 'config':cfg, 'seed':seed, 'phase':phase,
            'epochs':manifest['protocol']['epochs'], 'candidate_id':candidate['candidate_id']}


def task_binding(manifest, task, environment):
    return {'study_manifest_sha256':manifest['manifest_sha256'], 'task_sha256':json_hash(task),
            'split_sha256':manifest['split_file_sha256'], 'environment_sha256':json_hash(environment)}


def completed(out, task, manifest):
    folder = Path(out)/'tasks'/task['task_id']
    if not (folder/'COMPLETE.json').exists():
        return None
    result=verify_complete(folder, task_binding(manifest, task, load_json(Path(out)/'run_environment.json')))
    if result.get('task_sha256') != json_hash(task):
        raise ValueError('Completed task identity differs: '+task['task_id'])
    value=score(result)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Completed task has invalid macro-F1: '+task['task_id'])
    return result


def score(result):
    return result['metrics']['best_f1_metrics']['macro_f1']


def discovery_tasks(manifest):
    return [_task(manifest,f,c,manifest['protocol']['discovery_seed'],'discovery')
            for f in FAMILIES for c in manifest['candidates'][f]]


def choose_challengers(manifest, results):
    tolerance = manifest['protocol']['recipe_f1_tie_tolerance']
    choices = {}
    for family in FAMILIES:
        candidates = manifest['candidates'][family]
        nonanchors = candidates[1:]
        scores = [(c, score(results[_task(manifest,family,c,manifest['protocol']['discovery_seed'],'discovery')['task_id']])) for c in nonanchors]
        best = max(s for _,s in scores)
        choices[family] = min((c for c,s in scores if s >= best-tolerance), key=lambda c:c['candidate_id'])
    return choices


def confirmation_tasks(manifest, challengers):
    return [_task(manifest,f,c,manifest['protocol']['confirmation_seed'],'confirmation')
            for f in FAMILIES for c in (manifest['candidates'][f][0],challengers[f])]


def select_recipes(manifest, challengers, results):
    selected = {}
    for family in FAMILIES:
        anchor, challenger = manifest['candidates'][family][0], challengers[family]
        def values(candidate):
            return [score(results[_task(manifest,family,candidate,manifest['protocol'][key],phase)['task_id']])
                    for key,phase in [('discovery_seed','discovery'),('confirmation_seed','confirmation')]]
        a,c = values(anchor),values(challenger)
        winner = challenger if statistics.mean(c) > statistics.mean(a)+manifest['protocol']['recipe_f1_tie_tolerance'] else anchor
        selected[family] = {'candidate':winner, 'anchor_f1':a, 'challenger_f1':c,
                            'paired_f1_delta':[y-x for x,y in zip(a,c)],
                            'confirmation_training_seeds':[manifest['protocol']['discovery_seed'],manifest['protocol']['confirmation_seed']]}
    return {'selected':selected,'objective':'Mean selected inner macro-F1 across two matched seeds; 1e-6 tie prefers C0',
            'study_manifest_sha256':manifest['manifest_sha256'],
            'final_training_policy':manifest['protocol']['final_training_policy']}


def final_tasks(manifest, selection):
    return [_task(manifest,f,selection['selected'][f]['candidate'],seed,'final')
            for f in FAMILIES for seed in manifest['protocol']['final_seeds']]


def _all_known_tasks(out, manifest):
    tasks = discovery_tasks(manifest)
    if (Path(out)/'challengers.json').exists():
        challengers=load_json(Path(out)/'challengers.json')
        if set(challengers) != set(FAMILIES) or any(challengers[f] not in manifest['candidates'][f][1:] for f in FAMILIES):
            raise ValueError('Locked challenger is not a declared non-anchor candidate.')
        tasks += confirmation_tasks(manifest, challengers)
    if (Path(out)/'selection.json').exists():
        selection=load_json(Path(out)/'selection.json')
        if not (Path(out)/'challengers.json').exists() or selection.get('study_manifest_sha256') != manifest['manifest_sha256']:
            raise ValueError('Selection lacks its challenger lock or manifest binding.')
        if set(selection.get('selected',{})) != set(FAMILIES) or any(
                selection['selected'][f]['candidate'] not in (manifest['candidates'][f][0],challengers[f]) for f in FAMILIES):
            raise ValueError('Selected recipe is not the locked anchor/challenger.')
        tasks += final_tasks(manifest, selection)
    return tasks


def status(out, manifest=None):
    out = Path(out)
    manifest = manifest or load_json(out/'study_manifest.json')
    _validate_protocol(manifest['protocol'])
    if manifest.get('manifest_sha256') != json_hash({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Study manifest checksum differs.')
    if set(manifest['candidates']) != set(FAMILIES) or any(len(manifest['candidates'][f]) != (4 if f=='linear' else 8) for f in FAMILIES):
        raise ValueError('Frozen study must declare4Linear+8eachother family candidates.')
    _verify_splits(out,manifest)
    tasks = _all_known_tasks(out,manifest)
    states,results = [],{}
    for task in tasks:
        folder=out/'tasks'/task['task_id']
        done = completed(out,task,manifest)
        if done: results[task['task_id']]=done
        state = 'COMPLETE' if done else 'FAILED' if (folder/'FAILED.json').exists() else 'PARTIAL' if (folder/'COMMIT.json').exists() else 'PLANNED'
        states.append({'task_id':task['task_id'],'state':state})
    n = sum(t['state']=='COMPLETE' for t in states)
    if (out/'challengers.json').exists():
        if not all(t['task_id'] in results for t in discovery_tasks(manifest)):
            raise ValueError('Challenger lock exists before discovery completed.')
        challengers=load_json(out/'challengers.json')
        if challengers != choose_challengers(manifest,results):
            raise ValueError('Locked challengers differ from the completed discovery objective.')
    if (out/'selection.json').exists():
        if not all(t['task_id'] in results for t in confirmation_tasks(manifest,challengers)):
            raise ValueError('Selection lock exists before confirmation completed.')
        if load_json(out/'selection.json') != select_recipes(manifest,challengers,results):
            raise ValueError('Locked selection differs from the completed paired confirmation objective.')
    complete=False
    if (out/'STUDY_COMPLETE.json').exists():
        marker=load_json(out/'STUDY_COMPLETE.json')
        expected={t['task_id']:sha256_file(out/'tasks'/t['task_id']/'COMPLETE.json') for t in tasks if t['task_id'] in results}
        analyses={name:sha256_file(out/name) for name in ('analysis/summary.json','analysis/summary.csv')}
        if (n!=61 or len(tasks)!=61 or marker.get('status')!='COMPLETE'
                or marker.get('study_manifest_sha256')!=manifest['manifest_sha256']
                or marker.get('task_completion_sha256')!=expected
                or marker.get('analysis_sha256')!=analyses):
            raise ValueError('Study completion marker/summary/task hashes differ; preserve for diagnosis.')
        complete=True
    return {'status':'COMPLETE' if complete else 'INCOMPLETE',
            'completed_tasks':n,'maximum_tasks':61,'known_tasks':len(tasks),'tasks':states,
            'clock':load_json(out/'clock.json') if (out/'clock.json').exists() else {}}


class StudyLease:
    """Drive advisory lease, token-checked per epoch; serialized local acquisition.

    O_EXCL guards the short ownership read/modify/write transaction. An abrupt
    crash inside that guard fails closed instead of guessing that another writer
    is dead. Drive FUSE is not a distributed lock service: use one Colab runtime.
    """
    def __init__(self,out,accrue=True):
        self.out=Path(out); self.path=self.out/'ACTIVE_OWNER.json'; self.token=uuid.uuid4().hex
        self.guard=self.out/'OWNER_TRANSACTION.lock'; self.accrue=accrue
    @contextlib.contextmanager
    def _guard(self):
        # Reject a mistyped old output path before even adding ownership files.
        legacy_markers=('run_manifest.json','freeze_manifest.json','student_run_export.json')
        if any((self.out/name).exists() for name in legacy_markers):
            raise ValueError('This is a historical experiment folder. Use the separate V3 run directory.')
        manifest=self.out/'study_manifest.json'
        if manifest.exists() and load_json(manifest).get('schema_version')!=3:
            raise ValueError('This is a different historical study; V3 cannot write into it.')
        self.out.mkdir(parents=True,exist_ok=True)
        try:
            descriptor=os.open(self.guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as error:
            raise RuntimeError('Study ownership transaction is busy or was interrupted. Stop all other runtimes; '
                               'preserve/inspect OWNER_TRANSACTION.lock before removing a stale guard.') from error
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump({'token':self.token,'host':socket.gethostname(),'pid':os.getpid(),
                           'created_unix':time.time()},stream);stream.flush();os.fsync(stream.fileno())
            yield
        finally:
            self.guard.unlink(missing_ok=True)
    def __enter__(self):
        with self._guard():
            now=time.time()
            if self.path.exists():
                old=load_json(self.path)
                if old.get('active') and now-old['heartbeat_unix'] < 600:
                    raise RuntimeError('Another notebook owns this study, or a crashed lease is younger than10min. '
                                       'Stop other runtimes; retry after lease expires.')
            self.record={'token':self.token,'host':socket.gethostname(),'pid':os.getpid(),
                         'active':True,'heartbeat_unix':now}
            atomic_json(self.path,self.record)
            self.last=time.monotonic()
        return self
    def activate_clock(self):
        self.accrue=True;self.last=time.monotonic()
    def _heartbeat_unlocked(self):
        record=load_json(self.path)
        if record['token']!=self.token:
            raise RuntimeError('Study ownership changed; refusing concurrent writer.')
        now=time.monotonic()
        if self.accrue and (self.out/'clock.json').exists():
            clock=load_json(self.out/'clock.json')
            clock['active_run_seconds']+=now-self.last
            atomic_json(self.out/'clock.json',clock)
        self.last=now
        self.record['heartbeat_unix']=time.time();atomic_json(self.path,self.record)
    def heartbeat(self):
        with self._guard(): self._heartbeat_unlocked()
    def __exit__(self,exc_type,exc_value,traceback):
        try:
            with self._guard():
                self._heartbeat_unlocked()
                self.record['active']=False;atomic_json(self.path,self.record)
        except Exception:
            # Preserve the training/source/integrity failure that caused exit.
            if exc_type is None: raise
        return False


def _forecast_rates(out):
    """Conservative family estimates, updated with completed task wall time."""
    rates=dict(OLD_FIT_SECONDS)
    attempts = load_json(Path(out)/'attempts.json') if (Path(out)/'attempts.json').exists() else []
    for attempt in attempts:
        if attempt['completed']:
            rates[attempt['family']]=max(rates[attempt['family']],attempt['wall_seconds'])
    return rates


def _full_remaining_cost(manifest, results, out):
    """Diagnostic only: an estimate above the cap is not a reason to stop now."""
    remaining = {f:len(manifest['candidates'][f])+2+len(manifest['protocol']['final_seeds']) for f in FAMILIES}
    for task in _all_known_tasks(out,manifest):
        if task['task_id'] in results: remaining[task['family']]-=1
    rates=_forecast_rates(out)
    p=manifest['protocol']
    return sum(remaining[f]*rates[f] for f in FAMILIES)*(1+p['forecast_buffer_fraction'])+p['finalization_reserve_seconds']


def _remaining_cost(manifest, results, out):
    """Reserve the next discovery job and all paired confirmation/final jobs.

    A forecast for all remaining discovery candidates is advisory: demanding
    that it fit before every job caused an immediate stop after one slow job.
    Later phases still reserve every remaining confirmation/final seed. The
    process clock and per-task deadline continue to enforce the configured cap;
    this policy does not promise that all 61 jobs can finish in that time.
    """
    known=_all_known_tasks(out,manifest)
    pending=[task for task in known if task['task_id'] not in results]
    p=manifest['protocol']
    if not pending: return p['finalization_reserve_seconds']
    # Include undeclared-yet tasks: challengers/winners become known only after
    # their preceding phase completes, but their seed counts are already fixed.
    remaining={f:2+len(p['final_seeds']) for f in FAMILIES}
    for task in known:
        if task['phase'] in ('confirmation','final') and task['task_id'] in results:
            remaining[task['family']]-=1
    next_task=pending[0]
    if next_task['phase']=='discovery': remaining[next_task['family']]+=1
    elif next_task['phase'] not in ('confirmation','final'):
        raise ValueError('Unexpected task phase in budget reservation.')
    rates=_forecast_rates(out)
    return sum(remaining[f]*rates[f] for f in FAMILIES)*(1+p['forecast_buffer_fraction'])+p['finalization_reserve_seconds']


def execute(out, protocol_path, data_dir, device, max_tasks=None):
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks<1):
        raise ValueError('max_tasks must be a positive integer.')
    if str(device)!='cuda' or not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Full optimization is student-run on A100. CPU is only for smoke/software checks.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out=Path(out)
    with StudyLease(out,accrue=False) as lease:
        manifest=_prepare(out,protocol_path,data_dir,device)
        current=status(out,manifest)
        if current['status']=='COMPLETE':
            print('ALREADY COMPLETE: verified tasks skipped.',flush=True); return current
        dataset,_=load_training_data(data_dir,manifest['protocol']['expected_dataset_sha256'])
        with np.load(out/'split_ids.npz',allow_pickle=False) as saved:
            ids={k:saved[k].copy() for k in saved.files}
        results={t['task_id']:r for t in _all_known_tasks(out,manifest) if (r:=completed(out,t,manifest)) is not None}
        lease.activate_clock()
        n_new=0;warned_full_forecast=False
        def run_tasks(tasks):
            nonlocal n_new,warned_full_forecast
            for task in tasks:
                if task['task_id'] in results: continue
                lease.heartbeat();clock=load_json(out/'clock.json')
                left=clock['budget_seconds']-clock['active_run_seconds']
                forecast=_remaining_cost(manifest,results,out)
                full_forecast=_full_remaining_cost(manifest,results,out)
                if forecast>left:
                    atomic_json(out/'STOP_REASON.json',{'status':'INCOMPLETE_TIME_RESERVATION','seconds_left':left,
                                'projected_remaining_seconds':forecast,'full_projected_remaining_seconds':full_forecast,
                                'reservation_policy':'next_discovery_plus_all_confirmation_and_final'})
                    print('TIME RESERVATION: checkpoint preserved; export partial results.',flush=True);return False
                if full_forecast>left and not warned_full_forecast:
                    print('TIME FORECAST WARNING: all61 jobs may exceed the remaining budget. '
                          'Continuing while preserving confirmation/final reserves; '
                          f"configured {clock['budget_seconds']/3600:g}-hour cap unchanged.",flush=True)
                    warned_full_forecast=True
                if max_tasks is not None and n_new>=max_tasks: return False
                print(f"\nTASK {len(results)+1}/61: {task['task_id']}",flush=True)
                prefix='original_train' if task['phase']=='final' else 'inner_train'
                validation='old_validation' if task['phase']=='final' else 'inner_validation'
                start=time.monotonic()
                deadline=start+max(0,left-manifest['protocol']['finalization_reserve_seconds'])
                result=None;failure=None
                try:
                    result=train_task(out/'tasks'/task['task_id'],task,dataset,dataset,ids[prefix],ids[validation],
                                      task_binding(manifest,task,load_json(out/'run_environment.json')),device,
                                      stop_at=deadline,heartbeat=lease.heartbeat)
                except Exception as error:
                    failure={'type':type(error).__name__,'message':str(error)}
                    raise
                finally:
                    attempt={'task_id':task['task_id'],'family':task['family'],'wall_seconds':time.monotonic()-start,
                             'completed':result is not None,'failure':failure}
                    try:
                        path=out/'attempts.json';attempts=load_json(path) if path.exists() else [];attempts.append(attempt);atomic_json(path,attempts)
                    except Exception as ledger_error:
                        if failure is None: raise
                        print('Could not persist failed-attempt ledger: '+str(ledger_error),flush=True)
                lease.heartbeat()
                if result is None: return False
                results[task['task_id']]=result;n_new+=1
            return True
        if not run_tasks(discovery_tasks(manifest)): return status(out,manifest)
        challengers=choose_challengers(manifest,results)
        atomic_json(out/'challengers.json',challengers,immutable=True)
        if not run_tasks(confirmation_tasks(manifest,challengers)): return status(out,manifest)
        selection=select_recipes(manifest,challengers,results)
        atomic_json(out/'selection.json',selection,immutable=True)
        if not run_tasks(final_tasks(manifest,selection)): return status(out,manifest)
        summarize(out,manifest,selection,results)
        atomic_json(out/'STUDY_COMPLETE.json',{'status':'COMPLETE','study_manifest_sha256':manifest['manifest_sha256'],
                    'task_completion_sha256':{t['task_id']:sha256_file(out/'tasks'/t['task_id']/'COMPLETE.json') for t in _all_known_tasks(out,manifest)},
                    'analysis_sha256':{name:sha256_file(out/name) for name in ('analysis/summary.json','analysis/summary.csv')},
                    'study_type':'post_test_exploratory_validation_only'},immutable=True)
    return status(out,manifest)


def summarize(out,manifest,selection,results):
    rows=[]
    for f in FAMILIES:
        tasks=[t for t in final_tasks(manifest,selection) if t['family']==f]
        metrics=[results[t['task_id']]['metrics'] for t in tasks]
        row={'model':f,'candidate_id':selection['selected'][f]['candidate']['candidate_id'],'n':len(metrics)}
        for key in ('macro_f1','accuracy','loss'):
            values=[m['best_f1_metrics'][key] for m in metrics]
            row[key+'_mean']=statistics.mean(values);row[key+'_sample_sd']=statistics.stdev(values)
        row['selected_epochs']=[m['best_f1_epoch'] for m in metrics]
        row['parameters']=metrics[0]['parameters'];rows.append(row)
    atomic_json(Path(out)/'analysis/summary.json',{'final_exploratory_validation':rows,'inner_confirmation':selection,
                'limitations':['Previously seen validation selects final checkpoint; no new blind test','Inner selection uses only2seeds','Historical CE-selected references differ in protocol','No causal claim without fresh matched final anchors']})
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_bytes(Path(out)/'analysis/summary.csv',stream.getvalue().encode())


def smoke(out, manifest, data_dir, device):
    with StudyLease(out,accrue=False):
        return _smoke(out,manifest,data_dir,device)


def _smoke(out, manifest, data_dir, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    dataset,_=load_training_data(data_dir,manifest['protocol']['expected_dataset_sha256'])
    with np.load(Path(out)/'split_ids.npz',allow_pickle=False) as saved: ids=saved['inner_train'][:8]
    images=torch.stack([dataset[int(i)][0] for i in ids]).to(device)
    labels=torch.tensor([dataset[int(i)][1] for i in ids],dtype=torch.long,device=device)
    checks=[];torch.set_num_threads(manifest['protocol']['cpu_threads']);session=uuid.uuid4().hex
    for f in FAMILIES:
        # One tiny optimizer step per recipe, never a study training trajectory.
        for candidate in manifest['candidates'][f]:
            model=build_model(f,candidate['config'],101).to(device)
            cfg=candidate['config']
            optimizer_class={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[cfg['optimizer'].lower()]
            optimizer=optimizer_class(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            logits=model(images);loss=torch.nn.functional.cross_entropy(logits,labels)
            assert logits.shape==(8,10) and torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            checkpoint_check=None
            if candidate==manifest['candidates'][f][0]:
                model.eval()
                with torch.no_grad(): expected=model(images).detach().cpu()
                checkpoint=Path(out)/'smoke_checkpoints'/session/(f+'_C0.pt')
                digest=atomic_torch_save(checkpoint,{'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                                    'scope':'one_software_step_on8training_images_not_study_results'})
                restored=build_model(f,cfg,101).to(device)
                restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
                restored.eval()
                with torch.no_grad(): actual=restored(images).detach().cpu()
                assert torch.equal(expected,actual),'Smoke checkpoint reload predictions differ.'
                checkpoint_check={'path':checkpoint.relative_to(out).as_posix(),'sha256':digest,'exact_logits_equal':True}
                del restored
            checks.append({'family':f,'candidate_id':candidate['candidate_id'],'finite_forward_backward':True,
                           'finite_optimizer_step':True,'checkpoint_reload':checkpoint_check,
                           'parameters':sum(p.numel() for p in model.parameters())})
            del model,optimizer
    atomic_json(Path(out)/'smoke_checks.json',{'scope':'One8-image forward/backward/optimizer step per candidate; '
                'C0checkpoint roundtrip per family; not model-performance evidence','session':session,'checks':checks})
    print('SMOKE DONE:36 candidates optimizer checks,5checkpoint roundtrips; no full training or official-test inference.',flush=True)


def export_results(out):
    with StudyLease(out,accrue=False):
        return _export_results(out)


def _export_results(out):
    out=Path(out);manifest=load_json(out/'study_manifest.json')
    current=status(out,manifest)
    for relative,expected in manifest['source']['files_sha256'].items():
        source=ROOT/relative
        payload=source.read_bytes().replace(b'\r\n',b'\n') if source.suffix in ('.py','.json','.txt') else source.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected: raise ValueError('Source changed before export: '+relative)
        path=out/'source_snapshot'/relative
        if path.exists() and path.read_bytes()!=payload: raise ValueError('Source snapshot changed: '+relative)
        atomic_bytes(path,payload)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    atomic_bytes(out/'export_pip_freeze.txt',freeze.encode())
    atomic_json(out/'STUDY_PROGRESS.json',current)
    payloads=[p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()
              and p.name not in ('artifact_manifest.json','export_receipt.json','ACTIVE_OWNER.json','OWNER_TRANSACTION.lock')
              and not p.name.endswith('.tmp')]
    hashes={p.relative_to(out).as_posix():sha256_file(p) for p in payloads}
    atomic_json(out/'artifact_manifest.json',hashes)
    archive=out.parent/(out.name+'_results.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*payloads,out/'artifact_manifest.json']: z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise ValueError('Export ZIP CRC validation failed.')
        for name,expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=expected: raise ValueError('ZIP payload hash mismatch: '+name)
    os.replace(temporary,archive)
    receipt={'status':current['status'],'completed_tasks':current['completed_tasks'],'maximum_tasks':61,
             'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256_file(archive),
             'exported_payload_count':len(hashes),'official_test_evaluated':False}
    atomic_json(out/'export_receipt.json',receipt)
    return receipt
