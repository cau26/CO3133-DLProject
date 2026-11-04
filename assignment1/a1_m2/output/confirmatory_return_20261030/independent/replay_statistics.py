"""Independent statistics replay from committed predictions; no model inference.

Run from repository root. The original ZIP is read without modifying it. The
Fashion replay uses only the 60,000-image training label file, never test labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
ARCHIVE = None
FAMILIES = ['linear', 'mlp', 'cnn', 'gru', 'transformer']
SEEDS = [1009, 2027, 3037]
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def metrics(y, p, losses, classes=10):
    cm = np.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    true, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    f1 = np.divide(2 * tp, true + predicted, out=np.zeros(classes), where=true+predicted != 0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    recall = np.divide(tp, true, out=np.zeros(classes), where=true != 0)
    return {'accuracy': float(tp.sum()/len(y)), 'macro_f1': float(f1.mean()),
            'loss': float(np.mean(losses)), 'n_examples': len(y), 'confusion': cm.tolist(),
            'per_class': [{'label': CLASS_NAMES[i], 'index': i, 'support': int(true[i]),
                           'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                          for i in range(classes)]}


def mean_sd(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'values': values, 'n': len(values)}


def main():
    errors = []
    cells, trajectory_checks, prediction_cache = [], [], {}
    cell_index = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        read = lambda name: json.loads(z.read(name))
        manifest = read('fashion_confirmation/confirmation_manifest.json')
        summary = read('fashion_confirmation/analysis/summary.json')
        protocol = manifest['protocol']
        check(protocol['seeds'] == SEEDS and protocol['families'] == FAMILIES, 'Seed/family roster changed')
        check(protocol['epochs'] == 60 and protocol['primary_family'] == 'cnn', 'Primary endpoint changed')
        split = np.load(io.BytesIO(z.read('fashion_confirmation/split_ids.npz')), allow_pickle=False)
        train_ids, val_ids = split['train'], split['validation']
        check((len(train_ids), len(val_ids)) == (54000, 6000), 'Unexpected split sizes')
        check(np.array_equal(np.sort(np.r_[train_ids, val_ids]), np.arange(60000)), 'Split missing/overlapping IDs')
        labels_raw = (ROOT / 'data/FashionMNIST/raw/train-labels-idx1-ubyte').read_bytes()
        check(int.from_bytes(labels_raw[:4], 'big') == 2049, 'Bad training IDX magic')
        check(int.from_bytes(labels_raw[4:8], 'big') == 60000, 'Bad training IDX length')
        labels = np.frombuffer(labels_raw[8:], dtype=np.uint8)
        check(len(labels) == 60000, 'Truncated training labels')
        check(np.array_equal(val_ids, np.sort(val_ids)), 'Split validation order differs')
        support = np.bincount(labels[val_ids], minlength=10).tolist()

        for family in FAMILIES:
            for seed in SEEDS:
                for arm in ['C0', 'selected']:
                    task = f'{family}_{arm}_seed{seed}'
                    folder = f'fashion_confirmation/tasks/{task}'
                    history = [{k: float(v) for k,v in row.items()} for row in
                               csv.DictReader(io.StringIO(z.read(folder+'/engine/history.csv').decode()))]
                    check(len(history) == 60 and [r['epoch'] for r in history] == list(range(1,61)), task+' epochs')
                    check(all(math.isfinite(v) for r in history for v in r.values()), task+' nonfinite history')
                    check(all(r['train_n_examples'] == 54000 and r['val_n_examples'] == 6000 and
                              r['updates'] == r['epoch']*422 for r in history), task+' example/update count')
                    # Python min/max preserve earliest occurrence on exact ties.
                    expected_epochs = {'ce': min(history,key=lambda x:x['val_loss'])['epoch'],
                                       'f1': max(history,key=lambda x:(x['val_macro_f1'],-x['val_loss']))['epoch'],
                                       'final': 60}
                    engine_metrics = read(folder+'/engine/metrics.json')
                    check(engine_metrics['epochs_completed'] == 60 and engine_metrics['optimizer_updates'] == 25320,
                          task+' engine completion')
                    check(engine_metrics['best_ce_epoch'] == expected_epochs['ce'] and
                          engine_metrics['best_f1_epoch'] == expected_epochs['f1'], task+' selected epoch')
                    bound_task = read(folder+'/engine/TASK.json')['task']
                    check(bound_task['config'] == protocol['recipes'][family][arm]['config'], task+' recipe changed')
                    task_cells = read(folder+'/cells.json')
                    check(len(task_cells)==3 and {r['policy'] for r in task_cells}=={'ce','f1','final'}, task+' cell roster')
                    for cell in task_cells:
                        policy = cell['policy']
                        check(cell['checkpoint_epoch'] == expected_epochs[policy], task+'/'+policy+' epoch mismatch')
                        csv_data = z.read(folder+f'/predictions_{policy}.csv')
                        records = list(csv.DictReader(io.StringIO(csv_data.decode())))
                        ids = np.asarray([int(r['original_id']) for r in records])
                        y = np.asarray([int(r['true']) for r in records])
                        pred = np.asarray([int(r['pred']) for r in records])
                        losses = np.asarray([float(r['nll']) for r in records])
                        check(np.array_equal(ids,val_ids) and np.array_equal(y,labels[val_ids]), task+'/'+policy+' ID/label mismatch')
                        check(np.isin(pred,np.arange(10)).all() and np.isfinite(losses).all() and (losses>=0).all(),
                              task+'/'+policy+' prediction invalid')
                        actual = metrics(y,pred,losses)
                        epoch_row = history[int(cell['checkpoint_epoch'])-1]
                        for key in ['accuracy','macro_f1','loss']:
                            difference = abs(actual[key]-cell[key])
                            errors.append(difference)
                            check(difference < 2e-12, task+'/'+policy+'/'+key+' metric mismatch')
                            check(abs(actual[key]-epoch_row['val_'+key])<1e-7, task+'/'+policy+'/'+key+' history mismatch')
                        cm = np.asarray(actual['confusion'])
                        check(cm.sum()==6000 and np.array_equal(cm.sum(1),support),task+'/'+policy+' confusion invalid')
                        derived = {**{k:cell[k] for k in ['task_id','family','seed','arm','candidate_id','policy','checkpoint_epoch']},
                                   **actual, 'prediction_sha256': sha(csv_data)}
                        cells.append(derived)
                        cell_index[(family,seed,arm,policy)] = derived
                        prediction_cache[(family,seed,arm,policy)] = (y,pred,losses)
                    trajectory_checks.append({'task_id': task,'epochs_verified':60,
                        'optimizer_updates':25320,'ce_epoch':int(expected_epochs['ce']),
                        'f1_epoch':int(expected_epochs['f1']), 'parameter_count':engine_metrics['parameters'],
                        'fit_wall_seconds':engine_metrics['fit_wall_seconds'],
                        'train_final': {k:history[-1]['train_'+k] for k in ['accuracy','macro_f1','loss']},
                        'validation_final': {k:history[-1]['val_'+k] for k in ['accuracy','macro_f1','loss']},
                        'clip_fraction_mean':statistics.mean(r['clip_fraction'] for r in history),
                        'peak_gradient_norm':max(r['gradient_norm_max'] for r in history)})

        aggregate = []
        for family in FAMILIES:
            for arm in ['C0','selected']:
                for policy in ['ce','f1','final']:
                    rows = [cell_index[family,seed,arm,policy] for seed in SEEDS]
                    item = {'family':family,'arm':arm,'policy':policy,'n_seeds':3,
                            'checkpoint_epochs':[r['checkpoint_epoch'] for r in rows]}
                    for metric in ['accuracy','macro_f1','loss']:
                        item[metric] = mean_sd([r[metric] for r in rows])
                    item['per_class_mean_f1'] = [statistics.mean(r['per_class'][i]['f1'] for r in rows) for i in range(10)]
                    aggregate.append(item)
                    target = next(r for r in summary['cells'] if (r['family'],r['arm'],r['policy'])==(family,arm,policy))
                    for metric in ['accuracy','macro_f1','loss']:
                        check(abs(target[metric+'_mean']-item[metric]['mean'])<2e-12,'Aggregate mean replay failed')
                        check(abs(target[metric+'_sample_sd']-item[metric]['sample_sd'])<2e-12,'Aggregate SD replay failed')
        contrasts=[]
        definitions = {
            'recipe_fixed_final_primary':[('selected','final',1),('C0','final',-1)],
            'recipe_at_ce':[('selected','ce',1),('C0','ce',-1)],
            'recipe_at_f1':[('selected','f1',1),('C0','f1',-1)],
            'selector_within_C0':[('C0','f1',1),('C0','ce',-1)],
            'selector_within_selected':[('selected','f1',1),('selected','ce',-1)],
            'recipe_selector_interaction':[('selected','f1',1),('selected','ce',-1),('C0','f1',-1),('C0','ce',1)]}
        for family in FAMILIES:
            for name,terms in definitions.items():
                values=[sum(sign*cell_index[family,seed,arm,policy]['macro_f1'] for arm,policy,sign in terms) for seed in SEEDS]
                row={'family':family,'contrast':name,'seeds':SEEDS,**mean_sd(values),
                     'positive_seed_pairs':sum(v>0 for v in values),'negative_seed_pairs':sum(v<0 for v in values)}
                target=next(r for r in summary['paired_contrasts'] if r['family']==family and r['contrast']==name)
                check(np.allclose(values,target['macro_f1_delta_per_seed'],rtol=0,atol=2e-12),'Paired delta replay failed')
                check(abs(row['mean']-target['macro_f1_delta_mean'])<2e-12,'Paired mean replay failed')
                contrasts.append(row)
        cnn_errors=[]
        for seed in SEEDS:
            y,c0,_=prediction_cache['cnn',seed,'C0','final']
            _,selected,_=prediction_cache['cnn',seed,'selected','final']
            c0correct, selectedcorrect=c0==y, selected==y
            cnn_errors.append({'seed':seed,'both_correct':int((c0correct&selectedcorrect).sum()),
                'selected_correct_C0_wrong':int((~c0correct&selectedcorrect).sum()),
                'C0_correct_selected_wrong':int((c0correct&~selectedcorrect).sum()),
                'both_wrong':int((~c0correct&~selectedcorrect).sum()),
                'prediction_disagreement':int((c0!=selected).sum()),
                'delta_per_class_f1':[cell_index['cnn',seed,'selected','final']['per_class'][i]['f1']-
                                      cell_index['cnn',seed,'C0','final']['per_class'][i]['f1'] for i in range(10)]})
        recipes={family:{key:{'C0':protocol['recipes'][family]['C0']['config'].get(key),
                             'selected':protocol['recipes'][family]['selected']['config'].get(key)}
                         for key in set(protocol['recipes'][family]['C0']['config'])|set(protocol['recipes'][family]['selected']['config'])
                         if protocol['recipes'][family]['C0']['config'].get(key)!=protocol['recipes'][family]['selected']['config'].get(key)}
                 for family in FAMILIES}
        rgb_names=[n for n in z.namelist() if n.startswith('rgb_prospective/')]
        rgb_equal=[]
        for name in rgb_names:
            target=ROOT/'assets/prospective_completed'/name.removeprefix('rgb_prospective/')
            check(target.is_file() and sha(target.read_bytes())==sha(z.read(name)), 'Returned RGB differs from locked completed asset: '+name)
            rgb_equal.append({'path':name,'sha256':sha(z.read(name))})
        rgb=read('rgb_prospective/PROSPECTIVE_RESULTS.json')
        rgb_perclass=[]
        for i,label in enumerate(['shirt_blouse','top_tshirt_sweatshirt','jacket','pants','skirt','dress']):
            d={'label':label,'support':rgb['cohort']['class_support'][label]}
            for arm in ['C0','C4']:
                d[arm+'_mean_f1']=statistics.mean(m['raw']['per_class'][i]['f1'] for m in rgb['models'] if m['candidate_id']==arm)
            d['delta_mean_f1']=d['C4_mean_f1']-d['C0_mean_f1']
            rgb_perclass.append(d)
        rgb_mean_accuracy={arm:statistics.mean(m['raw']['accuracy'] for m in rgb['models'] if m['candidate_id']==arm) for arm in ['C0','C4']}
        result={
            'status':'PASS_INDEPENDENT_PREDICTION_AND_HISTORY_REPLAY',
            'archive_name':ARCHIVE.name,'archive_sha256':sha(ARCHIVE.read_bytes()),
            'method':'Independent NumPy confusion matrices, Macro-F1, accuracy and per-example NLL; no package summary function, fitting, inference, selection or access to official test labels.',
            'prediction_files_verified':90,'prediction_rows_verified':540000,
            'trajectories_verified':30,'history_epoch_rows_verified':1800,
            'maximum_metric_absolute_difference':max(errors),
            'training_label_sha256':sha(labels_raw),'split_support':support,
            'families':FAMILIES,'seeds':SEEDS,'primary_family':'cnn','primary_policy':'final',
            'official_test_evaluated':False,'environment':manifest['environment'],
            'core_clock':read('fashion_confirmation/clock.json'),
            'aggregation_scope':'Fixed reused validation split; three paired seeds. SD is descriptive; no population CI or significance claim.',
            'recipe_changes':recipes,'aggregate_cells':aggregate,'paired_contrasts':contrasts,
            'trajectory_checks':trajectory_checks,'cnn_fixed_epoch_error_pairs':cnn_errors,
            'rgb':{'returned_payloads_identical_to_frozen_completed_assets':len(rgb_equal),
                   'payload_hashes':rgb_equal,'primary':rgb['primary'],'cohort':rgb['cohort'],
                   'mean_accuracy':rgb_mean_accuracy,'mean_per_class_f1':rgb_perclass},
            'cells':cells}
    dump(OUT/'INDEPENDENT_STATISTICS.json',result)
    print(json.dumps({k:result[k] for k in ['status','prediction_files_verified','prediction_rows_verified',
        'trajectories_verified','history_epoch_rows_verified','maximum_metric_absolute_difference']},indent=2))
    print('Primary CNN',next(c for c in contrasts if c['family']=='cnn' and c['contrast']=='recipe_fixed_final_primary'))
    print('RGB per class',rgb_perclass)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=ROOT,
                        help='Repository with official training IDX labels and frozen RGB assets.')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    ARCHIVE, ROOT, OUT = args.archive, args.repo, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    main()
