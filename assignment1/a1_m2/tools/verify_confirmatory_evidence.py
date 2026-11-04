"""Verify the compact confirmation record with the Python standard library.

Run ``python tools/verify_confirmatory_evidence.py`` from the public checkout.
This reads hashes, JSON, CSV and numeric split IDs. It never deserializes a model,
loads test data, performs inference, trains, or modifies the evidence directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct
import sys
import zipfile

FAMILIES = ('linear', 'mlp', 'cnn', 'gru', 'transformer')
SEEDS = (1009, 2027, 3037)
POLICIES = ('ce', 'f1', 'final')
ARCHIVE_SHA256 = '0aa34ed4e5f51f008a404f414e4b789f4495150d85278499ea93bb5421d5e99f'
TOLERANCE = 2e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid_constant)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and '\\' not in relative,
            f'Nonportable evidence path: {relative!r}')
    name = PurePosixPath(relative)
    require(not name.is_absolute() and name.parts and
            all(part not in ('.', '..') and ':' not in part for part in name.parts),
            f'Unsafe evidence path: {relative!r}')
    target = root.joinpath(*name.parts)
    require(not target.is_symlink(), f'Symlink is not an evidence payload: {relative}')
    require(root.resolve() in target.resolve().parents,
            f'Evidence path escapes its root: {relative}')
    return target


def compare(actual, expected, context, tolerance=TOLERANCE):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f'{context}: calculated {actual!r}, recorded {expected!r}')


def read_numeric_ids(data):
    """Decode a one-dimensional, nonobject integer NPY without NumPy/pickle."""
    require(data[:6] == b'\x93NUMPY', 'Invalid split NPY signature')
    version = tuple(data[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported split NPY version')
    header_start = 10 if version == (1, 0) else 12
    size = struct.unpack('<H' if version == (1, 0) else '<I', data[8:header_start])[0]
    header = ast.literal_eval(data[header_start:header_start + size].decode(
        'utf-8' if version == (3, 0) else 'latin1'))
    require(set(header) == {'descr', 'fortran_order', 'shape'} and
            header['fortran_order'] is False and isinstance(header['shape'], tuple) and
            len(header['shape']) == 1 and type(header['shape'][0]) is int,
            'Split arrays must be one-dimensional numeric vectors')
    formats = {'<i8': '<q', '>i8': '>q', '<i4': '<i', '>i4': '>i', '|u1': '<B'}
    require(header['descr'] in formats, 'Unsupported/noninteger split NPY dtype')
    fmt = formats[header['descr']]
    values = data[header_start + size:]
    require(len(values) == header['shape'][0] * struct.calcsize(fmt), 'Truncated split NPY')
    return [item[0] for item in struct.iter_unpack(fmt, values)]


def verify_manifest(folder):
    compact = read_json(folder / 'COMPACT_MANIFEST.json')
    require(compact['schema_version'] == 1, 'Unknown compact evidence schema')
    require(compact['original_archive_sha256'] == ARCHIVE_SHA256,
            'Compact evidence identifies a different original return')
    hashes = compact['payloads_sha256']
    require(isinstance(hashes, dict) and hashes, 'Empty compact payload inventory')
    for name, expected in hashes.items():
        path = safe_path(folder, name)
        require(path.is_file(), f'Missing compact payload: {name}')
        require(digest(path) == expected, f'Compact payload checksum mismatch: {name}')
    original = read_json(folder / 'RESULT_MANIFEST.json')
    require(original['status'] == 'COMPLETE', 'Original result manifest was incomplete')
    omissions = compact['omissions']
    # The compact manifest is authoritative for present files; original records
    # also bind each retained raw payload while documenting intentional omissions.
    original_present, original_omitted = 0, 0
    for item in original['files']:
        name = item['path']
        path = safe_path(folder, name)
        if name in hashes:
            require(hashes[name] == item['sha256'] and path.stat().st_size == item['bytes'],
                    f'Compact/raw inventory disagreement: {name}')
            original_present += 1
        else:
            original_omitted += 1
    require(isinstance(omissions, list) and (not original_omitted or omissions),
            'Raw payload omissions are undocumented')
    return {'compact_payloads_verified': len(hashes),
            'original_payloads_retained': original_present,
            'original_payloads_omitted': original_omitted,
            'omissions': omissions, 'hashes': hashes}


def prediction_metrics(path, validation_ids, reference_labels=None):
    confusion = [[0] * 10 for _ in range(10)]
    nll = []
    labels = []
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['original_id', 'true', 'pred', 'nll'],
                f'Unexpected prediction columns: {path.name}')
        for index, row in enumerate(reader):
            require(index < len(validation_ids), 'Too many prediction rows')
            image_id, truth, prediction = (int(row[k]) for k in ('original_id', 'true', 'pred'))
            loss = float(row['nll'])
            require(image_id == validation_ids[index], 'Original validation ID/order mismatch')
            require(0 <= truth < 10 and 0 <= prediction < 10 and math.isfinite(loss) and loss >= 0,
                    'Invalid prediction class or NLL')
            if reference_labels is not None:
                require(truth == reference_labels[index], 'Validation labels differ between cells')
            labels.append(truth)
            confusion[truth][prediction] += 1
            nll.append(loss)
    require(len(labels) == 6000, 'Every checkpoint cell must contain 6,000 predictions')
    support = [sum(row) for row in confusion]
    predicted_support = [sum(row[k] for row in confusion) for k in range(10)]
    require(all(n > 0 for n in support), 'A Fashion-MNIST class is missing')
    f1 = [2 * confusion[k][k] / (support[k] + predicted_support[k]) for k in range(10)]
    return {'n_examples': len(labels),
            'accuracy': sum(confusion[k][k] for k in range(10)) / len(labels),
            'macro_f1': statistics.mean(f1), 'loss': math.fsum(nll) / len(labels),
            'confusion': confusion, 'class_support': support}, labels


def verify(root):
    root = root.resolve()
    folder = root / 'output/confirmatory_return_20261030'
    inventory = verify_manifest(folder)
    present = inventory.pop('hashes')
    fashion = folder / 'fashion_confirmation'
    manifest = read_json(fashion / 'confirmation_manifest.json')
    require(manifest['manifest_sha256'] == json_hash(
        {k: v for k, v in manifest.items() if k != 'manifest_sha256'}), 'Fashion manifest hash mismatch')
    protocol = manifest['protocol']
    require(tuple(protocol['families']) == FAMILIES and tuple(protocol['seeds']) == SEEDS and
            protocol['epochs'] == 60 and tuple(protocol['checkpoint_policies']) == POLICIES and
            protocol['primary_family'] == 'cnn' and
            protocol['primary_comparison'] == 'selected_minus_C0_at_fixed_final_epoch' and
            protocol['official_test_evaluated'] is False, 'Declared design/primary endpoint changed')
    require(digest(fashion / 'split_ids.npz') == manifest['split_file_sha256'], 'Split artifact mismatch')
    with zipfile.ZipFile(fashion / 'split_ids.npz') as archive:
        require(set(archive.namelist()) == {'train.npy', 'validation.npy'}, 'Unexpected split archive')
        train = read_numeric_ids(archive.read('train.npy'))
        validation = read_numeric_ids(archive.read('validation.npy'))
    require(len(train) == 54000 and len(validation) == 6000 and
            sorted(train + validation) == list(range(60000)), 'Invalid 54k/6k partition')
    split_bytes = b''.join(struct.pack('<q', n) for n in train + validation)
    require(hashlib.sha256(split_bytes).hexdigest() == protocol['expected_split_sha256'],
            'Original split identity mismatch')
    source = manifest['source']
    require(json_hash(source['files_sha256']) == source['sha256'], 'Source identity checksum mismatch')
    for relative, expected in source['files_sha256'].items():
        path = safe_path(folder / 'source_snapshot', relative)
        require(path.is_file() and hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                == expected, f'Locked source snapshot changed: {relative}')
    expected_binding = {'confirmation_manifest_sha256': manifest['manifest_sha256'],
        'source_sha256': source['sha256'], 'dataset_sha256': protocol['expected_dataset_sha256'],
        'split_sha256': protocol['expected_split_sha256'], 'environment': manifest['environment']}
    summary = read_json(fashion / 'analysis/summary.json')
    completion = read_json(fashion / 'STUDY_COMPLETE.json')
    lock = read_json(fashion / 'EVALUATION_LOCK.json')
    require(completion['status'] == 'COMPLETE' and completion['manifest_sha256'] == manifest['manifest_sha256'],
            'Study completion binding mismatch')
    require(completion['analysis_sha256'] == digest(fashion / 'analysis/summary.json') and
            completion['evaluation_lock_sha256'] == digest(fashion / 'EVALUATION_LOCK.json'),
            'Study summary/lock checksum mismatch')
    require(lock['lock_sha256'] == json_hash({k: v for k, v in lock.items() if k != 'lock_sha256'}) and
            lock['primary_family'] == 'cnn' and lock['primary_policy'] == 'final', 'Evaluation lock changed')
    lock_index = {(r['task_id'], r['policy']): r for r in lock['checkpoints']}
    require(len(lock_index) == len(lock['checkpoints']) == 90, 'Lock must identify 90 unique cells')
    index, history_rows, checkpoint_count, errors = {}, 0, 0, []
    labels = None
    task_ids = set()
    for family in FAMILIES:
        for seed in SEEDS:
            for arm in ('C0', 'selected'):
                task_id = f'{family}_{arm}_seed{seed}'
                task_ids.add(task_id)
                task_folder = fashion / 'tasks' / task_id
                task_binding = read_json(task_folder / 'engine/TASK.json')
                task = task_binding['task']
                require(task_binding['binding'] == expected_binding and task['task_id'] == task_id and
                        task['family'] == family and task['seed'] == seed and task['arm'] == arm and
                        task['epochs'] == 60 and task['config'] == protocol['recipes'][family][arm]['config'],
                        f'Task/config/environment binding differs: {task_id}')
                receipt_path = task_folder / 'TRAJECTORY_COMPLETE.json'
                require(completion['trajectory_receipts_sha256'][task_id] == digest(receipt_path),
                        f'Trajectory receipt hash differs: {task_id}')
                receipt = read_json(receipt_path)
                require(receipt['status'] == 'COMPLETE' and receipt['task_sha256'] == json_hash(task) and
                        receipt['binding_sha256'] == json_hash(expected_binding), f'Invalid trajectory: {task_id}')
                for relative, expected in receipt['artifacts_sha256'].items():
                    require(digest(safe_path(task_folder, relative)) == expected,
                            f'Trajectory payload mismatch: {task_id}/{relative}')
                engine = read_json(task_folder / 'engine/COMPLETE.json')
                require(engine['status'] == 'COMPLETE' and engine['task_sha256'] == json_hash(task) and
                        engine['binding_sha256'] == json_hash(expected_binding), 'Engine binding changed')
                for relative, expected in engine['artifacts_sha256'].items():
                    candidate = safe_path(task_folder / 'engine', relative)
                    if candidate.is_file():
                        require(digest(candidate) == expected, f'Engine payload mismatch: {task_id}/{relative}')
                    else:
                        require(relative.startswith('state-') or relative in ('COMMIT.json', 'COMMIT.previous.json'),
                                f'Unexpected missing engine evidence: {task_id}/{relative}')
                metrics = read_json(task_folder / 'engine/metrics.json')
                require(metrics == engine['metrics'] and metrics['epochs_completed'] == 60 and
                        metrics['optimizer_updates'] == 25320 and metrics['n_train'] == 54000 and
                        metrics['n_validation'] == 6000, f'Engine counts differ: {task_id}')
                with (task_folder / 'engine/history.csv').open(encoding='utf-8', newline='') as stream:
                    history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
                require(len(history) == 60 and [r['epoch'] for r in history] == list(range(1, 61)),
                        f'History is not 60 complete ordered epochs: {task_id}')
                require(all(math.isfinite(v) for row in history for v in row.values()) and
                        all(row['updates'] == row['epoch'] * 422 and row['train_n_examples'] == 54000 and
                            row['val_n_examples'] == 6000 for row in history), f'Invalid history: {task_id}')
                history_rows += len(history)
                epochs = {'ce': min(history, key=lambda r: r['val_loss'])['epoch'],
                          'f1': max(history, key=lambda r: (r['val_macro_f1'], -r['val_loss']))['epoch'],
                          'final': 60}
                require(metrics['best_ce_epoch'] == epochs['ce'] and metrics['best_f1_epoch'] == epochs['f1'],
                        f'History/checkpoint selector mismatch: {task_id}')
                cells = read_json(task_folder / 'cells.json')
                require(len(cells) == 3 and {r['policy'] for r in cells} == set(POLICIES) and
                        cells == receipt['cells'], f'Cell roster/receipt differs: {task_id}')
                for cell in cells:
                    policy = cell['policy']
                    require(cell['task_id'] == task_id and cell['family'] == family and cell['seed'] == seed and
                            cell['arm'] == arm and cell['checkpoint_epoch'] == epochs[policy],
                            f'Cell identity/epoch differs: {task_id}/{policy}')
                    checkpoint = safe_path(task_folder, cell['checkpoint'])
                    require(checkpoint.is_file() and digest(checkpoint) == cell['checkpoint_sha256'],
                            f'Checkpoint hash differs: {task_id}/{policy}')
                    locked = lock_index[task_id, policy]
                    require(all(locked[k] == cell[k] for k in
                            ('task_id', 'family', 'seed', 'arm', 'policy', 'checkpoint_epoch', 'checkpoint_sha256')) and
                            locked['checkpoint'] == f'tasks/{task_id}/{cell["checkpoint"]}',
                            f'Locked checkpoint identity differs: {task_id}/{policy}')
                    computed, truth = prediction_metrics(task_folder / f'predictions_{policy}.csv', validation, labels)
                    if labels is None:
                        labels = truth
                    for key in ('accuracy', 'macro_f1', 'loss'):
                        errors.append(abs(computed[key] - cell[key]))
                        compare(computed[key], cell[key], f'{task_id}/{policy}/{key}')
                        compare(computed[key], history[int(epochs[policy]) - 1]['val_' + key],
                                f'{task_id}/{policy}/history/{key}', tolerance=1e-7)
                    require(cell['n_examples'] == computed['n_examples'], 'Cell example count differs')
                    index[family, seed, arm, policy] = computed
                    checkpoint_count += 1
    require(set(completion['trajectory_receipts_sha256']) == task_ids, 'Unexpected/missing completed trajectory')
    require(len(summary['cells']) == 30 and len(summary['paired_contrasts']) == 30,
            'Unexpected aggregate/contrast roster')
    aggregate_index = {(r['family'], r['arm'], r['policy']): r for r in summary['cells']}
    require(len(aggregate_index) == 30, 'Duplicate aggregate cell')
    for family in FAMILIES:
        for arm in ('C0', 'selected'):
            for policy in POLICIES:
                recorded = aggregate_index[family, arm, policy]
                require(recorded['n_seeds'] == 3, 'Aggregate seed count differs')
                for metric in ('accuracy', 'macro_f1', 'loss'):
                    values = [index[family, seed, arm, policy][metric] for seed in SEEDS]
                    compare(statistics.mean(values), recorded[metric + '_mean'], 'Aggregate mean')
                    compare(statistics.stdev(values), recorded[metric + '_sample_sd'], 'Aggregate sample SD')
    terms = {
        'recipe_fixed_final_primary': [('selected', 'final', 1), ('C0', 'final', -1)],
        'recipe_at_ce': [('selected', 'ce', 1), ('C0', 'ce', -1)],
        'recipe_at_f1': [('selected', 'f1', 1), ('C0', 'f1', -1)],
        'selector_within_C0': [('C0', 'f1', 1), ('C0', 'ce', -1)],
        'selector_within_selected': [('selected', 'f1', 1), ('selected', 'ce', -1)],
        'recipe_selector_interaction': [('selected', 'f1', 1), ('selected', 'ce', -1), ('C0', 'f1', -1), ('C0', 'ce', 1)]}
    contrast_index = {(r['family'], r['contrast']): r for r in summary['paired_contrasts']}
    require(len(contrast_index) == 30, 'Duplicate paired contrast')
    for family in FAMILIES:
        for name, definition in terms.items():
            values = [sum(sign * index[family, seed, arm, policy]['macro_f1'] for arm, policy, sign in definition)
                      for seed in SEEDS]
            recorded = contrast_index[family, name]
            require(tuple(recorded['seeds']) == SEEDS and recorded['n_paired_seeds'] == 3 and
                    len(recorded['macro_f1_delta_per_seed']) == 3, 'Paired contrast seed roster differs')
            for calculated, expected in zip(values, recorded['macro_f1_delta_per_seed']):
                compare(calculated, expected, 'Paired seed contrast')
            compare(statistics.mean(values), recorded['macro_f1_delta_mean'], 'Paired mean')
            compare(statistics.stdev(values), recorded['macro_f1_delta_sample_sd'], 'Paired sample SD')
    require(summary['official_test_evaluated'] is False, 'Summary claims an official-test evaluation')
    original_label_check = 'Cross-cell labels and locked split verified; raw training labels are not distributed.'
    training_labels = root / 'data/FashionMNIST/raw/train-labels-idx1-ubyte'
    if training_labels.is_file():
        raw = training_labels.read_bytes()
        require(raw[:8] == struct.pack('>II', 2049, 60000) and len(raw) == 60008,
                'Invalid optional official training IDX labels')
        require([raw[8 + image_id] for image_id in validation] == labels,
                'Optional official training labels disagree with saved targets')
        original_label_check = 'Saved targets also match the optional official training IDX label file.'
    rgb = folder / 'rgb_prospective'
    rgb_marker = read_json(rgb / 'PROSPECTIVE_COMPLETE.json')
    require(rgb_marker['status'] == 'PROSPECTIVE_AUDIT_COMPLETE', 'RGB audit is incomplete')
    for key, relative in [('results_sha256', 'PROSPECTIVE_RESULTS.json'),
                          ('freeze_sha256', 'PROSPECTIVE_FREEZE.json'),
                          ('cohort_ready_sha256', 'prepared/COHORT_READY.json'),
                          ('inference_inputs_sha256', 'evaluation/INFERENCE_INPUTS.json')]:
        require(digest(rgb / relative) == rgb_marker[key], f'RGB completion binding differs: {key}')
    require(len(rgb_marker['prediction_payloads']) == 18, 'RGB must retain 18 prediction payloads')
    for item in rgb_marker['prediction_payloads']:
        require(digest(safe_path(rgb, item['path'])) == item['sha256'], 'RGB prediction hash differs')
    result_status = read_json(folder / 'RESULT_STATUS.json')
    require(result_status['status'] == 'COMPLETE' and result_status['fashion']['completed_trajectories'] == 30 and
            result_status['fashion']['completed_checkpoint_cells'] == 90 and
            result_status['new_blind_Fashion_MNIST_test'] is False, 'Bundle completion/scope differs')
    primary = contrast_index['cnn', 'recipe_fixed_final_primary']
    return {'status': 'PASS_COMPACT_EVIDENCE_VERIFICATION',
        'original_archive_sha256': ARCHIVE_SHA256, **inventory,
        'trajectories_verified': len(task_ids), 'checkpoint_hashes_verified': checkpoint_count,
        'prediction_files_recomputed': len(index), 'prediction_rows_recomputed': len(index) * 6000,
        'history_epochs_verified': history_rows, 'aggregate_cells_recomputed': len(aggregate_index),
        'paired_contrasts_recomputed': len(contrast_index), 'maximum_cell_metric_difference': max(errors),
        'primary_CNN_epoch60_mean_delta': primary['macro_f1_delta_mean'],
        'primary_CNN_epoch60_seed_deltas': primary['macro_f1_delta_per_seed'],
        'target_verification_scope': original_label_check,
        'RGB_scope': 'Completion and 18 saved prediction payload hashes verified; this standard-library command does not recompute RGB softmax or the NumPy bootstrap.',
        'scope': 'Read-only compact-artifact verification and Fashion CSV statistics. No model unpickling, inference, training, official-test access or claim of a new blind test. Checkpoint hashes verify file identity, not numerical replay or operator identity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Public repository checkout root (default: this script\'s parent checkout).')
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (KeyError, OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
        print(json.dumps({'status': 'FAIL_COMPACT_EVIDENCE_VERIFICATION', 'error': str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
