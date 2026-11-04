"""Replay class-conditional selective prediction from frozen saved RGB logits.

This post-hoc diagnostic never constructs a model, trains, fits calibration,
selects a threshold, or changes the prospectively locked experiment record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('shirt_blouse', 'top_tshirt_sweatshirt', 'jacket', 'pants', 'skirt', 'dress')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def policy_decisions(probabilities, targets, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if (probabilities.ndim != 2 or probabilities.shape[1] < 2 or len(probabilities) == 0
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Expected finite normalized nonempty N-by-K probabilities.')
    if (targets.shape != (len(probabilities),) or targets.dtype.kind not in 'iu'
            or (targets < 0).any() or (targets >= probabilities.shape[1]).any()):
        raise ValueError('Targets must be integer class indices aligned with probabilities.')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0,1].')
    prediction = probabilities.argmax(1)
    return prediction, probabilities.max(1) >= threshold


def summarize(targets, prediction, accepted, labels=LABELS):
    targets, prediction, accepted = np.asarray(targets), np.asarray(prediction), np.asarray(accepted, dtype=bool)
    if not (targets.ndim == prediction.ndim == accepted.ndim == 1
            and targets.shape == prediction.shape == accepted.shape and len(targets)):
        raise ValueError('Targets, predictions and decisions must have equal nonempty shapes.')
    if (targets.dtype.kind not in 'iu' or prediction.dtype.kind not in 'iu'
            or (targets < 0).any() or (prediction < 0).any()
            or (targets >= len(labels)).any() or (prediction >= len(labels)).any()):
        raise ValueError('Unknown class index.')
    correct = targets == prediction

    def row(mask, label, index=None):
        support = int(mask.sum())
        acc = mask & accepted
        rev = mask & ~accepted
        nacc, ncorrect = int(acc.sum()), int((acc & correct).sum())
        errors = int((mask & ~correct).sum())
        nerr = nacc - ncorrect
        reviewed_errors = int((rev & ~correct).sum())
        return {'class_index': index, 'label': label, 'support': support,
                'accepted_count': nacc, 'accepted_correct': ncorrect, 'accepted_errors': nerr,
                'review_count': int(rev.sum()), 'review_correct': int((rev & correct).sum()),
                'review_errors': reviewed_errors, 'full_correct': int((mask & correct).sum()),
                'full_errors': errors, 'coverage': ratio(nacc, support),
                'accepted_accuracy': ratio(ncorrect, nacc), 'accepted_risk': ratio(nerr, nacc),
                'full_accuracy': ratio(int((mask & correct).sum()), support),
                'review_error_capture': ratio(reviewed_errors, errors),
                'accepted_error_escape_fraction': ratio(nerr, errors)}

    return {'overall': row(np.ones(len(targets), dtype=bool), 'all'),
            'true_class': [row(targets == c, label, c) for c, label in enumerate(labels)],
            'predicted_class': [row(prediction == c, label, c) for c, label in enumerate(labels)],
            'accepted_confusion': np.bincount(targets[accepted] * len(labels) + prediction[accepted],
                                             minlength=len(labels)**2).reshape(len(labels), len(labels)).tolist()}


def component_intervals(targets, prediction, accepted, groups, labels=LABELS, replicates=2000, seed=20261030):
    """Pointwise descriptive component bootstrap, conditional on one fixed model."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    ng = int(inverse.max()) + 1
    correct = targets == prediction
    rng = np.random.default_rng(seed)
    tables = {}
    for kind, by in [('true_class', targets), ('predicted_class', prediction)]:
        support = np.zeros((ng, len(labels)), dtype=np.int64)
        count = np.zeros_like(support)
        right = np.zeros_like(support)
        np.add.at(support, (inverse, by), 1)
        np.add.at(count, (inverse[accepted], by[accepted]), 1)
        np.add.at(right, (inverse[accepted & correct], by[accepted & correct]), 1)
        tables[kind] = (support, count, right)
    draws = {kind: {'coverage': [], 'accepted_accuracy': []} for kind in tables}
    for start in range(0, replicates, 100):
        weights = rng.multinomial(ng, np.full(ng, 1 / ng), size=min(100, replicates-start))
        for kind, (support, count, right) in tables.items():
            n, a, c = weights @ support, weights @ count, weights @ right
            draws[kind]['coverage'].append(np.divide(a, n, out=np.full(a.shape, np.nan), where=n > 0))
            draws[kind]['accepted_accuracy'].append(np.divide(c, a, out=np.full(a.shape, np.nan), where=a > 0))
    output = {'method': 'source/duplicate-component percentile bootstrap', 'replicates': replicates,
              'seed': seed, 'resampling_components': ng, 'confidence_level': .95,
              'scope': 'Post-hoc pointwise descriptive intervals for one fixed C4 seed314 model. No simultaneous classwise coverage, seed-population, deployment or future-accuracy guarantee.'}
    for kind in draws:
        output[kind] = []
        all_draws = {name: np.concatenate(parts) for name, parts in draws[kind].items()}
        for c, label in enumerate(labels):
            row = {'class_index': c, 'label': label}
            for name, values in all_draws.items():
                valid = values[np.isfinite(values[:, c]), c]
                row[name + '_interval'] = np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None
                row[name + '_defined_replicates'] = int(len(valid))
            output[kind].append(row)
    return output


def pct(value):
    return 'undefined' if value is None else f'{100*value:.2f}%'


def markdown_table(rows):
    lines = ['| Class | Accepted / support | Correct / accepted | Review | Coverage | Accepted accuracy |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['accepted_count']}/{row['support']} | {row['accepted_correct']}/{row['accepted_count']} | {row['review_count']} | {pct(row['coverage'])} | {pct(row['accepted_accuracy'])} |")
    return '\n'.join(lines)


def run(source, output):
    source, output = Path(source), Path(output)
    paths = {'freeze': source/'PROSPECTIVE_FREEZE.json', 'original_results': source/'PROSPECTIVE_RESULTS.json',
             'logits': source/'evaluation/final_C4_seed314/logits.npz', 'labels': source/'prepared/labels.npy',
             'index': source/'prepared/index_manifest.json', 'completion': source/'PROSPECTIVE_COMPLETE.json',
             'cohort_ready': source/'prepared/COHORT_READY.json',
             'logits_receipt': source/'evaluation/final_C4_seed314/LOGITS_RECEIPT.json'}
    before = {key: digest(path) for key, path in paths.items()}
    completion = json.loads(paths['completion'].read_text(encoding='utf-8'))
    ready = json.loads(paths['cohort_ready'].read_text(encoding='utf-8'))
    receipt = json.loads(paths['logits_receipt'].read_text(encoding='utf-8'))
    bound_predictions = {r['path']: r['sha256'] for r in completion['prediction_payloads']}
    if (completion['status'] != 'PROSPECTIVE_AUDIT_COMPLETE'
            or completion['freeze_sha256'] != before['freeze']
            or completion['results_sha256'] != before['original_results']
            or completion['cohort_ready_sha256'] != before['cohort_ready']
            or ready['files']['labels.npy'] != before['labels']
            or ready['files']['index_manifest.json'] != before['index']
            or bound_predictions['evaluation/final_C4_seed314/logits.npz'] != before['logits']
            or bound_predictions['evaluation/final_C4_seed314/LOGITS_RECEIPT.json'] != before['logits_receipt']
            or receipt['logits_sha256'] != before['logits']):
        raise ValueError('Frozen cohort/logit completion bindings differ.')
    freeze = json.loads(paths['freeze'].read_text(encoding='utf-8'))
    policy = freeze['policies']['C4']
    if policy['deployment_seed'] != 314 or policy['calibration']['policy']['mode'] != 'confidence_threshold':
        raise ValueError('Expected the fixed C4 seed314 confidence policy.')
    temperature, threshold = policy['calibration']['temperature'], policy['calibration']['review_threshold']
    if temperature <= 0 or not np.isfinite(temperature) or threshold != .88:
        raise ValueError('Frozen policy differs from the analyzed release.')
    index = json.loads(paths['index'].read_text(encoding='utf-8'))
    if tuple(index['labels']) != LABELS:
        raise ValueError('Class order differs.')
    y = np.load(paths['labels'], allow_pickle=False)
    logits = np.load(paths['logits'], allow_pickle=False)['logits'].astype(np.float64)
    if logits.shape != (len(y), len(LABELS)) or not np.isfinite(logits).all():
        raise ValueError('Invalid logits.')
    records = index['records']
    if len(records) != len(y) or any(row['array_index'] != i or row['class_index'] != int(y[i]) for i, row in enumerate(records)):
        raise ValueError('Labels/index order differs.')
    scores = logits / temperature
    scores -= scores.max(1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(1, keepdims=True)
    prediction, accepted = policy_decisions(p, y, threshold)
    counts = summarize(y, prediction, accepted)
    original = json.loads(paths['original_results'].read_text(encoding='utf-8'))
    stored = next(m['frozen_policy'] for m in original['models'] if m['task_id'] == 'final_C4_seed314')
    for key in ('accepted_count', 'accepted_correct', 'coverage', 'accepted_accuracy'):
        if not np.isclose(counts['overall'][key], stored[key], atol=2e-12, rtol=0):
            raise ValueError('Original global policy metrics do not reconcile: ' + key)
    groups = [row['evaluation_group_id'] for row in records]
    result = {'schema': 'garmentlab_posthoc_class_policy_v1',
              'analysis_source_sha256': digest(Path(__file__)),
              'analysis_runtime': {'python_major_minor': list(sys.version_info[:2]), 'numpy': np.__version__},
              'scope': 'Post-hoc diagnostic from saved logits; same frozen model, temperature, threshold and cohort. No inference, training, fitting or threshold adaptation.',
              'model': 'C4', 'training_seed': 314, 'temperature': temperature, 'review_threshold': threshold,
              'weights_sha256': receipt['weights_sha256'],
              'decision': 'accept if max softmax(logits / temperature) >= review_threshold',
              'source_payloads_sha256': before, 'labels': list(LABELS), **counts,
              'uncertainty': component_intervals(y, prediction, accepted, groups),
              'definitions': {'true_class': 'Condition on the annotated true label, available only in offline evaluation.',
                              'predicted_class': 'Condition on the model output label. Coverage denominator includes every prediction of that label, including reviewed predictions. Accepted accuracy is precision among accepted emissions.',
                              'review': 'A referral flag only; no human review outcomes are measured.'},
              'limitations': ['Diagnostic classes were examined after cohort results. No prespecified classwise hypothesis or multiplicity-adjusted guarantee.',
                             'Component-bootstrap intervals are pointwise and conditional on one trained model, source grouping and this license-shift cohort.',
                             'The original global calibration procedure does not certify each class or transfer its risk bound to the new cohort.',
                             'True-class and predicted-class denominators answer different questions; class-specific thresholds require new calibration data and a separate untouched audit cohort.']}
    output.mkdir(parents=True, exist_ok=True)
    (output/'POLICY_BY_CLASS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    for kind in ('true_class', 'predicted_class'):
        with (output/(kind+'.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(counts[kind][0]));writer.writeheader();writer.writerows(counts[kind])
    with (output/'accepted_confusion.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream);writer.writerow(['true_label', *LABELS])
        for label, row in zip(LABELS, counts['accepted_confusion']):writer.writerow([label, *row])
    md = '''# Class-conditional selective prediction

This post-hoc diagnostic replays the preset **C4 seed314** policy on the 2,766-crop prospective cohort. It uses temperature **5.345757234366385** and confidence threshold **0.88**, retaining equality in the accepted set. No fitting, model inference, calibration or policy adaptation takes place. Results refer to one fixed serving model, rather than means over three training seeds.

## Conditioning on the true class

''' + markdown_table([counts['overall'], *counts['true_class']]) + '''

Coverage uses all crops with a given annotated label. Accepted accuracy uses only accepted crops of that true class. These labels are available during offline evaluation, not when routing a new unlabeled input. Shirt/blouse sends 149 of 182 crops to review, but still misclassifies 11 of 33 accepted crops. Pants accepts 361 of 476 crops with one accepted error. The pooled 95.95% accepted accuracy therefore does not describe uniform performance across garment classes.

## Conditioning on the predicted class

''' + markdown_table([counts['overall'], *counts['predicted_class']]) + '''

Here the groups use the emitted model label. For example, 22 of 24 accepted shirt/blouse outputs are correct (91.67%), while 22 of 33 accepted true shirt/blouse inputs are correct (66.67%). The denominators differ because the latter also includes confidently misclassified shirts assigned to other labels. Accepted accuracy in the predicted-label table is precision among the accepted outputs of each label. Predicted jacket outputs have 76 correct predictions among 88 accepted outputs (86.36%), revealing an operational weakness that the pooled score alone obscures.

The policy refers 1,260 crops for review. Of the 495 errors in the full-cohort model predictions, 434 are referred and 61 escape into the accepted set. Review also includes 826 already correct model predictions. These are routing diagnostics; the study does not measure human correction quality or end-to-end human-assisted accuracy.

## Uncertainty and interpretation

The JSON retains 2,000-replicate source/duplicate-component percentile intervals for each class's coverage and accepted accuracy. They are post-hoc pointwise descriptive intervals, conditional on this fixed model and cohort. They do not provide simultaneous classwise certification, estimate the population of training seeds or transfer the original calibration risk guarantee to the license shift. Small accepted subsets, particularly 24 emitted shirt/blouse predictions, limit precision. Conditioning on the true label also differs from conditioning on a forecast probability; a true-class accepted-accuracy value alone does not directly test a probability-calibration identity.

The threshold remains fixed. Changing it by class using the inspected cohort would be further development, requiring independent calibration and a new untouched assessment population. The current evidence supports reporting the unequal routing/error burden, not claiming the defect has been fixed by this analysis.
'''
    (output/'POLICY_ANALYSIS.md').write_text(md, encoding='utf-8')
    if before != {key: digest(path) for key, path in paths.items()}:
        raise ValueError('A frozen input changed during analysis.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'output/confirmatory_return_20261030/rgb_prospective')
    parser.add_argument('--out', type=Path, default=ROOT/'output/policy_by_class_20261030')
    args = parser.parse_args()
    result = run(args.source, args.out)
    print(json.dumps({'status': 'PASS_FROZEN_POLICY_REPLAY', 'overall': result['overall'],
                      'true_class': result['true_class'], 'predicted_class': result['predicted_class']}, indent=2))


if __name__ == '__main__':
    main()
