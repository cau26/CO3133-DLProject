"""Publish verified GarmentLab measurements as a static, local results page.

This command derives documents from saved results; it does not train, select a
model, fit calibration or publish anything to a remote service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_a1_pages import frame, table, source, add_garmentlab_v2_section

ROOT = Path(__file__).resolve().parents[1]
RESULT_SHA = '662b2fe873223abf25fe48c4ece51c1c3d5738af97a1602d893525de0d53f2a4'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(run: Path, release_summary: Path | None = None):
    archive_audit = json.loads((ROOT / 'verification/GARMENTLAB_RECOVERY_ARCHIVE_RETURN_20261028.json').read_text(encoding='utf-8'))
    if archive_audit['archive_sha256'] != RESULT_SHA or not archive_audit['all_payload_hashes_pass']:
        raise ValueError('Expected the fully audited returned recovery archive')
    manifest = json.loads((run / 'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['payloads']:
        path = run / entry['path']
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Returned evidence changed: ' + entry['path'])
    summary = json.loads((run / 'FINAL_SUMMARY.json').read_text(encoding='utf-8'))
    if summary['smoke_only'] or summary['task_count'] != 6:
        raise ValueError('Expected six real completed tasks')
    out = ROOT / 'output/garmentlab_final'
    (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    (out / 'evidence').mkdir(exist_ok=True)
    figures = {}
    for name in ['selection_curves.png', 'assessment_confusion.png', 'assessment_reliability.png', 'corruption_stress.png']:
        value = (run / 'evaluation/figures' / name).read_bytes()
        target = out / 'figures' / name
        target.write_bytes(value)
        figures[name] = target.relative_to(ROOT).as_posix()
    for name in ['FINAL_SUMMARY.json', 'EVALUATION_BACKEND.json', 'study_manifest.json',
                 'evaluation/SELECTION_FREEZE.json', 'evaluation/calibration/CALIBRATION.json',
                 'release_candidate/ONNX_PARITY.json', 'release_candidate/CPU_BENCHMARK.json']:
        target = out / 'evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((run / name).read_bytes())
    local = json.loads(release_summary.read_text(encoding='utf-8')) if release_summary else None
    benchmark = None
    if local:
        if local.get('status') != 'READY_LOCAL_CPU_EXPORT' or local.get('ready_for_local_service') is not True:
            raise ValueError('A supplied derived release must have completed its separate verification')
        for item in local['artifacts']:
            path = release_summary.parent / item['path']
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Derived release evidence changed: ' + item['path'])
        benchmark = json.loads((release_summary.parent / 'release/CPU_BENCHMARK.json').read_text(encoding='utf-8'))
    raw = summary['assessment'][summary['deployment_task_id']]
    calibrated = summary['deployment_calibrated_assessment']
    data = {
        'schema_version': 'garmentlab_verified_publication_v1',
        'original_recovery_archive_sha256': RESULT_SHA,
        'original_summary_sha256': sha(run / 'FINAL_SUMMARY.json'),
        'original_recovery_status': summary['status'],
        'original_ready_for_local_service': summary['ready_for_local_service'],
        'labels': summary['labels'], 'split_counts': {'fit': 13827, 'selection': 1838, 'calibration': 1835, 'assessment': 327},
        'assessment_source_groups': 221,
        'seeds': [314, 2718, 5772], 'deployment_seed': summary['deployment_seed'],
        'selected_recipe': summary['selected_recipe'], 'recipe_summary': summary['recipe_summary'],
        'fixed_seed_raw_assessment': raw, 'fixed_seed_calibrated_assessment': calibrated,
        'calibration': summary['calibration'], 'corruptions': summary['corruptions'],
        'original_gates': summary['gates'], 'figures': figures,
        'derived_local_release': local,
        'derived_cpu_benchmark': benchmark,
        'scope': 'A real-image extension to the five-family Fashion-MNIST assignment. Recorded studies and later CPU export validation remain separate.',
        'limitations': summary['limitations'],
    }
    (out / 'data/publication_data.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

    rows = []
    for recipe, label in [('frozen', 'Frozen backbone'), ('finetune', 'Fine-tuning')]:
        metrics = summary['recipe_summary'][recipe]['assessment']
        rows.append([label, f"{metrics['accuracy']['mean']*100:.2f} ± {metrics['accuracy']['sample_sd']*100:.2f}%",
                     f"{metrics['macro_f1']['mean']:.4f} ± {metrics['macro_f1']['sample_sd']:.4f}"])
    results = table(['Recipe', 'Assessment accuracy', 'Assessment Macro-F1'], rows,
                    'Mean ± sample SD over three training seeds on the same 327 held-out crops', numeric=(1, 2))
    calibration_table = table(['Metric', 'Raw fine-tune', 'Temperature scaled'],
        [[label, f'{raw[key]:.4f}', f'{calibrated[key]:.4f}'] for key, label in
         [('nll', 'NLL'), ('ece', 'ECE, 15 bins'), ('brier_score', 'Brier score')]],
        'Fixed seed 314 on assessment, with calibration fitted on its separate partition', numeric=(1, 2))
    corruption_table = table(['Condition', 'Severity', 'Macro-F1', 'Coverage'],
        [[r['condition']['kind'], r['condition']['severity'], f"{r['metrics']['macro_f1']:.4f}",
          f"{r['metrics']['acceptance']['coverage']:.2%}"] for r in summary['corruptions']],
        'Nine project-specific stress tests on the same assessment crops', numeric=(1, 2, 3))
    def plot(name, caption):
        return f'<figure class="figure"><img src="{figures[name]}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'
    accepted = calibrated['acceptance']
    local_note = (f'<p>A separate Windows CPU re-export disables exporter optimization and passes the unchanged limit. '
                  f'Maximum logit error against the original saved PyTorch reference is '
                  f'{local["archived_oracle_parity"]["maximum_absolute_error"]:.8f} in the original audit. '
                  'The full 327-case check preserves every predicted label and accepted/review decision. '
                  'PNG preprocessing and the actual local API contracts also pass. '
                  'This derived release is ready for the declared local crop-classification application. '
                  'The original returned candidate remains unchanged.</p>'
                  f'<p>New CPU forward-only medians are {benchmark["measurements"]["pytorch"]["median_ms"]:.2f} ms '
                  f'for PyTorch and {benchmark["measurements"]["onnxruntime"]["median_ms"]:.2f} ms for ONNX Runtime, '
                  'with one CPU thread, five warmups and 30 calls. These exclude image decoding and HTTP. '
                  'They do not establish a speedup over a different machine or the original export.</p>'
                  if local else '<p>The original candidate remains blocked for activation while export verification is incomplete.</p>')
    body = f'''<section class="hero container"><p class="eyebrow">CO3133 · Assignment 1 extension · G-M10</p>
<h1>GarmentLab.<br>From benchmark to garment crops.</h1>
<p class="lead">A paired three-seed comparison of a frozen ImageNet backbone and fine-tuning on six Fashionpedia garment classes, followed by calibration, stress tests and an audited export.</p>
<div class="actions"><a class="button" href="assignment1-final.html">Read the core assignment</a><a class="text-link" href="{source('docs/garmentlab/REPRODUCIBILITY.md')}">Method and results</a></div></section>
<section class="landing-section container"><h2>Four data roles.</h2><p>17,827 crops form fit (13,827), selection (1,838), calibration (1,835) and assessment (327) partitions. Source-group and duplicate checks quarantine detected cross-partition overlaps. Assessment contains 221 source groups and comes from the released validation images. It is held out from model and threshold choice.</p>
<p>MobileNetV3-Small uses ImageNet initialization. Both recipes run 18 epochs with paired seeds 314, 2718 and 5772. Recipe selection averages selection Macro-F1; deployment seed 314 is preset. This extension supplements the five Fashion-MNIST models trained from scratch.</p></section>
<section class="landing-section container"><h2>Fine-tuning improves classification.</h2>{results}
<p>The preset seed 314 scores <strong>{raw['accuracy']:.2%} accuracy</strong> and <strong>{raw['macro_f1']:.4f} Macro-F1</strong>. This is a single checkpoint result. The table reports seed means. Seed SD measures training variation on one split.</p>
{plot('assessment_confusion.png', 'Fixed seed 314 assessment confusion matrix. Shirt/blouse has only 19 examples and recall 9/19.')}
{plot('selection_curves.png', 'Training and selection curves. Fine-tuning nearly fits training perfectly while selection NLL rises.')}</section>
<section class="landing-section container"><h2>Better labels need calibrated confidence.</h2>{calibration_table}
<p>Temperature {summary['calibration']['temperature']:.4f} changes probability confidence while preserving the winning class. Its threshold {summary['calibration']['review_threshold']:.6f} accepts {accepted['accepted_count']}/327 assessment crops at <strong>{accepted['accepted_accuracy']:.2%} accepted accuracy</strong> and <strong>{accepted['coverage']:.2%} coverage</strong>. The empirical 95% calibration target does not guarantee 95% accuracy on new inputs. Temperature and threshold share one calibration partition.</p>
{plot('assessment_reliability.png', 'Assessment reliability after freezing the temperature and review threshold.')}</section>
<section class="landing-section container"><h2>Noise and blur expose remaining weaknesses.</h2>{corruption_table}
{plot('corruption_stress.png', 'Stress tests modify the already letterboxed 224-pixel arrays. They are not smartphone-photo validation.')}
<p>The model expects one selected garment crop in its supported label space. It has not been evaluated as an object detector, on arbitrary smartphone photographs, or in production traffic. Small class support and multiple crops from common sources limit statistical conclusions.</p></section>
<section class="landing-section container"><h2>Export checks caught a numerical mismatch.</h2>
<p>The returned A100 research run completed all evaluations. Its ONNX artifact exceeded the preset absolute-logit limit: 0.000154495 versus 0.0001, with no top-1 disagreements in the 64-case audit. The release correctly retained candidate status.</p>{local_note}
<p><a href="{source('verification/GARMENTLAB_ONNX_EXPORT_FIX_20261028.json')}">Exporter diagnosis and measured trade-off</a> · <a href="output/garmentlab_final/data/publication_data.json">Verified publication data</a>. The static site presents results. The inference application runs separately through the local API.</p></section>
<section id="materials" class="landing-section container"><h2>Materials and attribution.</h2>
<ul class="resource-list"><li><a href="{source('docs/garmentlab/REPRODUCIBILITY.md')}">Methods and result interpretation</a><span>Analysis</span></li>
<li><a href="{source('verification/GARMENTLAB_RECOVERY_METRICS_RETURN_20261028.json')}">326 independent numeric checks</a><span>Evidence</span></li>
<li><a href="{source('verification/GARMENTLAB_RECOVERY_PROTOCOL_RETURN_20261028.json')}">38 protocol checks</a><span>Evidence</span></li>
<li><a href="{source('docs/garmentlab/REPRODUCIBILITY.md')}">Local inference and CPU export reproduction</a><span>Guide</span></li></ul>
<p>AI assisted LaTeX writing and pipeline development. <a href="{source('AI_USAGE.md')}">Full disclosure</a>.</p>
<p>Sources: <a href="https://github.com/cvdfoundation/fashionpedia">Fashionpedia dataset</a>, <a href="https://arxiv.org/abs/1905.02244">MobileNetV3</a>, <a href="https://proceedings.mlr.press/v70/guo17a.html">temperature scaling</a>, <a href="https://docs.pytorch.org/docs/2.11/onnx.html">PyTorch ONNX exporter</a>.</p></section>'''
    page = frame('GarmentLab real-image extension', 'Paired garment classification, calibration, stress testing and export verification.', body, study=True)
    (ROOT / 'garmentlab-results.html').write_text(add_garmentlab_v2_section(page), encoding='utf-8')
    print(json.dumps({'page': 'garmentlab-results.html', 'data': str(out / 'data/publication_data.json'), 'original_status': summary['status'], 'has_separate_local_release': bool(local)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--local-release-summary', type=Path)
    args = parser.parse_args()
    build(args.run, args.local_release_summary)
