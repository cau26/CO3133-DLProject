"""Separate matched Fashion-MNIST study and prospective RGB audit.

Existing studies and photographs are inputs. New results always have a separate
identity. A complete archive requires both the 30-trajectory study and RGB audit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

from garmentlab.runtime import atomic_json, digest, utc_now

REFERENCE = ROOT / 'output/garmentlab_improve_return_20261028'
PUBLICATION = ROOT / 'output/a1_final/data/publication_data.json'
PREVIEWS = ROOT / 'assets/prospective_exposure'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protect(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing confirmation identity differs: ' + path.name)
    else:
        atomic_json(path, value)


class Budget:
    def __init__(self, out, seconds=18000, lease_check=None):
        self.path = Path(out) / 'BUNDLE_CLOCK.json'
        previous = read(self.path) if self.path.exists() else {}
        if previous and previous['budget_seconds'] != seconds:
            raise ValueError('The bundle budget is immutable.')
        self.previous = float(previous.get('active_seconds', 0))
        self.seconds, self.started, self.last = seconds, time.monotonic(), -100
        self.lease_check = lease_check

    def stopped(self, force=False):
        if self.lease_check is not None:
            self.lease_check()
        elapsed = self.previous + time.monotonic() - self.started
        if force or elapsed - self.last >= 10:
            atomic_json(self.path, {'active_seconds': elapsed, 'budget_seconds': self.seconds,
                'scope': 'Cumulative controller time including preparation/inference/training; setup excluded. Abrupt uncommitted tail may be unrecorded; graceful epoch shutdown can overrun.'})
            self.last = elapsed
        return elapsed >= self.seconds - 120


@contextlib.contextmanager
def bundle_owner(out):
    """Hold a heartbeat lease over preparation, child execution and export."""
    from a1confirm.study import RunLease
    done, errors = threading.Event(), []
    with RunLease(out, accrue=False) as lease:
        def check():
            if errors:
                raise RuntimeError('Bundle ownership heartbeat failed; preserving committed evidence.') from errors[0]
        def heartbeat():
            while not done.wait(20):
                try:
                    lease.heartbeat()
                except BaseException as error:
                    errors.append(error)
                    return
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield check
        finally:
            done.set()
            worker.join(timeout=60)
            if worker.is_alive():
                raise RuntimeError('Bundle heartbeat did not stop safely.')


def stop_child(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def raw_inputs(args):
    from garmentlab.images import SOURCES, download_cached
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    pins = read(REFERENCE / 'data_evidence/index_manifest.json')['binding']['source_sha256']
    for key, (url, filename, size, _) in SOURCES.items():
        supplied = getattr(args, key)
        if supplied:
            path = Path(supplied)
            if path.stat().st_size != size or digest(path) != pins[key]:
                raise ValueError('Raw reference mismatch: ' + key)
        else:
            path = download_cached(url, cache / filename, expected_bytes=size,
                expected_sha256=pins[key], is_zip=filename.endswith('.zip'))
        result[key] = path
    result.update(historical_index=REFERENCE / 'data_evidence/index_manifest.json',
                  duplicate_audit=REFERENCE / 'data_evidence/DUPLICATE_AUDIT.json',
                  preview_credits=tuple(sorted(PREVIEWS.glob('*.csv'))))
    if len(result['preview_credits']) != 4:
        raise ValueError('All four recorded image-exposure credit lists are required.')
    return result


def bind(out):
    out = Path(out)
    if any((out / name).exists() for name in ('FINAL_SUMMARY.json', 'FINAL_SUMMARY_V2.json', 'study_manifest.json')):
        raise ValueError('A new output folder is required; historical studies are immutable.')
    out.mkdir(parents=True, exist_ok=True)
    source_manifest = ROOT / 'SOURCE_MANIFEST.json'
    value = {'schema': 'co3133_confirmatory_bundle_v1', 'controller_sha256': digest(__file__),
        'source_package_manifest_sha256': digest(source_manifest) if source_manifest.exists() else None,
        'fashion': {'trajectories': 30, 'fresh_seeds': [1009, 2027, 3037], 'epochs': 60,
                    'recipe_factor': ['C0', 'selected'], 'checkpoint_factor': ['ce', 'f1'],
                    'primary': 'CNN selected minus C0 at fixed final epoch',
                    'scope': 'Previously used Fashion-MNIST validation; no new blind-test claim'},
        'rgb': {'models': 'Six fixed C0/C4 models', 'cohort': 'Project-unused Fashionpedia training-source license IDs3/5',
                'source_group_cap': 2000, 'scope': 'Prospectively locked same-source license-cohort shift; not external/IID'},
        'max_controller_seconds': 18000, 'new_rgb_training_or_calibration': False}
    protect(out / 'BUNDLE_MANIFEST.json', value)
    return out


def rgb_audit(args, budget):
    from garmentlab.prospective import (freeze_prospective, prepare_prospective, evaluate_prospective,
        verify_prospective_complete, _models, _verify_freeze)
    folder = Path(args.out) / 'rgb_prospective'
    recorded = ROOT / 'assets/prospective_completed'
    if recorded.is_dir() and (recorded / 'PROSPECTIVE_COMPLETE.json').is_file():
        verify_prospective_complete(recorded)
        frozen = _verify_freeze(recorded)
        if frozen.get('smoke_only') is not False:
            raise ValueError('Synthetic audit cannot replace prospective experiment evidence.')
        models, files = _models(REFERENCE)
        if models != frozen['models'] or files != frozen['study_files']:
            raise ValueError('Recorded prospective audit/checkpoint identity differs.')
        for source in recorded.rglob('*'):
            if not source.is_file():
                continue
            if source.name == 'images_uint8.npy' or source.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                raise ValueError('Recorded prospective package must not contain photographs.')
            target = folder / source.relative_to(recorded)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != digest(source):
                    raise ValueError('Existing prospective evidence differs from the recorded audit.')
            else:
                shutil.copyfile(source, target)
        verify_prospective_complete(folder)
        print('Verified completed local CPU prospective audit; no RGB fitting or repeat test selection.', flush=True)
        return read(folder / 'PROSPECTIVE_RESULTS.json')
    paths = raw_inputs(args)
    frozen = freeze_prospective(folder, REFERENCE, **paths,
        exposure_attestation={'recorded_exposure_inventory_complete': True,
            'basis': 'Available project prepared indices, raw metadata/license history, duplicate quarantines and four recorded preview lists were inventoried. No recorded license3/5 photograph or model-output exposure was found. Unrecorded browsing and ImageNet pretraining exposure cannot be verified.'})
    print('RGB MODELS AND COHORT RULE FROZEN:', len(frozen['models']), flush=True)
    prepare_prospective(folder, **paths, stop_check=budget.stopped)
    return evaluate_prospective(folder, REFERENCE, device=args.device,
                                batch_size=128, stop_check=budget.stopped)


def core_plan(args):
    from a1confirm.study import prepare
    return prepare(Path(args.out) / 'fashion_confirmation', PUBLICATION,
                   args.data_dir, args.device, time_budget_seconds=17100)


def train_guarded(args, budget):
    folder = Path(args.out) / 'fashion_confirmation'
    command = [sys.executable, '-u', '-m', 'a1confirm', 'run', '--out', str(folder),
               '--data-dir', str(args.data_dir), '--device', args.device]
    with (Path(args.out) / 'FASHION_CONSOLE.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if budget.stopped():
                    stop_child(process)
                    atomic_json(Path(args.out) / 'STOP_REASON.json', {'status': 'INCOMPLETE_BUNDLE_TIME_BUDGET'})
                    return
                completed = len(list((folder / 'tasks').glob('*/TRAJECTORY_COMPLETE.json')))
                print(f'Fashion confirmation: {completed}/30 trajectories committed.', flush=True)
                time.sleep(20)
            if process.returncode:
                raise RuntimeError('Fashion controller failed; inspect FASHION_CONSOLE.log. Completed evidence is retained.')
        except BaseException:
            stop_child(process)
            raise


def progress(out):
    out = Path(out)
    core = out / 'fashion_confirmation'
    if (core / 'confirmation_manifest.json').exists():
        from a1confirm.study import status
        fashion = status(core)
    else:
        fashion = {'status': 'NOT_STARTED'}
    rgb_path = out / 'rgb_prospective/PROSPECTIVE_RESULTS.json'
    rgb = {'status': 'NOT_COMPLETE'}
    marker_path = rgb_path.parent / 'PROSPECTIVE_COMPLETE.json'
    if marker_path.exists():
        from garmentlab.prospective import verify_prospective_complete
        rgb = verify_prospective_complete(rgb_path.parent)
        if rgb.get('smoke_only') is not False:
            raise ValueError('Prospective completion identity differs or contains synthetic measurements.')
    complete = fashion['status'] == 'COMPLETE' and rgb['status'] == 'PROSPECTIVE_AUDIT_COMPLETE'
    return {'status': 'COMPLETE' if complete else 'INCOMPLETE', 'fashion': fashion,
            'rgb_status': rgb['status'], 'new_blind_Fashion_MNIST_test': False,
            'new_project_unused_RGB_cohort': rgb['status'] == 'PROSPECTIVE_AUDIT_COMPLETE'}


def export(out):
    out = Path(out)
    state = progress(out)
    atomic_json(out / 'RESULT_STATUS.json', state)
    for folder in ('a1', 'a1opt', 'a1confirm', 'garmentlab'):
        for source in (ROOT / folder).glob('*.py'):
            target = out / 'source_snapshot' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(source):
                raise ValueError('Result source snapshot changed.')
            if not target.exists():
                shutil.copyfile(source, target)
    excluded = {'RESULT_MANIFEST.json', 'CONFIRMATION_OWNER.json', 'CONFIRMATION_OWNER_TRANSACTION.lock'}
    payload = [p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()
        and p.name not in excluded and not p.name.endswith(('.tmp', '.lock'))
        and p.name != 'images_uint8.npy' and not ('rgb_prospective' in p.parts and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))]
    records = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in payload]
    atomic_json(out / 'RESULT_MANIFEST.json', {'status': state['status'], 'files': records,
                'photographs_exported': False, 'source_package_identity': read(out / 'BUNDLE_MANIFEST.json')})
    archive = out.parent / (out.name + '_results.zip')
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for p in [*payload, out / 'RESULT_MANIFEST.json']:
            zipped.write(p, p.relative_to(out).as_posix())
    with zipfile.ZipFile(temporary) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Result archive CRC failed.')
        for item in records:
            if hashlib.sha256(zipped.read(item['path'])).hexdigest() != item['sha256']:
                raise ValueError('Result archive payload changed.')
    os.replace(temporary, archive)
    return {**state, 'archive': str(archive), 'sha256': digest(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'rgb', 'train', 'export', 'status', 'all'), default='all')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('/content/fashion_confirmation_data'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for key in ('train_metadata', 'train_archive', 'assessment_metadata', 'assessment_archive'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path)
    args = parser.parse_args()
    if any((args.out / name).exists() for name in ('FINAL_SUMMARY.json','FINAL_SUMMARY_V2.json','study_manifest.json')):
        raise ValueError('A separate confirmation output folder is required.')
    with bundle_owner(args.out) as lease_check:
        bind(args.out)
        budget = Budget(args.out, lease_check=lease_check)
        result = None
        try:
            if args.stage in ('plan', 'train', 'all'):
                core_plan(args)
            if args.stage == 'all':
                from a1confirm.study import smoke
                smoke(args.out / 'fashion_confirmation', args.data_dir, args.device)
            if args.stage in ('rgb', 'all'):
                result = rgb_audit(args, budget)
            if args.stage in ('train', 'all') and not budget.stopped():
                train_guarded(args, budget)
            result = progress(args.out) if args.stage != 'export' else None
        finally:
            budget.stopped(force=True)
            if args.stage not in ('plan', 'status'):
                result = export(args.out)
            budget.stopped(force=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
