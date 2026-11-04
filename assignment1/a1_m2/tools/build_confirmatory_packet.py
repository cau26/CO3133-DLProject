"""Build the standalone confirmation notebook and exact inference/training inputs."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEST = ROOT.parent / 'deliverables/CO3133_Confirmatory_20261029'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_notebook(source_sha):
    recorded_rgb = (ROOT/'assets/prospective_completed/PROSPECTIVE_COMPLETE.json').exists()
    cells = []
    def add(kind, value):
        source = textwrap.dedent(value).strip() + '\n'
        item = {'cell_type': kind, 'id': f'confirm-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            ast.parse(source)
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    add('markdown', '''
    # CO3133 · Matched confirmation and prospective RGB audit

    Thirty matched Fashion-MNIST training trajectories separate the combined
    recipe and checkpoint-selection effects. Each trajectory retains CE, F1
    and fixed-final checkpoints. The Fashion-MNIST validation set remains
    retrospective; the official test is not loaded.

    Six frozen RGB models are evaluated on a predeclared, project-unused
    Fashionpedia license cohort. It is a same-source distribution-shift audit,
    not an external or IID test. There is no RGB refitting or recalibration.

    The default training cap is 4.75 active hours; the combined controller has
    a five-hour soft cap including data preparation. Completion is determined
    by committed tasks and audit receipts. The notebook supports Run all and
    resumes the same versioned Drive directory after interruption.
    ''')
    add('markdown', '''
    ## 1. Runtime, source identity and experiment store
    Verify the exact source/checkpoint package, preserve the CUDA library pair,
    install the declared analysis dependencies and locate the raw-data cache.
    ''')
    add('code', r'''
    import os, sys, json, hashlib, zipfile, subprocess
    from pathlib import Path, PurePosixPath
    from importlib.metadata import version
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    from google.colab import drive, files
    import torch
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'This experiment requires an A100 runtime.'
    drive.mount('/content/drive')
    SOURCE_SHA256 = '__SHA__'
    STORE = Path('/content/drive/MyDrive/G-M10_Confirmatory_20261029')
    STORE.mkdir(parents=True, exist_ok=True)
    def file_sha(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    SOURCE = STORE/'source'/SOURCE_SHA256/'CO3133_Confirmatory_Source_20261029.zip'
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        print('Source archive: CO3133_Confirmatory_Source_20261029.zip')
        uploaded = files.upload()
        matches = [b for b in uploaded.values() if hashlib.sha256(b).hexdigest() == SOURCE_SHA256]
        assert len(matches) == 1, 'The source archive does not match this notebook.'
        SOURCE.write_bytes(matches[0])
        del uploaded, matches
    assert file_sha(SOURCE) == SOURCE_SHA256, 'Stored source identity differs.'
    REPO = Path('/content/CO3133Confirm_'+SOURCE_SHA256[:12])
    REPO.mkdir(exist_ok=True)
    with zipfile.ZipFile(SOURCE) as archive:
        manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
        paths = {p['path'] for p in manifest['payloads']} | {'SOURCE_MANIFEST.json'}
        assert len(paths) == len(archive.namelist()) and paths == set(archive.namelist())
        assert sum(p['bytes'] for p in manifest['payloads']) < 256*1024*1024
        for item in manifest['payloads']:
            path = PurePosixPath(item['path'])
            assert not path.is_absolute() and '..' not in path.parts and ':' not in str(path) and '\\' not in str(path)
            value = archive.read(str(path))
            assert len(value) == item['bytes'] and hashlib.sha256(value).hexdigest() == item['sha256']
            target = REPO.joinpath(*path.parts)
            assert target.resolve().is_relative_to(REPO.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert file_sha(target) == item['sha256'], 'Local source differs from the locked package.'
            else:
                target.write_bytes(value)
        (REPO/'SOURCE_MANIFEST.json').write_bytes(archive.read('SOURCE_MANIFEST.json'))
    before = {p:version(p) for p in ('torch','torchvision')}
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(REPO/'requirements-confirmatory-colab.txt')],check=True)
    assert before == {p:version(p) for p in before}, 'CUDA torch/torchvision pair changed.'
    caches = [Path('/content/drive/MyDrive/G-M10_GarmentLab_ImproveToday/cache'),
              Path('/content/drive/MyDrive/G-M10_GarmentLab_A100_Deadline/cache')]
    CACHE = next((p for p in caches if (p/'train2020.zip').is_file()), STORE/'cache')
    CACHE.mkdir(parents=True,exist_ok=True)
    OUT = STORE/'runs'/('g_m10_confirmatory_run01_'+SOURCE_SHA256[:12])
    command = [sys.executable,'-u',str(REPO/'tools/run_confirmatory.py'),
               '--out',str(OUT),'--cache',str(CACHE),
               '--data-dir','/content/fashion_confirmation_data','--device','cuda']
    print('GPU:',torch.cuda.get_device_name(0))
    print('Outputs:',OUT)
    print('Raw cache:',CACHE)
    ''')
    cells[-1]['source'] = cells[-1]['source'].replace('__SHA__', source_sha)
    add('markdown', '''
    ## 2. Software checks and matched-study plan
    Exercise synthetic recovery, source/tamper gates and audit exclusions.
    Freeze the real recipes, seeds, split and runtime before model comparisons.
    A one-batch A100 check verifies execution without producing study scores.
    ''')
    add('code', '''
    subprocess.run([sys.executable,'-m','pytest','-q',
        'tests/test_a1confirm_study.py','tests/test_garmentlab_prospective.py',
        'tests/test_confirmatory_controller.py',
        '--basetemp','/content/confirmatory_tests_'+SOURCE_SHA256[:12]],cwd=REPO,check=True)
    subprocess.run(command+['--stage','plan'],cwd=REPO,check=True)
    subprocess.run([sys.executable,'-m','a1confirm','smoke','--out',str(OUT/'fashion_confirmation'),
        '--data-dir','/content/fashion_confirmation_data','--device','cuda'],cwd=REPO,check=True)
    print('Software checks and A100 batch execution passed.')
    ''')
    add('markdown', '''
    ## 3. Prospective RGB audit and matched training
    Commit the fresh cohort before target-label filtering, remove historical
    source/duplicate overlap and evaluate every frozen RGB checkpoint. Execute
    the 30 matched Fashion-MNIST trajectories and preserve epoch transactions.
    Progress and complete/partial evidence are saved directly to Drive.
    ''')
    add('code', '''
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'NOTEBOOK_CONSOLE.log').open('a',encoding='utf-8') as log:
        process = subprocess.Popen(command+['--stage','all'],cwd=REPO,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line); log.flush()
            returncode = process.wait()
        except BaseException:
            import signal
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=120)
            except subprocess.TimeoutExpired: process.terminate()
            raise
    assert returncode == 0, 'Controller stopped with an error. Saved logs and partial evidence identify the failure.'
    ''')
    add('markdown', '''
    ## 4. Result identity and export
    Verify the result archive against its committed payload hashes. Report the
    actual completion status; a partial ZIP does not establish a completed study.
    Photographs and prepared image arrays are excluded from the export.
    ''')
    add('code', '''
    subprocess.run(command+['--stage','export'],cwd=REPO,check=True)
    ARCHIVE = OUT.parent/(OUT.name+'_results.zip')
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        receipt = json.loads(archive.read('RESULT_MANIFEST.json'))
        for item in receipt['files']:
            value = archive.read(item['path'])
            assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
        status = json.loads(archive.read('RESULT_STATUS.json'))
    print('STATUS:',status['status'])
    print('FASHION:',status['fashion'].get('completed_trajectories',0),'/30')
    print('RGB:',status['rgb_status'])
    print('RESULT:',ARCHIVE)
    print('SHA256:',file_sha(ARCHIVE))
    files.download(str(ARCHIVE))
    ''')
    if recorded_rgb:
        for cell in cells:
            if cell['cell_type'] == 'markdown':
                cell['source'] = cell['source'].replace(
                    'Six frozen RGB models are evaluated on a predeclared, project-unused',
                    'The package retains the completed local CPU audit of six frozen RGB models on a prospectively locked, project-unused')
                cell['source'] = cell['source'].replace(
                    '## 3. Prospective RGB audit and matched training\nCommit the fresh cohort before target-label filtering, remove historical\nsource/duplicate overlap and evaluate every frozen RGB checkpoint. Execute',
                    '## 3. Verified RGB audit and matched training\nVerify and reuse the completed local CPU prospective audit. Its original\nmodel/cohort lock, prediction arrays and statistical receipts are retained;\nthis verification is not another blind audit or a new RGB fit. Execute')
                cell['source'] = cell['source'].replace(
                    'install the declared analysis dependencies and locate the raw-data cache.',
                    'install the declared analysis dependencies and locate the raw-data cache.\nThe included completed RGB audit avoids downloading source photographs for this run.')
    return {'cells': cells, 'metadata': {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'colab':{'name':'CO3133_Confirmatory_A100_20261029.ipynb'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    selected = {}
    def add(path):
        p = ROOT / path
        if p.is_dir():
            for item in p.rglob('*'):
                if item.is_file() and '__pycache__' not in item.parts and item.suffix not in ('.pyc','.jpg','.jpeg','.png'):
                    selected[item.relative_to(ROOT).as_posix()] = item
        else:
            assert p.is_file(), p
            selected[Path(path).as_posix()] = p
    for name in ('a1','a1opt','a1confirm','garmentlab','assets/a1_optimization_reference','assets/prospective_exposure'):
        add(name)
    recorded = ROOT/'assets/prospective_completed'
    if recorded.is_dir():
        from garmentlab.prospective import verify_prospective_complete
        result = verify_prospective_complete(recorded)
        assert result['smoke_only'] is False
        assert not any(p.name=='images_uint8.npy' or p.suffix.lower() in ('.jpg','.jpeg','.png','.webp') for p in recorded.rglob('*') if p.is_file())
        add('assets/prospective_completed')
    for name in ('tools/run_confirmatory.py','requirements-confirmatory-colab.txt',
        'docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md',
        'docs/a1_extension_research/PROSPECTIVE_RGB_AUDIT_20261029.md',
        'docs/a1_extension_research/CONFIRMATORY_AI_USAGE.md',
        'output/a1_final/data/publication_data.json','tests/test_a1confirm_study.py',
        'tests/test_garmentlab_prospective.py','tests/test_confirmatory_controller.py'):
        add(name)
    reference='output/garmentlab_improve_return_20261028'
    for name in ('FINAL_SELECTION.json','study_v2_manifest.json','study_v2_state.json',
        'release_candidate/release.json','data_evidence/index_manifest.json',
        'data_evidence/DUPLICATE_AUDIT.json','data_evidence/DATA_READINESS.json'):
        add(reference+'/'+name)
    for candidate in ('C0','C4'):
        for seed in (314,2718,5772):
            for filename in ('best.pt','task.json','history.json','COMPLETE.json'):
                add(f'{reference}/tasks/final_{candidate}_seed{seed}/{filename}')
    payloads=[{'path':name,'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(selected.items())]
    manifest={'schema':'co3133_confirmatory_source_v1','scope':'Executable source, frozen comparison inputs, six RGB checkpoints and recorded exposure metadata. No source photographs or optimizer recovery states.',
        'new_experiment_measurements_included':(ROOT/'assets/prospective_completed/PROSPECTIVE_COMPLETE.json').exists(),
        'student_A100_confirmation_completed':False,'payloads':payloads}
    archive=DEST/'CO3133_Confirmatory_Source_20261029.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for name,p in sorted(selected.items()): zipped.write(p,name)
        zipped.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert len(set(zipped.namelist()))==len(zipped.namelist())
        for item in payloads:
            data=zipped.read(item['path'])
            assert hashlib.sha256(data).hexdigest()==item['sha256']
    notebook=make_notebook(sha(archive))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    target=DEST/'CO3133_Confirmatory_A100_20261029.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    shutil.copyfile(ROOT/'docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md',DEST/'PROTOCOL.md')
    shutil.copyfile(ROOT/'docs/a1_extension_research/PROSPECTIVE_RGB_AUDIT_20261029.md',DEST/'PROSPECTIVE_RGB_AUDIT.md')
    receipt={'status':'SOURCE_HASH_CRC_AND_NOTEBOOK_SYNTAX_VERIFIED','source_sha256':sha(archive),
        'source_bytes':archive.stat().st_size,'payloads':len(payloads),'notebook_sha256':sha(target),
        'student_A100_study_completed':False,'new_RGB_audit_completed':(recorded/'PROSPECTIVE_COMPLETE.json').exists()}
    (DEST/'PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    build()
