"""Measure a real local GarmentLab HTTP service using preserved source JPEGs.

Input bytes are read once before timing. The client measurements include multipart
request construction, loopback upload, server decode/crop/normalization/inference,
JSON response and loopback download. No photo or tensor is written to the receipt.
This is a finite closed-loop local benchmark, not production capacity validation.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize(rows, elapsed_seconds):
    """Report successful-prediction latency separately from fast rejections."""
    import numpy as np
    valid = [r for r in rows if r['status_code'] == 200 and r['contract_passed']]
    counts = Counter(str(r['status_code']) for r in rows)
    def stats(values):
        if not values:
            return None
        return {'count': len(values), 'p50_ms': float(np.percentile(values, 50)),
                'p95_ms': float(np.percentile(values, 95)),
                'p99_ms': float(np.percentile(values, 99)), 'maximum_ms': float(max(values))}
    return {'requests': len(rows), 'elapsed_seconds': elapsed_seconds,
            'attempted_requests_per_second': len(rows) / elapsed_seconds,
            'successful_predictions_per_second': len(valid) / elapsed_seconds,
            'status_counts': dict(counts),
            'successful_prediction_client_e2e': stats([r['client_e2e_ms'] for r in valid]),
            'all_response_client_e2e': stats([r['client_e2e_ms'] for r in rows]),
            'predictor_decode_preprocess': stats([r['server_timing_ms']['decode_preprocess'] for r in valid]),
            'predictor_forward': stats([r['server_timing_ms']['forward'] for r in valid]),
            'busy_responses': sum(r.get('error_code') == 'service_busy' for r in rows),
            'unexpected_failures': sum(r['status_code'] not in (200, 503) or
                (r['status_code'] == 503 and r.get('error_code') != 'service_busy') or
                (r['status_code'] == 200 and not r['contract_passed']) for r in rows)}


def load_cases(index_path, image_zip, indices):
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    records = index['records']
    cases = []
    with zipfile.ZipFile(image_zip) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if name in members:
                raise ValueError('Source ZIP contains duplicate basenames')
            members[name] = info
        for array_index in indices:
            row = records[array_index]
            if row['array_index'] != array_index or row.get('role') != 'assessment':
                raise ValueError('Benchmark records must be original assessment rows')
            name = row['file_name']
            if Path(name).name != name or name not in members:
                raise ValueError('Missing bounded source image')
            info = members[name]
            if info.file_size > 5 * 1024**2:
                raise ValueError('Benchmark image exceeds service upload contract')
            image = archive.read(info)
            if hashlib.sha256(image).hexdigest() != row['image_bytes_sha256']:
                raise ValueError('Source JPEG SHA differs from prepared record')
            crop = row['crop_xyxy']
            if len(crop) != 4 or any(type(v) is not int for v in crop):
                raise ValueError('Recorded crop must contain four integers')
            cases.append({'array_index': array_index, 'image': image, 'crop': crop,
                'image_sha256': row['image_bytes_sha256'], 'image_bytes': len(image),
                'width': row['width'], 'height': row['height'],
                'source_group_id': row['source_group_id']})
    if not cases:
        raise ValueError('At least one case is required')
    return cases


def _serve(release, port, threads):
    import torch
    import uvicorn
    from garmentlab.api import create_app
    from garmentlab.inference import Predictor
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.set_num_threads(threads)
    predictor = Predictor(release)
    uvicorn.run(create_app(garment_predictor=predictor), host='127.0.0.1', port=port,
                access_log=False, log_level='warning')


async def run_protocol(url, cases, release_id, requests, warmup, concurrency, overload, soak_seconds=0):
    import httpx
    limits = httpx.Limits(max_connections=max(overload, max(concurrency)), max_keepalive_connections=max(overload, max(concurrency)))
    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limits, trust_env=False) as client:
        for _ in range(300):
            try:
                health = await client.get('/health')
                models = await client.get('/v1/models')
                if health.status_code == 200 and health.json().get('garment_ready'):
                    if models.json()['garment']['metadata']['model_version'] != release_id:
                        raise ValueError('Unexpected service is listening on the benchmark port')
                    break
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(.1)
        else:
            raise RuntimeError('Local service did not become healthy within 30 seconds')
        async def predict(case):
            start = time.perf_counter()
            try:
                response = await client.post('/v1/predict',
                    files={'file': ('benchmark.jpg', case['image'], 'image/jpeg')},
                    data={'crop': json.dumps(case['crop'])})
                elapsed = (time.perf_counter() - start) * 1000
                body = response.json()
                result = body.get('result', {})
                best = result.get('predicted_class', {})
                expected = case.get('expected')
                passed = (response.status_code == 200 and result.get('model_version') == release_id
                    and result.get('calibrated') is True and result.get('release_status') == 'ready'
                    and response.headers.get('x-request-id') == body.get('request_id')
                    and (expected is None or (best.get('index') == expected['index']
                        and result.get('review_required') == expected['review_required']
                        and abs(best.get('probability', -1) - expected['confidence']) <= 1e-4)))
                return {'array_index': case['array_index'], 'status_code': response.status_code,
                    'client_e2e_ms': elapsed, 'request_id': body.get('request_id'),
                    'contract_passed': passed, 'predicted_index': best.get('index'),
                    'confidence': best.get('probability'), 'review_required': result.get('review_required'),
                    'server_timing_ms': result.get('timing_ms'),
                    'error_code': body.get('error', {}).get('code')}
            except (httpx.HTTPError, ValueError) as error:
                return {'array_index': case['array_index'], 'status_code': 0,
                        'client_e2e_ms': (time.perf_counter() - start) * 1000,
                        'contract_passed': False, 'exception_type': type(error).__name__}
        preflight = []
        for case in cases:
            row = await predict(case)
            if not row['contract_passed']:
                raise ValueError('Actual source JPEG failed preflight prediction')
            case['expected'] = {'index': row['predicted_index'], 'confidence': row['confidence'],
                                'review_required': row['review_required']}
            preflight.append(row)
        phases = []
        for load in [*concurrency, overload]:
            for i in range(warmup):
                row = await predict(cases[i % len(cases)])
                if not row['contract_passed']:
                    raise ValueError('Sequential warmup failed')
            count = requests if load in concurrency else max(24, load * 3)
            gate = asyncio.Semaphore(load)
            async def bounded(i):
                async with gate:
                    return await predict(cases[i % len(cases)])
            started = time.perf_counter()
            rows = await asyncio.gather(*(bounded(i) for i in range(count)))
            elapsed = time.perf_counter() - started
            phases.append({'kind': 'supported_concurrency' if load in concurrency else 'overload_probe',
                           'concurrency': load, 'warmup_requests': warmup, 'summary': summarize(rows, elapsed), 'rows': rows})
        if soak_seconds:
            import math
            load=max(concurrency)
            for i in range(warmup):
                if not (await predict(cases[i % len(cases)]))['contract_passed']:
                    raise ValueError('Soak warmup failed')
            started=time.perf_counter()
            stop=started+soak_seconds
            rows=[]
            async def soak_worker(offset):
                number=offset
                while time.perf_counter()<stop:
                    row=await predict(cases[number % len(cases)])
                    row['completed_after_s']=time.perf_counter()-started
                    rows.append(row)
                    number+=load
            await asyncio.gather(*(soak_worker(i) for i in range(load)))
            elapsed=time.perf_counter()-started
            windows=[]
            count=math.ceil(soak_seconds/60)
            for i in range(count):
                cohort=[r for r in rows if min(int(r['completed_after_s']//60),count-1)==i]
                width=min(60,soak_seconds-i*60)
                windows.append({'start_s':i*60,'nominal_end_s':i*60+width,
                                'summary':summarize(cohort,width)})
            phases.append({'kind':'soak_supported_concurrency','concurrency':load,
                'requested_duration_seconds':soak_seconds,'warmup_requests':warmup,
                'summary':summarize(rows,elapsed),'sixty_second_windows':windows,'rows':rows})
        contracts = []
        sample = cases[0]
        for name, files, data, expected in [
            ('missing_crop', {'file': ('case.jpg', sample['image'], 'image/jpeg')}, {}, 400),
            ('invalid_jpeg', {'file': ('case.jpg', b'not-a-jpeg', 'image/jpeg')}, {'crop':'[0,0,20,20]'}, 400),
            ('unsupported_media_type', {'file': ('case.gif', b'bad', 'image/gif')}, {}, 415),
            ('oversized_upload', {'file': ('case.jpg', b'x'*(5*1024**2+1), 'image/jpeg')}, {}, 413),
        ]:
            response = await client.post('/v1/predict', files=files, data=data)
            contracts.append({'name': name, 'expected_status': expected, 'observed_status': response.status_code,
                              'passed': response.status_code == expected})
        return {'preflight': preflight, 'phases': phases, 'invalid_input_contracts': contracts}


def benchmark(args):
    from importlib.metadata import version
    from garmentlab.release import validate_release
    release = Path(args.release).resolve(strict=True)
    checked = validate_release(release, require_ready=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new benchmark receipt path')
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.index, args.image_zip, args.indices)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    protocol = {'schema': 'garmentlab_real_http_benchmark_v1', 'requests_per_supported_phase': args.requests,
        'warmup_requests_per_phase': args.warmup, 'supported_concurrency': args.concurrency,
        'overload_concurrency': args.overload, 'pytorch_threads': args.threads,
        'soak_seconds': args.soak_seconds,
        'input_selection': 'Preset original recorded source JPEG/crop cases, no generated photos',
        'clock_scope': 'Client timer includes multipart construction, loopback upload, server decode/crop/normalization/forward, JSON serialization and loopback download. Input disk/ZIP reads occur before timing.',
        'latency_population': 'Successful predictions separately from all responses; busy503 must never artificially improve the success latency or throughput.'}
    protocol_file = output.with_suffix('.protocol.json')
    protocol_file.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    log_path = output.with_suffix('.server.log')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    result = {'status':'INCOMPLETE', 'protocol':protocol}
    with log_path.open('wb') as log:
        process = subprocess.Popen([sys.executable, '-I', '-u', str(Path(__file__).resolve()), '--serve',
            '--release', str(release), '--port', str(port), '--threads', str(args.threads)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            measured = asyncio.run(run_protocol(f'http://127.0.0.1:{port}', cases,
                checked['manifest']['release_id'], args.requests, args.warmup, args.concurrency, args.overload,args.soak_seconds))
            success = (all(row['passed'] for row in measured['invalid_input_contracts']) and
                all(p['summary']['unexpected_failures']==0 and
                    (p['kind'] not in ('supported_concurrency','soak_supported_concurrency') or p['summary']['status_counts'].get('200',0)==p['summary']['requests'])
                    for p in measured['phases']))
            result.update({'status':'PASS_FINITE_LOCAL_HTTP_BENCHMARK' if success else 'FAIL_HTTP_CONTRACT', **measured})
        except Exception as error:
            result.update({'status':'FAILED_BENCHMARK_EXECUTION', 'exception_type':type(error).__name__,
                           'message':str(error)[:1500]})
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
    result.update({'release_id':checked['manifest']['release_id'], 'weights_sha256':checked['manifest']['weights']['sha256'],
        'release_manifest_sha256':sha256_file(release/'release.json'),
        'index_sha256':sha256_file(args.index), 'source_image_zip_sha256':sha256_file(args.image_zip),
        'source_sha256':{n:sha256_file(ROOT/n) for n in ['tools/benchmark_garmentlab_service.py','garmentlab/api.py','garmentlab/inference.py','garmentlab/release.py','a1/cnn.py']},
        'inputs':[{k:v for k,v in c.items() if k not in {'image','expected'}} for c in cases],
        'environment':{'python':sys.version, 'platform':platform.platform(), 'versions':{n:version(n) for n in ['torch','torchvision','numpy','Pillow','fastapi','starlette','uvicorn','httpx']}},
        'limitations':['Same Windows machine and loopback network, not hosted or wide-area latency.',
            'Two repeated JPEG/crop cases do not estimate new-image quality or a general traffic distribution.',
            'Finite closed-loop requests and an optional short soak are not a production capacity or long-duration reliability validation.',
            'Percentiles are empirical for the recorded sample sizes and machine state.'],
        'server_log_sha256':sha256_file(log_path), 'protocol_sha256':sha256_file(protocol_file)})
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','release_id']},indent=2))
    return 0 if result['status'].startswith('PASS') else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--index',type=Path)
    p.add_argument('--image-zip',type=Path)
    p.add_argument('--indices',type=int,nargs='+',default=[17500,17504])
    p.add_argument('--output',type=Path)
    p.add_argument('--requests',type=int,default=120)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2])
    p.add_argument('--overload',type=int,default=8)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--soak-seconds',type=int,default=0)
    p.add_argument('--serve',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--port',type=int,default=0,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.serve:
        if not 1<=a.port<=65535 or not 1<=a.threads<=32:p.error('Invalid local server settings')
        return _serve(a.release,a.port,a.threads)
    if not a.index or not a.image_zip or not a.output:p.error('--index, --image-zip and --output are required')
    if not 20<=a.requests<=5000 or not 0<=a.warmup<=100 or not a.concurrency or any(c not in (1,2) for c in a.concurrency) or not 3<=a.overload<=32 or not 1<=a.threads<=32 or not 0<=a.soak_seconds<=600:
        p.error('Invalid finite benchmark protocol')
    return benchmark(a)


if __name__=='__main__':
    raise SystemExit(main())
