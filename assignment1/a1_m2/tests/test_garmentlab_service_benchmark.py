"""Benchmark accounting must not mistake fast rejected requests for throughput."""
import importlib.util
import ast
import fnmatch
import re
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('service_benchmark',Path(__file__).resolve().parents[1]/'tools/benchmark_garmentlab_service.py')
BENCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_success_latency_and_throughput_exclude_fast_busy_responses():
    rows=[{'status_code':200,'contract_passed':True,'client_e2e_ms':100.0,'server_timing_ms':{'decode_preprocess':20.0,'forward':50.0}},
          {'status_code':503,'contract_passed':False,'client_e2e_ms':1.0,'error_code':'service_busy'},
          {'status_code':200,'contract_passed':True,'client_e2e_ms':200.0,'server_timing_ms':{'decode_preprocess':30.0,'forward':60.0}}]
    s=BENCH.summarize(rows,1.0)
    assert s['successful_predictions_per_second']==2
    assert s['attempted_requests_per_second']==3
    assert s['successful_prediction_client_e2e']['p50_ms']==150
    assert s['busy_responses']==1 and s['unexpected_failures']==0


def test_contract_failure_and_unexpected_503_are_reported():
    rows=[{'status_code':200,'contract_passed':False,'client_e2e_ms':1},
          {'status_code':503,'contract_passed':False,'error_code':'prediction_failed','client_e2e_ms':1},
          {'status_code':0,'contract_passed':False,'client_e2e_ms':1}]
    s=BENCH.summarize(rows,2)
    assert s['unexpected_failures']==3 and s['successful_predictions_per_second']==0
    assert s['successful_prediction_client_e2e'] is None


def test_both_ci_events_track_shared_pooling_source_and_benchmark():
    workflow=(Path(__file__).resolve().parents[1]/'.github/workflows/garmentlab-draft.yml').read_text(encoding='utf-8')
    filters=re.findall(r'paths:\s*(\[[^\n]+\])',workflow)
    assert len(filters)==2
    for raw in filters:
        patterns=ast.literal_eval(raw)
        for changed in ['a1/cnn.py','garmentlab/inference.py','tools/benchmark_garmentlab_service.py',
                        'requirements-garmentlab-cpu-lock.txt']:
            assert any(fnmatch.fnmatchcase(changed,pattern) for pattern in patterns),changed
