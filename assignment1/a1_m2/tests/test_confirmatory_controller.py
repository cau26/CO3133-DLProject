"""Bundle identity, budget accounting and photograph-free result export."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools import run_confirmatory as c


def test_historical_folder_is_rejected_without_rewriting(tmp_path):
    original=tmp_path/'FINAL_SUMMARY_V2.json'
    original.write_text('{"historical":true}')
    with pytest.raises(ValueError,match='historical'):
        c.bind(tmp_path)
    assert original.read_text()=='{"historical":true}'
    assert not (tmp_path/'BUNDLE_MANIFEST.json').exists()


def test_bundle_identity_cannot_silently_change(tmp_path):
    c.bind(tmp_path)
    path=tmp_path/'BUNDLE_MANIFEST.json'
    value=json.loads(path.read_text())
    value['fashion']['fresh_seeds']=[42,43,44]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='identity differs'):
        c.bind(tmp_path)


def test_budget_reentry_counts_previous_time_and_locks_cap(tmp_path,monkeypatch):
    now=[100.0]
    monkeypatch.setattr(c.time,'monotonic',lambda:now[0])
    first=c.Budget(tmp_path,seconds=1000)
    now[0]=400.0
    assert first.stopped() is False
    second=c.Budget(tmp_path,seconds=1000)
    now[0]=1000.0
    assert second.stopped() is True
    with pytest.raises(ValueError,match='immutable'):
        c.Budget(tmp_path,seconds=2000)


def test_export_keeps_partial_identity_and_excludes_photographs(tmp_path,monkeypatch):
    out=tmp_path/'run'
    c.bind(out)
    prepared=out/'rgb_prospective/prepared'
    prepared.mkdir(parents=True)
    (prepared/'images_uint8.npy').write_bytes(b'private images')
    (prepared/'labels.npy').write_bytes(b'labels')
    (prepared/'preview.jpg').write_bytes(b'private photograph')
    partial={'status':'INCOMPLETE','fashion':{'status':'NOT_STARTED'},'rgb_status':'NOT_COMPLETE'}
    monkeypatch.setattr(c,'progress',lambda _:partial)
    source=tmp_path/'source'
    source.mkdir()
    monkeypatch.setattr(c,'ROOT',source)
    result=c.export(out)
    assert result['status']=='INCOMPLETE'
    with zipfile.ZipFile(result['archive']) as archive:
        assert archive.testzip() is None
        assert 'rgb_prospective/prepared/images_uint8.npy' not in archive.namelist()
        assert not any(n.endswith('.jpg') for n in archive.namelist())
        assert 'rgb_prospective/prepared/labels.npy' in archive.namelist()
        manifest=json.loads(archive.read('RESULT_MANIFEST.json'))
        assert manifest['status']=='INCOMPLETE'
        assert manifest['photographs_exported'] is False
        for item in manifest['files']:
            data=archive.read(item['path'])
            assert len(data)==item['bytes']
            assert hashlib.sha256(data).hexdigest()==item['sha256']


def test_result_filename_alone_does_not_promote_an_incomplete_audit(tmp_path):
    folder=tmp_path/'rgb_prospective'
    folder.mkdir()
    (folder/'PROSPECTIVE_RESULTS.json').write_text('{"status":"PROSPECTIVE_AUDIT_COMPLETE"}')
    state=c.progress(tmp_path)
    assert state['status']=='INCOMPLETE'
    assert state['rgb_status']=='NOT_COMPLETE'


def test_bundle_owner_rejects_second_controller_and_releases(tmp_path):
    with c.bundle_owner(tmp_path):
        with pytest.raises(RuntimeError,match='owns this study'):
            with c.bundle_owner(tmp_path):
                pass
    assert json.loads((tmp_path/'CONFIRMATION_OWNER.json').read_text())['active'] is False
    with c.bundle_owner(tmp_path):
        pass


def test_graceful_budget_flush_keeps_subtensecond_tail(tmp_path,monkeypatch):
    now=[0.0]
    monkeypatch.setattr(c.time,'monotonic',lambda:now[0])
    budget=c.Budget(tmp_path,seconds=1000)
    budget.stopped()
    now[0]=7.0
    budget.stopped(force=True)
    assert json.loads((tmp_path/'BUNDLE_CLOCK.json').read_text())['active_seconds']==7.0


def test_child_is_reaped_before_export_after_escalated_stop():
    class Child:
        def __init__(self):self.events=[];self.waits=0
        def poll(self):return None
        def send_signal(self,value):self.events.append('interrupt')
        def terminate(self):self.events.append('terminate')
        def kill(self):self.events.append('kill')
        def wait(self,timeout):
            self.events.append('wait');self.waits+=1
            if self.waits<3:raise c.subprocess.TimeoutExpired('child',timeout)
            return -9
    child=Child()
    c.stop_child(child)
    assert child.events==['interrupt','wait','terminate','wait','kill','wait']
