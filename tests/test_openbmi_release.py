import hashlib,json
import pytest
from p3a.data.openbmi_manifest import build_subject_manifest
from p3a.data.openbmi_release import load_execution_release

def setup(tmp_path,status='frozen_independent_execution'):
    p=tmp_path/'protocol.json';m=tmp_path/'manifest.json';r=tmp_path/'receipt.json'
    p.write_text(json.dumps(dict(status=status,seeds=[20261013,20261014,20261015],subject_roles={'evaluation':list(range(37,55))})))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    train=[f'openbmi_candidate:s37_session1:train:{i:03d}' for i in range(33)]
    test=[f'openbmi_candidate:s37_session1:test:{i:03d}' for i in range(36)]
    manifest=build_subject_manifest(37,train,test,'a'*64,digest(p))
    m.write_text(json.dumps(manifest));r.write_text(json.dumps(dict(status=status,protocol_sha256=digest(p),manifest_sha256=digest(m))))
    return p,m,r,digest

def test_candidate_is_closed(tmp_path):
    p,m,r,_=setup(tmp_path,'candidate')
    with pytest.raises(ValueError,match='CLOSED'):load_execution_release(p,m,r,37,20261013,'a'*64)

def test_release_binds_identity_targets_and_content(tmp_path):
    p,m,r,digest=setup(tmp_path)
    assert load_execution_release(p,m,r,37,20261013,'a'*64)[0]['subject']==37
    with pytest.raises(ValueError):load_execution_release(p,m,r,37,123,'a'*64)
    with pytest.raises(ValueError):load_execution_release(p,m,r,37,20261013,'b'*64)
    value=json.loads(m.read_text());value['targets'][0]['target']=value['targets'][0]['intended'];m.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='hash'):load_execution_release(p,m,r,37,20261013,'a'*64)
    receipt=json.loads(r.read_text());receipt['manifest_sha256']=digest(m);r.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='targets'):load_execution_release(p,m,r,37,20261013,'a'*64)


def test_validation_integration_cannot_unlock_sealed(tmp_path):
    p,m,r,digest=setup(tmp_path,'validation_integration_only')
    train=[f'openbmi_candidate:s19_session1:train:{i:03d}' for i in range(33)]
    test=[f'openbmi_candidate:s19_session1:test:{i:03d}' for i in range(36)]
    m.write_text(json.dumps(build_subject_manifest(19,train,test,'a'*64,digest(p))))
    receipt=json.loads(r.read_text());receipt['manifest_sha256']=digest(m);r.write_text(json.dumps(receipt))
    assert load_execution_release(p,m,r,19,20261013,'a'*64,validation_integration=True)[0]['role']=='validation'
    with pytest.raises(ValueError):load_execution_release(p,m,r,19,20261013,'a'*64)
    with pytest.raises(ValueError):load_execution_release(p,m,r,37,20261013,'a'*64,validation_integration=True)
