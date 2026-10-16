import pytest
from p3a.data.openbmi_manifest import build_subject_manifest,validate_evaluation_binding

def ids(subject,phase,n):return [f'openbmi_candidate:s{subject:02d}_session1:{phase}:{i:03d}' for i in range(n)]
def build(subject=37):return build_subject_manifest(subject,ids(subject,'train',33),ids(subject,'test',36),'a'*64,'b'*64)

def test_manifest_identity_target_and_binding():
    m=build()
    assert m==build()
    assert len(m['targets'])==36
    assert all(0<=t['target']<36 and t['target']!=t['intended'] for t in m['targets'])
    assert validate_evaluation_binding(m,'b'*64,37,ids(37,'test',36))
    with pytest.raises(ValueError):validate_evaluation_binding(m,'c'*64,37,ids(37,'test',36))
    with pytest.raises(ValueError):validate_evaluation_binding(m,'b'*64,37,list(reversed(ids(37,'test',36))))
    with pytest.raises(ValueError):validate_evaluation_binding(build(19),'b'*64,19,ids(19,'test',36))

def test_manifest_rejects_unavailable_or_fabricated_identity():
    with pytest.raises(ValueError):build(52)
    with pytest.raises(ValueError):build_subject_manifest(37,ids(38,'train',33),ids(37,'test',36),'a'*64,'b'*64)
