import pytest
from p3a.data.openbmi_partitions import candidate_character_partition,validate_partition,candidate_subject_role

def ids(): return [f's1:train:{i}' for i in range(33)],[f's1:test:{i}' for i in range(36)]

def test_disjoint_reproducible_and_phase_preserved():
    a,b=ids();p=candidate_character_partition(a,b,20261013)
    assert p==candidate_character_partition(a,b,20261013)
    assert set(p['normalization']+p['calibration'])==set(a) and p['evaluation']==b
    validate_partition(p)

def test_rejects_overlap_and_role_leakage():
    a,b=ids();b[0]=a[0]
    with pytest.raises(ValueError): candidate_character_partition(a,b,1)
    a,b=ids();p=candidate_character_partition(a,b,1)
    p['calibration'][0]=p['normalization'][0]
    with pytest.raises(ValueError): validate_partition(p)

def test_subject_boundaries():
    assert [candidate_subject_role(i) for i in (1,18,19,36,37,54)]==['development','development','validation','validation','evaluation','evaluation']
    with pytest.raises(ValueError): candidate_subject_role(55)
