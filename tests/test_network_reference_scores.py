import numpy as np
import pytest
from p3a.defenses.network_reference_scores import dropout_uncertainty,local_intrinsic_dimension,NetworkMahalanobis

def test_uncertainty_zero_and_disagreement():
    p=np.array([[[.9,.1],[.9,.1]],[[.9,.1],[.1,.9]]])
    u=dropout_uncertainty(p)
    assert u[0]==0 and u[1]>0

def test_lid_formula_and_self_exclusion():
    expected=-3/np.log(np.array([1,2,4])/4).sum()
    assert np.allclose(local_intrinsic_dimension([[0]],[[1],[2],[4]],3),expected)
    with pytest.raises(ValueError): local_intrinsic_dimension([[0]],[[0],[1],[2]],2)

def test_mahalanobis_far_point_and_source_class_order():
    x=np.array([[0,0],[0,.1],[1,1],[1,1.1]])
    m=NetworkMahalanobis().fit(x,[0,0,1,1])
    assert m.score([[20,20]])[0]>m.score([[0,0]])[0]
    n=NetworkMahalanobis().fit(x,[1,1,0,0])
    assert np.allclose(m.score(x),n.score(x))
