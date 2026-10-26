"""Regression for Python-epsilon conversion, without tolerant comparisons."""
import pytest
import torch
from p3a.attacks.adaptive_evasion import project_applied_delta

@pytest.mark.parametrize('epsilon',[4.5227703094482425,.3,.1,.05])
def test_strict_python_bound_after_raw_readdition(epsilon):
    raw=torch.tensor([0.,1.,-1.,100.,-100.,10000.,-10000.],dtype=torch.float32)
    for sign in [-1,1]:
        delta=project_applied_delta(raw,torch.full_like(raw,sign*epsilon*2),epsilon)
        assert bool(((((raw+delta)-raw).double().abs())<=epsilon).all())

def test_old_scalar_comparison_can_accept_an_upward_rounded_bound():
    epsilon=4.5227703094482425
    rounded=torch.tensor(epsilon,dtype=torch.float32)
    assert not bool(rounded>epsilon)
    assert float(rounded)>epsilon
    raw=torch.zeros(1,dtype=torch.float32)
    corrected=project_applied_delta(raw,rounded.reshape(1),epsilon)
    assert float(((raw+corrected)-raw).abs().max())<=epsilon
