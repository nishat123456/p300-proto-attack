import torch
import pytest
from p3a.attacks.adaptive_evasion import defense_aware_pgd

def logits(x):
    z=x.flatten(1).mean(dim=1);return torch.stack([-z,z],dim=1)
def score(x):return x.flatten(1).square().mean(dim=1)
def run(x,eps,weight=0):
    return defense_aware_pgd(x,torch.ones(len(x),dtype=torch.long),logits,score,0.,1.,eps,weight,20,2,7)

def test_zero_budget_and_reproducible_projection():
    x=torch.zeros(2,1,8)
    assert torch.equal(run(x,0),x)
    d=run(x,.5);assert d.abs().max()<=.5 and torch.equal(d,run(x,.5))
    assert (logits(x+d).argmax(dim=1)==1).all()

def test_detector_penalty_changes_tradeoff():
    x=torch.zeros(2,1,8)
    plain=run(x,.5);aware=run(x,.5,100)
    assert score(x+aware).mean()<score(x+plain).mean()

def test_invalid_scale_rejected():
    with pytest.raises(ValueError):defense_aware_pgd(torch.zeros(1,1,8),torch.ones(1,dtype=torch.long),logits,score,0.,0.,.1,1.,10,1,0)


def test_returned_budget_with_large_float32_baseline():
    x=torch.full((2,1,8),10000.,dtype=torch.float32)
    eps=.3678136825561524
    delta=run(x,eps)
    assert torch.isfinite(delta).all()
    assert delta.abs().max()<=eps
