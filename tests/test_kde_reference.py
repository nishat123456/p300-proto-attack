import numpy as np
import torch
import pytest
from p3a.defenses.kde_reference import kde_anomaly,torch_kde_anomaly

def test_negative_exponent_density_and_torch_parity_gradient():
    r=np.array([[0.,0.],[.1,.2],[1.,1.],[1.1,1.2]]);labels=np.array([0,0,1,1])
    x=np.array([[0.,0.],[4.,4.]])
    expected=kde_anomaly(x,r,labels,[0,0],.5)
    assert expected[1]>expected[0]
    q=torch.tensor(x,requires_grad=True)
    actual=torch_kde_anomaly(q,torch.tensor(r),torch.tensor(labels),torch.tensor([0,0]),.5)
    assert np.allclose(actual.detach().numpy(),expected)
    assert torch.isfinite(torch.autograd.grad(actual.sum(),q)[0]).all()

def test_missing_class_rejected():
    with pytest.raises(ValueError):kde_anomaly([[0]],[[1]],[0],[1],.5)
