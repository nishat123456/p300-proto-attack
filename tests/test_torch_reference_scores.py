import numpy as np
import torch
from p3a.defenses.network_reference_scores import local_intrinsic_dimension,NetworkMahalanobis
from p3a.defenses.torch_reference_scores import lid_score,mahalanobis_score,predicted_character_anomaly,dropout_uncertainty_score
from p3a.defenses.network_character import dropout_probability_draws
from p3a.defenses.network_reference_scores import dropout_uncertainty
from torch import nn

def test_numpy_parity_and_query_gradient():
    r=np.array([[0.,0.],[.1,.2],[1.,1.],[1.1,.9]])
    q=torch.tensor([[2.,3.]],dtype=torch.float64,requires_grad=True)
    md=NetworkMahalanobis().fit(r,[0,0,1,1])
    a=mahalanobis_score(q,torch.tensor(md.means),torch.tensor(md.precision))
    b=lid_score(q,torch.tensor(r),3)
    assert np.allclose(a.detach().numpy(),md.score(q.detach().numpy()))
    assert np.allclose(b.detach().numpy(),local_intrinsic_dimension(q.detach().numpy(),r,3))
    g=torch.autograd.grad((a+b).sum(),q)[0]
    assert torch.isfinite(g).all() and g.abs().sum()>0

def test_actual_predicted_character_gradient():
    evidence=torch.tensor([[0.,4.,0.,4.]])
    scores=torch.tensor([[9.,1.,9.,3.]],requires_grad=True)
    groups=torch.tensor([[[1,0],[0,1],[1,0],[0,1]]],dtype=torch.bool)
    value=predicted_character_anomaly(evidence,scores,groups)
    assert value.item()==2
    assert torch.equal(torch.autograd.grad(value.sum(),scores)[0],torch.tensor([[0.,.5,0.,.5]]))

def test_eot_parity_and_gradient():
    class Model(nn.Module):
        def __init__(self):super().__init__();self.drop=nn.Dropout(.5)
        def forward(self,x):return self.drop(x).flatten(1).sum(dim=1)
    m=Model();x=torch.ones(4,2,2,requires_grad=True)
    expected=dropout_uncertainty(dropout_probability_draws(m,x,8,7))
    actual=dropout_uncertainty_score(m,x,8,7)
    assert np.allclose(actual.detach().numpy(),expected)
    assert torch.isfinite(torch.autograd.grad(actual.sum(),x)[0]).all()


def test_probability_controls_analytic_extremes_and_gradient():
    from p3a.defenses.torch_reference_scores import probability_anomaly
    z=torch.tensor([-1000.,-1.,0.,1.,1000.],requires_grad=True)
    confidence=probability_anomaly(z,'confidence')
    entropy=probability_anomaly(z,'entropy')
    assert torch.isfinite(confidence).all() and torch.isfinite(entropy).all()
    assert confidence[2]==.5 and torch.allclose(entropy[2],torch.tensor(2.).log())
    assert confidence[0]==0 and confidence[-1]==0
    assert entropy[0]==0 and entropy[-1]==0
    assert torch.allclose(confidence[1],confidence[3])
    assert torch.allclose(entropy[1],entropy[3])
    gradient=torch.autograd.grad((entropy+confidence).sum(),z)[0]
    assert torch.isfinite(gradient).all() and gradient[1]!=0 and gradient[3]!=0
