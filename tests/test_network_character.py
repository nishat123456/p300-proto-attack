import numpy as np
import torch
from torch import nn
import pytest
from p3a.defenses.network_character import dropout_probability_draws,predicted_character_score

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm1d(2);self.drop=nn.Dropout(.5)
    def forward(self,x): return self.drop(self.bn(x[:,:,0])).sum(axis=1)

def test_dropout_freezes_bn_restores_modes_and_rng():
    m=Toy();m.train();x=torch.ones(8,2,1)
    before=m.bn.running_mean.clone();rng=torch.get_rng_state().clone()
    draws=dropout_probability_draws(m,x,8,7)
    assert draws.shape==(8,8,2) and np.allclose(draws.sum(axis=2),1)
    assert torch.equal(before,m.bn.running_mean) and m.training and m.bn.training
    assert torch.equal(rng,torch.get_rng_state())
    assert np.allclose(draws,dropout_probability_draws(m,x,8,7))
    assert draws.var(axis=0).sum()>0

def test_character_selected_by_evidence_not_anomaly():
    g=np.array([[True,False],[False,True],[True,False],[False,True]])
    r=predicted_character_score([0,4,0,4],[9,1,9,3],g)
    assert r['predicted_index']==1 and r['score']==2
    with pytest.raises(ValueError): predicted_character_score([0,1],[1,2],np.zeros((2,2),dtype=bool))
