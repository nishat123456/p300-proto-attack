from types import SimpleNamespace
import numpy as np
import torch
import pytest
from p3a.attacks.source_template import fit_source_template


def trials():
    return [SimpleNamespace(subject_id=f's{s:02d}_session1',source_split='train',eeg=np.zeros((2,8+s%2),dtype=np.float32),
        onsets=np.array([0,3]),flash_groups=np.eye(2,dtype=bool),layout=SimpleNamespace(index=lambda x:0),target_symbol='A',trial_id=f's{s:02d}:train') for s in range(1,19)]


def logits(raw,on,groups):
    z=raw.mean((1,2));return torch.stack([z,-z],dim=1)


def test_source_template_variable_lengths_and_reproducibility():
    kwargs=dict(eps_raw=.2,seed=7,updates=2,batch_size=3,template_len=3)
    template,info=fit_source_template(trials(),logits,**kwargs)
    other,_=fit_source_template(trials(),logits,**kwargs)
    assert template.shape==(2,3) and torch.isfinite(template).all() and template.abs().max()<=1
    assert torch.equal(template,other) and len(info['losses'])==2


def test_template_rejects_target_subject_or_phase():
    t=trials();t[0].subject_id='s37_session1'
    with pytest.raises(ValueError):fit_source_template(t,logits,.2,7,updates=1)
    t=trials();t[0].source_split='test'
    with pytest.raises(ValueError):fit_source_template(t,logits,.2,7,updates=1)
