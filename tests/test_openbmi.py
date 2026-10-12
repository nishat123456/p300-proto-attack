import numpy as np
import pytest
from p3a.data.loaders.openbmi import build_trials

def fixture():
    rows=np.arange(1,37).reshape(6,6)
    schedule=[np.concatenate([rows,rows.T])]*5
    mask=np.tile(np.r_[True,np.zeros(5,dtype=bool),True,np.zeros(5,dtype=bool)],5)
    t=np.arange(201,801,10); t[0],t[1]=t[1],t[0]
    return dict(x=np.zeros((2000,2)),t=t,y_dec=np.where(mask,1,2),fs=100,chan=['Cz','Pz']),schedule

def test_preserves_order_and_index():
    data,s=fixture(); trials=build_trials(data,s,'A',1,1,'train')
    assert len(trials)==1 and trials[0].onsets[0]>trials[0].onsets[1]
    assert trials[0].n_flashes==60 and trials[0].target_flash_mask().sum()==10

@pytest.mark.parametrize('kind',['negative','duplicate','labels','end'])
def test_rejects_invalid(kind):
    d,s=fixture()
    if kind=='negative': d['t'][0]=0
    if kind=='duplicate': d['t'][0]=d['t'][1]
    if kind=='labels': d['y_dec'][0]=1 if d['y_dec'][0]==2 else 2
    if kind=='end': d['x']=d['x'][:800]
    with pytest.raises(ValueError): build_trials(d,s,'A',1,1,'train')
