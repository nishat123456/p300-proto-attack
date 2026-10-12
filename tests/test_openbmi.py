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

@pytest.mark.parametrize('kind',['fs_zero','fs_nan','label3','time_nan','fraction','channels','schedule_fraction','schedule_short','phrase'])
def test_input_contract(kind):
    d,s=fixture()
    phrase='A'
    if kind=='fs_zero': d['fs']=0
    if kind=='fs_nan': d['fs']=float('nan')
    if kind=='label3': d['y_dec'][0]=3
    if kind=='time_nan': d['t']=d['t'].astype(float);d['t'][0]=float('nan')
    if kind=='fraction': d['t']=d['t'].astype(float);d['t'][0]+=.5
    if kind=='channels': d['chan']=['Cz']
    if kind=='schedule_fraction': s=[a.astype(float) for a in s];s[0][0,0]=1.5
    if kind=='schedule_short': s=s[:4]
    if kind=='phrase': phrase='?'
    with pytest.raises(ValueError): build_trials(d,s,phrase,1,1,'train')
