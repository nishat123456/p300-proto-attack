"""Bounded OpenBMI character loader candidate; stored event order preserved."""
import numpy as np
from p3a.data.layout import Layout
from p3a.data.trial import Trial

LAYOUT = Layout('openbmi_schedule36','ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_',6,6)


def build_trials(data, schedule, phrase, subject, session, split):
    groups_raw=np.concatenate([np.asarray(a,dtype=int) for a in schedule[:5]],axis=0)
    if groups_raw.shape != (60,6) or groups_raw.min()<1 or groups_raw.max()>36:
        raise ValueError('invalid five-sequence schedule')
    groups=np.zeros((60,36),dtype=bool)
    for i,row in enumerate(groups_raw):
        if len(set(row))!=6: raise ValueError('duplicate symbol within flash')
        groups[i,row-1]=True
    if not np.all(groups.sum(axis=0)==10): raise ValueError('invalid symbol coverage')
    x=np.asarray(data['x']); t=np.asarray(data['t']).ravel()
    labels=np.asarray(data['y_dec']).ravel(); fs=float(data['fs'])
    if not np.all(t==np.floor(t)) or len(t)!=60*len(phrase) or len(labels)!=len(t):
        raise ValueError('unaligned integer event/character counts')
    t=t.astype(np.int64)-1
    if x.ndim!=2 or not np.isfinite(x).all() or np.any(t<0) or np.any(t>=len(x)):
        raise ValueError('invalid signal/index bounds')
    if len(set(t.tolist()))!=len(t): raise ValueError('duplicate event indices')
    trials=[]
    for i,symbol in enumerate(phrase):
        on=t[i*60:(i+1)*60]; lab=labels[i*60:(i+1)*60]
        if not np.array_equal(groups[:,LAYOUT.index(symbol)],lab==1):
            raise ValueError('schedule and true-character labels disagree')
        start=max(0,int(on.min())-int(.2*fs)); end=int(on.max())+int(fs)+1
        if end>len(x): raise ValueError('post-flash epoch exceeds recording')
        trials.append(Trial(eeg=x[start:end].T.astype(np.float32),fs=fs,
            onsets=on-start,flash_groups=groups.copy(),flash_codes=np.tile(np.arange(1,13),5),
            target_symbol=symbol,layout=LAYOUT,subject_id=f's{subject:02d}_session{session}',
            dataset='openbmi_candidate',trial_index=i,source_split=split,
            channel_names=[str(v) for v in np.asarray(data['chan']).ravel()]))
    return trials
