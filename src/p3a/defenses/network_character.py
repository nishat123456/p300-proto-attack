"""Network-to-character reference adapters, without true-character input."""
import numpy as np
import torch
from torch import nn


def dropout_probability_draws(model,epochs,draws,seed):
    """Sample binary probabilities with dropout only; preserve BN and RNG state."""
    if not isinstance(draws,int) or draws<2 or epochs.ndim!=3:
        raise ValueError('at least two draws and [N,C,L] epochs required')
    modules=list(model.modules()); flags=[m.training for m in modules]
    if not any(isinstance(m,(nn.Dropout,nn.Dropout1d,nn.Dropout2d,nn.Dropout3d)) for m in modules):
        raise ValueError('model has no explicit dropout modules')
    devices=[] if epochs.device.type=='cpu' else [epochs.device.index or 0]
    try:
        model.eval()
        for m in modules:
            if isinstance(m,(nn.Dropout,nn.Dropout1d,nn.Dropout2d,nn.Dropout3d)): m.train()
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            torch.manual_seed(seed)
            result=[]
            for _ in range(draws):
                logits=model(epochs)
                if logits.shape!=(len(epochs),): raise ValueError('binary flash logits required')
                p=torch.sigmoid(logits)
                result.append(torch.stack([1-p,p],dim=-1).cpu().numpy())
            return np.stack(result)
    finally:
        for m,flag in zip(modules,flags): m.training=flag


def encoder_features(model,epochs):
    """Deterministic EEGNet encoder activations; explicit architecture adapter."""
    flags=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            values=model.encoder(epochs.unsqueeze(1))
            if values.ndim!=2: raise ValueError('flat encoder activations required')
            return values.cpu().numpy()
    finally:
        for m,flag in flags: m.training=flag


def predicted_character_score(flash_evidence,flash_anomaly,groups):
    """Mean anomaly on flashes supporting decoder-selected character.

    Candidate aggregation, not a reproduction or new detector claim. True intent
    is not an argument. Larger aggregate means more anomalous.
    """
    evidence=np.asarray(flash_evidence,dtype=float)
    anomaly=np.asarray(flash_anomaly,dtype=float); g=np.asarray(groups)
    if evidence.ndim!=1 or anomaly.shape!=evidence.shape or g.ndim!=2 or g.shape[0]!=len(evidence):
        raise ValueError('aligned flash scores/groups required')
    if g.dtype!=bool or not g.shape[1] or np.any(g.sum(axis=0)==0):
        raise ValueError('boolean groups must cover every character')
    if not np.isfinite(evidence).all() or not np.isfinite(anomaly).all():
        raise ValueError('finite scores required')
    symbol_evidence=(g.T@evidence)/g.sum(axis=0)
    predicted=int(np.argmax(symbol_evidence))
    return dict(predicted_index=predicted,score=float(anomaly[g[:,predicted]].mean()))
