"""Differentiable counterparts of reference statistics for white-box evaluation."""
import torch
from torch import nn


def mahalanobis_score(features,means,precision):
    delta=features[:,None,:]-means[None,:,:]
    return torch.einsum('ncd,df,ncf->nc',delta,precision,delta).min(dim=1).values


def lid_score(features,reference,k):
    if features.ndim!=2 or reference.ndim!=2 or features.shape[1]!=reference.shape[1] or not 2<=k<=len(reference):
        raise ValueError('aligned features and valid neighborhood required')
    distances=torch.cdist(features,reference).topk(k,largest=False,sorted=True).values
    if (distances<=0).any():raise ValueError('zero distance requires explicit exclusion')
    denominator=(distances/distances[:,-1:]).log().sum(dim=1)
    if (denominator>=0).any():raise ValueError('degenerate neighborhood')
    return -k/denominator


def dropout_uncertainty_score(model,epochs,draws,seed):
    """EOT probability variance with autograd enabled, BatchNorm fixed."""
    if draws<2:raise ValueError('multiple EOT draws required')
    flags=[(m,m.training) for m in model.modules()]
    if not any(isinstance(m,(nn.Dropout,nn.Dropout1d,nn.Dropout2d,nn.Dropout3d)) for m,_ in flags):
        raise ValueError('explicit dropout modules required')
    devices=[] if epochs.device.type=='cpu' else [epochs.device.index or 0]
    try:
        model.eval()
        for m,_ in flags:
            if isinstance(m,(nn.Dropout,nn.Dropout1d,nn.Dropout2d,nn.Dropout3d)):m.train()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            probabilities=[]
            for _ in range(draws):
                p=model(epochs).sigmoid();probabilities.append(torch.stack([1-p,p],dim=-1))
            return torch.stack(probabilities).var(dim=0,unbiased=False).sum(dim=1)
    finally:
        for m,flag in flags:m.training=flag


def predicted_character_anomaly(flash_evidence,flash_scores,groups):
    """Piecewise differentiable actual predicted-character score [B]."""
    if flash_evidence.ndim!=2 or flash_scores.shape!=flash_evidence.shape or groups.shape[:2]!=flash_scores.shape or groups.dtype!=torch.bool:
        raise ValueError('aligned batch/flash/group tensors required')
    counts=groups.sum(dim=1)
    if (counts<=0).any():raise ValueError('each symbol requires coverage')
    means=torch.bmm(flash_evidence.unsqueeze(1),groups.to(flash_scores.dtype)).squeeze(1)/counts
    predicted=means.argmax(dim=1)
    mask=groups.gather(2,predicted[:,None,None].expand(-1,groups.shape[1],1)).squeeze(2)
    return (flash_scores*mask).sum(dim=1)/mask.sum(dim=1)


def probability_anomaly(logits,kind):
    """Conventional binary confidence/entropy controls, higher means anomalous."""
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError('finite floating point decoder logits required')
    p=logits.sigmoid()
    if kind=='confidence':return 1-torch.maximum(p,1-p)
    if kind=='entropy':return torch.nn.functional.softplus(logits)-logits*p
    raise ValueError('unknown probability control')
