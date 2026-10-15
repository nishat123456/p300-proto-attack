"""Class-conditional Gaussian KDE anomaly score, explicit negative exponent.

Adaptation: Zhang/Gu2024 printedEq7 has positive exponent inconsistent with
its low-density description. Use-logmeanexp(-squared_distance/bandwidth**2).
Not certified exact reproduction of unavailable author implementation.
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def kde_anomaly(features,reference,reference_labels,predicted_labels,bandwidth):
    x=np.asarray(features,dtype=float);r=np.asarray(reference,dtype=float)
    y=np.asarray(reference_labels);p=np.asarray(predicted_labels)
    if not np.isfinite(bandwidth) or bandwidth<=0 or x.ndim!=2 or r.ndim!=2 or x.shape[1]!=r.shape[1] or y.shape!=(len(r),) or p.shape!=(len(x),):
        raise ValueError('valid bandwidth,features and aligned labels required')
    if not np.isfinite(x).all() or not np.isfinite(r).all():raise ValueError('finite features required')
    scores=np.empty(len(x))
    for c in np.unique(p):
        ref=r[y==c]
        if not len(ref):raise ValueError('missing source reference class')
        d2=cdist(x[p==c],ref,metric='sqeuclidean')
        scores[p==c]=-(logsumexp(-d2/bandwidth**2,axis=1)-np.log(len(ref)))
    return scores


def torch_kde_anomaly(features,reference,reference_labels,predicted_labels,bandwidth):
    if not np.isfinite(bandwidth) or bandwidth<=0:raise ValueError('positive finite bandwidth required')
    scores=torch.zeros(len(features),device=features.device,dtype=features.dtype)
    for c in predicted_labels.unique():
        mask=predicted_labels==c;ref=reference[reference_labels==c]
        if not len(ref):raise ValueError('missing source reference class')
        d2=torch.cdist(features[mask],ref).square()
        scores[mask]=-(torch.logsumexp(-d2/bandwidth**2,dim=1)-np.log(len(ref)))
    return scores
