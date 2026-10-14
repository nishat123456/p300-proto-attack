"""Reference detector statistics on source-trained network activations.

Adaptations of BU/LID/Mahalanobis families, not certified paper reproductions.
No target labels or attacked target features enter reference fitting.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf


def _matrix(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():
        raise ValueError('finite nonempty feature matrix required')
    return x


def dropout_uncertainty(probability_draws):
    p=np.asarray(probability_draws,dtype=float)
    if p.ndim!=3 or p.shape[0]<2 or p.shape[2]<2 or not np.isfinite(p).all():
        raise ValueError('draws x examples x classes probability tensor required')
    if np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=2),1):
        raise ValueError('normalized probabilities required')
    return p.var(axis=0).sum(axis=1)


def local_intrinsic_dimension(query,reference,k):
    x,r=_matrix(query),_matrix(reference)
    if x.shape[1]!=r.shape[1] or not isinstance(k,int) or not 2<=k<=len(r):
        raise ValueError('compatible features and valid integer k required')
    distances=np.sort(cdist(x,r),axis=1)[:,:k]
    if np.any(distances<=0):
        raise ValueError('zero distances require explicit self-exclusion/duplicate policy')
    denominator=np.log(distances/distances[:,-1:]).sum(axis=1)
    if np.any(denominator>=0): raise ValueError('degenerate equal-distance neighborhood')
    return -k/denominator


class NetworkMahalanobis:
    """Shared source-class residual covariance with fixed shrinkage.

    Ledoit-Wolf covariance is an explicit adaptation, not a silent reproduction.
    Returns nearest-class squared distance; larger values mean more anomalous.
    """
    def fit(self,source_features,source_labels):
        x=_matrix(source_features); y=np.asarray(source_labels)
        if y.ndim!=1 or len(y)!=len(x) or len(np.unique(y))<2:
            raise ValueError('aligned multi-class source labels required')
        self.classes=np.unique(y)
        self.means=np.stack([x[y==c].mean(axis=0) for c in self.classes])
        residual=np.empty_like(x)
        for c,mean in zip(self.classes,self.means): residual[y==c]=x[y==c]-mean
        self.precision=LedoitWolf().fit(residual).precision_
        return self

    def score(self,features):
        x=_matrix(features)
        if x.shape[1]!=self.means.shape[1]: raise ValueError('feature dimension mismatch')
        delta=x[:,None,:]-self.means[None,:,:]
        return np.einsum('ncd,df,ncf->nc',delta,self.precision,delta).min(axis=1)
