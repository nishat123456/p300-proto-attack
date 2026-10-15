"""Raw differentiable decoder/detector callback for adaptive evaluation.

References and preprocessing must be fitted on allowed source/clean data before
construction. This module fits nothing and accepts no true-character labels.
"""
import torch
from torch import nn
from p3a.data.epochs import gather_epochs,downsample_onsets,epoch_window
from p3a.models.letter_head import symbol_evidence
from p3a.defenses.torch_reference_scores import mahalanobis_score,lid_score,dropout_uncertainty_score,predicted_character_anomaly


class RawDetectorPipeline(nn.Module):
    def __init__(self,preprocessor,decoder,onsets,groups,reference_features,means,precision,k,draws,seed,adjustment_epsilon,reference_labels=None,kde_bandwidth=.5):
        super().__init__()
        self.preprocessor=preprocessor;self.decoder=decoder
        self.register_buffer('onsets',onsets.long());self.register_buffer('groups',groups.bool())
        for name,value in [('reference_features',reference_features),('means',means),('precision',precision)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer('reference_labels',torch.as_tensor([] if reference_labels is None else reference_labels,dtype=torch.long))
        self.kde_bandwidth=kde_bandwidth
        self.k=k;self.draws=draws;self.seed=seed;self.adjustment_epsilon=adjustment_epsilon
        if adjustment_epsilon<0:raise ValueError('nonnegative reference adjustment required')
        self.eval()
        for p in self.parameters():p.requires_grad_(False)

    def _epochs(self,raw):
        signal=self.preprocessor(raw)
        epochs=gather_epochs(signal,downsample_onsets(self.onsets,self.preprocessor.q),epoch_window(self.preprocessor.fs_out))
        b,f,c,l=epochs.shape
        return epochs.reshape(b*f,c,l),b,f

    def logits(self,raw):
        epochs,b,f=self._epochs(raw)
        return symbol_evidence(self.decoder(epochs).reshape(b,f),self.groups)

    def _features(self,epochs):return self.decoder.encoder(epochs.unsqueeze(1))

    def score(self,raw,kind):
        epochs,b,f=self._epochs(raw)
        evidence=self.decoder(epochs).reshape(b,f)
        if kind in ('confidence','entropy'):
            from p3a.defenses.torch_reference_scores import probability_anomaly
            values=probability_anomaly(evidence.reshape(-1),kind)
        elif kind=='bu':
            values=dropout_uncertainty_score(self.decoder,epochs,self.draws,self.seed)
        elif kind=='kde':
            from p3a.defenses.kde_reference import torch_kde_anomaly
            if len(self.reference_labels)!=len(self.reference_features):raise ValueError('source reference labels required for KDE')
            flash_predictions=(self.decoder(epochs)>0).long()
            values=torch_kde_anomaly(self._features(epochs),self.reference_features,self.reference_labels,flash_predictions,self.kde_bandwidth)
        elif kind=='lid':values=lid_score(self._features(epochs),self.reference_features,self.k)
        elif kind in ('md','md_adjusted'):
            if kind=='md_adjusted':
                with torch.enable_grad():
                    x=epochs if epochs.requires_grad else epochs.detach().requires_grad_(True)
                    distance=mahalanobis_score(self._features(x),self.means,self.precision)
                    gradient=torch.autograd.grad(distance.sum(),x,create_graph=True,retain_graph=True)[0]
                    adjusted=x-self.adjustment_epsilon*gradient.sign()
                    values=mahalanobis_score(self._features(adjusted),self.means,self.precision)
            else:values=mahalanobis_score(self._features(epochs),self.means,self.precision)
        else:raise ValueError('unknown detector kind')
        return predicted_character_anomaly(evidence,values.reshape(b,f),self.groups)
