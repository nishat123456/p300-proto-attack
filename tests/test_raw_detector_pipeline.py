import torch
from torch import nn
from p3a.defenses.raw_detector_pipeline import RawDetectorPipeline

class Pre(nn.Module):
    q=1;fs_out=5.
    def forward(self,x):return x
class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Flatten(start_dim=1);self.drop=nn.Dropout(.25)
    def forward(self,x):return self.drop(self.encoder(x.unsqueeze(1))).mean(dim=1)

def test_raw_input_gradients_reach_all_detector_paths():
    torch.manual_seed(17)
    refs=torch.randn(12,8);means=torch.stack([refs[:6].mean(0),refs[6:].mean(0)])
    p=RawDetectorPipeline(Pre(),Decoder(),torch.tensor([[0,4]]),torch.tensor([[[1,0],[0,1]]],dtype=torch.bool),refs,means,torch.eye(8),3,8,7,.001)
    raw=torch.randn(1,2,10,requires_grad=True)
    assert p.logits(raw).shape==(1,2)
    for kind in ('bu','lid','md','md_adjusted'):
        score=p.score(raw,kind)
        gradient=torch.autograd.grad(score.sum(),raw,retain_graph=True)[0]
        assert score.shape==(1,) and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert not any(param.requires_grad for param in p.parameters())
