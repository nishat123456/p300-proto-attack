"""Defense-aware raw-signal PGD engine; caller supplies differentiable score.

Evaluation stress test, not a new attack contribution. Stochastic defenses
require caller-specified EOT in score_fn and a fixed audit of gradient behavior.
"""
import math
import torch
import torch.nn.functional as F


def defense_aware_pgd(eeg,target,logits_fn,score_fn,threshold,score_scale,
                      eps_raw,weight,steps,restarts,seed):
    if eeg.ndim!=3 or not eeg.is_floating_point() or not torch.isfinite(eeg).all():
        raise ValueError('finite raw [B,C,T] input required')
    b=len(eeg)
    if target.shape!=(b,) or target.dtype!=torch.long:raise ValueError('aligned integer targets required')
    if not all(math.isfinite(v) and v>=0 for v in (eps_raw,weight)) or steps<1 or restarts<1:
        raise ValueError('nonnegative finite budget/weight and positive counts required')
    threshold=torch.as_tensor(threshold,device=eeg.device,dtype=eeg.dtype).expand(b)
    scale=torch.as_tensor(score_scale,device=eeg.device,dtype=eeg.dtype).expand(b)
    if not torch.isfinite(threshold).all() or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('finite thresholds and positive calibration-derived scale required')
    rng=torch.Generator(device=eeg.device).manual_seed(seed)
    best=eeg.detach().clone();best_loss=torch.full((b,),float('inf'),device=eeg.device)
    step=eps_raw/10
    for restart in range(restarts):
        x=eeg.detach()+torch.empty_like(eeg).uniform_(-eps_raw,eps_raw,generator=rng)
        for iteration in range(steps+1):
            x=x.detach().requires_grad_(True)
            logits=logits_fn(x);score=score_fn(x)
            if score.shape!=(b,):raise ValueError('one differentiable detector score per character required')
            ce=F.cross_entropy(logits,target,reduction='none')
            loss=ce+weight*F.softplus((score-threshold)/scale)
            if not torch.isfinite(loss).all():raise ValueError('nonfinite adaptive objective')
            better=loss.detach()<best_loss
            best[better]=x.detach()[better];best_loss=torch.minimum(best_loss,loss.detach())
            if iteration==steps:break
            gradient=torch.autograd.grad(loss.sum(),x)[0]
            if not torch.isfinite(gradient).all():raise ValueError('nonfinite adaptive gradient')
            x=x-step*gradient.sign()
            x=eeg+(x-eeg).clamp(-eps_raw,eps_raw)
    # Subtracting large float32 baselines can round beyond the projected bound.
    # Enforce the declared perturbation budget on the returned delta itself.
    return (best-eeg).detach().clamp(-eps_raw,eps_raw)
