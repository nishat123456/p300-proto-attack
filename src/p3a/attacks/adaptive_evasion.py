"""Defense-aware raw-signal PGD engine; caller supplies differentiable score.

Evaluation stress test, not a new attack contribution. Stochastic defenses
require caller-specified EOT in score_fn and a fixed audit of gradient behavior.
"""
import math
import torch
import torch.nn.functional as F


def defense_aware_pgd(eeg,target,logits_fn,score_fn,threshold,score_scale,
                      eps_raw,weight,steps,restarts,seed,audit=None):
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
        x=eeg.detach()+project_applied_delta(eeg,torch.empty_like(eeg).uniform_(-eps_raw,eps_raw,generator=rng),eps_raw)
        for iteration in range(steps+1):
            x=x.detach().requires_grad_(True)
            logits=logits_fn(x);score=score_fn(x)
            if score.shape!=(b,):raise ValueError('one differentiable detector score per character required')
            ce=F.cross_entropy(logits,target,reduction='none')
            loss=ce+weight*F.softplus((score-threshold)/scale)
            if not torch.isfinite(loss).all():raise ValueError('nonfinite adaptive objective')
            better=loss.detach()<best_loss
            best[better]=x.detach()[better];best_loss=torch.minimum(best_loss,loss.detach())
            if audit is not None and (iteration%20==0 or iteration==steps):
                audit(dict(restart=restart,iteration=iteration,ce=ce.detach().cpu().tolist(),score=score.detach().cpu().tolist(),best_objective=best_loss.cpu().tolist()))
            if iteration==steps:break
            gradient=torch.autograd.grad(loss.sum(),x)[0]
            if not torch.isfinite(gradient).all():raise ValueError('nonfinite adaptive gradient')
            x=x-step*gradient.sign()
            x=eeg+project_applied_delta(eeg,x-eeg,eps_raw)
    # Subtracting large float32 baselines can round beyond the projected bound.
    # Enforce the declared perturbation budget on the returned delta itself.
    return project_applied_delta(eeg,best-eeg,eps_raw)


def project_applied_delta(eeg,delta,eps_raw):
    """Enforce the budget on the actually representable raw sum, not only delta."""
    if eeg.shape!=delta.shape or not math.isfinite(eps_raw) or eps_raw<0:
        raise ValueError('aligned raw delta and finite nonnegative budget required')
    with torch.no_grad():
        applied=eeg+delta.clamp(-eps_raw,eps_raw)
        # One rounding ULP inward repairs addition/subtraction cancellation.
        excess=(applied-eeg).abs()>eps_raw
        applied=torch.where(excess,torch.nextafter(applied,eeg),applied)
        result=applied-eeg
        if not torch.isfinite(result).all() or (result.abs()>eps_raw).any():raise ValueError('raw projection failed')
        return result
