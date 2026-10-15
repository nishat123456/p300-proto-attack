"""Character-level rejection accounting and subject-cluster uncertainty."""
import numpy as np
from scipy.stats import beta


def exact_binomial_interval(alarms,total,confidence=.95):
    if not isinstance(alarms,int) or not isinstance(total,int) or not 0<=alarms<=total or total<1 or not 0<confidence<1:
        raise ValueError('valid integer independent counts/confidence required')
    alpha=1-confidence
    return (0. if alarms==0 else float(beta.ppf(alpha/2,alarms,total-alarms+1)),
            1. if alarms==total else float(beta.ppf(1-alpha/2,alarms+1,total-alarms)))


def clean_rejection_accounting(correct,rejected):
    correct=np.asarray(correct);rejected=np.asarray(rejected)
    if correct.dtype!=bool or rejected.dtype!=bool or correct.ndim!=1 or correct.shape!=rejected.shape or not len(correct):
        raise ValueError('aligned nonempty boolean character outcomes required')
    accepted=~rejected
    return dict(n_characters=len(correct),n_rejected=int(rejected.sum()),
        clean_accuracy=float(correct.mean()),observed_far=float(rejected.mean()),
        accepted_correct_rate=float((accepted&correct).mean()),
        accepted_wrong_rate=float((accepted&~correct).mean()),
        conditional_accepted_accuracy=None if not accepted.any() else float(correct[accepted].mean()),
        iid_retry_overhead=None if not accepted.any() else float(1/accepted.mean()-1))


def paired_subject_bootstrap(left,right,seed,n_resamples=2000):
    """Input one seed-averaged summary per subject, paired across methods.

    This estimates subject-population uncertainty under exchangeable subjects;
    not per-user bounds or independent repeated-target observations.
    """
    left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    if left.ndim!=1 or left.shape!=right.shape or len(left)<2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('paired finite subject summaries required')
    if not isinstance(n_resamples,int) or n_resamples<100:raise ValueError('at least100 resamples required')
    delta=left-right;rng=np.random.default_rng(seed)
    means=delta[rng.integers(0,len(delta),size=(n_resamples,len(delta)))].mean(axis=1)
    return dict(mean_difference=float(delta.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n_subjects=len(delta),seed=seed,n_resamples=n_resamples)
