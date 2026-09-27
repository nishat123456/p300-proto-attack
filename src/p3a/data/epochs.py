"""Differentiable flash epoch extraction and fixed-shape trial batching."""
from __future__ import annotations

import math
import operator

import numpy as np
import torch


def downsample_onsets(onsets, q):
    """Floor raw sample indices onto the decimated sample grid."""
    q = operator.index(q)
    if q <= 0:
        raise ValueError("q must be positive")
    if isinstance(onsets, torch.Tensor):
        if onsets.is_floating_point() or onsets.dtype == torch.bool:
            raise ValueError("onsets must be integer sample indices")
        if (onsets < 0).any():
            raise ValueError("onsets must be nonnegative")
        return torch.div(onsets.to(torch.int64), q, rounding_mode="floor")
    onsets = np.asarray(onsets)
    if not np.issubdtype(onsets.dtype, np.integer) or (onsets < 0).any():
        raise ValueError("onsets must be nonnegative integer sample indices")
    return onsets.astype(np.int64) // q


def epoch_window(fs_out, tmax=0.8):
    """Number of samples in the half-open interval [0, tmax)."""
    if not math.isfinite(fs_out) or not math.isfinite(tmax) or fs_out <= 0 or tmax <= 0:
        raise ValueError("fs_out and tmax must be finite and positive")
    length = int(round(tmax * fs_out))
    if length < 1:
        raise ValueError("epoch window must contain at least one sample")
    return length


def gather_epochs(x, onsets_ds, L):
    """Gather [B,F,C,L] epochs from [B,C,T] without padding or truncation."""
    L = operator.index(L)
    if x.ndim != 3 or onsets_ds.ndim != 2 or x.shape[0] != onsets_ds.shape[0]:
        raise ValueError("expected x [B,C,T] and onsets_ds [B,F]")
    if L <= 0 or onsets_ds.dtype != torch.long:
        raise ValueError("L must be positive and onsets_ds must be int64")
    if onsets_ds.device != x.device:
        raise ValueError("x and onsets_ds must be on the same device")
    if (onsets_ds < 0).any() or (onsets_ds + L > x.shape[-1]).any():
        raise ValueError("epoch extends outside the signal")
    indices = onsets_ds[..., None] + torch.arange(L, device=x.device)
    b, c, _ = x.shape
    f = onsets_ds.shape[1]
    values = x.gather(2, indices.reshape(b, 1, f * L).expand(b, c, f * L))
    return values.reshape(b, c, f, L).permute(0, 2, 1, 3)


def batch_trials(trials, device):
    """Return (eeg, onsets, groups, targets) for compatible Trial objects."""
    if not trials:
        raise ValueError("cannot batch an empty trial list")
    first = trials[0]
    for trial in trials:
        if trial.layout != first.layout or trial.fs != first.fs:
            raise ValueError("trials must share a layout and sampling rate")
        if (trial.eeg.shape != first.eeg.shape or trial.onsets.shape != first.onsets.shape
                or trial.flash_groups.shape != first.flash_groups.shape):
            raise ValueError("trials must have equal C, T, F, and K")
        if (trial.eeg.ndim != 2 or trial.onsets.ndim != 1
                or trial.flash_groups.shape != (len(trial.onsets), trial.layout.n_symbols)):
            raise ValueError("invalid trial dimensions")
    return (
        torch.as_tensor(np.stack([t.eeg for t in trials]), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack([t.onsets for t in trials]), dtype=torch.long, device=device),
        torch.as_tensor(np.stack([t.flash_groups for t in trials]), dtype=torch.bool, device=device),
        torch.tensor([t.layout.index(t.target_symbol) for t in trials], dtype=torch.long, device=device),
    )
