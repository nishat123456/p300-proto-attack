"""Convert flash logits into mean evidence for arbitrary symbol groups."""
import math
import operator

import torch


def symbol_evidence(flash_scores, groups):
    """Mean score over flashes containing each symbol, returning [B,K].

    Reject symbols with no observations instead of silently assigning evidence.
    """
    if (flash_scores.ndim != 2 or groups.ndim != 3
            or flash_scores.shape != groups.shape[:2]):
        raise ValueError("expected flash_scores [B,F] and groups [B,F,K]")
    if not flash_scores.is_floating_point() or groups.dtype != torch.bool:
        raise ValueError("scores must be floating point and groups must be boolean")
    if flash_scores.device != groups.device:
        raise ValueError("scores and groups must be on the same device")
    counts = groups.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("every symbol must occur in at least one flash")
    weights = groups.to(flash_scores.dtype)
    return torch.bmm(flash_scores.unsqueeze(1), weights).squeeze(1) / counts


def symbol_logits(flash_scores, groups, tau=1.0):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    return symbol_evidence(flash_scores, groups) / tau


def first_reps(onsets, groups, flash_scores, n_reps, flashes_per_rep):
    """Return the first complete repetitions in their existing flash order."""
    n_reps = operator.index(n_reps)
    flashes_per_rep = operator.index(flashes_per_rep)
    if (onsets.ndim != 2 or flash_scores.shape != onsets.shape
            or groups.ndim != 3 or groups.shape[:2] != onsets.shape):
        raise ValueError("expected batched, aligned onsets, groups, and scores")
    n = n_reps * flashes_per_rep
    if n_reps <= 0 or flashes_per_rep <= 0 or n > onsets.shape[1]:
        raise ValueError("requested repetitions must fit in the available flashes")
    return onsets[:, :n], groups[:, :n], flash_scores[:, :n]
