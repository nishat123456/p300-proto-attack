"""Targeted-attack success for the P300 speller (Prompt 10).

Targeted success is the fraction of (source trial, attacker target) pairs where
the speller's symbol argmax equals the attacker target. The experiment reports
it at full 15-repetition evidence and at the reduced 5-repetition setting, for
the surrogate and for every victim, using the same fixed pre-registered targets
for every method (PLAN.md Sections 5 and 6).

Everything here is batched and works on tensors, so the caller can keep the
computation on the device the models run on. The reduced-repetition path reuses
the flash scores already computed by the speller and only re-runs the cheap
letter head, so evaluating 5 reps costs no extra speller forward passes.
"""
from __future__ import annotations

import torch

from p3a.models.letter_head import symbol_logits

__all__ = [
    "FLASHES_PER_REP",
    "REPS",
    "multi_rep_symbol_choice",
    "reduced_rep_choice",
    "targeted_success",
]

# Comp III runs 15 repetitions of 12 flashes per character trial.
FLASHES_PER_REP = 12
REPS = (15, 5)


def _combined_groups(groups: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
    """Union a base [B, F, K] group mask with any extra [B, F, K2] symbols.

    The victims and the surrogate share the Comp III 36-symbol layout, so the
    base mask already covers every target. ``extra`` lets a caller score a speller
    whose layout has different symbols by widening the group matrix, which keeps
    this module dataset agnostic without any special casing.
    """
    if extra is None:
        return groups
    return torch.cat([groups, extra.to(groups.dtype)], dim=2)


def reduced_rep_choice(flash_scores: torch.Tensor, groups: torch.Tensor,
                       on_batch, n_reps: int, flashes_per_rep: int = FLASHES_PER_REP
                       ) -> torch.Tensor:
    """Argmax symbol using only the first ``n_reps`` complete repetitions.

    ``flash_scores`` is [B, F] and ``groups`` is [B, F, K] for one batch of
    trials. ``on_batch`` maps a flash-score tensor to a victim's score tensor of
    the same shape when the victim differs from the surrogate; pass ``None`` to
    reuse the surrogate's scores. Returns the chosen symbol index per trial.
    """
    if flash_scores.ndim != 2 or groups.ndim != 3:
        raise ValueError("flash_scores must be [B, F] and groups [B, F, K]")
    if flash_scores.shape[:2] != groups.shape[:2]:
        raise ValueError("flash_scores and groups must align on (B, F)")
    n = n_reps * flashes_per_rep
    if n > flash_scores.shape[1]:
        raise ValueError("requested repetitions exceed the available flashes")
    scores = flash_scores[:, :n]
    if on_batch is not None:
        scores = on_batch(scores)
    sub_groups = groups[:, :n]
    return symbol_logits(scores, sub_groups).argmax(dim=1)


def multi_rep_symbol_choice(flash_scores: torch.Tensor, groups: torch.Tensor,
                            on_batch=None) -> dict[int, torch.Tensor]:
    """Chosen symbol index per trial for each repetition count in ``REPS``."""
    return {
        reps: reduced_rep_choice(flash_scores, groups, on_batch, reps)
        for reps in REPS
    }


def targeted_success(choice: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of trials whose chosen symbol equals the attacker target."""
    if choice.ndim != 1 or target.ndim != 1 or choice.shape != target.shape:
        raise ValueError("choice and target must be aligned 1-D tensors")
    if choice.numel() == 0:
        return float("nan")
    return float((choice == target).float().mean())


def _as_long(tensor, device) -> torch.Tensor:
    if torch.is_tensor(tensor):
        return tensor.to(device=device, dtype=torch.long)
    return torch.as_tensor(tensor, dtype=torch.long, device=device)


def success_for_speller(speller, eeg: torch.Tensor, onsets: torch.Tensor,
                        groups: torch.Tensor, target, n_reps: int,
                        max_flash_batch: int | None = None) -> float:
    """Targeted success of one speller at ``n_reps`` repetitions.

    Runs the speller once, then restricts the letter head to the first ``n_reps``
    repetitions. This is the single place the surrogate and every victim score
    success, so the numbers in the results table are computed identically.
    """
    target = _as_long(target, eeg.device)
    speller.eval()
    with torch.no_grad():
        out = speller(eeg, onsets, groups, max_flash_batch=max_flash_batch)
        scores = out["flash_scores"]
        n = n_reps * FLASHES_PER_REP
        if n > scores.shape[1]:
            n = scores.shape[1]
        choice = symbol_logits(scores[:, :n], groups[:, :n]).argmax(dim=1)
    return targeted_success(choice, target)


def success_all_reps(speller, eeg: torch.Tensor, onsets: torch.Tensor,
                     groups: torch.Tensor, target, max_flash_batch: int | None = None
                     ) -> dict[int, float]:
    """Targeted success for every repetition count in ``REPS``."""
    return {
        reps: success_for_speller(speller, eeg, onsets, groups, target, reps,
                                  max_flash_batch=max_flash_batch)
        for reps in REPS
    }
