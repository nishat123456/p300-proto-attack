"""Shared toy speller and trial builders for evaluation tests (CPU only).

The evaluation tests need a speller whose forward pass is cheap, whose logits
depend on the raw EEG (so an attack can actually move the decision), and whose
preprocessing is a real `Preprocessor` (so the stored windows and ERPs exercise
the same code path as the experiment). The affine decoder below satisfies all
three: the symbol evidence is a linear function of the flashed raw samples, so a
large enough perturbation provably flips the argmax.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from p3a.data.layout import COMP3, rc_code_to_group
from p3a.data.trial import Trial
from p3a.models.speller import Speller
from p3a.preproc.pipeline import Preprocessor

FS = 240.0
N_FLASHES = 180
FLASHES_PER_REP = 12


class AffineFlashDecoder(nn.Module):
    """One logit per epoch, affine in the flattened preprocessed epoch."""

    def __init__(self, n_channels: int, L: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(n_channels * L, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1)).squeeze(1)


def make_trial(index: int, *, C: int = 4, T: int = 2000, split: str = "test",
               subject: str = "A", dataset: str = "synthetic",
               target_symbol: str | None = None, seed: int = 0) -> Trial:
    """A tiny random trial with the full 180-flash Comp III schedule."""
    rng = np.random.default_rng(seed * 1000 + index)
    codes = np.tile(np.arange(1, 13), N_FLASHES // FLASHES_PER_REP)
    onsets = 5 + 10 * np.arange(N_FLASHES)
    groups = np.stack([rc_code_to_group(COMP3, int(c)) for c in codes])
    symbol = target_symbol or COMP3.symbols[index % COMP3.n_symbols]
    return Trial(
        eeg=rng.normal(size=(C, T)).astype(np.float32), fs=FS, onsets=onsets,
        flash_groups=groups, flash_codes=codes, target_symbol=symbol, layout=COMP3,
        subject_id=subject, dataset=dataset, trial_index=index, source_split=split,
    )


def make_trials(n: int, **kwargs) -> list[Trial]:
    return [make_trial(i, **kwargs) for i in range(n)]


def make_speller(trials: list[Trial], L: int = 48, seed: int = 0) -> Speller:
    """A toy Speller: real Preprocessor, affine flash decoder, eval mode."""
    pre = Preprocessor(n_channels=trials[0].eeg.shape[0], fs=FS)
    pre.fit_scaling([t.eeg for t in trials])
    return Speller(pre, AffineFlashDecoder(pre.n_channels, L, seed=seed), L).eval()


def make_loadable_speller(trials: list[Trial], L: int = 48, seed: int = 0) -> Speller:
    """A Speller with a registry EEGNet decoder, so it survives save/load.

    The evaluation entry point loads its surrogate from a checkpoint, which
    rebuilds the decoder through the registry; the affine toy decoder above is
    deliberately not registered, so reloadable tests use this one instead.
    """
    from p3a.models.eegnet import EEGNet

    pre = Preprocessor(n_channels=trials[0].eeg.shape[0], fs=FS)
    pre.fit_scaling([t.eeg for t in trials])
    torch.manual_seed(seed)
    return Speller(pre, EEGNet(pre.n_channels, L), L).eval()


def batch(trials: list[Trial], device="cpu"):
    eeg = torch.as_tensor(np.stack([t.eeg for t in trials]), dtype=torch.float32, device=device)
    onsets = torch.as_tensor(np.stack([t.onsets for t in trials]), dtype=torch.long, device=device)
    groups = torch.as_tensor(np.stack([t.flash_groups for t in trials]), dtype=torch.bool, device=device)
    return eeg, onsets, groups
