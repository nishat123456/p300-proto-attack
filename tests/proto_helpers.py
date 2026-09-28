"""Shared synthetic fixtures for prototype tests (no real EEG, CPU only)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from p3a.data.layout import COMP3, rc_code_to_group
from p3a.data.trial import Trial

FS = 240.0


class FakeSpeller(nn.Module):
    """Minimal stand-in exposing the one method the bank uses: `epochs`.

    Returns deterministic per-trial epochs so feature extraction is exercised
    without a real decoder or preprocessing.
    """

    def __init__(self, n_channels: int, L: int):
        super().__init__()
        self.n_channels = n_channels
        self.L = L

    def epochs(self, eeg: torch.Tensor, onsets: torch.Tensor) -> torch.Tensor:
        b, c, _ = eeg.shape
        f = onsets.shape[1]
        window = eeg[:, :, : self.L]  # [B, C, L]
        base = window * 0.001 + 1.0  # per-trial constant, nonzero
        return base[:, None, :, :].expand(b, f, c, self.L).clone()


def synthetic_trial(subject: str, index: int, *, split: str = "train",
                    target_symbol: str | None = None, seed: int = 0,
                    dataset: str = "synthetic", C: int = 4, T: int = 200) -> Trial:
    rng = np.random.default_rng(seed * 100 + index)
    eeg = rng.normal(size=(C, T)).astype(np.float32)
    codes = list(range(1, 13)) * 2  # two full row/column sweeps
    onsets = np.arange(len(codes)) * 10 + 5
    groups = np.stack([rc_code_to_group(COMP3, c) for c in codes])
    symbol = target_symbol if target_symbol is not None else COMP3.symbols[index % 36]
    return Trial(
        eeg=eeg, fs=FS, onsets=onsets, flash_groups=groups,
        flash_codes=np.asarray(codes), target_symbol=symbol, layout=COMP3,
        subject_id=subject, dataset=dataset, trial_index=index,
        source_split=split,
    )


def trials_for(subject: str, n: int, *, split: str = "train", start: int = 0,
               dataset: str = "synthetic") -> list[Trial]:
    return [synthetic_trial(subject, start + i, split=split, seed=start + i,
                            dataset=dataset) for i in range(n)]
