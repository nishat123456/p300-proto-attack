"""Unified trial representation shared by every dataset loader."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Layout


@dataclass
class Trial:
    """One character-selection trial.

    eeg:          raw EEG [channels, time], float32, before any preprocessing
    onsets:       flash onset sample indices [F]
    flash_groups: bool [F, n_symbols], symbols lit by each flash
    flash_codes:  dataset-native stimulus code per flash [F] (row/col code for Comp III)
    """

    eeg: np.ndarray
    fs: float
    onsets: np.ndarray
    flash_groups: np.ndarray
    flash_codes: np.ndarray
    target_symbol: str
    layout: Layout
    subject_id: str
    dataset: str
    trial_index: int
    source_split: str  # official split in the source release: "train" or "test"
    channel_names: list[str] = field(default_factory=list)

    @property
    def trial_id(self) -> str:
        return f"{self.dataset}:{self.subject_id}:{self.source_split}:{self.trial_index:03d}"

    @property
    def n_flashes(self) -> int:
        return len(self.onsets)

    def target_flash_mask(self, symbol: str | None = None) -> np.ndarray:
        """Flashes that contain `symbol` (default: the trial's own target)."""
        k = self.layout.index(symbol or self.target_symbol)
        return self.flash_groups[:, k]
