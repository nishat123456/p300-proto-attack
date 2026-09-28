"""Float16 NPZ artifact storage for the attack evaluation (Prompt 10).

The detectors in Prompt 11 read their inputs from these files, so the schema is
explicit and self-describing. Two artifact kinds exist:

- ``CleanArtifacts``: one file per attacked subject. Per trial it holds the
  genuine decision ERP at the trial's true symbol, the clean decision ERP at
  each pre-registered attacker target, and four one-second windows of the
  preprocessed clean signal at positions drawn once per trial.
- ``AttackArtifacts``: one file per method/generator/bank/epsilon combination.
  Per trial and attacker target it holds the decision ERP of the attacked input
  ``x + delta``, plus four one-second windows of the preprocessed attacked
  signal at the same per-trial positions used for the clean file. Because the
  perturbation depends on the attacker target, attacked windows are stored per
  (trial, target); the clean windows are stored once per trial and shared by
  every method and target. Window positions line up index for index across
  methods, so detectors compare like with like.

All floating point arrays are stored as float16 to keep the kill test within the
Kaggle disk budget; metadata (indices, symbols, window starts) stays exact.
Clean windows are stored once per trial. Attacked windows are stored per target
because each target produces a different perturbation. The decision ERP is the
target-conditioned ERP of the *attacker* target, computed with the surrogate's
own differentiable preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from p3a.prototypes.features import target_conditioned_erp

__all__ = [
    "N_WINDOWS",
    "STORE_FORMAT",
    "WINDOW_SECONDS",
    "ArtifactSpec",
    "AttackArtifacts",
    "CleanArtifacts",
    "attack_npz_path",
    "clean_npz_path",
    "decision_erp",
    "preprocessed_windows",
    "sample_window_starts",
    "windows_from_preprocessed",
]

STORE_FORMAT = 1
N_WINDOWS = 4
WINDOW_SECONDS = 1.0
_F16 = np.float16


@dataclass(frozen=True)
class ArtifactSpec:
    """Identity of one attack artifact on disk."""

    subject: str
    method: str
    generator: str
    bank: str
    eps_multiplier: float

    @property
    def label(self) -> str:
        """Short stable label combining the method, generator and bank."""
        bank = self.bank or "none"
        return f"{self.method}_{self.generator}_{bank}"

    def filename(self) -> str:
        eps = f"{self.eps_multiplier:.2f}".replace(".", "p")
        return f"{self.label}_eps{eps}.npz"


def attack_npz_path(out_dir, spec: ArtifactSpec) -> Path:
    return Path(out_dir) / spec.filename()


def clean_npz_path(out_dir, subject: str) -> Path:
    return Path(out_dir) / f"{str(subject).upper()}_clean.npz"


def sample_window_starts(fs_out: float, n_samples: int, n_windows: int = N_WINDOWS,
                         seconds: float = WINDOW_SECONDS, seed: int = 0) -> np.ndarray:
    """Deterministic one-second window start samples for one preprocessed trial.

    Positions are drawn once per (seed, trial length) and reused for the clean
    and every attacked version, so windows are comparable across methods. If the
    trial is too short for the window, every start collapses to 0.
    """
    width = int(round(seconds * fs_out))
    if width < 1:
        raise ValueError("window width must be at least one sample")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    max_start = max(n_samples - width, 0)
    if max_start == 0:
        return np.zeros(n_windows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, max_start + 1, size=n_windows).astype(np.int64)


@torch.no_grad()
def decision_erp(speller, eeg: torch.Tensor, onsets: torch.Tensor,
                 groups: torch.Tensor, target_idx: int) -> np.ndarray:
    """Target-conditioned decision ERP [C, L] float16 for one trial and target.

    ``eeg`` is a single raw trial [1, C, T]. The ERP is the mean preprocessed
    epoch over the flashes that light ``target_idx``.
    """
    if eeg.ndim != 3 or eeg.shape[0] != 1:
        raise ValueError("eeg must be a single-trial [1, C, T] tensor")
    epochs = speller.epochs(eeg, onsets)[0]  # [F, C, L]
    feature = target_conditioned_erp(epochs, groups[0], int(target_idx))
    return feature.detach().to(torch.float32).cpu().numpy().astype(_F16)


@torch.no_grad()
def preprocessed_windows(speller, eeg: torch.Tensor, starts: np.ndarray,
                         seconds: float = WINDOW_SECONDS) -> np.ndarray:
    """One-second preprocessed windows [n_windows, C, W] float16 for one trial.

    ``starts`` are preprocessed-sample indices; a window that would run past the
    end is right-aligned to the final sample instead of padded.
    """
    if eeg.ndim != 3 or eeg.shape[0] != 1:
        raise ValueError("eeg must be a single-trial [1, C, T] tensor")
    x = speller.preprocessor(eeg)
    return windows_from_preprocessed(
        x, np.asarray(starts)[None], speller.preprocessor.fs_out, seconds=seconds
    )[0]


def windows_from_preprocessed(x_pre: torch.Tensor, starts: np.ndarray, fs_out: float,
                              seconds: float = WINDOW_SECONDS) -> np.ndarray:
    """One-second windows from preprocessed signals, [B, n_windows, C, W].

    ``x_pre`` is [B, C, T'] and ``starts`` is [B, n_windows] preprocessed-sample
    indices. A window that would run past the end is right-aligned to the final
    sample instead of padded, so every stored window has the same width.
    """
    if x_pre.ndim != 3:
        raise ValueError("x_pre must be [B, C, T']")
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 2 or starts.shape[0] != x_pre.shape[0]:
        raise ValueError("starts must be [B, n_windows] aligned with x_pre")
    width = int(round(seconds * float(fs_out)))
    n_samples = x_pre.shape[-1]
    if width < 1 or n_samples < 1:
        raise ValueError("signal too short for the requested window")
    batch, n_windows = starts.shape
    out = np.empty((batch, n_windows, x_pre.shape[1], width), dtype=_F16)
    for b in range(batch):
        row = x_pre[b]
        for w in range(n_windows):
            begin = int(min(max(starts[b, w], 0), max(n_samples - width, 0)))
            out[b, w] = row[:, begin:begin + width].to(torch.float32).cpu().numpy().astype(_F16)
    return out


@dataclass
class CleanArtifacts:
    """Genuine and clean duplicate-resistant references for one subject."""

    subject: str
    trial_index: np.ndarray        # [T]
    trial_id: np.ndarray           # [T] object
    true_idx: np.ndarray           # [T]
    true_erp: np.ndarray           # [T, C, L] f16 (genuine, at true symbol)
    target_idx: np.ndarray         # [T, K] attacker targets
    target_erp: np.ndarray         # [T, K, C, L] f16 (clean, at attacker target)
    window_starts: np.ndarray      # [T, n_windows]
    windows: np.ndarray            # [T, n_windows, C, W] f16
    meta: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, store_format=np.int64(STORE_FORMAT), subject=np.asarray(self.subject),
            trial_index=self.trial_index.astype(np.int64),
            trial_id=self.trial_id.astype(object),
            true_idx=self.true_idx.astype(np.int64),
            true_erp=self.true_erp.astype(_F16),
            target_idx=self.target_idx.astype(np.int64),
            target_erp=self.target_erp.astype(_F16),
            window_starts=self.window_starts.astype(np.int64),
            windows=self.windows.astype(_F16),
        )
        return path

    @classmethod
    def load(cls, path) -> "CleanArtifacts":
        with np.load(Path(path), allow_pickle=True) as data:
            if int(data["store_format"]) != STORE_FORMAT:
                raise ValueError("unsupported clean artifact format")
            return cls(
                subject=str(data["subject"]),
                trial_index=np.asarray(data["trial_index"], dtype=np.int64),
                trial_id=np.asarray(data["trial_id"], dtype=object),
                true_idx=np.asarray(data["true_idx"], dtype=np.int64),
                true_erp=np.asarray(data["true_erp"], dtype=_F16),
                target_idx=np.asarray(data["target_idx"], dtype=np.int64),
                target_erp=np.asarray(data["target_erp"], dtype=_F16),
                window_starts=np.asarray(data["window_starts"], dtype=np.int64),
                windows=np.asarray(data["windows"], dtype=_F16),
            )


@dataclass
class AttackArtifacts:
    """Attacked decision ERPs and windows for one method/generator/bank/eps."""

    subject: str
    trial_index: np.ndarray        # [T]
    trial_id: np.ndarray           # [T] object
    true_idx: np.ndarray           # [T]
    target_idx: np.ndarray         # [T, K]
    target_symbol: np.ndarray      # [T, K] object
    decision_erp: np.ndarray       # [T, K, C, L] f16
    window_starts: np.ndarray      # [T, n_windows]
    windows: np.ndarray            # [T, K, n_windows, C, W] f16
    meta: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, store_format=np.int64(STORE_FORMAT), subject=np.asarray(self.subject),
            trial_index=self.trial_index.astype(np.int64),
            trial_id=self.trial_id.astype(object),
            true_idx=self.true_idx.astype(np.int64),
            target_idx=self.target_idx.astype(np.int64),
            target_symbol=self.target_symbol.astype(object),
            decision_erp=self.decision_erp.astype(_F16),
            window_starts=self.window_starts.astype(np.int64),
            windows=self.windows.astype(_F16),
        )
        return path

    @classmethod
    def load(cls, path) -> "AttackArtifacts":
        with np.load(Path(path), allow_pickle=True) as data:
            if int(data["store_format"]) != STORE_FORMAT:
                raise ValueError("unsupported attack artifact format")
            return cls(
                subject=str(data["subject"]),
                trial_index=np.asarray(data["trial_index"], dtype=np.int64),
                trial_id=np.asarray(data["trial_id"], dtype=object),
                true_idx=np.asarray(data["true_idx"], dtype=np.int64),
                target_idx=np.asarray(data["target_idx"], dtype=np.int64),
                target_symbol=np.asarray(data["target_symbol"], dtype=object),
                decision_erp=np.asarray(data["decision_erp"], dtype=_F16),
                window_starts=np.asarray(data["window_starts"], dtype=np.int64),
                windows=np.asarray(data["windows"], dtype=_F16),
            )
