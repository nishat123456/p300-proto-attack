"""Target-conditioned ERP features and the prototype feature scaler.

A target-conditioned ERP is the mean of a trial's flashes that contain one
chosen symbol. It is the query feature for retrieval (the attacker target, not
the trial's true symbol) and the reference feature stored in a prototype bank
(the trial's true symbol). The torch path is differentiable so it can appear in
the attack loss; the numpy path matches it exactly for offline use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _validate(epochs, groups, symbol_idx):
    """Shared dimension checks for the torch and numpy paths."""
    if epochs.ndim != 3:
        raise ValueError("expected epochs [F, C, L]")
    if groups.ndim != 2:
        raise ValueError("expected groups [F, K]")
    if groups.shape[0] != epochs.shape[0]:
        raise ValueError("epochs and groups must share the flash dimension F")
    if not (0 <= symbol_idx < groups.shape[1]):
        raise ValueError(f"symbol_idx {symbol_idx} outside [0, {groups.shape[1]})")


def target_conditioned_erp(epochs, groups, symbol_idx):
    """Mean ERP over flashes that contain `symbol_idx`, shape [C, L].

    Torch in, torch out (differentiable in `epochs`); numpy in, numpy out.
    Raises if the symbol is never flashed, so callers cannot silently average
    over an empty selection.
    """
    if isinstance(epochs, torch.Tensor):
        if not isinstance(groups, torch.Tensor):
            raise TypeError("epochs and groups must be the same array type")
        if not epochs.is_floating_point() or groups.dtype != torch.bool:
            raise ValueError("epochs must be floating point and groups boolean")
        if epochs.device != groups.device:
            raise ValueError("epochs and groups must be on the same device")
        _validate(epochs, groups, symbol_idx)
        mask = groups[:, symbol_idx]
        if not bool(mask.any()):
            raise ValueError(f"symbol_idx {symbol_idx} is never flashed")
        selected = epochs[mask]  # [n_sel, C, L]
        return selected.mean(dim=0)

    epochs = np.asarray(epochs)
    groups = np.asarray(groups)
    if not np.issubdtype(epochs.dtype, np.floating):
        raise ValueError("epochs must be floating point")
    if groups.dtype != np.bool_:
        raise ValueError("groups must be boolean")
    _validate(epochs, groups, symbol_idx)
    mask = groups[:, symbol_idx]
    if not mask.any():
        raise ValueError(f"symbol_idx {symbol_idx} is never flashed")
    return epochs[mask].mean(axis=0)


class FeatureScaler:
    """Per-(channel, time) standardization fitted on bank members only.

    The scaler is deliberately independent of any attack or victim: it is fit
    on genuine prototype features from the bank, saved and loaded byte-exact,
    and applied to both prototype and query features so retrieval distances are
    comparable across subjects and sessions.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mean.shape

    @classmethod
    def fit(cls, features) -> "FeatureScaler":
        """Fit per-(c, t) mean and std over a list of [C, L] features.

        Channels/times with (near) zero variance get std 1 so transform stays
        finite and invertible instead of dividing by zero.
        """
        if features is None:
            raise ValueError("fit requires at least one feature")
        arrays = [np.asarray(f, dtype=np.float64) for f in features]
        if not arrays:
            raise ValueError("fit requires at least one feature")
        shape = arrays[0].shape
        if len(shape) != 2:
            raise ValueError("features must be [C, L]")
        for arr in arrays:
            if arr.shape != shape:
                raise ValueError("all features must share the same [C, L] shape")
        stack = np.stack(arrays, axis=0)  # [N, C, L]
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, feature) -> np.ndarray:
        arr = np.asarray(feature, dtype=np.float32)
        if arr.shape != self.mean.shape:
            raise ValueError(
                f"feature shape {arr.shape} does not match scaler {self.mean.shape}"
            )
        return (arr - self.mean) / self.std

    def inverse_transform(self, feature) -> np.ndarray:
        arr = np.asarray(feature, dtype=np.float32)
        if arr.shape != self.mean.shape:
            raise ValueError(
                f"feature shape {arr.shape} does not match scaler {self.mean.shape}"
            )
        return arr * self.std + self.mean

    def save(self, path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        return path

    @classmethod
    def load(cls, path) -> "FeatureScaler":
        path = Path(path)
        if not path.exists() and path.suffix != ".npz":
            path = path.with_suffix(".npz")
        with np.load(path) as data:
            return cls(data["mean"], data["std"])

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureScaler":
        return cls(np.asarray(payload["mean"], np.float32),
                   np.asarray(payload["std"], np.float32))
