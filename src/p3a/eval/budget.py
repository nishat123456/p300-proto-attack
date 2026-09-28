"""Perturbation budget metrics and matched-success interpolation (Prompt 10).

The attack evaluation reports, for every perturbation, how large it is in raw
EEG units and how much it stands out from the clean signal. Four quantities are
used throughout the experiment (see PLAN.md Section 6):

- L-infinity norm: the budget the pre-registration fixes and every attack
  respects per sample.
- L2 norm and RMS: the same perturbation measured in aggregate.
- SNR in dB: how much smaller the perturbation energy is than the clean signal
  energy, ``10 * log10(sum(clean^2) / sum(delta^2))``.

Two fallback helpers live here as well, because they operate on the budget
axis: interpolating a cost at a target success level without ever extrapolating
past the measured grid, and the normalized area under the success-vs-budget
curve. Both tie directly to the always-reported fallbacks in PLAN.md.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "budget_metrics",
    "interpolate_at_success",
    "l2_norm",
    "linf_norm",
    "rms_norm",
    "sample_scalars",
    "snr_db",
    "success_curve_area",
]

# Floor for the perturbation energy so a zero-budget sample yields a large but
# finite SNR instead of an infinity.
_EPS = 1e-12


def _check_pair(clean: torch.Tensor, delta: torch.Tensor) -> None:
    if not torch.is_tensor(clean) or not torch.is_tensor(delta):
        raise TypeError("clean and delta must be torch tensors")
    if clean.ndim != 3 or delta.ndim != 3:
        raise ValueError("clean and delta must be [B, C, T]")
    if clean.shape != delta.shape:
        raise ValueError("clean and delta must share a shape")


def linf_norm(delta: torch.Tensor) -> torch.Tensor:
    """Per-sample L-infinity norm over all channels and time, [B]."""
    if not torch.is_tensor(delta) or delta.ndim != 3:
        raise ValueError("delta must be [B, C, T]")
    return delta.flatten(1).abs().amax(dim=1)


def l2_norm(delta: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm over all channels and time, [B]."""
    if not torch.is_tensor(delta) or delta.ndim != 3:
        raise ValueError("delta must be [B, C, T]")
    return torch.linalg.vector_norm(delta.flatten(1), dim=1)


def rms_norm(delta: torch.Tensor) -> torch.Tensor:
    """Per-sample root-mean-square over all channels and time, [B]."""
    if not torch.is_tensor(delta) or delta.ndim != 3:
        raise ValueError("delta must be [B, C, T]")
    return delta.flatten(1).pow(2).mean(dim=1).sqrt()


def snr_db(clean: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Signal-to-perturbation ratio in dB, per sample, [B].

    ``10 * log10(sum(clean^2) / sum(delta^2))`` with the denominator floored so a
    zero perturbation (zero budget) returns a large finite value.
    """
    _check_pair(clean, delta)
    signal = clean.flatten(1).pow(2).sum(dim=1)
    noise = delta.flatten(1).pow(2).sum(dim=1).clamp_min(_EPS)
    return 10.0 * torch.log10(signal.clamp_min(_EPS) / noise)


def budget_metrics(clean: torch.Tensor, delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """All four per-sample budget metrics keyed by name."""
    _check_pair(clean, delta)
    return {
        "linf": linf_norm(delta),
        "l2": l2_norm(delta),
        "rms": rms_norm(delta),
        "snr_db": snr_db(clean, delta),
    }


def sample_scalars(values) -> dict[str, float]:
    """Mean over samples of each per-sample metric tensor or array.

    Accepts a dict of 1D tensors or arrays (the output of ``budget_metrics`` on a
    batch) and returns plain floats, which is what a results row stores.
    """
    summary = {}
    for key, value in values.items():
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        arr = arr.reshape(-1)
        summary[key] = float(np.mean(arr)) if arr.size else float("nan")
    return summary


def interpolate_at_success(eps, success, level: float = 0.8):
    """Budget interpolated at a target success level, or None if not reached.

    ``eps`` and ``success`` are aligned 1D sequences, ``eps`` ascending. If the
    method never reaches ``level`` the function returns None so the caller
    reports the method's maximum success with a footnote instead of
    extrapolating, exactly as the pre-registered fallback requires.
    """
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    success = np.asarray(success, dtype=np.float64).reshape(-1)
    if eps.size != success.size:
        raise ValueError("eps and success must have the same length")
    if eps.size == 0:
        raise ValueError("eps and success must not be empty")
    if not np.all(np.diff(eps) > 0):
        raise ValueError("eps must be strictly increasing")
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    if success.max() < level:
        return None
    if success[0] >= level:
        return float(eps[0])
    order = np.argsort(eps)
    eps, success = eps[order], success[order]
    for i in range(1, len(eps)):
        lo, hi = success[i - 1], success[i]
        if lo < level <= hi and hi > lo:
            frac = (level - lo) / (hi - lo)
            return float(eps[i - 1] + frac * (eps[i] - eps[i - 1]))
    return float(eps[-1])


def success_curve_area(eps, success) -> float:
    """Normalized area under the success-vs-budget curve.

    The curve is integrated over the normalized epsilon range with the trapezoid
    rule, so the result lies in [0, 1] regardless of the budget units and can be
    compared across subjects and methods.
    """
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    success = np.asarray(success, dtype=np.float64).reshape(-1)
    if eps.size != success.size:
        raise ValueError("eps and success must have the same length")
    if eps.size < 2:
        raise ValueError("at least two points are needed for a curve area")
    span = eps[-1] - eps[0]
    if span <= 0:
        raise ValueError("eps must span a positive range")
    return float(np.trapezoid(success, eps) / span)
