"""Synthetic Prompt 10 artifacts for detector tests (CPU only).

These helpers write a miniature ``results/attacks_eval/<S>/`` tree with the same
NPZ schema and parquet columns as ``scripts/evaluate_attacks.py``, so the
detector modules can be exercised end to end without any real EEG or attack.

The synthetic induced ERPs carry a detectable signature (a scaled, shifted copy
of the genuine ERP plus a high-frequency ripple), so a learned detector can
separate genuinely from induced and the tests can assert the metric helpers
return sensible values instead of noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p3a.eval.store import (
    ArtifactSpec,
    AttackArtifacts,
    CleanArtifacts,
    attack_npz_path,
    clean_npz_path,
)


def genuine_erp(n_samples: int, n_channels: int, length: int, fs: float = 60.0,
                seed: int = 0, trial_index: int = 0) -> np.ndarray:
    """A deterministic ERP-like waveform [n_samples, C, L].

    A P300 bump in 250-500 ms plus a slow trend, scaled by channel position, so
    correlations, peak latency and topography all have something to measure.
    """
    rng = np.random.default_rng(seed * 1000 + trial_index)
    t = np.arange(length) / fs
    bump = np.exp(-((t - 0.35) ** 2) / (2 * 0.06 ** 2))
    trend = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    channel_gain = np.linspace(0.5, 1.5, n_channels)
    base = np.outer(channel_gain, bump + trend)
    noise = 0.01 * rng.standard_normal((n_samples, n_channels, length))
    return (base[None] + noise).astype(np.float32)


def induced_erp(clean: np.ndarray, seed: int = 0, scale: float = 1.6,
                ripple: float = 0.5) -> np.ndarray:
    """A perturbed ERP: over-scaled, delayed and with added high-frequency ripple.

    The clean genuine ERP is stretched in the time axis to shift the peak, then a
    20-30 Hz ripple is added. Both changes are exactly what the plausibility and
    band-power metrics are meant to catch.
    """
    x = np.asarray(clean, dtype=np.float32)
    n, c, length = x.shape
    idx = np.clip(np.round(np.arange(length) * 0.92).astype(int), 0, length - 1)
    stretched = x[:, :, idx] * scale
    rng = np.random.default_rng(seed + 7)
    t = np.arange(length)
    high = ripple * np.sin(2 * np.pi * 25.0 / 60.0 * t)[None, None, :] * (
        1.0 + 0.1 * rng.standard_normal((n, 1, 1))
    )
    return (stretched + high).astype(np.float32)


def write_subject(results_root, subject: str = "A", n_trials: int = 6,
                  n_targets: int = 4, n_channels: int = 4, length: int = 48,
                  window_width: int = 20, n_windows: int = 2,
                  methods=None, eps=(0.20,), seed: int = 0,
                  trial_index=None) -> Path:
    """Write a compact clean NPZ, attack NPZs and one attacks.parquet.

    ``methods`` is a list of (method, generator, bank, success_rate). The
    induced/success flags drive which attacked samples the defense treats as
    induced; a success rate of 1 makes every pair induced. Trial indices default
    to ``0..n_trials-1`` so both parity folds are covered.
    """
    methods = methods or [
        ("pgd", "none", "none", 1.0),
        ("target_only", "signal", "cross_subject", 1.0),
        ("noise", "none", "none", 1.0),
        ("tiny_noise", "none", "none", 1.0),
    ]
    directory = Path(results_root) / "attacks_eval" / subject
    directory.mkdir(parents=True, exist_ok=True)
    if trial_index is None:
        trial_index = np.arange(n_trials, dtype=np.int64)
    trial_index = np.asarray(trial_index, dtype=np.int64)
    if trial_index.size != n_trials:
        raise ValueError("trial_index must have one entry per trial")
    trial_id = np.array([f"comp3:{subject}:test:{i:03d}" for i in trial_index], dtype=object)
    true_idx = (np.arange(n_trials) % 36).astype(np.int64)

    rng = np.random.default_rng(seed)
    true_erp = np.stack([
        genuine_erp(1, n_channels, length, seed=seed, trial_index=int(ti))[0]
        for ti in trial_index
    ]).astype(np.float16)
    target_idx = np.stack([
        (true_idx[i] + 1 + np.arange(n_targets)) % 36 for i in range(n_trials)
    ]).astype(np.int64)
    target_erp = np.stack([
        np.stack([
            genuine_erp(1, n_channels, length, seed=seed + 100,
                        trial_index=int(ti))[0]
            for _ in range(n_targets)
        ]) for ti in trial_index
    ]).astype(np.float16)
    starts = rng.integers(0, max(window_width, 1), size=(n_trials, n_windows)).astype(np.int64)
    clean_windows = rng.normal(size=(n_trials, n_windows, n_channels, window_width)).astype(np.float16)
    CleanArtifacts(
        subject=subject, trial_index=trial_index, trial_id=trial_id,
        true_idx=true_idx, true_erp=true_erp, target_idx=target_idx,
        target_erp=target_erp, window_starts=starts, windows=clean_windows,
    ).save(clean_npz_path(directory, subject))

    rows = []
    for method, generator, bank, success_rate in methods:
        for mult in eps:
            decision = np.empty((n_trials, n_targets, n_channels, length), np.float16)
            windows = np.empty((n_trials, n_targets, n_windows, n_channels, window_width), np.float16)
            for ti in range(n_trials):
                attack_erp = induced_erp(
                    target_erp[ti].astype(np.float32), seed=seed + ti
                )
                decision[ti] = attack_erp.astype(np.float16)
                attack_win = induced_erp(
                    clean_windows[ti].astype(np.float32).reshape(
                        1, n_channels, n_windows * window_width
                    ), seed=seed + 50 + ti
                )
                attack_win = attack_win.reshape(n_windows, n_channels, window_width)
                windows[ti] = np.repeat(attack_win[None], n_targets, axis=0).astype(np.float16)
                for k in range(n_targets):
                    ok = bool(rng.random() < success_rate)
                    rows.append({
                        "subject": subject, "method": method, "generator": generator,
                        "bank": bank, "eps_multiplier": float(mult), "eps_raw": 1.0,
                        "trial_index": int(trial_index[ti]), "trial_id": str(trial_id[ti]),
                        "true_symbol": "A", "true_idx": int(true_idx[ti]),
                        "target_symbol": "B", "target_idx": int(target_idx[ti, k]),
                        "retrieval_distance": float("nan"), "attack_ok": True,
                        "success_surrogate_15": float(ok), "success_surrogate_5": float(ok),
                        "success_deepconvnet_15": float(ok), "success_deepconvnet_5": float(ok),
                        "linf": 0.2, "l2": 1.0, "rms": 0.1, "snr_db": 10.0,
                    })
            spec = ArtifactSpec(subject, method, generator, bank, float(mult))
            AttackArtifacts(
                subject=subject, trial_index=trial_index, trial_id=trial_id,
                true_idx=true_idx,
                target_idx=target_idx,
                target_symbol=np.array([["B"] * n_targets] * n_trials, dtype=object),
                decision_erp=decision, window_starts=starts, windows=windows,
            ).save(attack_npz_path(directory, spec))
    pd.DataFrame(rows).to_parquet(directory / "attacks.parquet", index=False)
    return directory
