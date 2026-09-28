"""Simple anomaly detectors: band power and ERP Mahalanobis (Prompt 11).

Two cheap, training-light defenders that need no attack labels and serve as
sanity baselines against the learned detectors.

- ``band_power_anomaly``: per window, the power in the 20-30 Hz band and in the
  0-2 Hz band is turned into a z-score against the clean windows of the same
  subject. A perturbation that adds high-frequency ripple or a slow drift shows
  up here.
- ``erp_mahalanobis``: the clean decision ERPs define a Gaussian; a decision ERP
  is scored by its Mahalanobis distance after a PCA to 20 dimensions, with the
  covariance shrunk by the Ledoit-Wolf estimator so the inverse is stable.

Both return per-sample anomaly scores (higher is more anomalous) so the shared
``auc`` and ``tpr_at_fpr`` helpers can summarize them.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from p3a.defenses.metrics import auc, folds_by_parity, tpr_at_fpr

__all__ = [
    "band_power_anomaly",
    "erp_mahalanobis",
    "evaluate_simple_detectors",
]

# Analysis bands in Hz: the ripple band the perturbation tends to add and the
# slow-drift band it can bias.
RIPPLE_BAND = (20.0, 30.0)
DRIFT_BAND = (0.0, 2.0)
PCA_DIMS = 20


def _band_power(windows: np.ndarray, fs_out: float, band: tuple[float, float]) -> np.ndarray:
    """Mean band power per sample, shape [N].

    ``windows`` is [N, n_windows, C, W]. Power is averaged over windows and
    channels after a Welch PSD per window, so a single quiet channel cannot hide
    the effect.
    """
    arr = np.asarray(windows, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("windows must be [N, n_windows, C, W]")
    nperseg = min(arr.shape[-1], max(int(fs_out), 8))
    freqs, psd = signal.welch(arr, fs=float(fs_out), nperseg=nperseg, axis=-1)
    if band[0] <= 0:
        band_mask = (freqs < band[1])
    else:
        band_mask = (freqs >= band[0]) & (freqs < band[1])
    if not band_mask.any():
        raise ValueError(f"no PSD bins fall inside band {band}")
    power = psd[..., band_mask].mean(axis=-1)  # [N, n_windows, C]
    return power.mean(axis=(1, 2))


def band_power_anomaly(clean_windows: np.ndarray, eval_windows: np.ndarray,
                       fs_out: float) -> np.ndarray:
    """Combined ripple and drift z-score above the clean distribution, [N].

    The clean windows define the mean and standard deviation of each band's
    power; the eval windows are converted to z-scores and the two bands are
    summed. A higher value means the window is more anomalous in either band.
    """
    clean_windows = np.asarray(clean_windows)
    eval_windows = np.asarray(eval_windows)
    squared = np.zeros(eval_windows.shape[0], dtype=np.float64)
    for band in (RIPPLE_BAND, DRIFT_BAND):
        clean_power = _band_power(clean_windows, fs_out, band)
        mean = float(np.mean(clean_power))
        std = float(np.std(clean_power))
        std = std if std > 1e-12 else 1.0
        eval_power = _band_power(eval_windows, fs_out, band)
        z = (eval_power - mean) / std
        squared += z ** 2
    return np.sqrt(squared)


def erp_mahalanobis(clean_erp: np.ndarray, eval_erp: np.ndarray,
                    n_components: int = PCA_DIMS) -> np.ndarray:
    """Mahalanobis distance of eval ERPs from the clean ERP Gaussian, [N].

    A PCA to ``n_components`` (or fewer when the sample count is small) is fit
    on the clean ERPs, a Ledoit-Wolf covariance is estimated in that subspace,
    and each eval ERP is scored by its distance to the clean mean. Higher means
    further from genuine.
    """
    clean = np.asarray(clean_erp, dtype=np.float64)
    ev = np.asarray(eval_erp, dtype=np.float64)
    if clean.ndim != 3 or ev.ndim != 3:
        raise ValueError("clean_erp and eval_erp must be [N, C, L]")
    if clean.shape[1:] != ev.shape[1:]:
        raise ValueError("clean and eval ERPs must share C and L")
    flat_clean = clean.reshape(clean.shape[0], -1)
    flat_eval = ev.reshape(ev.shape[0], -1)
    k = int(min(n_components, flat_clean.shape[0], flat_clean.shape[1]))
    if k < 1:
        raise ValueError("not enough clean samples to fit a PCA")
    pca = PCA(n_components=k, random_state=0).fit(flat_clean)
    z_clean = pca.transform(flat_clean)
    z_eval = pca.transform(flat_eval)
    cov = LedoitWolf().fit(z_clean)
    return cov.mahalanobis(z_eval)


def evaluate_simple_detectors(clean_erp: np.ndarray, attacked_erp: np.ndarray,
                              clean_windows: np.ndarray, attacked_windows: np.ndarray,
                              fs_out: float, trial_index=None,
                              fpr: float = 0.05) -> dict:
    """Two-fold held-out AUC and TPR for both simple detectors."""
    n = len(clean_erp)
    if not (len(attacked_erp) == len(clean_windows) == len(attacked_windows) == n):
        raise ValueError("clean and attacked arrays must align")
    trial_index = np.arange(n) if trial_index is None else np.asarray(trial_index)
    folds = folds_by_parity(trial_index)
    rows = []
    for fold, test in folds.items():
        train = ~test
        if not train.any() or not test.any():
            continue
        labels = np.concatenate([np.ones(int(test.sum())), np.zeros(int(test.sum()))])
        mahal_scores = np.concatenate([
            erp_mahalanobis(clean_erp[train], attacked_erp[test]),
            erp_mahalanobis(clean_erp[train], clean_erp[test]),
        ])
        band_scores = np.concatenate([
            band_power_anomaly(clean_windows[train], attacked_windows[test], fs_out),
            band_power_anomaly(clean_windows[train], clean_windows[test], fs_out),
        ])
        rows.append({
            "fold": int(fold),
            "mahalanobis_auc": auc(mahal_scores, labels),
            "mahalanobis_tpr_at_fpr": tpr_at_fpr(mahal_scores, labels, fpr)[0],
            "band_power_auc": auc(band_scores, labels),
            "band_power_tpr_at_fpr": tpr_at_fpr(band_scores, labels, fpr)[0],
        })
    if not rows:
        return {"mahalanobis_auc": float("nan"),
                "mahalanobis_tpr_at_fpr": float("nan"),
                "band_power_auc": float("nan"),
                "band_power_tpr_at_fpr": float("nan"), "folds": []}
    return {
        "mahalanobis_auc": float(np.nanmean([r["mahalanobis_auc"] for r in rows])),
        "mahalanobis_tpr_at_fpr": float(np.nanmean(
            [r["mahalanobis_tpr_at_fpr"] for r in rows])),
        "band_power_auc": float(np.nanmean([r["band_power_auc"] for r in rows])),
        "band_power_tpr_at_fpr": float(np.nanmean(
            [r["band_power_tpr_at_fpr"] for r in rows])),
        "folds": rows,
    }
