"""Fast CPU tests for the simple anomaly detectors (Prompt 11)."""
import numpy as np
import pytest

from p3a.defenses.simple import (
    band_power_anomaly,
    erp_mahalanobis,
    evaluate_simple_detectors,
)


def _erp_bank(n=30, c=4, length=48, seed=0, scale=1.0, ripple=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 60.0
    bump = np.exp(-((t - 0.35) ** 2) / (2 * 0.06 ** 2))
    base = scale * np.outer(np.linspace(0.5, 1.5, c), bump)
    noise = 0.01 * rng.standard_normal((n, c, length))
    high = ripple * np.sin(2 * np.pi * 25.0 * t)[None, None, :]
    return (base[None] + noise + high).astype(np.float32)


def test_mahalanobis_is_zero_at_clean_mean_and_positive_off_manifold():
    clean = _erp_bank(seed=0)
    d_clean = erp_mahalanobis(clean, clean.mean(axis=0, keepdims=True))
    assert d_clean[0] == pytest.approx(0.0, abs=1e-6)
    shifted = clean + 5.0
    d_shift = erp_mahalanobis(clean, shifted)
    assert d_shift.mean() > d_clean.mean() + 1.0


def test_mahalanobis_rejects_shape_mismatch():
    clean = _erp_bank(seed=0)
    with pytest.raises(ValueError, match="share C and L"):
        erp_mahalanobis(clean, clean[:, :, :-1])


def test_band_power_anomaly_rises_with_ripple():
    rng = np.random.default_rng(0)
    clean_win = rng.normal(size=(20, 2, 4, 40)).astype(np.float32)
    t = np.arange(40)
    ripple = 2.0 * np.sin(2 * np.pi * 25.0 / 60.0 * t)
    attacked_win = clean_win + ripple[None, None, None, :]
    scores = band_power_anomaly(clean_win, attacked_win, fs_out=60.0)
    baseline = band_power_anomaly(clean_win, clean_win, fs_out=60.0)
    assert scores.mean() > baseline.mean() + 1.0


def test_evaluate_simple_detectors_keys_and_ranges():
    clean_erp = _erp_bank(seed=1)
    attacked_erp = _erp_bank(seed=1, scale=1.8, ripple=0.4)
    rng = np.random.default_rng(1)
    clean_win = rng.normal(size=(30, 2, 4, 40)).astype(np.float32)
    attacked_win = clean_win + 1.0
    result = evaluate_simple_detectors(
        clean_erp, attacked_erp, clean_win, attacked_win, fs_out=60.0
    )
    assert set(result) == {
        "mahalanobis_auc", "mahalanobis_tpr_at_fpr",
        "band_power_auc", "band_power_tpr_at_fpr", "folds",
    }
    for key, value in result.items():
        if key == "folds":
            continue
        assert 0.0 <= value <= 1.0
    assert len(result["folds"]) == 2
