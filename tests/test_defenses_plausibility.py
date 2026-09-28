"""Fast CPU tests for the ERP plausibility metrics (Prompt 11)."""
import numpy as np
import pytest

from p3a.defenses.metrics import ChannelMap
from p3a.defenses.plausibility import (
    P300_WINDOW,
    plausibility_metrics,
    target_genuine_templates,
)


def _erp(scale=1.0, shift=0.0, ripple=0.0, c=4, length=48, seed=0, n=1):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 60.0
    bump = np.exp(-((t - 0.35 - shift) ** 2) / (2 * 0.06 ** 2))
    base = scale * np.outer(np.linspace(0.5, 1.5, c), bump)
    high = ripple * np.sin(2 * np.pi * 25.0 * t)[None, :]
    noise = 0.005 * rng.standard_normal((n, c, length))
    return (base[None] + high[None] + noise).astype(np.float32)


def test_templates_cover_every_referenced_symbol():
    true_idx = np.array([1, 2, 2, 3, 4, 5])
    clean = np.stack([_erp(seed=t)[0] for t in range(6)]).astype(np.float16)
    templates = target_genuine_templates(clean, true_idx)
    for symbol in np.unique(true_idx):
        assert int(symbol) in templates
        assert templates[int(symbol)].shape == (4, 48)


def test_templates_average_only_true_symbol_erps():
    clean = np.stack([_erp(seed=0)[0], _erp(seed=1)[0]]).astype(np.float16)
    templates = target_genuine_templates(clean, np.array([7, 7]))
    np.testing.assert_allclose(templates[7], clean.astype(np.float64).mean(axis=0))


def test_identical_erp_has_high_correlation_and_low_psd_distance():
    cmap = ChannelMap(pz_index=2, cz_index=1, source="provided")
    genuine = _erp(seed=0).astype(np.float16)
    templates = {0: genuine[0].astype(np.float64)}
    records = plausibility_metrics(genuine, np.array([0]),
                                   np.array([0]), templates, fs_out=60.0,
                                   channel_map=cmap)
    record = records[0]
    assert record["waveform_corr_pz"] == pytest.approx(1.0, abs=1e-3)
    assert record["waveform_corr_cz"] == pytest.approx(1.0, abs=1e-3)
    assert record["topography_corr"] == pytest.approx(1.0, abs=1e-3)
    # Both the attacked ERP and the clean template pass through the same float16
    # storage path, so an identical waveform must have a near-zero PSD distance.
    assert record["psd_distance_db"] == pytest.approx(0.0, abs=1e-3)
    assert record["channel_map_source"] == "provided"


def test_shifted_and_rippled_erp_scores_worse():
    cmap = ChannelMap(pz_index=2, cz_index=1, source="provided")
    genuine = _erp(seed=0)
    templates = {0: genuine[0].astype(np.float64)}
    attacked = _erp(scale=1.8, shift=0.1, ripple=0.5, seed=1).astype(np.float16)
    good = plausibility_metrics(genuine.astype(np.float16), np.array([0]),
                                np.array([0]), templates, fs_out=60.0, channel_map=cmap)[0]
    bad = plausibility_metrics(attacked, np.array([0]), np.array([0]), templates,
                               fs_out=60.0, channel_map=cmap)[0]
    assert bad["waveform_corr_pz"] < good["waveform_corr_pz"]
    assert bad["psd_distance_db"] > good["psd_distance_db"]


def test_p300_latency_in_window_and_amplitude_sign():
    cmap = ChannelMap(pz_index=0, cz_index=1, source="provided")
    genuine = _erp(seed=0).astype(np.float64)
    templates = {0: genuine[0]}
    record = plausibility_metrics(genuine.astype(np.float16), np.array([0]),
                                  np.array([0]), templates, fs_out=60.0,
                                  channel_map=cmap)[0]
    lo_ms, hi_ms = 1000.0 * P300_WINDOW[0], 1000.0 * P300_WINDOW[1]
    assert lo_ms <= record["p300_latency_ms"] <= hi_ms
    assert record["p300_amplitude"] > 0


def test_missing_template_yields_nan_row():
    cmap = ChannelMap(pz_index=2, cz_index=1, source="provided")
    attacked = _erp(seed=0)
    records = plausibility_metrics(attacked, np.array([5]), np.array([9]), {},
                                   fs_out=60.0, channel_map=cmap)
    assert np.isnan(records[0]["waveform_corr_pz"])
    assert records[0]["reference_symbol_idx"] == 9
