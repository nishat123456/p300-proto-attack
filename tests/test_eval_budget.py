"""Fast CPU tests for the budget metrics and matched-success fallbacks (Prompt 10)."""
import numpy as np
import pytest
import torch

from p3a.eval.budget import (
    budget_metrics,
    interpolate_at_success,
    l2_norm,
    linf_norm,
    rms_norm,
    sample_scalars,
    snr_db,
    success_curve_area,
)


def _pair(batch=3, C=4, T=50, seed=0):
    torch.manual_seed(seed)
    clean = torch.randn(batch, C, T)
    delta = torch.randn(batch, C, T) * 0.1
    return clean, delta


def test_linf_l2_rms_match_manual_values():
    clean, delta = _pair()
    assert torch.allclose(linf_norm(delta), delta.flatten(1).abs().amax(dim=1))
    assert torch.allclose(l2_norm(delta), delta.flatten(1).norm(dim=1), atol=1e-5)
    assert torch.allclose(
        rms_norm(delta), delta.flatten(1).pow(2).mean(dim=1).sqrt(), atol=1e-6
    )


def test_snr_db_is_twenty_for_ten_percent_perturbation():
    clean = torch.ones(2, 3, 20)
    delta = clean * 0.1
    value = snr_db(clean, delta)
    assert torch.allclose(value, torch.full((2,), 20.0), atol=1e-4)


def test_snr_db_zero_perturbation_is_large_but_finite():
    clean = torch.randn(2, 3, 20)
    value = snr_db(clean, torch.zeros_like(clean))
    assert torch.isfinite(value).all()
    assert bool((value > 100).all())


def test_budget_metrics_keys_and_zero_delta():
    clean, _ = _pair()
    metrics = budget_metrics(clean, torch.zeros_like(clean))
    assert set(metrics) == {"linf", "l2", "rms", "snr_db"}
    assert torch.allclose(metrics["linf"], torch.zeros(3))
    assert torch.allclose(metrics["rms"], torch.zeros(3))
    scaled = metrics["linf"] * 100.0
    assert torch.allclose(scaled, torch.zeros(3))


def test_budget_metrics_reject_mismatched_shapes():
    clean, delta = _pair()
    with pytest.raises(ValueError, match="share a shape"):
        budget_metrics(clean, delta[:, :, :-1])


def test_sample_scalars_averages_tensors_and_arrays():
    clean, delta = _pair()
    metrics = budget_metrics(clean, delta)
    summary = sample_scalars(metrics)
    assert set(summary) == {"linf", "l2", "rms", "snr_db"}
    for key, value in summary.items():
        assert value == pytest.approx(float(metrics[key].mean()))


def test_interpolate_reaches_level_between_grid_points():
    eps = np.array([0.1, 0.2, 0.3, 0.4])
    success = np.array([0.2, 0.5, 0.9, 0.95])
    # 0.65 sits between (0.2, 0.5) and (0.3, 0.9): fraction 0.375.
    value = interpolate_at_success(eps, success, 0.65)
    assert value == pytest.approx(0.2 + 0.375 * 0.1)


def test_interpolate_returns_none_when_level_not_reached():
    eps = np.array([0.1, 0.2, 0.3])
    success = np.array([0.1, 0.2, 0.3])
    assert interpolate_at_success(eps, success, 0.8) is None


def test_interpolate_clamps_to_first_grid_point_when_already_above():
    eps = np.array([0.1, 0.2, 0.3])
    success = np.array([0.9, 0.92, 0.95])
    assert interpolate_at_success(eps, success, 0.8) == pytest.approx(0.1)


def test_interpolate_rejects_non_increasing_eps():
    with pytest.raises(ValueError, match="increasing"):
        interpolate_at_success(np.array([0.2, 0.1]), np.array([0.5, 0.9]), 0.8)


def test_success_curve_area_is_one_for_perfect_curve():
    eps = np.array([0.1, 0.2, 0.3])
    assert success_curve_area(eps, np.ones(3)) == pytest.approx(1.0)


def test_success_curve_area_is_half_for_linear_curve():
    eps = np.array([0.0, 1.0])
    assert success_curve_area(eps, np.array([0.0, 1.0])) == pytest.approx(0.5)
