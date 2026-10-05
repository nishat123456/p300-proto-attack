import numpy as np
import pytest
from p3a.defenses.spatiotemporal_features import rich_features


def test_channel_and_time_identity_retained_with_summary_control():
    rng = np.random.default_rng(21)
    erp = rng.normal(size=(2, 4, 48)).astype(np.float32)
    windows = rng.normal(size=(2, 5, 4, 30)).astype(np.float32)
    original = rich_features(erp, windows)
    changed = rich_features(erp[:, ::-1], windows[:, :, ::-1])
    assert original.shape == (2, 36+4*12)
    assert np.allclose(original[:, :36], changed[:, :36], atol=1e-5)
    assert not np.allclose(original[:, 36:], changed[:, 36:])
    reversed_time = rich_features(erp[..., ::-1], windows)
    assert not np.allclose(original[:, 36:], reversed_time[:, 36:])
    assert np.isfinite(original).all()


def test_rich_features_reject_insufficient_temporal_samples():
    with pytest.raises(ValueError):
        rich_features(np.zeros((1, 4, 3)), np.zeros((1, 2, 4, 30)))
