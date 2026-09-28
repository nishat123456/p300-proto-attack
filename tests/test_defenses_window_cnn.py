"""Fast CPU tests for the clean-vs-attacked window CNN (Prompt 11)."""
import numpy as np
import pytest

from p3a.defenses.window_cnn import (
    WindowCNN,
    evaluate_window_detector,
    score_window_model,
    train_window_model,
)


def _separable_windows(n=24, n_windows=2, c=4, width=20, seed=0):
    """Clean windows are low-amplitude noise; attacked windows carry ripple."""
    rng = np.random.default_rng(seed)
    clean = 0.1 * rng.standard_normal((n, n_windows, c, width)).astype(np.float32)
    t = np.arange(width)
    ripple = 1.0 * np.sin(2 * np.pi * 8.0 / 60.0 * t)  # inside the window band
    attacked = (clean + ripple[None, None, None, :]).astype(np.float32)
    return attacked, clean


def test_window_cnn_forward_shape():
    model = WindowCNN(n_channels=4, width=20)
    x = __import__("torch").zeros(5, 1, 4, 20)
    logits = model(x)
    assert logits.shape == (5,)


def test_train_window_model_reduces_loss_and_separates():
    attacked, clean = _separable_windows(seed=0)
    train_mask = np.ones(len(attacked), dtype=bool)
    det = train_window_model(attacked, clean, train_mask, 4, 20, seed=0, epochs=12)
    pos = score_window_model(det.model, attacked)
    neg = score_window_model(det.model, clean)
    # The attacked windows must score above the clean ones on average.
    assert pos.mean() > neg.mean()


def test_evaluate_window_detector_auc_on_separable_data():
    attacked, clean = _separable_windows(n=30, seed=1)
    trial_index = np.arange(30)  # spans both parities so both folds exist
    summary, oof = evaluate_window_detector(
        attacked, clean, trial_index, seed=0, epochs=10
    )
    assert summary["auc"] > 0.7
    assert np.isfinite(summary["tpr_at_fpr"])
    assert len(summary["folds"]) == 2
    assert oof.shape[0] == 30


def test_evaluate_window_detector_shapes_mismatch_raises():
    attacked, clean = _separable_windows(n=8)
    with pytest.raises(ValueError, match="share a shape"):
        evaluate_window_detector(attacked, clean[:, :, :, :-1], np.arange(8))


def test_score_is_order_invariant_over_windows():
    attacked, clean = _separable_windows(n=6)
    det = train_window_model(attacked, clean, np.ones(6, bool), 4, 20, seed=0, epochs=3)
    a = score_window_model(det.model, attacked)
    shuffled = attacked[:, ::-1].copy()
    b = score_window_model(det.model, shuffled)
    np.testing.assert_allclose(a, b, atol=1e-5)
