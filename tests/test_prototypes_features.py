"""Fast CPU tests for target-conditioned features and the feature scaler."""
import numpy as np
import pytest
import torch

from p3a.data.layout import COMP3
from p3a.prototypes.features import FeatureScaler, target_conditioned_erp

F, C, L, K = 6, 4, 5, COMP3.n_symbols


def _groups_with(symbol_idx):
    """Groups where flashes 0 and 2 light `symbol_idx`, flashes 1 and 3 do not."""
    groups = np.zeros((F, K), dtype=bool)
    groups[0, symbol_idx] = True
    groups[2, symbol_idx] = True
    groups[1, (symbol_idx + 1) % K] = True
    groups[3, (symbol_idx + 2) % K] = True
    return groups


def test_torch_and_numpy_paths_match_and_average_right_flashes():
    rng = np.random.default_rng(0)
    epochs_np = rng.normal(size=(F, C, L)).astype(np.float32)
    groups = _groups_with(7)
    expected = epochs_np[[0, 2]].mean(axis=0)

    np_out = target_conditioned_erp(epochs_np, groups, 7)
    torch_out = target_conditioned_erp(
        torch.as_tensor(epochs_np), torch.as_tensor(groups), 7
    )
    np.testing.assert_allclose(np_out, expected)
    np.testing.assert_allclose(torch_out.numpy(), expected)
    assert torch_out.dtype == torch.float32


def test_torch_path_is_differentiable():
    epochs = torch.randn(F, C, L, requires_grad=True)
    groups = torch.as_tensor(_groups_with(3))
    out = target_conditioned_erp(epochs, groups, 3)
    out.sum().backward()
    assert epochs.grad is not None
    # Only the two selected flashes carry gradient, each weighted 1/2.
    assert torch.allclose(epochs.grad[1], torch.zeros(C, L))
    assert torch.allclose(epochs.grad[0], torch.full((C, L), 0.5))
    assert torch.allclose(epochs.grad[2], torch.full((C, L), 0.5))


def test_missing_symbol_is_rejected():
    epochs = np.zeros((F, C, L), np.float32)
    groups = np.zeros((F, K), dtype=bool)
    groups[0, 0] = True
    with pytest.raises(ValueError, match="never flashed"):
        target_conditioned_erp(epochs, groups, 5)
    with pytest.raises(ValueError, match="never flashed"):
        target_conditioned_erp(torch.zeros(F, C, L), torch.as_tensor(groups), 5)


def test_dimension_and_type_validation():
    epochs = np.zeros((F, C, L), np.float32)
    groups = np.zeros((F - 1, K), dtype=bool)
    with pytest.raises(ValueError, match="flash dimension"):
        target_conditioned_erp(epochs, groups, 0)
    with pytest.raises(ValueError, match="outside"):
        target_conditioned_erp(epochs, _groups_with(1), K)
    with pytest.raises(ValueError, match="boolean"):
        target_conditioned_erp(epochs, np.zeros((F, K), np.int64), 0)
    with pytest.raises(ValueError, match="floating"):
        target_conditioned_erp(np.zeros((F, C, L), np.int64), _groups_with(1), 1)
    with pytest.raises(TypeError):
        target_conditioned_erp(torch.zeros(F, C, L),
                               np.zeros((F, K), dtype=bool), 0)


def test_scaler_roundtrip_and_zero_variance_safety():
    rng = np.random.default_rng(1)
    feats = [rng.normal(size=(C, L)).astype(np.float32) for _ in range(5)]
    # Pin one (channel, time) location to a constant across all members so its
    # variance is exactly zero and cannot be standardized.
    for f in feats:
        f[0, 0] = 3.0
    scaler = FeatureScaler.fit(feats)

    assert scaler.mean.shape == (C, L) and scaler.std.shape == (C, L)
    assert np.all(scaler.std > 0)
    assert scaler.std[0, 0] == 1.0
    out = scaler.transform(feats[0])
    assert out.dtype == np.float32 and out.shape == (C, L)
    np.testing.assert_allclose(scaler.inverse_transform(out), feats[0], atol=1e-4)
    # The zero-variance location maps to 0 rather than dividing by zero.
    assert out[0, 0] == 0.0


def test_scaler_save_load_is_exact(tmp_path):
    rng = np.random.default_rng(2)
    scaler = FeatureScaler.fit([rng.normal(size=(C, L)).astype(np.float32) for _ in range(4)])
    path = scaler.save(tmp_path / "scaler.npz")
    loaded = FeatureScaler.load(path)
    np.testing.assert_array_equal(loaded.mean, scaler.mean)
    np.testing.assert_array_equal(loaded.std, scaler.std)
    sample = rng.normal(size=(C, L)).astype(np.float32)
    np.testing.assert_array_equal(loaded.transform(sample), scaler.transform(sample))


def test_scaler_save_adds_npz_suffix(tmp_path):
    scaler = FeatureScaler.fit([np.zeros((C, L), np.float32)])
    path = scaler.save(tmp_path / "scaler")
    assert path.name == "scaler.npz" and path.exists()


def test_scaler_shape_mismatch_rejected():
    scaler = FeatureScaler.fit([np.zeros((C, L), np.float32)])
    with pytest.raises(ValueError, match="does not match"):
        scaler.transform(np.zeros((C, L + 1), np.float32))
    with pytest.raises(ValueError):
        FeatureScaler.fit([])


def test_query_uses_attacker_target_not_true_symbol():
    """A source trial whose true symbol differs: average the attacker flashes."""
    rng = np.random.default_rng(3)
    epochs = rng.normal(size=(F, C, L)).astype(np.float32)
    true_idx, attacker_idx = 0, 5
    groups = np.zeros((F, K), dtype=bool)
    # flashes 0,1 light the true symbol; flashes 2,4 light the attacker target.
    groups[0, true_idx] = groups[1, true_idx] = True
    groups[2, attacker_idx] = groups[4, attacker_idx] = True
    query = target_conditioned_erp(epochs, groups, attacker_idx)
    true_feature = target_conditioned_erp(epochs, groups, true_idx)
    np.testing.assert_allclose(query, epochs[[2, 4]].mean(axis=0))
    assert not np.allclose(query, true_feature)
