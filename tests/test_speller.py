"""CPU tests for the Speller wrapper and its checkpoint roundtrip."""
import numpy as np
import pytest
import torch

from p3a.models.eegnet import EEGNet
from p3a.models.speller import Speller
from p3a.models.xdawn_lda import XdawnLDA
from p3a.preproc.pipeline import Preprocessor
from p3a.data.layout import COMP3, rc_code_to_group

FS = 240.0
C = 8
T = 4000
L = 48
F = 12
K = 36


def _trial_batch():
    rng = np.random.default_rng(0)
    eeg = torch.as_tensor(rng.normal(size=(2, C, T)), dtype=torch.float32)
    step = 200
    onsets = torch.tensor([[10 + step * f for f in range(F)],
                           [30 + step * f for f in range(F)]], dtype=torch.long)
    # One full row/column sweep (codes 1..12) lights every symbol exactly twice,
    # so symbol_evidence has a nonzero count for all 36 symbols.
    codes = list(range(1, 13))
    groups = torch.stack([torch.as_tensor(np.stack([rc_code_to_group(COMP3, c) for c in codes]))
                          for _ in range(2)])
    assert (groups.sum(dim=1) > 0).all()
    return eeg, onsets, groups


def _fitted_preprocessor():
    rng = np.random.default_rng(1)
    pre = Preprocessor(n_channels=C, fs=FS)
    pre.fit_scaling([rng.normal(size=(C, T)).astype(np.float32) for _ in range(3)])
    return pre


def test_epochs_shape_and_raw_gradient():
    pre = _fitted_preprocessor()
    speller = Speller(pre, EEGNet(C, L), L)
    eeg, onsets, _ = _trial_batch()
    eeg.requires_grad_()
    ep = speller.epochs(eeg, onsets)
    assert ep.shape == (2, F, C, L)
    assert ep.dtype == torch.float32
    ep.sum().backward()
    assert eeg.grad is not None and torch.isfinite(eeg.grad).all()


def test_forward_returns_flash_and_symbol_shapes():
    pre = _fitted_preprocessor()
    speller = Speller(pre, EEGNet(C, L), L)
    eeg, onsets, groups = _trial_batch()
    out = speller(eeg, onsets, groups)
    assert set(out) == {"flash_scores", "logits"}
    assert out["flash_scores"].shape == (2, F)
    assert out["logits"].shape == (2, K)
    assert torch.isfinite(out["logits"]).all()


def test_torch_checkpoint_roundtrip(tmp_path):
    pre = _fitted_preprocessor()
    speller = Speller(pre, EEGNet(C, L), L).eval()
    eeg, onsets, groups = _trial_batch()
    before = speller(eeg, onsets, groups)["logits"]
    path = speller.save(tmp_path / "eegnet.pt", training_config={"model": "eegnet"}, seed=7)
    assert path.exists()

    loaded = Speller.load(path).eval()
    assert loaded.L == L
    assert loaded.decoder_name == "eegnet"
    after = loaded(eeg, onsets, groups)["logits"]
    torch.testing.assert_close(before, after)
    # Scaling buffers must survive the roundtrip, not reset to 0/1.
    torch.testing.assert_close(loaded.preprocessor.ch_mean, pre.ch_mean)
    torch.testing.assert_close(loaded.preprocessor.ch_std, pre.ch_std)


def test_xdawn_checkpoint_roundtrip(tmp_path):
    pre = _fitted_preprocessor()
    rng = np.random.default_rng(2)
    epochs = rng.normal(size=(40, C, L))
    y = np.zeros(40, dtype=bool)
    y[:20] = True
    epochs[y] += 0.5
    decoder = XdawnLDA().fit(epochs, y)
    speller = Speller(pre, decoder, L)

    eeg, onsets, groups = _trial_batch()
    before = speller(eeg, onsets, groups)["flash_scores"]
    path = speller.save(tmp_path / "xdawn.joblib", seed=7)
    loaded = Speller.load(path)
    assert loaded.decoder_name == "xdawn_lda"
    after = loaded(eeg, onsets, groups)["flash_scores"]
    np.testing.assert_allclose(before.numpy(), after.numpy(), rtol=1e-5, atol=1e-5)


def test_load_rejects_bad_format():
    with pytest.raises(ValueError, match="format"):
        Speller.from_checkpoint({"speller_format": 99})
