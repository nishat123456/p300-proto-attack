from pathlib import Path

import numpy as np
import pytest
import torch

from p3a.data.epochs import batch_trials, downsample_onsets, epoch_window, gather_epochs
from p3a.data.loaders.comp3 import load_subject
from p3a.models.letter_head import first_reps, symbol_evidence, symbol_logits


def test_arbitrary_groups_normalization_and_gradient():
    scores = torch.tensor([[1., 3., 8.]], requires_grad=True)
    groups = torch.tensor([[[True, False], [True, True], [False, True]]])
    torch.testing.assert_close(symbol_evidence(scores, groups), torch.tensor([[2., 5.5]]))
    symbol_logits(scores, groups, tau=2).sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[0.25, 0.5, 0.25]]))
    with pytest.raises(ValueError):
        symbol_logits(scores, groups, tau=0)
    with pytest.raises(ValueError, match='every symbol'):
        symbol_evidence(scores, torch.zeros_like(groups))


def test_repetition_truncation():
    onsets = torch.arange(24).reshape(1, 24)
    groups = torch.ones(1, 24, 2, dtype=torch.bool)
    scores = onsets.float().requires_grad_()
    o, g, s = first_reps(onsets, groups, scores, 1, 12)
    assert o.tolist() == [list(range(12))] and g.shape == (1, 12, 2)
    s.sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[1.] * 12 + [0.] * 12]))
    with pytest.raises(ValueError):
        first_reps(onsets, groups, scores, 3, 12)


@pytest.fixture(scope='module')
def trials_a():
    root = Path(__file__).parents[1] / 'data/raw/comp3'
    if not (root / 'Subject_A_Train.mat').exists():
        pytest.skip('Subject A train not downloaded')
    return load_subject(root, 'A', 'train')


def test_real_targets_all_85_trials(trials_a):
    groups = torch.tensor(np.stack([t.flash_groups for t in trials_a]))
    scores = torch.tensor(np.stack([t.target_flash_mask() for t in trials_a]), dtype=torch.float32)
    expected = torch.tensor([t.layout.index(t.target_symbol) for t in trials_a])
    assert len(trials_a) == 85
    assert torch.equal(symbol_logits(scores, groups).argmax(-1), expected)


def test_real_row_column_equivalence(trials_a):
    rng = np.random.default_rng(12)
    for trial in trials_a:
        scores = rng.normal(size=trial.n_flashes).astype(np.float32)
        means = np.array([scores[trial.flash_codes == c].mean() for c in range(1, 13)])
        expected = (means[6:, None] + means[None, :6]).reshape(-1) / 2
        actual = symbol_evidence(torch.tensor(scores)[None], torch.tensor(trial.flash_groups)[None])[0]
        np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7)
        assert actual.argmax().item() == expected.argmax()


def test_real_epoch_boundary(trials_a):
    # Check all actual onsets against the available decimated signal length.
    for trial in trials_a:
        assert int(downsample_onsets(trial.onsets, 4).max()) + epoch_window(60) <= (trial.eeg.shape[1] + 3) // 4
    eeg, onsets, _, _ = batch_trials(trials_a[:2], 'cpu')
    epochs = gather_epochs(eeg[..., ::4], downsample_onsets(onsets, 4), epoch_window(60))
    assert epochs.shape == (2, 180, 64, 48)
