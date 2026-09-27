from dataclasses import replace

import numpy as np
import pytest
import torch

from p3a.data.epochs import batch_trials, downsample_onsets, epoch_window, gather_epochs
from p3a.data.layout import COMP3, rc_code_to_group
from p3a.data.trial import Trial


def test_epoch_slices_and_overlap_gradients():
    x = torch.arange(24.0).reshape(2, 2, 6).requires_grad_()
    onsets = torch.tensor([[0, 2], [1, 3]])
    y = gather_epochs(x, onsets, 3)
    assert y.shape == (2, 2, 2, 3)
    for b in range(2):
        for f in range(2):
            torch.testing.assert_close(y[b, f], x[b, :, onsets[b, f]:onsets[b, f] + 3])
    y.sum().backward()
    expected = torch.tensor([[1., 1., 2., 1., 1., 0.], [0., 1., 1., 2., 1., 1.]])
    torch.testing.assert_close(x.grad, expected[:, None].expand(2, 2, 6))


@pytest.mark.parametrize('onset', [-1, 4])
def test_out_of_bounds_rejected(onset):
    with pytest.raises(ValueError, match='outside'):
        gather_epochs(torch.zeros(1, 2, 6), torch.tensor([[onset]]), 3)


def test_onset_grid_and_window():
    values = [0, 3, 4, 7, 8]
    assert downsample_onsets(np.array(values), 4).tolist() == [0, 0, 1, 1, 2]
    result = downsample_onsets(torch.tensor(values, dtype=torch.int32), 4)
    assert result.dtype == torch.int64
    assert result.tolist() == [0, 0, 1, 1, 2]
    assert epoch_window(60) == 48
    assert epoch_window(64) == 51
    with pytest.raises(ValueError):
        downsample_onsets(np.array(values), 0)
    with pytest.raises(ValueError):
        epoch_window(0)


def test_batch_trials():
    trial = Trial(np.zeros((2, 12), np.float32), 240, np.array([0, 4]),
                  np.stack([rc_code_to_group(COMP3, c) for c in (1, 7)]),
                  np.array([1, 7]), 'A', COMP3, 'A', 'synthetic', 0, 'train')
    eeg, onsets, groups, targets = batch_trials([trial, replace(trial, target_symbol='B')], 'cpu')
    assert eeg.shape == (2, 2, 12) and eeg.dtype == torch.float32
    assert onsets.shape == (2, 2) and onsets.dtype == torch.long
    assert groups.shape == (2, 2, 36) and groups.dtype == torch.bool
    assert targets.tolist() == [0, 1]
    with pytest.raises(ValueError):
        batch_trials([], 'cpu')
    with pytest.raises(ValueError):
        batch_trials([trial, replace(trial, eeg=np.zeros((2, 11)))], 'cpu')
