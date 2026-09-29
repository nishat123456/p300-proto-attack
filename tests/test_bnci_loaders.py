from pathlib import Path

import numpy as np
import scipy.io as sio

from p3a.data.loaders.bnci2014_008 import (
    CHANNEL_NAMES as CHANNELS_008,
    load_subject as load_008,
    row_first_code_to_group,
)
from p3a.data.loaders.bnci2015_003 import (
    load_subject as load_003,
    singleton_code_to_group,
)


def _write_008(root: Path):
    n_trials, spacing, lead = 2, 4, 100
    trial_width = 120 * spacing + 40
    starts = lead + np.arange(n_trials) * trial_width
    n_samples = int(starts[-1] + trial_width)
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(n_samples, 8)).astype(np.float32)
    stim = np.zeros(n_samples, dtype=np.int16)
    labels = np.zeros(n_samples, dtype=np.uint8)
    targets = [(2, 9), (5, 12)]
    for trial, start in enumerate(starts):
        codes = np.tile(np.arange(1, 13), 10)
        for j, code in enumerate(codes):
            pos = int(start + j * spacing)
            stim[pos:pos + 2] = code
            labels[pos:pos + 2] = 2 if code in targets[trial] else 1
    sio.savemat(root / "A01.mat", {"data": {
        "channels": np.asarray(CHANNELS_008, dtype=object),
        "X": signal,
        "y": labels,
        "y_stim": stim,
        "trial": starts + 1,
        "classes": np.asarray(["NonTarget", "Target"], dtype=object),
        "classes_stim": np.asarray(
            [f"Row{i}" for i in range(1, 7)] + [f"Col{i}" for i in range(1, 7)],
            dtype=object,
        ),
    }})


def _matrix_003(target_offset: int = 0, n_trials: int = 5):
    flashes, spacing, gap, lead = 540, 2, 40, 50
    n_samples = lead + n_trials * (flashes * spacing + gap) + 300
    matrix = np.zeros((11, n_samples), dtype=np.float64)
    matrix[0] = np.arange(n_samples) / 256.0
    rng = np.random.default_rng(2 + target_offset)
    matrix[1:9] = rng.normal(size=(8, n_samples))
    cursor = lead
    for trial in range(n_trials):
        codes = np.tile(np.arange(1, 37), 15)
        target_code = 1 + ((trial + target_offset) % 36)
        for j, code in enumerate(codes):
            pos = cursor + j * spacing
            matrix[9, pos] = code
            matrix[10, pos] = int(code == target_code)
        cursor += flashes * spacing + gap
    return matrix


def _write_003(root: Path):
    sio.savemat(root / "s1.mat", {"s1": {
        "train": _matrix_003(0),
        "test": _matrix_003(10, n_trials=2),
    }})


def test_008_row_first_mapping():
    assert np.flatnonzero(row_first_code_to_group(1)).tolist() == list(range(6))
    assert np.flatnonzero(row_first_code_to_group(7)).tolist() == [0, 6, 12, 18, 24, 30]


def test_008_loader_recovers_trials_targets_and_counts(tmp_path):
    _write_008(tmp_path)
    trials = load_008(tmp_path, "A01")
    assert len(trials) == 2
    assert [trial.target_symbol for trial in trials] == ["I", "3"]
    assert all(trial.n_flashes == 120 for trial in trials)
    assert all(trial.target_flash_mask().sum() == 20 for trial in trials)
    assert trials[0].channel_names == CHANNELS_008
    assert trials[0].trial_id == "bnci2014_008:A01:all:000"


def test_003_single_symbol_mapping():
    assert np.flatnonzero(singleton_code_to_group(1)).tolist() == [0]
    assert np.flatnonzero(singleton_code_to_group(36)).tolist() == [35]


def test_003_loader_preserves_official_split_and_singleton_flashes(tmp_path):
    _write_003(tmp_path)
    train = load_003(tmp_path, 1, "train")
    test = load_003(tmp_path, "s1", "test")
    assert len(train) == 5
    assert len(test) == 2
    assert [trial.target_symbol for trial in train] == list("ABCDE")
    assert [trial.target_symbol for trial in test] == list("KL")
    assert all(trial.n_flashes == 540 for trial in train + test)
    assert all(trial.target_flash_mask().sum() == 15 for trial in train + test)
    assert all(np.all(trial.flash_groups.sum(axis=1) == 1) for trial in train + test)
    assert train[0].trial_id == "bnci2015_003:s1:train:000"
    assert test[0].trial_id == "bnci2015_003:s1:test:000"


def test_003_row_column_protocol(tmp_path):
    matrix = _matrix_003(0, n_trials=1)
    matrix[9:] = 0
    codes = np.tile(np.arange(1, 13), 15)
    for i, code in enumerate(codes):
        pos = 50 + i * 2
        matrix[9, pos] = code
        matrix[10, pos] = int(code in (3, 8))
    sio.savemat(tmp_path / "s3.mat", {"s3": {"train": matrix, "test": matrix}})
    trial = load_003(tmp_path, "s3", "train")[0]
    assert trial.n_flashes == 180
    assert trial.target_symbol == "I"
    assert trial.target_flash_mask().sum() == 30
    assert np.all(trial.flash_groups.sum(axis=1) == 6)
