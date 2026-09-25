from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from p3a.data.layout import symbol_rc_codes
from p3a.data.loaders.comp3 import flash_onsets, load_subject

ROOT = Path(__file__).parents[1] / "data/raw/comp3"
needs_a = pytest.mark.skipif(not (ROOT / "Subject_A_Train.mat").exists(), reason="Subject A train not downloaded")


def test_flash_onsets_edge_cases():
    assert flash_onsets(np.array([1, 1, 0, 0, 1, 1, 0])).tolist() == [0, 4]
    assert flash_onsets(np.array([0, 1, 0, 1])).tolist() == [1, 3]


@pytest.fixture(scope="module")
def trials_a():
    return load_subject(ROOT, "A", "train")


@needs_a
def test_counts(trials_a):
    assert len(trials_a) == 85
    for t in trials_a:
        assert t.eeg.shape == (64, 7794) and t.eeg.dtype == np.float32
        assert t.n_flashes == 180
        assert t.target_flash_mask().sum() == 30
        assert np.bincount(t.flash_codes, minlength=13)[1:].tolist() == [15] * 12


@needs_a
def test_soa_is_42_samples(trials_a):
    d = np.diff(trials_a[0].onsets)
    assert np.median(d) == 42


@needs_a
def test_target_mask_matches_stimulus_type(trials_a):
    mat = sio.loadmat(ROOT / "Subject_A_Train.mat", variable_names=["StimulusType"])
    for t in trials_a:
        stype = mat["StimulusType"][t.trial_index][t.onsets].astype(bool)
        assert (stype == t.target_flash_mask()).all()
        r, c = symbol_rc_codes(t.layout, t.target_symbol)
        assert set(t.flash_codes[stype]) == {r, c}
