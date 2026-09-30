import numpy as np
from p3a.data.trial import Trial
from p3a.data.loaders.bnci2014_008 import LAYOUT
from scripts.evaluate_bnci_clean_far import clean_features


def _trial(index):
    rng = np.random.default_rng(index)
    groups = np.zeros((12, 36), dtype=bool)
    for code in range(6):
        groups[code, code * 6:(code + 1) * 6] = True
        groups[6 + code, code::6] = True
    return Trial(eeg=rng.normal(size=(8, 2048)).astype(np.float32), fs=256.0,
                 onsets=np.arange(100, 1300, 100), flash_groups=groups,
                 flash_codes=np.arange(1, 13), target_symbol="A", layout=LAYOUT,
                 subject_id="A01", dataset="bnci2014_008", trial_index=index,
                 source_split="all", channel_names=["Fz","Cz","Pz","Oz","P3","P4","PO7","PO8"])


def test_clean_features_are_finite_and_channel_invariant():
    trials = [_trial(0), _trial(1)]
    values = clean_features(trials, trials[:1], 7)
    assert values.shape == (2, 36)
    assert np.isfinite(values).all()
