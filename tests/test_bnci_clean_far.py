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


def test_train_normalization_excludes_calibration_and_test(monkeypatch):
    from scripts import evaluate_bnci_clean_far as module
    trials = [_trial(i) for i in range(6)]
    split = {name: [trial.trial_id for trial in group] for name, group in
             zip(("train", "val", "test"), (trials[:2], trials[2:4], trials[4:]))}
    fitted = []

    def features(selected, fit_trials, seed):
        assert [t.trial_id for t in fit_trials] == split["train"]
        return np.array([[t.trial_index] * 36 for t in selected], dtype=float)

    class Scaler:
        def __init__(self, **kwargs):
            pass

        def fit(self, values):
            fitted.extend(values[:, 0].tolist())
            return self

        def transform(self, values):
            return values

    class Model:
        def decision_function(self, values):
            return values[:, 0]

    monkeypatch.setattr(module, "clean_features", features)
    monkeypatch.setattr(module, "RobustScaler", Scaler)
    result = module.evaluate_subject(Model(), trials, split, 0.05, 7, "train")
    assert fitted == [0.0, 1.0]
    assert result["threshold"] == 3.0
    assert result["n_normalization"] == 2
    assert result["normalization_fit"] == "train"
