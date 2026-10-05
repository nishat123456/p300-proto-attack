import numpy as np
import pytest
from p3a.defenses.feature_baselines import fit_detector


@pytest.mark.parametrize("kind", ["logistic", "forest", "mahalanobis"])
def test_score_orientation_and_finite_values(kind):
    rng = np.random.default_rng(12)
    clean = rng.normal(0, 0.1, (40, 3))
    induced = rng.normal(3, 0.1, (40, 3))
    model = fit_detector(kind, np.concatenate([clean, induced]),
                         np.repeat([0, 1], 40), 12)
    scores = model.decision_function(np.array([[0., 0., 0.], [3., 3., 3.]]))
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_distance_model_does_not_fit_attack_values():
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(30, 3))
    labels = np.repeat([0, 1], 30)
    first = fit_detector("mahalanobis", np.concatenate([clean, clean+10]), labels, 4)
    second = fit_detector("mahalanobis", np.concatenate([clean, clean+100]), labels, 4)
    assert np.allclose(first.decision_function(clean), second.decision_function(clean))


def test_normalization_excludes_calibration_and_evaluation(monkeypatch):
    from p3a.defenses import authenticity as module
    from sklearn.preprocessing import RobustScaler
    fitted = []

    class SpyScaler(RobustScaler):
        def fit(self, values, y=None):
            fitted.append(values.copy())
            return super().fit(values, y)

    monkeypatch.setattr(module, "RobustScaler", SpyScaler)
    indices = np.arange(25)
    ids = np.array([str(i) for i in indices])
    def features(offset):
        return module.FeatureSet(np.column_stack([indices+offset, indices*.01]), indices, ids)
    module.cross_subject_feature_sets(
        [(features(103), features(100))], {"attack": (features(3), features(0))},
        seed=3, independent_normalization=True,
    )
    assert len(fitted) == 6
    for outer, values in enumerate(fitted[1:]):
        fitted_indices = values[:, 0].astype(int)
        assert len(fitted_indices) == 15
        assert not np.any(fitted_indices % 5 == outer)
        assert not np.any(fitted_indices % 5 == (outer+1) % 5)


def test_score_summary_separates_ties_from_strict_exceedances():
    from p3a.defenses.feature_baselines import score_summary
    result = score_summary(np.array([0.1, 0.5, 0.5, 0.8]), {0.05: 0.5})
    assert result["thresholds"]["0.05"] == {
        "value": 0.5, "n_ge": 3, "n_gt": 1, "n_equal": 2}


def test_current_features_discard_channel_identity_and_window_order():
    from p3a.defenses.authenticity import authenticity_features
    rng = np.random.default_rng(8)
    erp = rng.normal(size=(3, 4, 48)).astype(np.float32)
    windows = rng.normal(size=(3, 5, 4, 30)).astype(np.float32)
    original = authenticity_features(erp, windows)
    permuted = authenticity_features(erp[:, ::-1], windows[:, ::-1, ::-1])
    assert np.allclose(original, permuted, atol=1e-5)
