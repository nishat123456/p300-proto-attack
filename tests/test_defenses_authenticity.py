import numpy as np

from p3a.defenses import artifacts as art
from p3a.defenses.authenticity import (
    authenticity_features,
    cross_validate_authenticity,
)

from defense_helpers import write_subject


def _combos(tmp_path):
    methods = [
        ("pgd", "none", "none", 1.0),
        ("target_only", "signal", "cross_subject", 1.0),
        ("noise", "none", "none", 1.0),
        ("tiny_noise", "none", "none", 1.0),
    ]
    write_subject(tmp_path, "A", n_trials=20, n_targets=2, methods=methods,
                  eps=(0.2,))
    return {
        spec.label: art.combo_to_inputs(art.load_combo(tmp_path, "A", spec))
        for spec in art.available_combos(tmp_path, "A")
    }


def test_authenticity_features_are_finite_and_fixed_width():
    rng = np.random.default_rng(1)
    erp = rng.normal(size=(6, 4, 48)).astype(np.float32)
    windows = rng.normal(size=(6, 3, 4, 60)).astype(np.float32)
    features = authenticity_features(erp, windows)
    assert features.shape[0] == 6
    assert 20 <= features.shape[1] <= 50
    assert np.isfinite(features).all()


def test_nested_authenticity_evaluation_reports_calibrated_operating_points(tmp_path):
    combos = _combos(tmp_path)
    pool = [value for key, value in combos.items()
            if key.startswith(("pgd_", "target_only_", "noise_"))]
    result = cross_validate_authenticity(pool, combos, n_folds=5, seed=2)
    assert set(result) == set(combos)
    for record in result.values():
        assert record["n_folds"] == 5
        assert np.isfinite(record["auc"])
        for suffix in ("0p01", "0p05", "0p1"):
            assert 0.0 <= record[f"tpr_{suffix}"] <= 1.0
            assert 0.0 <= record[f"far_{suffix}"] <= 1.0
        for fold in record["folds"]:
            assert fold["fold"] != fold["calibration_fold"]
