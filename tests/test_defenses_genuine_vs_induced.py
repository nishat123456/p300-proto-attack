"""Fast CPU tests for the genuine-vs-induced ERP defense (Prompt 11)."""
import numpy as np
import pytest

from p3a.defenses.genuine_vs_induced import (
    DEFAULT_TRAIN_POOL,
    ErpDefense,
    cross_validate_defense,
    fit_erp_defense,
    pool_samples,
    unique_genuine,
)

from tests.defense_helpers import write_subject
from p3a.defenses import artifacts as art


def _load(tmp_path, subject="A"):
    combo = {}
    for spec in art.available_combos(tmp_path, subject):
        combo[spec.label] = art.combo_to_inputs(art.load_combo(tmp_path, subject, spec))
    return combo


def _pool_fixture(tmp_path):
    methods = [("pgd", "none", "none", 1.0), ("target_only", "signal", "cross_subject", 1.0),
               ("noise", "none", "none", 1.0), ("tiny_noise", "none", "none", 1.0)]
    # Trial indices default to 0..11, so both parity folds are populated.
    write_subject(tmp_path, "A", n_trials=12, n_targets=3, methods=methods,
                  eps=(0.20,))
    return _load(tmp_path)


def test_unique_genuine_drops_duplicates(tmp_path):
    write_subject(tmp_path, "A", n_trials=5, n_targets=4,
                  methods=[("pgd", "none", "none", 1.0)])
    inputs = _load(tmp_path)["pgd_none_none"]
    genuine = unique_genuine(inputs)
    assert len(genuine.erp) == 5
    assert len(set(genuine.trial_id.tolist())) == 5


def test_pool_samples_restrict_by_parity(tmp_path):
    combo = _pool_fixture(tmp_path)
    pool = [combo["pgd_none_none"], combo["noise_none_none"]]
    ind_all, gen_all = pool_samples(pool)
    ind_even, gen_even = pool_samples(pool, parity=0)
    ind_odd, gen_odd = pool_samples(pool, parity=1)
    assert len(ind_even.erp) + len(ind_odd.erp) == len(ind_all.erp)
    assert len(gen_even.erp) + len(gen_odd.erp) == len(gen_all.erp)
    assert np.all(ind_even.trial_index % 2 == 0)
    assert np.all(ind_odd.trial_index % 2 == 1)


def test_pool_samples_counts_each_genuine_trial_once_across_attacks(tmp_path):
    combo = _pool_fixture(tmp_path)
    pool = [combo["pgd_none_none"], combo["target_only_signal_cross_subject"],
            combo["noise_none_none"]]
    _, genuine = pool_samples(pool)
    assert len(genuine.erp) == 12
    assert len(set(genuine.trial_id.tolist())) == 12


def test_fit_erp_defense_scores_and_threshold(tmp_path):
    combo = _pool_fixture(tmp_path)
    pool = [combo["pgd_none_none"], combo["target_only_signal_cross_subject"],
            combo["noise_none_none"]]
    det = fit_erp_defense(pool, seed=0)
    assert isinstance(det, ErpDefense)
    assert det.kind in ("logreg", "cnn")
    induced = det.score(combo["pgd_none_none"].attacked_erp)
    genuine = det.score(combo["pgd_none_none"].genuine_erp)
    assert np.isfinite(induced).all() and np.isfinite(genuine).all()
    # The threshold is the 95th percentile of genuine training scores, so the
    # false alarm on training genuine is close to 5 percent.
    from p3a.defenses.metrics import false_alarm_rate
    far = false_alarm_rate(
        np.concatenate([induced, genuine]),
        np.concatenate([np.ones(len(induced)), np.zeros(len(genuine))]),
        det.threshold,
    )
    assert np.isnan(far) or far <= 0.25


def test_cross_validate_defense_reports_unseen_methods(tmp_path):
    combo = _pool_fixture(tmp_path)
    pool = [combo["pgd_none_none"], combo["target_only_signal_cross_subject"],
            combo["noise_none_none"]]
    evals = {label: combo[label] for label in combo}
    summary = cross_validate_defense(pool, evals, seed=0)
    # Every evaluation label gets a record, including the held-out tiny_noise.
    assert set(summary) == set(combo)
    for label, record in summary.items():
        assert set(record) >= {"auc", "tpr_at_fpr", "far", "folds"}
    assert np.isfinite(summary["tiny_noise_none_none"]["auc"]) or \
        np.isnan(summary["tiny_noise_none_none"]["auc"])


def test_default_train_pool_excludes_held_out_families():
    assert set(DEFAULT_TRAIN_POOL) == {"pgd", "target_only", "noise"}
    assert "tiny_noise" not in DEFAULT_TRAIN_POOL
    assert not any("proto" in name for name in DEFAULT_TRAIN_POOL)


def test_fit_requiring_both_classes(tmp_path):
    write_subject(tmp_path, "A", n_trials=6, n_targets=2,
                  methods=[("pgd", "none", "none", 0.0)])  # no successes
    inputs = _load(tmp_path)["pgd_none_none"]
    with pytest.raises(ValueError, match="genuine and induced"):
        fit_erp_defense([inputs])
