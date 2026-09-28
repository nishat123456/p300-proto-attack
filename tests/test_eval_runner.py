"""Fast CPU tests for the resumable evaluation runner (Prompt 10).

These tests exercise ``run_evaluation`` end to end on a toy speller: real
Preprocessor, affine decoder, synthetic trials, and the noise / PGD providers
(the two methods that need no bank or trained generator). They check the two
properties the experiment depends on: the parquet rows are complete and
correct, and a second run resumes by skipping every completed combination.
"""
import numpy as np
import pandas as pd
import pytest

from p3a.data.layout import COMP3
from p3a.eval.evaluate import (
    MethodSpec,
    NoiseProvider,
    PGDProvider,
    default_method_specs,
    load_attack_targets,
    run_evaluation,
)
from p3a.eval.store import ArtifactSpec, AttackArtifacts, attack_npz_path, clean_npz_path

from tests.eval_helpers import make_speller, make_trials


def _targets(trials, n=10):
    targets = []
    for t in trials:
        true = t.target_symbol
        pool = [s for s in COMP3.symbols if s != true][:n]
        targets.append(pool)
    return targets


def _noise_specs():
    return [MethodSpec("noise", "noise"), MethodSpec("pgd", "pgd")]


def test_default_specs_cover_all_required_methods():
    labels = {s.label() for s in default_method_specs()}
    required = {
        "noise_none_none", "pgd_none_none", "tiny_noise_none_none",
        "target_only_signal_cross_subject", "target_only_causal_cross_subject",
        "proto_random_signal_cross_subject", "proto_random_causal_cross_subject",
        "proto_medoid_signal_cross_subject", "proto_medoid_causal_cross_subject",
        "proto_nearest_signal_cross_subject", "proto_nearest_causal_cross_subject",
        "proto_wrong_signal_cross_subject", "proto_wrong_causal_cross_subject",
        "proto_nearest_signal_same_subject", "proto_nearest_causal_same_subject",
    }
    assert required <= labels


def test_run_evaluation_row_count_and_columns(tmp_path):
    trials = make_trials(3)
    speller = make_speller(trials)
    targets = _targets(trials)
    providers = {
        "noise_none_none": NoiseProvider(),
        "pgd_none_none": PGDProvider(steps=3),
    }
    result = run_evaluation(
        trials=trials, targets_per_trial=targets, models={"surrogate": speller},
        providers=providers, eps_scale=0.5, out_dir=tmp_path,
        seed=0, eps_multipliers=(0.10, 0.20), specs=_noise_specs(), device="cpu",
    )
    # 2 methods x 2 eps x 3 trials x 10 targets.
    assert result.n_combos == 4
    assert result.n_combos_done == 4
    assert result.n_rows == 2 * 2 * 3 * 10
    frame = pd.read_parquet(result.parquet)
    assert len(frame) == result.n_rows
    required_cols = {
        "subject", "method", "generator", "bank", "eps_multiplier", "eps_raw",
        "trial_index", "trial_id", "true_symbol", "true_idx", "target_symbol",
        "target_idx", "retrieval_distance", "attack_ok", "linf", "l2", "rms",
        "snr_db", "success_surrogate_15", "success_surrogate_5",
    }
    assert required_cols <= set(frame.columns)
    assert set(frame["method"].unique()) == {"noise", "pgd"}
    # Every pair carries a success value in {0, 1} and a non-negative budget.
    assert frame["success_surrogate_15"].isin([0.0, 1.0]).all()
    assert (frame["linf"] >= 0).all()
    assert (frame["l2"] >= 0).all()
    # The L-inf budget is exactly eps_raw for the methods that use it (pgd), and
    # never exceeds it.
    pgd_rows = frame[frame["method"] == "pgd"]
    assert (pgd_rows["linf"] <= pgd_rows["eps_raw"] + 1e-5).all()


def test_run_evaluation_npz_shapes_and_float16(tmp_path):
    trials = make_trials(2)
    speller = make_speller(trials)
    targets = _targets(trials)
    providers = {"noise_none_none": NoiseProvider()}
    result = run_evaluation(
        trials=trials, targets_per_trial=targets, models={"surrogate": speller},
        providers=providers, eps_scale=0.4, out_dir=tmp_path,
        seed=0, eps_multipliers=(0.20,), specs=[MethodSpec("noise", "noise")],
        device="cpu",
    )
    assert result.n_rows == 2 * 10
    spec = ArtifactSpec("A", "noise", "none", "none", 0.20)
    art = AttackArtifacts.load(attack_npz_path(tmp_path, spec))
    n_targets = len(targets[0])
    assert art.decision_erp.shape[:2] == (2, n_targets)
    assert art.decision_erp.dtype == np.float16
    assert art.windows.shape[:2] == (2, n_targets)
    assert art.windows.dtype == np.float16
    assert art.window_starts.shape[0] == 2
    clean = np.load(clean_npz_path(tmp_path, "A"), allow_pickle=True)
    assert int(clean["store_format"]) == 1
    assert clean["true_erp"].dtype == np.float16
    assert clean["target_erp"].shape[:2] == (2, n_targets)


def test_run_evaluation_is_resumable(tmp_path):
    trials = make_trials(2)
    speller = make_speller(trials)
    targets = _targets(trials)
    providers = {"noise_none_none": NoiseProvider(), "pgd_none_none": PGDProvider(steps=2)}
    kwargs = dict(
        trials=trials, targets_per_trial=targets, models={"surrogate": speller},
        providers=providers, eps_scale=0.3, out_dir=tmp_path, seed=0,
        eps_multipliers=(0.10, 0.20), specs=_noise_specs(), device="cpu",
    )
    first = run_evaluation(**kwargs)
    assert first.n_combos_done == 4 and first.n_combos_skipped == 0
    second = run_evaluation(**kwargs)
    assert second.n_combos_done == 0 and second.n_combos_skipped == 4
    assert second.n_rows == first.n_rows
    forced = run_evaluation(**kwargs, force=True)
    assert forced.n_combos_done == 4 and forced.n_combos_skipped == 0


def test_resume_recomputes_when_trial_set_changes(tmp_path):
    trials = make_trials(2)
    speller = make_speller(trials)
    spec = [MethodSpec("noise", "noise")]
    providers = {"noise_none_none": NoiseProvider()}
    common = dict(models={"surrogate": speller}, providers=providers,
                  eps_scale=0.3, out_dir=tmp_path, seed=0,
                  eps_multipliers=(0.20,), specs=spec, device="cpu")
    first = run_evaluation(trials=trials[:1], targets_per_trial=_targets(trials[:1]),
                           **common)
    assert first.n_rows == 10
    second = run_evaluation(trials=trials, targets_per_trial=_targets(trials), **common)
    assert second.n_combos_done == 1
    assert second.n_combos_skipped == 0
    assert second.n_rows == 20


def test_run_evaluation_batch_size_does_not_change_results(tmp_path):
    """Rows and attacked windows are identical for one sample or all at once."""
    trials = make_trials(3)
    speller = make_speller(trials)
    targets = _targets(trials)
    providers = {
        "noise_none_none": NoiseProvider(),
        "pgd_none_none": PGDProvider(steps=3),
    }
    common = dict(
        trials=trials, targets_per_trial=targets, models={"surrogate": speller},
        providers=providers, eps_scale=0.5, seed=0, eps_multipliers=(0.20,),
        specs=_noise_specs(), device="cpu",
    )
    run_evaluation(out_dir=tmp_path / "small", batch_trials=1, **common)
    run_evaluation(out_dir=tmp_path / "large", batch_trials=30, **common)
    a = pd.read_parquet(tmp_path / "small" / "attacks.parquet").sort_values(
        ["method", "trial_index", "target_idx"]).reset_index(drop=True)
    b = pd.read_parquet(tmp_path / "large" / "attacks.parquet").sort_values(
        ["method", "trial_index", "target_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    spec = ArtifactSpec("A", "pgd", "none", "none", 0.20)
    wins_a = AttackArtifacts.load(attack_npz_path(tmp_path / "small", spec)).windows
    wins_b = AttackArtifacts.load(attack_npz_path(tmp_path / "large", spec)).windows
    np.testing.assert_array_equal(wins_a, wins_b)


def test_run_evaluation_requires_surrogate():
    trials = make_trials(1)
    with pytest.raises(ValueError, match="surrogate"):
        run_evaluation(
            trials=trials, targets_per_trial=_targets(trials), models={},
            providers={}, eps_scale=0.3, out_dir="/tmp/unused", device="cpu",
        )


def test_run_evaluation_victim_columns_present(tmp_path):
    trials = make_trials(2)
    surrogate = make_speller(trials, seed=0)
    victim = make_speller(trials, seed=1)
    targets = _targets(trials)
    providers = {"noise_none_none": NoiseProvider()}
    result = run_evaluation(
        trials=trials, targets_per_trial=targets,
        models={"surrogate": surrogate, "deepconvnet": victim},
        providers=providers, eps_scale=0.2, out_dir=tmp_path, seed=0,
        eps_multipliers=(0.20,), specs=[MethodSpec("noise", "noise")], device="cpu",
    )
    frame = pd.read_parquet(result.parquet)
    assert {"success_surrogate_15", "success_surrogate_5",
            "success_deepconvnet_15", "success_deepconvnet_5"} <= set(frame.columns)


def test_load_attack_targets_matches_frozen_split():
    from p3a.data.splits import load as load_split
    from p3a.attacks.train import repo_root

    targets = load_attack_targets("A")
    split = load_split(repo_root() / "splits/comp3_A.json")
    assert len(targets) == len(split["test"]) == 100
    # Every test trial has a pre-registered subset of 10 distinct targets.
    for trial_id in split["test"]:
        subset = targets[trial_id]
        assert len(subset) == 10 and len(set(subset)) == 10
