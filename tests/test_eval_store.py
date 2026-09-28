"""Fast CPU tests for the float16 NPZ artifact store (Prompt 10)."""
import numpy as np

from p3a.eval.store import (
    N_WINDOWS,
    STORE_FORMAT,
    ArtifactSpec,
    AttackArtifacts,
    CleanArtifacts,
    attack_npz_path,
    clean_npz_path,
    decision_erp,
    preprocessed_windows,
    sample_window_starts,
    windows_from_preprocessed,
)

from tests.eval_helpers import make_speller, make_trials, batch


def test_sample_window_starts_are_deterministic_and_bounded():
    a = sample_window_starts(60.0, 300, seed=3)
    b = sample_window_starts(60.0, 300, seed=3)
    c = sample_window_starts(60.0, 300, seed=4)
    assert a.shape == (N_WINDOWS,)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    width = int(round(60.0))
    assert (a >= 0).all() and (a <= 300 - width).all()


def test_sample_window_starts_collapse_on_short_signal():
    starts = sample_window_starts(60.0, 30, seed=0)
    assert np.array_equal(starts, np.zeros(N_WINDOWS, dtype=np.int64))


def test_decision_erp_shape_and_dtype():
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    erp = decision_erp(speller, eeg[:1], onsets[:1], groups[:1], 5)
    assert erp.ndim == 2 and erp.shape[0] == eeg.shape[1]
    assert erp.dtype == np.float16


def test_preprocessed_windows_shape_and_float16():
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, _, _ = batch(trials)
    starts = sample_window_starts(speller.preprocessor.fs_out, 30, seed=0)
    windows = preprocessed_windows(speller, eeg[:1], starts)
    assert windows.shape == (N_WINDOWS, eeg.shape[1], int(round(60.0)))
    assert windows.dtype == np.float16


def test_windows_from_preprocessed_aligns_with_direct_path():
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, _, _ = batch(trials)
    starts = sample_window_starts(speller.preprocessor.fs_out, 30, seed=1)
    direct = preprocessed_windows(speller, eeg[:1], starts)
    x_pre = speller.preprocessor(eeg[:1])
    batched = windows_from_preprocessed(
        x_pre, starts[None], speller.preprocessor.fs_out
    )[0]
    np.testing.assert_array_equal(direct, batched)


def test_artifact_spec_filename_and_paths(tmp_path):
    spec = ArtifactSpec("A", "proto_nearest", "causal", "cross_subject", 0.20)
    assert spec.label == "proto_nearest_causal_cross_subject"
    assert spec.filename() == "proto_nearest_causal_cross_subject_eps0p20.npz"
    assert attack_npz_path(tmp_path, spec).name == spec.filename()
    assert clean_npz_path(tmp_path, "a").name == "A_clean.npz"


def test_clean_artifacts_roundtrip(tmp_path):
    trials = make_trials(2)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    erps, targets = [], []
    for i in range(2):
        erps.append(decision_erp(speller, eeg[i:i + 1], onsets[i:i + 1], groups[i:i + 1], i))
        targets.append([i, (i + 1) % 36])
    starts = np.stack([
        sample_window_starts(speller.preprocessor.fs_out, 30, seed=i) for i in range(2)
    ])
    x_pre = speller.preprocessor(eeg)
    windows = windows_from_preprocessed(x_pre, starts, speller.preprocessor.fs_out)
    clean = CleanArtifacts(
        subject="A", trial_index=np.array([0, 1]),
        trial_id=np.array(["a", "b"], dtype=object),
        true_idx=np.array([0, 1]), true_erp=np.stack(erps).astype(np.float16),
        target_idx=np.array(targets, dtype=np.int64),
        target_erp=np.stack([np.stack([e, e]) for e in erps]).astype(np.float16),
        window_starts=starts, windows=windows,
    )
    path = clean.save(tmp_path / "A_clean.npz")
    loaded = CleanArtifacts.load(path)
    assert loaded.subject == "A"
    np.testing.assert_array_equal(loaded.trial_index, clean.trial_index)
    np.testing.assert_array_equal(loaded.true_erp, clean.true_erp)
    assert loaded.true_erp.dtype == np.float16
    assert loaded.windows.dtype == np.float16


def test_attack_artifacts_roundtrip_and_format(tmp_path):
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    erp = decision_erp(speller, eeg, onsets, groups, 3)
    starts = sample_window_starts(speller.preprocessor.fs_out, 30, seed=0)
    windows = windows_from_preprocessed(
        speller.preprocessor(eeg), starts[None], speller.preprocessor.fs_out
    )
    art = AttackArtifacts(
        subject="A", trial_index=np.array([0]), trial_id=np.array(["a"], dtype=object),
        true_idx=np.array([0]), target_idx=np.array([[3]]),
        target_symbol=np.array([["D"]], dtype=object),
        decision_erp=erp[None, None].astype(np.float16),
        window_starts=starts[None], windows=windows[:, None],  # [T, K, n_windows, C, W]
    )
    path = art.save(tmp_path / "x.npz")
    loaded = AttackArtifacts.load(path)
    assert loaded.decision_erp.shape == (1, 1, erp.shape[0], erp.shape[1])
    assert loaded.decision_erp.dtype == np.float16
    assert loaded.windows.shape == (1, 1, N_WINDOWS, eeg.shape[1], int(round(60.0)))
    assert loaded.windows.dtype == np.float16
    np.testing.assert_array_equal(loaded.target_symbol, art.target_symbol)


def test_loading_wrong_format_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, store_format=np.int64(-1))
    import pytest

    with pytest.raises(ValueError, match="format"):
        AttackArtifacts.load(path)
    assert STORE_FORMAT == 1
