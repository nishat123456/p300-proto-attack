"""Fast CPU tests for the shared detector metrics and channel map (Prompt 11)."""
import numpy as np
import pytest

from p3a.defenses.metrics import (
    ChannelMap,
    auc,
    false_alarm_rate,
    folds_by_parity,
    tpr_at_fpr,
)


def test_auc_perfect_and_reversed():
    labels = np.array([1, 1, 0, 0])
    assert auc([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(1.0)
    assert auc([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(0.0)


def test_auc_single_class_is_nan():
    assert np.isnan(auc([0.1, 0.2], [1, 1]))
    assert np.isnan(auc([], []))


def test_tpr_at_fpr_threshold_and_far():
    # 100 negatives and 100 positives, positives all above the negatives.
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    scores = np.concatenate([np.linspace(2, 3, 100), np.linspace(0, 1, 100)])
    tpr, threshold = tpr_at_fpr(scores, labels, 0.05)
    assert tpr == pytest.approx(1.0)
    far = false_alarm_rate(scores, labels, threshold)
    assert far <= 0.05 + 1e-9


def test_tpr_at_fpr_rejects_bad_rate():
    with pytest.raises(ValueError, match="fpr"):
        tpr_at_fpr([0.1, 0.2], [1, 0], 1.5)


def test_folds_by_parity_splits_even_and_odd():
    trial_index = np.array([0, 1, 2, 3, 4, 5])
    folds = folds_by_parity(trial_index)
    np.testing.assert_array_equal(folds[1], [True, False, True, False, True, False])
    np.testing.assert_array_equal(folds[2], [False, True, False, True, False, True])


def test_channel_map_from_names_is_case_insensitive():
    names = ["Fz", "Cz", "Pz", "Oz"]
    cmap = ChannelMap.from_names(names)
    assert cmap.pz_index == 2 and cmap.cz_index == 1
    assert cmap.source == "provided"


def test_channel_map_from_names_missing_raises():
    with pytest.raises(ValueError, match="Pz"):
        ChannelMap.from_names(["Fz", "Cz", "Oz"])


def test_comp3_official_channel_indices():
    cmap = ChannelMap.comp3(64)
    assert cmap.pz_index == 50
    assert cmap.cz_index == 10
    assert cmap.source == "comp3_montage"


def test_channel_map_autodetect_picks_largest_p300_energy():
    erp = np.zeros((3, 4, 48), dtype=np.float32)
    # Channel 2 has the strongest 250-500 ms energy, channel 3 the next.
    erp[:, 2, 15:30] = 5.0
    erp[:, 3, 15:30] = 2.0
    erp[:, 0, 15:30] = 0.5
    erp[:, 1, 15:30] = 0.1
    cmap = ChannelMap.autodetect(erp, fs_out=60.0)
    assert cmap.source == "autodetected"
    assert cmap.pz_index == 2 and cmap.cz_index == 3
