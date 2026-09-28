"""Fast CPU tests for deterministic prototype retrieval."""
import numpy as np
import pytest

from p3a.prototypes.bank import BankEntry, PrototypeBank
from p3a.prototypes.features import FeatureScaler
from p3a.prototypes.retrieval import RetrievalResult, medoid_index, retrieve
from tests.proto_helpers import trials_for
from p3a.prototypes.bank import PrototypeBank as _Bank

C, L = 2, 4


def _bank(entries):
    """Three-symbol bank built directly from entries (no speller needed)."""
    bank = PrototypeBank(
        mode="same_subject", attacked_subject="A", dataset="synthetic",
        allowed_subjects={"A"}, allowed_splits={"train"},
        layout_id="rc6x6_comp3", n_symbols=36, n_channels=C, epoch_len=L,
    )
    bank.entries = list(entries)
    return bank


def _entry(feature, symbol, index):
    return BankEntry(np.asarray(feature, np.float32), symbol, "A", "synthetic",
                     f"comp3:A:train:{index:03d}", "train")


def _fixture_bank():
    """Symbol 0: rows [0,0] and [2,0]; symbol 1: [10,0]; symbol 2: [0,10]."""
    entries = [
        _entry(np.array([[0.0] * L, [0.0] * L]), 0, 0),
        _entry(np.array([[2.0] * L, [0.0] * L]), 0, 1),
        _entry(np.array([[10.0] * L, [0.0] * L]), 1, 2),
        _entry(np.array([[0.0] * L, [10.0] * L]), 2, 3),
    ]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    return bank, scaler


def test_medoid_is_deterministic_and_correct():
    bank, scaler = _fixture_bank()
    # Entries 0 and 1 are the symbol-0 cluster: entry 1 is central (distances
    # to 0 is 2, symmetric), but the tie is broken by lowest index.
    result = retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "medoid")
    assert result.index == medoid_index(bank, 0, scaler)
    # Deterministic across repeated calls and cache reuse.
    assert retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "medoid").index == result.index


def test_medoid_tie_breaks_to_lowest_index():
    # Three collinear points with a symmetric middle: index 0 and 2 tie, 1 is
    # the medoid, so this checks the expected single answer.
    entries = [
        _entry(np.full((C, L), 0.0, np.float32), 0, 0),
        _entry(np.full((C, L), 1.0, np.float32), 0, 1),
        _entry(np.full((C, L), 2.0, np.float32), 0, 2),
    ]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    assert medoid_index(bank, 0, scaler) == 1


def test_medoid_minimizes_sum_of_distances_not_squared_distances():
    # For 0, 1, 2, 4, Euclidean-distance sums tie at 1 and 2, so index 1 wins.
    # Squared-distance sums would incorrectly choose index 2.
    entries = [
        _entry(np.full((C, L), value, np.float32), 0, i)
        for i, value in enumerate((0.0, 1.0, 2.0, 4.0))
    ]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    assert medoid_index(bank, 0, scaler) == 1


def test_nearest_uses_standardized_distance():
    bank, scaler = _fixture_bank()
    query = np.full((C, L), 1.0, np.float32)
    result = retrieve(bank, scaler, query, 0, "nearest")
    assert result.index in (0, 1)
    assert result.distance is not None and np.isfinite(result.distance)
    # Nearest to the target cluster, never a wrong-letter entry.
    assert bank.entries[result.index].symbol_idx == 0


def test_nearest_tie_breaks_to_lowest_index():
    entries = [
        _entry(np.full((C, L), 0.0, np.float32), 0, 0),
        _entry(np.full((C, L), 2.0, np.float32), 0, 1),
    ]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    # Query at 1.0 is equidistant; the lower bank index must win.
    result = retrieve(bank, scaler, np.full((C, L), 1.0, np.float32), 0, "nearest")
    assert result.index == 0


def test_wrong_letter_returns_a_different_symbol():
    bank, scaler = _fixture_bank()
    query = np.array([[10.0] * L, [0.0] * L], np.float32)  # sits on symbol 1
    result = retrieve(bank, scaler, query, 0, "wrong_letter")
    assert result.index == 2  # symbol 1 is the nearest non-target entry
    assert bank.entries[result.index].symbol_idx != 0


def test_wrong_letter_returns_none_when_target_pool_is_empty():
    entries = [_entry(np.zeros((C, L), np.float32), 1, 0)]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    assert retrieve(
        bank, scaler, np.zeros((C, L), np.float32), 0, "wrong_letter"
    ) is None


def test_random_is_seeded_and_in_target_pool():
    bank, scaler = _fixture_bank()
    rng = np.random.default_rng(123)
    query = np.zeros((C, L), np.float32)
    result = retrieve(bank, scaler, query, 0, "random", rng)
    assert bank.entries[result.index].symbol_idx == 0
    assert np.isfinite(result.distance)
    index, distance = result
    assert index == result.index and distance == result.distance
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    first = retrieve(bank, scaler, query, 0, "random", rng_a).index
    second = retrieve(bank, scaler, query, 0, "random", rng_b).index
    assert first == second


def test_none_only_when_pool_absent():
    entries = [_entry(np.zeros((C, L), np.float32), 1, 0)]
    bank = _bank(entries)
    scaler = FeatureScaler.fit([e.feature for e in entries])
    assert retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "nearest") is None
    assert retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "medoid") is None
    assert retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "random",
                    np.random.default_rng(0)) is None
    assert retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "wrong_letter") is None


def test_invalid_mode_and_symbol_rejected():
    bank, scaler = _fixture_bank()
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retrieve(bank, scaler, np.zeros((C, L), np.float32), 0, "bogus")
    with pytest.raises(ValueError, match="outside"):
        retrieve(bank, scaler, np.zeros((C, L), np.float32), 999, "nearest")
    with pytest.raises(ValueError, match="requires an rng"):
        retrieve(bank, scaler, None, 0, "random")
