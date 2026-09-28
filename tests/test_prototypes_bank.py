"""Fast CPU tests for PrototypeBank leakage guards and HDF5 roundtrip."""
import numpy as np
import pytest

from p3a.prototypes.bank import BankEntry, PrototypeBank
from tests.proto_helpers import FakeSpeller, synthetic_trial, trials_for

C, L = 4, 48


def build(mode, attacked, subjects, splits, trials, **kw):
    speller = FakeSpeller(C, L)
    return PrototypeBank.build(
        trials, speller, subjects, splits, mode=mode,
        attacked_subject=attacked, device="cpu", **kw,
    )


def test_same_subject_only_uses_frozen_train_ids():
    trials = trials_for("A", 8, split="train")
    allowed = {t.trial_id for t in trials[:5]}
    bank = build("same_subject", "A", {"A"}, {"train"}, trials, allowed_trial_ids=allowed)
    assert len(bank) == 5
    assert {e.split for e in bank.entries} == {"train"}
    assert all(e.trial_id in allowed for e in bank.entries)
    assert all(e.subject_id == "A" for e in bank.entries)


def test_same_subject_refuses_test_trials():
    trials = trials_for("A", 3, split="test")
    with pytest.raises(ValueError, match="never use test"):
        build("same_subject", "A", {"A"}, {"train", "test"}, trials)


def test_same_subject_requires_exactly_attacked_subject():
    trials = trials_for("A", 3)
    with pytest.raises(ValueError, match="exactly the attacked subject"):
        build("same_subject", "A", {"A", "B"}, {"train"}, trials)


def test_cross_subject_excludes_attacked_subject_at_build():
    trials = trials_for("A", 3) + trials_for("B", 4)
    bank = build("cross_subject", "A", {"B"}, {"train"}, trials)
    assert len(bank) == 4
    assert {e.subject_id for e in bank.entries} == {"B"}
    with pytest.raises(ValueError, match="attacked subject"):
        build("cross_subject", "A", {"A", "B"}, {"train"}, trials)


def test_cross_subject_refuses_attacked_trial_present_in_input():
    """Subject list says B only, but A trials are mixed in: refuse, do not drop."""
    trials = trials_for("B", 2) + trials_for("A", 2)
    # Subject filter would ignore A; force the issue by passing A in the list.
    with pytest.raises(ValueError, match="attacked subject"):
        build("cross_subject", "A", {"A", "B"}, {"train"}, trials)


def test_build_requires_a_dataset_and_layout():
    a = synthetic_trial("A", 0, dataset="d1")
    b = synthetic_trial("A", 1, dataset="d2")
    with pytest.raises(ValueError, match="multiple datasets"):
        build("same_subject", "A", {"A"}, {"train"}, [a, b])


def test_cross_subject_load_refuses_tampered_attacked_entry(tmp_path):
    trials = trials_for("B", 3)
    bank = build("cross_subject", "A", {"B"}, {"train"}, trials)
    path = bank.save(tmp_path / "bank.h5")

    import h5py
    with h5py.File(path, "r+") as f:
        subjects = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in f["subject_id"][()].tolist()]
        subjects[0] = "A"
        del f["subject_id"]
        f.create_dataset("subject_id", data=np.array(subjects, dtype=h5py.string_dtype()))
    with pytest.raises(AssertionError, match="attacked-subject"):
        PrototypeBank.load(path)


def test_same_subject_load_refuses_test_split(tmp_path):
    trials = trials_for("A", 3, split="train")
    bank = build("same_subject", "A", {"A"}, {"train"}, trials)
    path = bank.save(tmp_path / "bank.h5")
    import h5py
    with h5py.File(path, "r+") as f:
        splits = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["split"][()].tolist()]
        splits[0] = "test"
        del f["split"]
        f.create_dataset("split", data=np.array(splits, dtype=h5py.string_dtype()))
    with pytest.raises(AssertionError, match="test entry"):
        PrototypeBank.load(path)


def test_hdf5_roundtrip_preserves_entries_and_policy(tmp_path):
    trials = trials_for("B", 5, split="train") + trials_for("B", 2, split="test", start=10)
    bank = build("cross_subject", "A", {"B"}, {"train", "test"}, trials)
    path = bank.save(tmp_path / "bank.h5")
    loaded = PrototypeBank.load(path)

    assert loaded.mode == "cross_subject"
    assert loaded.attacked_subject == "A"
    assert loaded.dataset == bank.dataset
    assert loaded.n_symbols == bank.n_symbols
    assert loaded.allowed_splits == bank.allowed_splits
    assert len(loaded) == len(bank)
    for original, restored in zip(bank.entries, loaded.entries):
        np.testing.assert_array_equal(original.feature, restored.feature)
        assert original.symbol_idx == restored.symbol_idx
        assert original.trial_id == restored.trial_id
        assert original.subject_id == restored.subject_id
        assert original.split == restored.split


def test_load_rejects_missing_metadata(tmp_path):
    trials = trials_for("B", 2)
    bank = build("cross_subject", "A", {"B"}, {"train"}, trials)
    path = bank.save(tmp_path / "bank.h5")
    import h5py
    with h5py.File(path, "r+") as f:
        del f.attrs["mode"]
    with pytest.raises(ValueError, match="missing metadata"):
        PrototypeBank.load(path)


def test_validate_rejects_duplicate_trial_identity():
    trials = trials_for("A", 3)
    bank = build("same_subject", "A", {"A"}, {"train"}, trials)
    bank.entries.append(BankEntry(bank.entries[0].feature, bank.entries[0].symbol_idx,
                                  "A", bank.dataset, bank.entries[0].trial_id, "train"))
    with pytest.raises(ValueError, match="duplicate trial identity"):
        bank.validate()


def test_add_entry_enforces_cross_subject_policy():
    trials = trials_for("B", 2)
    bank = build("cross_subject", "A", {"B"}, {"train"}, trials)
    bad = BankEntry(np.zeros((C, L), np.float32), 0, "A", bank.dataset, "comp3:A:train:999", "train")
    with pytest.raises(AssertionError, match="attacked-subject"):
        bank.add_entry(bad)


def test_symbols_and_counts():
    trials = [synthetic_trial("A", i, target_symbol="ABC"[i % 3]) for i in range(6)]
    bank = build("same_subject", "A", {"A"}, {"train"}, trials)
    counts = bank.counts_per_symbol()
    assert counts.sum() == 6
    assert counts.sum() == len(bank)
    assert set(bank.symbols_available()) == {bank.entries[0].symbol_idx,
                                             bank.entries[1].symbol_idx,
                                             bank.entries[2].symbol_idx}
