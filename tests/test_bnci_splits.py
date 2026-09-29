import numpy as np

from p3a.data.bnci_splits import (
    attack_targets,
    chronological_70_15_15,
    official_train_test,
)
from p3a.data.layout import COMP3
from p3a.data.trial import Trial


def _trials(n, split="all"):
    trials = []
    for index in range(n):
        trials.append(Trial(
            eeg=np.zeros((2, 10), np.float32), fs=10.0,
            onsets=np.array([0]), flash_groups=np.ones((1, 36), bool),
            flash_codes=np.array([1]), target_symbol=COMP3.symbols[index % 36],
            layout=COMP3, subject_id="X", dataset="d", trial_index=index,
            source_split=split,
        ))
    return trials


def test_chronological_split_counts_and_order():
    split = chronological_70_15_15(_trials(35))
    assert {key: len(value) for key, value in split.items()} == {
        "train": 24, "val": 5, "test": 6,
    }
    assert split["train"][0].endswith(":000")
    assert split["test"][-1].endswith(":034")


def test_official_split_is_preserved():
    split = official_train_test(_trials(5, "train"), _trials(5, "test"))
    assert len(split["train"]) == 5 and len(split["test"]) == 5
    assert split["val"] == []
    assert all(":train:" in value for value in split["train"])
    assert all(":test:" in value for value in split["test"])


def test_attack_targets_are_reproducible_distinct_and_non_true():
    trials = _trials(4, "test")
    first = attack_targets(trials, seed=9)
    second = attack_targets(trials, seed=9)
    assert first == second
    for trial in trials:
        values = first[trial.trial_id]
        assert len(values) == len(set(values)) == 10
        assert trial.target_symbol not in values
