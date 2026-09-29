"""Frozen external-dataset split and attack-target construction."""
from __future__ import annotations

import numpy as np

from p3a.data.trial import Trial


def chronological_70_15_15(trials: list[Trial]) -> dict[str, list[str]]:
    """Split ordered trials with floor cut points at 70 and 85 percent."""
    if not trials:
        raise ValueError("trials must be non-empty")
    ordered = sorted(trials, key=lambda trial: trial.trial_index)
    n = len(ordered)
    train_end = int(np.floor(0.70 * n))
    val_end = int(np.floor(0.85 * n))
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError(f"not enough trials for a 70/15/15 split: {n}")
    ids = [trial.trial_id for trial in ordered]
    return {"train": ids[:train_end], "val": ids[train_end:val_end],
            "test": ids[val_end:]}


def official_train_test(train: list[Trial], test: list[Trial]) -> dict[str, list[str]]:
    """Preserve BNCI 2015-003 official recordings without a derived val set."""
    if not train or not test:
        raise ValueError("official train and test trials must be non-empty")
    return {"train": [trial.trial_id for trial in train], "val": [],
            "test": [trial.trial_id for trial in test]}


def attack_targets(test_trials: list[Trial], seed: int,
                   n_targets: int = 10) -> dict[str, str]:
    """Pre-draw distinct non-true target symbols for every test trial."""
    rng = np.random.default_rng(seed)
    result = {}
    for trial in sorted(test_trials, key=lambda value: value.trial_id):
        pool = [symbol for symbol in trial.layout.symbols
                if symbol != trial.target_symbol]
        if n_targets > len(pool):
            raise ValueError("n_targets exceeds the available non-true symbols")
        result[trial.trial_id] = "".join(
            rng.choice(pool, size=n_targets, replace=False).tolist()
        )
    return result
