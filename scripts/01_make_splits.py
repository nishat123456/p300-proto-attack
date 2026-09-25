"""Create the frozen Comp III splits and pre-registered attack target subsets.

Official train (85 chars) -> train/val; official test (100 chars) -> test.
Run once. Re-running verifies the files are unchanged.
"""
from pathlib import Path

import numpy as np

from p3a.data import splits
from p3a.data.layout import COMP3

ROOT = Path(__file__).parents[1]
SEED = 0
TARGET_SEED = 20260925
N_VAL = 15
N_TARGETS = 10

for subject in ("A", "B"):
    train_ids = [f"comp3:{subject}:train:{i:03d}" for i in range(85)]
    test_ids = [f"comp3:{subject}:test:{i:03d}" for i in range(100)]
    s = splits.make_train_val(train_ids, N_VAL, SEED)
    s["test"] = test_ids
    h = splits.write_once(ROOT / f"splits/comp3_{subject}.json", s)
    print(subject, {k: len(v) for k, v in s.items()}, h)

    # Pre-registered attack targets: 10 symbols per test trial, never the true symbol.
    labels = (ROOT / f"data/raw/comp3/true_labels_{subject.lower()}.txt").read_text().strip()
    rng = np.random.default_rng(TARGET_SEED + ord(subject))
    targets = {}
    for i, true in enumerate(labels):
        pool = [c for c in COMP3.symbols if c != true]
        targets[test_ids[i]] = "".join(rng.choice(pool, N_TARGETS, replace=False))
    h = splits.write_once(ROOT / f"splits/comp3_{subject}_attack_targets.json", targets)
    print(subject, "attack targets", h)
