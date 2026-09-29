"""BNCI 2015-003 visual P300 speller recordings.

Verified against the official ``s1.mat`` and dataset description on 2026-09-29:
the first row is time at 256 Hz, rows 1 through 8 are EEG, row 9 contains
single-symbol stimulus codes 1 through 36, and row 10 is the target flag. Each
recorded character presents all 36 symbols 15 times, giving 540 flashes per
character. Recording lengths vary by subject and split. The older description's
row/column wording does not match the released trigger values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from ..layout import Layout
from ..trial import Trial
from .comp3 import flash_onsets, find_file

FS = 256.0
N_REPETITIONS = 15
N_FLASHES = 36 * N_REPETITIONS
CHANNEL_NAMES = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8"]
LAYOUT = Layout("single36_bnci2015_003", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6, 6)


def singleton_code_to_group(code: int) -> np.ndarray:
    """Map the released 1 through 36 code to one flashed symbol."""
    if not 1 <= int(code) <= 36:
        raise ValueError(f"invalid BNCI 2015-003 stimulus code {code}")
    group = np.zeros(36, dtype=bool)
    group[int(code) - 1] = True
    return group


def load_subject(root: str | Path, subject: str | int, split: str) -> list[Trial]:
    """Load every character in the official train or test recording."""
    split = str(split).lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    number = int(str(subject).lower().removeprefix("s"))
    if not 1 <= number <= 10:
        raise ValueError("subject must be 1 through 10")
    subject_id = f"s{number}"
    mat = sio.loadmat(find_file(root, f"{subject_id}.mat"), squeeze_me=True,
                      struct_as_record=False)
    matrix = np.asarray(getattr(mat[subject_id], split), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 11:
        raise ValueError(f"expected [11,samples], got {matrix.shape}")
    step = np.diff(matrix[0, :min(matrix.shape[1], 1000)])
    if not np.allclose(step, 1.0 / FS, atol=1e-9):
        raise ValueError("time row does not indicate 256 Hz sampling")
    eeg = matrix[1:9].astype(np.float32)
    stim = matrix[9].astype(np.int64)
    target = matrix[10].astype(np.int64)
    all_onsets = flash_onsets(stim != 0)
    if not len(all_onsets) or len(all_onsets) % N_FLASHES:
        raise ValueError(
            f"flash count must be a positive multiple of {N_FLASHES}, "
            f"got {len(all_onsets)}"
        )
    n_trials = len(all_onsets) // N_FLASHES

    trials = []
    padding = int(round(0.2 * FS))
    tail = int(round(0.8 * FS))
    for index in range(n_trials):
        onset_abs = all_onsets[index * N_FLASHES:(index + 1) * N_FLASHES]
        codes = stim[onset_abs].astype(np.int64)
        flags = target[onset_abs].astype(np.int64)
        values, counts = np.unique(codes, return_counts=True)
        if not np.array_equal(values, np.arange(1, 37)) or \
                not np.all(counts == N_REPETITIONS):
            raise ValueError(f"trial {index}: stimulus counts are not 15 per symbol")
        target_codes = np.unique(codes[flags == 1])
        if len(target_codes) != 1 or int(np.sum(flags)) != N_REPETITIONS:
            raise ValueError(f"trial {index}: expected one target code repeated 15 times")
        start = max(int(onset_abs[0]) - padding, 0)
        stop = min(int(onset_abs[-1]) + tail, eeg.shape[1])
        trials.append(Trial(
            eeg=np.ascontiguousarray(eeg[:, start:stop], dtype=np.float32),
            fs=FS,
            onsets=onset_abs - start,
            flash_groups=np.stack([singleton_code_to_group(int(code)) for code in codes]),
            flash_codes=codes,
            target_symbol=LAYOUT.symbols[int(target_codes[0]) - 1],
            layout=LAYOUT,
            subject_id=subject_id,
            dataset="bnci2015_003",
            trial_index=index,
            source_split=split,
            channel_names=CHANNEL_NAMES.copy(),
        ))
    return trials
