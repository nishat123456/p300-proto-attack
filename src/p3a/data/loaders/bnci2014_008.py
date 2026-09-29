"""BNCI 2014-008 P300 speller recordings from eight participants with ALS.

The official description defines ``data.trial`` as MATLAB one-based character
starts. Stimulus codes 1 through 6 are rows and 7 through 12 are columns, which
is the reverse of the BCI Competition III convention. Each character contains
10 repetitions of all 12 groups, for 120 flashes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from ..layout import Layout
from ..trial import Trial
from .comp3 import flash_onsets, find_file

FS = 256.0
N_FLASHES = 120
CHANNEL_NAMES = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
LAYOUT = Layout("rc6x6_bnci2014_008", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6, 6)


def row_first_code_to_group(code: int) -> np.ndarray:
    """Map BNCI row-first codes to a Boolean mask over the 6 by 6 layout."""
    group = np.zeros(LAYOUT.n_symbols, dtype=bool)
    if 1 <= code <= 6:
        row = code - 1
        group[row * 6:(row + 1) * 6] = True
    elif 7 <= code <= 12:
        col = code - 7
        group[col::6] = True
    else:
        raise ValueError(f"invalid BNCI 2014-008 stimulus code {code}")
    return group


def _target_symbol(codes: np.ndarray, target: np.ndarray) -> str:
    selected = np.unique(codes[target == 2])
    rows = selected[(selected >= 1) & (selected <= 6)]
    cols = selected[(selected >= 7) & (selected <= 12)]
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError(f"expected one target row and column, got {selected.tolist()}")
    index = (int(rows[0]) - 1) * 6 + (int(cols[0]) - 7)
    return LAYOUT.symbols[index]


def load_subject(root: str | Path, subject: str) -> list[Trial]:
    """Load all 35 character trials for one subject, preserving order."""
    subject = str(subject).upper()
    if subject.isdigit():
        subject = f"A{int(subject):02d}"
    if not (len(subject) == 3 and subject[0] == "A" and subject[1:].isdigit()):
        raise ValueError("subject must look like A01 through A08")
    mat = sio.loadmat(find_file(root, f"{subject}.mat"), squeeze_me=True,
                      struct_as_record=False)
    data = mat["data"]
    starts = np.asarray(data.trial, dtype=np.int64).reshape(-1) - 1
    signal = np.asarray(data.X, dtype=np.float32)
    stim_all = np.asarray(data.y_stim, dtype=np.int64).reshape(-1)
    target_all = np.asarray(data.y, dtype=np.int64).reshape(-1)
    channels = [str(value) for value in np.asarray(data.channels).reshape(-1)]
    if signal.ndim != 2 or signal.shape[1] != 8:
        raise ValueError(f"expected signal [samples,8], got {signal.shape}")
    if channels != CHANNEL_NAMES:
        raise ValueError(f"unexpected channel order {channels}")

    trials = []
    for index, start in enumerate(starts):
        stop = int(starts[index + 1]) if index + 1 < len(starts) else len(signal)
        stim = stim_all[start:stop]
        target = target_all[start:stop]
        onsets = flash_onsets(stim != 0)
        codes = stim[onsets].astype(np.int64)
        labels = target[onsets].astype(np.int64)
        if len(onsets) != N_FLASHES:
            raise ValueError(f"trial {index}: expected {N_FLASHES} flashes, got {len(onsets)}")
        values, counts = np.unique(codes, return_counts=True)
        if not np.array_equal(values, np.arange(1, 13)) or not np.all(counts == 10):
            raise ValueError(f"trial {index}: stimulus counts are not 10 per row/column")
        trials.append(Trial(
            eeg=np.ascontiguousarray(signal[start:stop].T, dtype=np.float32),
            fs=FS,
            onsets=onsets,
            flash_groups=np.stack([row_first_code_to_group(int(code)) for code in codes]),
            flash_codes=codes,
            target_symbol=_target_symbol(codes, labels),
            layout=LAYOUT,
            subject_id=subject,
            dataset="bnci2014_008",
            trial_index=index,
            source_split="all",
            channel_names=CHANNEL_NAMES.copy(),
        ))
    return trials
