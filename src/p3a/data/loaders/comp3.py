"""BCI Competition III Dataset II (Wadsworth P300 speller), subjects A and B.

Train files carry TargetChar; test targets come from the published true_labels_{a,b}.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from ..layout import COMP3, rc_code_to_group
from ..trial import Trial

FS = 240.0
N_FLASHES = 180


def flash_onsets(flashing: np.ndarray) -> np.ndarray:
    """Rising edges of the Flashing channel, including sample 0 if the trial starts mid-flash."""
    f = flashing.astype(bool)
    onsets = np.flatnonzero(f[1:] & ~f[:-1]) + 1
    if f[0]:
        onsets = np.insert(onsets, 0, 0)
    return onsets


def load_subject(root: str | Path, subject: str, split: str) -> list[Trial]:
    root = Path(root)
    subject = subject.upper()
    mat = sio.loadmat(root / f"Subject_{subject}_{split.capitalize()}.mat")
    signal = mat["Signal"]  # [trials, time, channels]
    if split == "train":
        targets = str(mat["TargetChar"][0])
    else:
        targets = (root / f"true_labels_{subject.lower()}.txt").read_text().strip()
    if len(targets) != signal.shape[0]:
        raise ValueError(f"{len(targets)} targets for {signal.shape[0]} trials")

    trials = []
    for i in range(signal.shape[0]):
        onsets = flash_onsets(mat["Flashing"][i])
        codes = mat["StimulusCode"][i][onsets].astype(int)
        if len(onsets) != N_FLASHES or (codes == 0).any():
            raise ValueError(f"trial {i}: {len(onsets)} flashes, zero codes={int((codes == 0).sum())}")
        trials.append(
            Trial(
                eeg=np.ascontiguousarray(signal[i].T, dtype=np.float32),
                fs=FS,
                onsets=onsets,
                flash_groups=np.stack([rc_code_to_group(COMP3, c) for c in codes]),
                flash_codes=codes,
                target_symbol=targets[i],
                layout=COMP3,
                subject_id=subject,
                dataset="comp3",
                trial_index=i,
                source_split=split,
            )
        )
    return trials
