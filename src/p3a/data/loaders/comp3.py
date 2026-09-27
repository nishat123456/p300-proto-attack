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


def find_file(root: str | Path, name: str) -> Path:
    """Return root/name if it exists, else the first recursive match under root.

    Kaggle attaches datasets in nested folders such as
    "Dataset p300/Subject_A_Train.mat/Subject_A_Train.mat", so a plain join
    fails there while local flat layouts keep working.
    """
    root = Path(root)
    direct = root / name
    # Kaggle layouts can contain a *directory* named like the .mat file, so a
    # bare exists() check would return a folder; require an actual file.
    if direct.is_file():
        return direct
    # Discard same-named directories: Kaggle's unzipped layout can contain a
    # folder called "Subject_A_Train.mat" that holds the real file.
    matches = sorted(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"{name!r} not found under {root}")
    return matches[0]


def load_subject(
    root: str | Path,
    subject: str,
    split: str,
    labels_dir: str | Path | None = None,
) -> list[Trial]:
    root = Path(root)
    subject = subject.upper()
    mat = sio.loadmat(find_file(root, f"Subject_{subject}_{split.capitalize()}.mat"))
    signal = mat["Signal"]  # [trials, time, channels]
    if split == "train":
        targets = str(mat["TargetChar"][0])
    else:
        labels_root = Path(labels_dir) if labels_dir is not None else root
        targets = find_file(labels_root, f"true_labels_{subject.lower()}.txt").read_text().strip()
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
