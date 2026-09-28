"""Shared metrics, fold logic and channel resolution for the detectors.

Every detector in this package reports the same two numbers on a held-out fold:
the ROC AUC and the true positive rate at a fixed false positive rate, plus the
false alarm rate on held-out genuine trials. This module centralizes that math
so the window CNN, the genuine-vs-induced defense and the simple detectors can
never disagree on how a threshold is drawn or a fold is split.

Folds are defined by TEST TRIAL ID parity, exactly as Prompt 11 fixes: fold 1 is
the even trial index and fold 2 is the odd. A detector is trained on one fold
and scored on the other, and the reported number is the mean of the two
directions.

The channel map resolves the Pz and Cz positions the plausibility metrics need.
The Comp III release carries no channel labels, so Pz/Cz can only come from an
explicit montage. When the caller supplies channel names the map resolves them
by name; when not, the map is marked ``autodetected`` and picks the channel with
the largest genuine P300 energy as a data-driven Pz proxy and the next distinct
channel as a Cz proxy, and that provenance is written to the output so a result
can never be mistaken for a true Pz/Cz measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "ChannelMap",
    "DEFAULT_FPR",
    "auc",
    "folds_by_parity",
    "tpr_at_fpr",
]

DEFAULT_FPR = 0.05

# Common spellings for the two midline sites the plausibility metrics use.
_PZ_ALIASES = ("pz", "p_z", "pz.")
_CZ_ALIASES = ("cz", "c_z", "cz.")


def auc(scores, labels) -> float:
    """ROC AUC, or NaN when only one class is present.

    A fold that happens to hold a single class cannot define an AUC; returning
    NaN lets the caller drop the fold instead of inventing a number.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length")
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores, labels, fpr: float = DEFAULT_FPR) -> tuple[float, float]:
    """True positive rate at ``fpr`` plus the threshold that achieves it.

    The threshold is the ``1 - fpr`` quantile of the negative scores, so at most
    ``fpr`` of the negatives exceed it; the TPR is the fraction of positives at
    or above that threshold. Higher scores mean "more positive". When either
    class is absent the TPR is NaN.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length")
    if not (0.0 < fpr < 1.0):
        raise ValueError("fpr must lie strictly between 0 and 1")
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    threshold = float(np.quantile(neg, 1.0 - fpr))
    tpr = float((pos >= threshold).mean())
    return tpr, threshold


def false_alarm_rate(scores, labels, threshold: float) -> float:
    """Fraction of negative (genuine) samples at or above ``threshold``."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    neg = scores[labels == 0]
    if neg.size == 0:
        return float("nan")
    return float((neg >= threshold).mean())


def folds_by_parity(trial_index) -> dict[int, np.ndarray]:
    """Boolean masks for the two folds, keyed 1 (even) and 2 (odd).

    ``trial_index`` is the frozen test-trial index of each sample; a sample that
    belongs to trial 12 is in fold 1 and trial 13 in fold 2, so both folds span
    the whole recording instead of clustering in time.
    """
    trial_index = np.asarray(trial_index, dtype=np.int64).reshape(-1)
    return {1: (trial_index % 2 == 0), 2: (trial_index % 2 == 1)}


@dataclass
class ChannelMap:
    """Resolve the Pz and Cz channel indices used by the plausibility metrics.

    ``source`` records whether indices came from supplied channel names, the
    official Comp III montage, or a data-driven fallback.
    """

    pz_index: int
    cz_index: int
    source: str
    names: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.pz_index < 0 or self.cz_index < 0:
            raise ValueError("channel indices must be nonnegative")
        if self.source not in ("provided", "comp3_montage", "autodetected"):
            raise ValueError(f"unknown channel map source {self.source!r}")

    @classmethod
    def from_names(cls, names) -> "ChannelMap":
        """Resolve Pz and Cz from a channel-name list, case-insensitively."""
        names = [str(n) for n in names]
        lowered = [n.strip().lower() for n in names]

        def find(aliases):
            for i, name in enumerate(lowered):
                if name in aliases:
                    return i
            return None

        pz, cz = find(_PZ_ALIASES), find(_CZ_ALIASES)
        if pz is None or cz is None:
            missing = []
            if pz is None:
                missing.append("Pz")
            if cz is None:
                missing.append("Cz")
            raise ValueError(f"channel names do not contain {', '.join(missing)}")
        return cls(pz, cz, "provided", tuple(names))

    @classmethod
    def comp3(cls, n_channels: int) -> "ChannelMap":
        """Pz/Cz from the official 64-channel Dataset II assignment.

        The competition documentation numbers Cz as channel 11 and Pz as
        channel 51, hence zero-based indices 10 and 50.
        """
        if int(n_channels) != 64:
            raise ValueError("the Comp III montage requires 64 channels")
        return cls(pz_index=50, cz_index=10, source="comp3_montage")

    @classmethod
    def autodetect(cls, genuine_erp: np.ndarray, fs_out: float,
                   window=(0.25, 0.5)) -> "ChannelMap":
        """Pick Pz as the largest genuine P300 channel and Cz as the runner-up.

        This is a documented proxy for an unknown montage. ``genuine_erp`` is
        [N, C, L] and the P300 energy is the mean squared amplitude in
        ``window`` seconds at ``fs_out``.
        """
        erp = np.asarray(genuine_erp, dtype=np.float64)
        if erp.ndim != 3:
            raise ValueError("genuine_erp must be [N, C, L]")
        lo = int(round(window[0] * fs_out))
        hi = int(round(window[1] * fs_out))
        hi = min(max(hi, lo + 1), erp.shape[2])
        energy = (erp[:, :, lo:hi] ** 2).mean(axis=(0, 2))  # [C]
        order = np.argsort(energy)[::-1]
        pz = int(order[0])
        cz = int(order[1]) if order.size > 1 else int(order[0])
        return cls(pz, cz, "autodetected")
