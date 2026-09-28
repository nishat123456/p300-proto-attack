"""Genuine-vs-induced ERP detector, the defense (Prompt 11).

Input is the decision ERP [C, L]: the target-conditioned response the speller's
decision relied on. Genuine samples are clean test trials' ERPs at their true
symbol; induced samples are attacked ERPs at the attacker target where the
surrogate attack succeeded.

Training respects the held-out families fixed by the pre-registration: the pool
is {pgd, target_only (both generators), noise} only, and Tiny Noise plus every
prototype-guided variant are evaluated as unseen attacks. Two model kinds are
compared, a logistic regression on the standardized flattened ERP and a small
1D CNN, and the one with the better inner-validation AUC inside the training
fold is used. The decision threshold is the 95th percentile of the genuine
training scores, which targets a 5 percent false alarm rate by construction.

Folds are by test trial parity, as Prompt 11 fixes: each model is trained on one
fold and scored on the other, and every reported number is the mean over the two
directions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from p3a.defenses.metrics import (
    DEFAULT_FPR,
    auc,
    false_alarm_rate,
    tpr_at_fpr,
)
from p3a.defenses.artifacts import AttackInputs
from p3a.utils.repro import set_seed

__all__ = [
    "DEFAULT_TRAIN_POOL",
    "ErpCNN",
    "ErpDefense",
    "cross_validate_defense",
    "fit_erp_defense",
    "unique_genuine",
]

# The attack families the defense is trained on; everything else is unseen.
DEFAULT_TRAIN_POOL = ("pgd", "target_only", "noise")


@dataclass(frozen=True)
class _SampleSet:
    """Flattened ERPs with a per-sample trial index for fold assignment."""

    erp: np.ndarray      # [N, C, L] float32
    trial_index: np.ndarray
    trial_id: np.ndarray


def unique_genuine(inputs: AttackInputs) -> _SampleSet:
    """One genuine true-symbol ERP per distinct trial, in first-seen order.

    ``inputs.genuine_erp`` repeats a trial's genuine ERP once per attacker
    target; the defense needs it once, so duplicates are dropped by trial id.
    """
    seen, keep = set(), []
    for i, trial_id in enumerate(inputs.trial_id):
        key = str(trial_id)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    return _SampleSet(
        erp=np.asarray(inputs.genuine_erp, dtype=np.float32)[keep],
        trial_index=np.asarray(inputs.trial_index, dtype=np.int64)[keep],
        trial_id=np.asarray(inputs.trial_id, dtype=object)[keep],
    )


def _induced(inputs: AttackInputs) -> _SampleSet:
    """Attacked ERPs at the attacker target where the surrogate succeeded."""
    mask = np.asarray(inputs.success, dtype=bool)
    return _SampleSet(
        erp=np.asarray(inputs.attacked_erp, dtype=np.float32)[mask],
        trial_index=np.asarray(inputs.trial_index, dtype=np.int64)[mask],
        trial_id=np.asarray(inputs.trial_id, dtype=object)[mask],
    )


class ErpCNN(nn.Module):
    """Small temporal CNN over the [C, L] decision ERP."""

    def __init__(self, n_channels: int, length: int, hidden: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_channels)
        self.conv1 = nn.Conv1d(n_channels, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.head = nn.Linear(hidden * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(self.bn0(x))))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.drop(self.pool(h)).flatten(1)
        return self.head(h).squeeze(1)


@dataclass
class ErpDefense:
    """A fitted ERP detector: a model, its standardizer and its threshold."""

    kind: str
    model: object
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    n_channels: int
    length: int

    def _standardize(self, erp: np.ndarray) -> np.ndarray:
        arr = np.asarray(erp, dtype=np.float32)
        return (arr - self.mean) / self.std

    def score(self, erp: np.ndarray) -> np.ndarray:
        """Induced-ness logits, higher meaning more likely induced."""
        arr = np.asarray(erp, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("erp must be [N, C, L]")
        if self.kind == "logreg":
            flat = self._standardize(arr).reshape(arr.shape[0], -1)
            return self.model.decision_function(flat).reshape(-1)
        model = self.model
        was_training = model.training
        dev = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(self._standardize(arr), dtype=torch.float32,
                                     device=dev)
            logits = model(tensor).cpu()
        if was_training:
            model.train()
        return logits.numpy().reshape(-1)

    def predict(self, erp: np.ndarray) -> np.ndarray:
        """Boolean induced prediction at the configured threshold."""
        return self.score(erp) >= self.threshold


def _fit_logreg(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> ErpDefense:
    flat = x_train.reshape(x_train.shape[0], -1)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(flat, y_train)
    return ErpDefense("logreg", clf, np.zeros(1, dtype=np.float32),
                      np.ones(1, dtype=np.float32), 0.0,
                      x_train.shape[1], x_train.shape[2])


def _fit_cnn(x_train: np.ndarray, y_train: np.ndarray, seed: int,
             epochs: int = 30, lr: float = 1e-3, batch: int = 64,
             device: str = "cpu") -> ErpDefense:
    set_seed(seed)
    dev = torch.device(device)
    model = ErpCNN(x_train.shape[1], x_train.shape[2]).to(dev)
    n = x_train.shape[0]
    n_pos = int(y_train.sum())
    n_neg = n - n_pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(max(n_neg / max(n_pos, 1), 1e-6), device=dev)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            optimizer.zero_grad()
            loss = criterion(model(x[idx].to(dev)), y[idx].to(dev))
            loss.backward()
            optimizer.step()
    return ErpDefense("cnn", model, np.zeros(1, dtype=np.float32),
                      np.ones(1, dtype=np.float32), 0.0,
                      x_train.shape[1], x_train.shape[2])


def _standardized_defense(genuine: _SampleSet, induced: _SampleSet, kind: str,
                          seed: int, fpr: float, epochs: int = 30,
                          device: str = "cpu") -> ErpDefense:
    """Fit one ERP detector and set its threshold from the genuine training set."""
    x = np.concatenate([induced.erp, genuine.erp], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(len(induced.erp)), np.zeros(len(genuine.erp))])
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    xs = (x - mean) / std

    if kind == "logreg":
        det = _fit_logreg(xs, y, seed)
    elif kind == "cnn":
        det = _fit_cnn(xs, y, seed, epochs=epochs, device=device)
    else:
        raise ValueError(f"unknown detector kind {kind!r}")
    det.mean, det.std = mean, std
    genuine_scores = det.score(genuine.erp)
    det.threshold = float(np.quantile(genuine_scores, 1.0 - fpr))
    return det


def _choose_kind(genuine: _SampleSet, induced: _SampleSet, seed: int,
                 kinds=("logreg", "cnn"), val_fraction: float = 0.25,
                 device: str = "cpu") -> str:
    """Pick the model kind by inner-validation AUC inside the training fold.

    Validation trials are drawn by trial id so no trial can appear on both
    sides, which matters because an induced sample repeats its trial across
    attacker targets. Both kinds are fit on the inner training ids and scored on
    the inner validation ids; the higher AUC wins. A kind that cannot be scored
    (single class in the inner validation) is skipped. The split is seeded, so
    the choice is reproducible.
    """
    ids = np.unique(np.concatenate([induced.trial_id, genuine.trial_id]).astype(str))
    if len(ids) < 2:
        return "logreg"
    n_val = max(int(round(len(ids) * val_fraction)), 1)
    n_val = min(n_val, len(ids) - 1)
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(ids, size=n_val, replace=False).tolist())

    def mask(sample_set):
        return np.array([str(t) in val_ids for t in sample_set.trial_id], dtype=bool)

    ind_val, gen_val = mask(induced), mask(genuine)
    ind_train = _SampleSet(induced.erp[~ind_val], induced.trial_index[~ind_val], induced.trial_id[~ind_val])
    gen_train = _SampleSet(genuine.erp[~gen_val], genuine.trial_index[~gen_val], genuine.trial_id[~gen_val])
    ind_eval = _SampleSet(induced.erp[ind_val], induced.trial_index[ind_val], induced.trial_id[ind_val])
    gen_eval = _SampleSet(genuine.erp[gen_val], genuine.trial_index[gen_val], genuine.trial_id[gen_val])
    if len(ind_eval.erp) == 0 or len(gen_eval.erp) == 0:
        return "logreg"

    best_kind, best_auc = "logreg", -1.0
    for kind in kinds:
        try:
            det = _standardized_defense(gen_train, ind_train, kind, seed, fpr=0.05,
                                        device=device)
            scores = np.concatenate([det.score(ind_eval.erp), det.score(gen_eval.erp)])
            labels = np.concatenate([np.ones(len(ind_eval.erp)), np.zeros(len(gen_eval.erp))])
            value = auc(scores, labels)
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(value) and value > best_auc:
            best_kind, best_auc = kind, value
    return best_kind


def pool_samples(pool: list[AttackInputs], parity: int | None = None
                 ) -> tuple[_SampleSet, _SampleSet]:
    """Concatenate a training pool into (induced, genuine) sample sets.

    ``parity`` restricts both sets to trials with ``trial_index % 2 == parity``,
    which is how the two-fold cross-validation keeps a held-out fold out of
    training. A pool chunk's induced set is its successful attacked ERPs and its
    genuine set is its unique true-symbol ERPs.
    """
    ind_parts, gen_parts = [], []
    for chunk in pool:
        ind_parts.append(_induced(chunk))
        gen_parts.append(unique_genuine(chunk))
    ind = _concat(ind_parts)
    gen = _deduplicate_by_trial(_concat(gen_parts))
    if parity is not None:
        ind = _restrict(ind, int(parity))
        gen = _restrict(gen, int(parity))
    return ind, gen


def _concat(parts: list[_SampleSet]) -> _SampleSet:
    if not parts:
        return _SampleSet(np.zeros((0, 0, 0), np.float32), np.zeros(0, np.int64),
                          np.zeros(0, dtype=object))
    return _SampleSet(
        erp=np.concatenate([p.erp for p in parts], axis=0),
        trial_index=np.concatenate([p.trial_index for p in parts]),
        trial_id=np.concatenate([p.trial_id for p in parts]),
    )


def _deduplicate_by_trial(sample_set: _SampleSet) -> _SampleSet:
    """Keep the first genuine sample for each trial across pooled attacks."""
    seen, keep = set(), []
    for i, trial_id in enumerate(sample_set.trial_id):
        key = str(trial_id)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    return _SampleSet(sample_set.erp[keep], sample_set.trial_index[keep],
                      sample_set.trial_id[keep])


def _restrict(sample_set: _SampleSet, parity: int) -> _SampleSet:
    keep = sample_set.trial_index % 2 == parity
    return _SampleSet(sample_set.erp[keep], sample_set.trial_index[keep],
                      sample_set.trial_id[keep])


def fit_erp_defense(pool: list[AttackInputs], fpr: float = DEFAULT_FPR,
                    seed: int = 0, kinds=("logreg", "cnn"), device: str = "cpu",
                    parity: int | None = None) -> ErpDefense:
    """Fit the defense on a training pool, optionally restricted to one fold.

    ``pool`` is a list of ``AttackInputs`` for the training attack families. The
    induced set is their successful attacked ERPs and the genuine set is their
    unique true-symbol ERPs. ``parity`` keeps only trials of that parity, which
    the two-fold cross-validation uses to train on the held-out fold's
    opposite.
    """
    ind, gen = pool_samples(pool, parity=parity)
    if len(gen.erp) == 0 or len(ind.erp) == 0:
        raise ValueError("defense training needs both genuine and induced samples")
    kind = _choose_kind(gen, ind, seed, kinds=kinds, device=device)
    return _standardized_defense(gen, ind, kind, seed, fpr=fpr, device=device)


def cross_validate_defense(pool: list[AttackInputs], eval_chunks: dict[str, AttackInputs],
                           fpr: float = DEFAULT_FPR, seed: int = 0,
                           kinds=("logreg", "cnn"), device: str = "cpu") -> dict:
    """Two-fold defense evaluation over the eval chunks.

    ``pool`` supplies the training families; ``eval_chunks`` maps a label to its
    aligned inputs (any method and epsilon). For each held-out fold a defense is
    trained on the other parity of the pool and scored on the held-out induced
    and genuine samples of every eval chunk. Returns mean AUC, mean TPR at the
    training threshold, and mean false alarm rate, plus per-chunk fold records.
    """
    if not pool or not eval_chunks:
        raise ValueError("cross_validate_defense needs a pool and eval chunks")
    records: dict[str, list] = {label: [] for label in eval_chunks}

    for test_parity in (0, 1):
        train_parity = 1 - test_parity
        try:
            det = fit_erp_defense(pool, fpr=fpr, seed=seed, kinds=kinds,
                                  device=device, parity=train_parity)
        except ValueError:
            continue
        for label, chunk in eval_chunks.items():
            ind = _restrict(_induced(chunk), test_parity)
            gen = _restrict(unique_genuine(chunk), test_parity)
            if len(ind.erp) == 0 or len(gen.erp) == 0:
                continue
            ind_scores = det.score(ind.erp)
            gen_scores = det.score(gen.erp)
            labels = np.concatenate([np.ones(len(ind_scores)), np.zeros(len(gen_scores))])
            scores = np.concatenate([ind_scores, gen_scores])
            records[label].append({
                "fold": int(test_parity),
                "auc": auc(scores, labels),
                "tpr_at_fpr": tpr_at_fpr(scores, labels, fpr)[0],
                "far": false_alarm_rate(scores, labels, det.threshold),
                "n_induced": int(len(ind_scores)),
                "n_genuine": int(len(gen_scores)),
                "threshold": det.threshold,
                "model_kind": det.kind,
            })

    summary = {}
    for label, rows in records.items():
        if not rows:
            summary[label] = {"auc": float("nan"), "tpr_at_fpr": float("nan"),
                              "far": float("nan"), "model_kind": None, "folds": []}
            continue
        summary[label] = {
            "auc": float(np.nanmean([r["auc"] for r in rows])),
            "tpr_at_fpr": float(np.nanmean([r["tpr_at_fpr"] for r in rows])),
            "far": float(np.nanmean([r["far"] for r in rows])),
            "model_kind": rows[0]["model_kind"],
            "folds": rows,
        }
    return summary
