"""Prototype retrieval: pick one bank entry to guide a perturbation.

All distances are standardized Euclidean in feature space: both the prototype
features and the query feature are transformed by the same `FeatureScaler`
fit on bank members. Ties are broken deterministically by the lowest entry
index, so a seeded run is reproducible regardless of platform sort behavior.

Modes
-----
- ``random``: a uniform entry among those of the target symbol.
- ``medoid``: the entry with the smallest sum of distances to the other entries
  of that symbol; cached per (bank identity, symbol) because it is O(n^2).
- ``nearest``: smallest distance to the query among target-symbol entries.
- ``wrong_letter``: smallest distance to the query among entries whose symbol
  differs from the target (the causal control).

The query feature is the target-conditioned ERP of the SOURCE trial at the
ATTACKER target flashes, never at the trial's true symbol.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import numpy as np

MODES = ("random", "medoid", "nearest", "wrong_letter")

# Medoid sums depend on the bank features and scaler only, not on the query or
# rng. Keys are (content fingerprint, symbol) so two banks cannot collide and
# editing a bank cannot serve a stale medoid.
_MEDOID_CACHE: dict[tuple, int] = {}


def bank_fingerprint(bank) -> str:
    """Content fingerprint so a medoid cache can never outlive its bank.

    Includes trial identity and feature bytes, so appending an entry or
    replacing a feature changes the fingerprint and invalidates cached medoids
    instead of returning a stale index.
    """
    hasher = hashlib.sha256()
    for entry in bank.entries:
        hasher.update(entry.trial_id.encode())
        hasher.update(str(entry.symbol_idx).encode())
        hasher.update(entry.subject_id.encode())
        hasher.update(entry.dataset.encode())
        hasher.update(entry.split.encode())
        hasher.update(np.ascontiguousarray(entry.feature).tobytes())
    return hasher.hexdigest()[:16]


def scaler_fingerprint(scaler) -> str:
    """Fingerprint of the standardization stats, or a sentinel when unscaled.

    Medoids are computed on standardized features, so the cache key must change
    when the scaler changes even if the bank bytes do not.
    """
    if scaler is None:
        return "raw"
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(scaler.mean, dtype=np.float32).tobytes())
    hasher.update(np.ascontiguousarray(scaler.std, dtype=np.float32).tobytes())
    return hasher.hexdigest()[:16]


class RetrievalResult(NamedTuple):
    """The chosen entry and its standardized Euclidean distance to the query.

    As a named tuple, the result supports both attribute access and the
    ``index, distance = retrieve(...)`` API specified by the experiment plan.
    """

    index: int
    distance: float


def _sq_dists(features: np.ndarray, query: np.ndarray) -> np.ndarray:
    diff = features - query[None]
    return np.einsum("nct,nct->n", diff, diff)


def _pairwise_sq_dists(features: np.ndarray) -> np.ndarray:
    """Squared distance between every pair of feature rows [N, N]."""
    diff = features[:, None] - features[None, :]
    return np.einsum("ijct,ijct->ij", diff, diff)


def _argmin_first(values: np.ndarray) -> int:
    """Lowest index attaining the minimum, deterministic under ties."""
    return int(np.argmin(values))


def medoid_index(bank, symbol_idx: int, scaler=None) -> int | None:
    """Bank index of the medoid of `symbol_idx`, or None if absent.

    The medoid is the entry minimizing the sum of standardized Euclidean
    distances to the other entries of the same symbol. The result is cached by
    content fingerprint, so an unchanged bank re-computes nothing while an
    edited bank cannot return a stale index.
    """
    idx = [i for i, e in enumerate(bank.entries) if e.symbol_idx == symbol_idx]
    if not idx:
        return None
    if scaler is None:
        feats = np.stack([bank.entries[i].feature for i in idx], axis=0).astype(np.float32)
    else:
        feats = np.stack([scaler.transform(bank.entries[i].feature) for i in idx], axis=0)
    key = (bank_fingerprint(bank), scaler_fingerprint(scaler), int(symbol_idx))
    cached = _MEDOID_CACHE.get(key)
    if cached is not None and cached in idx:
        return cached
    distances = np.sqrt(_pairwise_sq_dists(feats))
    sums = distances.sum(axis=1)
    chosen = idx[_argmin_first(sums)]
    _MEDOID_CACHE[key] = chosen
    return chosen


def retrieve(bank, scaler, query_feature, target_idx, mode, rng=None):
    """Return a `RetrievalResult`, or None when the relevant pool is empty.

    The pool is the target-symbol entries for ``random`` / ``medoid`` /
    ``nearest``, and every entry of a different symbol for ``wrong_letter``.
    None is returned only when that pool is empty.
    """
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    target_idx = int(target_idx)
    if not (0 <= target_idx < bank.n_symbols):
        raise ValueError(f"target_idx {target_idx} outside bank symbols")

    target_pool = [i for i, e in enumerate(bank.entries) if e.symbol_idx == target_idx]
    if not target_pool:
        return None
    if mode == "wrong_letter":
        pool = [i for i, e in enumerate(bank.entries) if e.symbol_idx != target_idx]
        if not pool:
            return None
    else:
        pool = target_pool

    if mode == "random":
        if rng is None:
            raise ValueError("random retrieval requires an rng")
        if query_feature is None:
            raise ValueError("random retrieval requires a query feature for distance")
        chosen = pool[int(rng.integers(len(pool)))]
        query = np.asarray(scaler.transform(query_feature), dtype=np.float32)
        feature = np.asarray(
            scaler.transform(bank.entries[chosen].feature), dtype=np.float32
        )
        distance = float(np.sqrt(_sq_dists(feature[None], query)[0]))
        return RetrievalResult(chosen, distance)

    query = np.asarray(scaler.transform(query_feature), dtype=np.float32)

    if mode == "medoid":
        # medoid_index returns a bank index (always an entry of target_idx).
        feats = np.stack([scaler.transform(e.feature) for e in bank.entries], axis=0)
        chosen = medoid_index(bank, target_idx, scaler)
        dist = float(np.sqrt(_sq_dists(feats[chosen:chosen + 1], query)[0]))
        return RetrievalResult(chosen, dist)

    feats = np.stack([scaler.transform(bank.entries[i].feature) for i in pool], axis=0)
    sq = _sq_dists(feats, query)
    local = _argmin_first(sq)
    return RetrievalResult(pool[local], float(np.sqrt(sq[local])))
