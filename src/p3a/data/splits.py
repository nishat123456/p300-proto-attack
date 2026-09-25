"""Frozen splits. Written once to splits/*.json and never regenerated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def make_train_val(trial_ids: list[str], n_val: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trial_ids))
    val = sorted(trial_ids[i] for i in order[:n_val])
    train = sorted(trial_ids[i] for i in order[n_val:])
    return {"train": train, "val": val}


def split_hash(split: dict) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:16]


def write_once(path: str | Path, split: dict) -> str:
    """Write a split file; refuse to overwrite a different existing split."""
    path = Path(path)
    payload = {"split": split, "hash": split_hash(split)}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["hash"] != payload["hash"]:
            raise RuntimeError(f"{path} exists with a different split; splits are frozen")
        return existing["hash"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload["hash"]


def load(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    if split_hash(payload["split"]) != payload["hash"]:
        raise RuntimeError(f"{path} hash mismatch; split file was edited")
    return payload["split"]
