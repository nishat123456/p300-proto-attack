"""Load the Prompt 10 evaluation outputs for the detectors.

The detectors consume exactly what ``scripts/evaluate_attacks.py`` wrote: one
clean NPZ per subject, one attacked NPZ per (method, generator, bank, epsilon),
and a single ``attacks.parquet`` with one row per (method, generator, bank, eps,
trial, target). This module turns those files into aligned arrays and a
per-sample manifest, so the detector modules do not each re-implement the file
layout.

The key alignment guarantee is that an attacked decision ERP at index (t, k)
corresponds to the clean duplicate at the same (t, k), and the clean genuine ERP
at trial t is ``clean.true_erp[t]``. The manifest carries the trial index (for
the parity folds), the trial id, the attacker target, and whether the surrogate
attack succeeded (the induced subset the defense trains and is scored on).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from p3a.eval.store import (
    ArtifactSpec,
    AttackArtifacts,
    CleanArtifacts,
    attack_npz_path,
    clean_npz_path,
)

__all__ = [
    "AttackInputs",
    "ComboInputs",
    "available_combos",
    "clean_path",
    "combine_samples",
    "load_clean",
    "load_combo",
    "load_manifest",
    "result_dir",
]


def result_dir(results_root, subject: str) -> Path:
    """The per-subject output directory written by evaluate_attacks."""
    return Path(results_root) / "attacks_eval" / str(subject).upper()


def clean_path(results_root, subject: str) -> Path:
    return clean_npz_path(result_dir(results_root, subject), subject)


@dataclass
class ComboInputs:
    """One (method, generator, bank, eps) combination of attacked artifacts."""

    spec: ArtifactSpec
    clean: CleanArtifacts
    attack: AttackArtifacts
    eps_raw: float = float("nan")
    frame: pd.DataFrame | None = None

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def eps_multiplier(self) -> float:
        return float(self.spec.eps_multiplier)

    @property
    def n_trials(self) -> int:
        return int(self.attack.decision_erp.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.attack.decision_erp.shape[1])

    def flat_decision_erp(self) -> np.ndarray:
        """[N, C, L] attacked decision ERPs flattened over (trial, target)."""
        art = self.attack.decision_erp
        return art.reshape(self.n_trials * self.n_targets, *art.shape[2:])

    def flat_clean_decision_erp(self) -> np.ndarray:
        """[N, C, L] clean decision ERPs at the attacker target, same order."""
        erp = self.clean.target_erp
        return erp.reshape(self.n_trials * self.n_targets, *erp.shape[2:])

    def flat_windows(self) -> np.ndarray:
        """[N, n_windows, C, W] attacked windows flattened over (trial, target)."""
        wins = self.attack.windows
        return wins.reshape(self.n_trials * self.n_targets, *wins.shape[2:])

    def flat_clean_windows(self) -> np.ndarray:
        """[N, n_windows, C, W] clean windows tiled over targets, same order."""
        tiled = np.repeat(self.clean.windows[:, None], self.n_targets, axis=1)
        return tiled.reshape(self.n_trials * self.n_targets, *self.clean.windows.shape[1:])

    def success_mask(self, models=("surrogate",)) -> np.ndarray:
        """[N] bool: every listed model chose the target on that pair.

        Reads this combination's rows from the manifest; without them it assumes
        success, which is the correct default for the induced set when a caller
        only has the NPZ files.
        """
        n = self.n_trials * self.n_targets
        if self.frame is None or self.frame.empty:
            return np.ones(n, dtype=bool)
        rows = self.rows_frame()
        mask = np.ones(len(rows), dtype=bool)
        for model in models:
            column = f"success_{model}_15"
            if column in rows.columns:
                mask &= rows[column].to_numpy() >= 0.5
        if len(mask) != n:
            raise ValueError("manifest rows do not align with the artifact arrays")
        return mask

    def rows_frame(self) -> pd.DataFrame:
        """This combination's parquet rows aligned to the NPZ pair order."""
        if self.frame is None or self.frame.empty:
            return pd.DataFrame()
        frame = self.frame
        rows = frame[
            (frame["method"] == self.spec.method)
            & (frame["generator"] == self.spec.generator)
            & (frame["bank"] == self.spec.bank)
            & np.isclose(frame["eps_multiplier"], self.spec.eps_multiplier)
        ]
        if rows.empty:
            return rows
        keys = ["trial_index", "target_idx"]
        if rows.duplicated(keys).any():
            raise ValueError("duplicate (trial_index, target_idx) rows in manifest")
        expected = pd.MultiIndex.from_arrays([
            np.repeat(self.attack.trial_index, self.n_targets),
            self.attack.target_idx.reshape(-1),
        ], names=keys)
        indexed = rows.set_index(keys)
        missing = expected.difference(indexed.index)
        if len(missing):
            raise ValueError("manifest rows do not align with the artifact arrays")
        return indexed.loc[expected].reset_index()

    def sample_manifest(self) -> pd.DataFrame:
        """Per-(trial, target) manifest aligned with the flattened arrays."""
        frame = self.rows_frame()
        if not frame.empty:
            base = frame[["trial_index", "trial_id", "target_idx", "target_symbol"]].copy()
            base = base.reset_index(drop=True)
        else:
            base = pd.DataFrame({
                "trial_index": np.repeat(self.attack.trial_index, self.n_targets),
                "trial_id": np.repeat(self.attack.trial_id, self.n_targets),
                "target_idx": self.attack.target_idx.reshape(-1),
                "target_symbol": self.attack.target_symbol.reshape(-1),
            })
        base["success"] = self.success_mask()
        base["label"] = self.label
        base["eps_multiplier"] = self.eps_multiplier
        return base


@dataclass
class AttackInputs:
    """Aligned arrays and metadata for one detector evaluation.

    ``attacked_erp`` and ``clean_duplicate_erp`` are [N, C, L]; ``genuine_erp``
    is the clean ERP at the true symbol for each sample's trial, expanded over
    the attacker targets; ``windows`` and ``clean_windows`` are [N, n_windows,
    C, W]. ``success`` flags the pairs where the surrogate chose the attacker
    target, the induced subset the defense uses.
    """

    trial_index: np.ndarray
    trial_id: np.ndarray
    true_idx: np.ndarray
    target_idx: np.ndarray
    target_symbol: np.ndarray
    attacked_erp: np.ndarray
    clean_duplicate_erp: np.ndarray
    genuine_erp: np.ndarray
    windows: np.ndarray
    clean_windows: np.ndarray
    success: np.ndarray
    meta: dict = field(default_factory=dict)


def load_manifest(results_root, subject: str) -> pd.DataFrame:
    """Read attacks.parquet for one subject, or an empty frame when absent."""
    path = result_dir(results_root, subject) / "attacks.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_clean(results_root, subject: str) -> CleanArtifacts:
    return CleanArtifacts.load(clean_path(results_root, subject))


def listed_specs(manifest: pd.DataFrame) -> list[ArtifactSpec]:
    """Unique (method, generator, bank, eps) specs present in the manifest."""
    if manifest.empty:
        return []
    keys = manifest[["method", "generator", "bank", "eps_multiplier"]].drop_duplicates()
    subject = str(manifest["subject"].iloc[0])
    return [
        ArtifactSpec(subject=subject, method=str(row["method"]),
                     generator=str(row["generator"]), bank=str(row["bank"]),
                     eps_multiplier=float(row["eps_multiplier"]))
        for _, row in keys.iterrows()
    ]


def available_combos(results_root, subject: str, eps_multipliers=None) -> list[ArtifactSpec]:
    """Specs whose attacker NPZ exists on disk, optionally filtered by eps."""
    directory = result_dir(results_root, subject)
    keep = []
    wanted = None if eps_multipliers is None else {float(m) for m in eps_multipliers}
    for spec in listed_specs(load_manifest(results_root, subject)):
        if wanted is not None and float(spec.eps_multiplier) not in wanted:
            continue
        if attack_npz_path(directory, spec).is_file():
            keep.append(spec)
    return keep


def load_combo(results_root, subject: str, spec: ArtifactSpec) -> ComboInputs:
    """Load one attacked combination and its aligned clean references."""
    directory = result_dir(results_root, subject)
    clean = load_clean(results_root, subject)
    attack = AttackArtifacts.load(attack_npz_path(directory, spec))
    eps_raw = _eps_raw_from_manifest(results_root, subject, spec)
    return ComboInputs(spec=spec, clean=clean, attack=attack, eps_raw=eps_raw,
                       frame=load_manifest(results_root, subject))


def _eps_raw_from_manifest(results_root, subject: str, spec: ArtifactSpec) -> float:
    manifest = load_manifest(results_root, subject)
    if manifest.empty:
        return float("nan")
    match = manifest[
        (manifest["method"] == spec.method)
        & (manifest["generator"] == spec.generator)
        & (manifest["bank"] == spec.bank)
        & np.isclose(manifest["eps_multiplier"], spec.eps_multiplier)
    ]
    if match.empty:
        return float("nan")
    return float(match["eps_raw"].iloc[0])


def combo_to_inputs(combo: ComboInputs) -> AttackInputs:
    """Build the aligned ``AttackInputs`` bundle for one combination."""
    n_trials, n_targets = combo.n_trials, combo.n_targets
    true_erp = combo.clean.true_erp  # [T, C, L]
    genuine = np.repeat(true_erp[:, None], n_targets, axis=1).reshape(
        n_trials * n_targets, *true_erp.shape[1:]
    )
    return AttackInputs(
        trial_index=np.repeat(combo.attack.trial_index, n_targets),
        trial_id=np.repeat(combo.attack.trial_id, n_targets),
        true_idx=np.repeat(combo.attack.true_idx, n_targets),
        target_idx=combo.attack.target_idx.reshape(-1),
        target_symbol=combo.attack.target_symbol.reshape(-1),
        attacked_erp=combo.flat_decision_erp(),
        clean_duplicate_erp=combo.flat_clean_decision_erp(),
        genuine_erp=genuine,
        windows=combo.flat_windows(),
        clean_windows=combo.flat_clean_windows(),
        success=combo.success_mask(),
        meta={"label": combo.label, "eps_multiplier": combo.eps_multiplier,
              "eps_raw": combo.eps_raw},
    )


def combine_samples(chunks: list[AttackInputs]) -> AttackInputs:
    """Concatenate aligned bundles along the sample axis (training pools)."""
    if not chunks:
        raise ValueError("combine_samples needs at least one chunk")

    def cat(key):
        return np.concatenate([getattr(c, key) for c in chunks], axis=0)

    return AttackInputs(
        trial_index=cat("trial_index"), trial_id=cat("trial_id"),
        true_idx=cat("true_idx"),
        target_idx=cat("target_idx"), target_symbol=cat("target_symbol"),
        attacked_erp=cat("attacked_erp"), clean_duplicate_erp=cat("clean_duplicate_erp"),
        genuine_erp=cat("genuine_erp"), windows=cat("windows"),
        clean_windows=cat("clean_windows"), success=cat("success"),
        meta={"labels": [c.meta.get("label") for c in chunks]},
    )
