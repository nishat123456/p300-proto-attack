"""Prototype bank: genuine target-conditioned ERPs with strict leakage guards.

One entry per genuine character trial. The stored feature is the
target-conditioned ERP of the trial's *true* symbol, computed with a trained
Speller's preprocessing and epoch window. Two modes exist and both are enforced
at build time and again at load time:

- ``same_subject``: only the attacked subject, and only trials in the frozen
  train portion of the official train split (``splits/comp3_X.json``). The val
  and test portions of that subject are excluded, as is the entire test file.
- ``cross_subject``: every trial (train and test) of the *other* subjects of the
  same dataset. Any entry whose subject equals the attacked subject is refused.

The bank carries its mode metadata into the HDF5 file so exclusion rules are
re-checked on load rather than trusted from the file contents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from p3a.prototypes.features import target_conditioned_erp

MODES = ("same_subject", "cross_subject")

# Metadata keys written to HDF5 so load can re-validate without guessing.
_META_KEYS = (
    "bank_format", "mode", "attacked_subject", "dataset", "layout_id",
    "n_symbols", "n_channels", "epoch_len", "train_split_only", "feature_kind",
)


def _build_entries(trials, speller, device: str):
    """Yield a BankEntry for each eligible trial using the true-symbol ERP."""
    was_training = speller.training
    speller.eval()
    try:
        with torch.no_grad():
            for trial in trials:
                eeg = torch.as_tensor(trial.eeg[None], dtype=torch.float32, device=device)
                onsets = torch.as_tensor(trial.onsets[None], dtype=torch.long, device=device)
                groups = torch.as_tensor(trial.flash_groups, dtype=torch.bool, device=device)
                epochs = speller.epochs(eeg, onsets)[0]  # [F, C, L]
                symbol_idx = trial.layout.index(trial.target_symbol)
                feature = target_conditioned_erp(epochs, groups, symbol_idx)
                yield BankEntry(
                    feature.detach().cpu().numpy(), symbol_idx,
                    trial.subject_id, trial.dataset, trial.trial_id, trial.source_split,
                )
    finally:
        if was_training:
            speller.train()


@dataclass
class BankEntry:
    """One stored prototype: a true-symbol target ERP plus provenance."""

    feature: np.ndarray  # [C, L] float32
    symbol_idx: int
    subject_id: str
    dataset: str
    trial_id: str
    split: str

    def __post_init__(self):
        self.feature = np.ascontiguousarray(self.feature, dtype=np.float32)
        self.symbol_idx = int(self.symbol_idx)
        self.subject_id = str(self.subject_id)
        self.dataset = str(self.dataset)
        self.trial_id = str(self.trial_id)
        self.split = str(self.split)
        if self.feature.ndim != 2:
            raise ValueError("entry feature must be [C, L]")


@dataclass
class PrototypeBank:
    """A set of genuine target ERPs gated by an explicit leakage policy.

    Parameters
    ----------
    mode : "same_subject" or "cross_subject".
    attacked_subject : subject the attack targets; never allowed in the bank
        when mode is cross_subject.
    dataset : dataset identifier all entries must share.
    allowed_subjects : subjects the bank may contain.
    allowed_splits : source splits the bank may contain ("train"/"test").
    allowed_trial_ids : optional allow-list of trial ids (used by same_subject
        to pin the frozen train portion). None means any id within the other
        constraints.
    """

    mode: str
    attacked_subject: str
    dataset: str
    allowed_subjects: frozenset
    allowed_splits: frozenset
    allowed_trial_ids: frozenset | None = None
    entries: list[BankEntry] = field(default_factory=list)
    layout_id: str = ""
    n_symbols: int = 0
    n_channels: int = 0
    epoch_len: int = 0
    feature_kind: str = "target_conditioned_erp"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown bank mode {self.mode!r}")
        self.attacked_subject = str(self.attacked_subject)
        self.allowed_subjects = frozenset(str(s) for s in self.allowed_subjects)
        self.allowed_splits = frozenset(str(s) for s in self.allowed_splits)
        if self.allowed_trial_ids is not None:
            self.allowed_trial_ids = frozenset(self.allowed_trial_ids)
        if self.mode == "cross_subject" and self.attacked_subject in self.allowed_subjects:
            raise ValueError("cross_subject bank cannot include the attacked subject")
        if self.mode == "same_subject" and self.allowed_subjects != {self.attacked_subject}:
            # same_subject is by definition one subject; refuse a subject list
            # that does not name exactly the attacked subject.
            raise ValueError("same_subject bank must name exactly the attacked subject")

    # -- build -------------------------------------------------------------

    @classmethod
    def build(
        cls,
        trials,
        speller,
        subject_ids,
        allowed_splits,
        *,
        mode: str,
        attacked_subject: str,
        allowed_trial_ids=None,
        device: str = "cpu",
    ) -> "PrototypeBank":
        """Compute features for eligible genuine trials and freeze the policy.

        Eligibility is the intersection of the requested subjects, the allowed
        source splits, and (when given) the frozen trial-id allow-list. Any
        eligible trial that violates the bank's mode is refused with an error
        instead of being silently dropped, so a caller mistake cannot produce a
        leaking bank.
        """
        if mode not in MODES:
            raise ValueError(f"unknown bank mode {mode!r}")
        subject_ids = {str(s) for s in subject_ids}
        allowed_splits = {str(s) for s in allowed_splits}
        allowed_trial_ids = (
            None if allowed_trial_ids is None else {str(t) for t in allowed_trial_ids}
        )
        attacked_subject = str(attacked_subject)

        if mode == "cross_subject":
            if attacked_subject in subject_ids:
                raise ValueError(
                    "refusing to build a cross_subject bank that includes the "
                    f"attacked subject {attacked_subject!r}"
                )
        if mode == "same_subject" and subject_ids != {attacked_subject}:
            raise ValueError(
                "same_subject bank must be built from exactly the attacked subject"
            )

        eligible = []
        for trial in trials:
            if trial.subject_id not in subject_ids:
                continue
            if trial.source_split not in allowed_splits:
                continue
            if allowed_trial_ids is not None and trial.trial_id not in allowed_trial_ids:
                continue
            if mode == "cross_subject" and trial.subject_id == attacked_subject:
                raise ValueError(
                    f"refusing attacked-subject trial {trial.trial_id!r} in a "
                    "cross_subject bank"
                )
            if mode == "same_subject" and trial.source_split == "test":
                raise ValueError("same_subject bank must never use test trials")
            eligible.append(trial)

        if not eligible:
            raise ValueError("no trials matched the bank's eligibility filter")
        datasets = {t.dataset for t in eligible}
        if len(datasets) != 1:
            raise ValueError(f"bank trials span multiple datasets: {sorted(datasets)}")
        dataset = datasets.pop()
        layout_ids = {t.layout.layout_id for t in eligible}
        if len(layout_ids) != 1:
            raise ValueError("bank trials span multiple layouts")
        n_symbols = {t.layout.n_symbols for t in eligible}
        if len(n_symbols) != 1:
            raise ValueError("bank trials span multiple symbol counts")
        n_symbols = n_symbols.pop()

        bank = cls(
            mode=mode,
            attacked_subject=attacked_subject,
            dataset=dataset,
            allowed_subjects=subject_ids,
            allowed_splits=allowed_splits,
            allowed_trial_ids=allowed_trial_ids,
            layout_id=layout_ids.pop(),
            n_symbols=n_symbols,
        )
        bank.entries = list(_build_entries(eligible, speller, device))
        bank._infer_shape()
        bank.validate()
        return bank

    # -- queries -----------------------------------------------------------

    def add_entry(self, entry: BankEntry) -> None:
        """Append a precomputed entry, enforcing the bank's policy immediately."""
        if self.mode == "cross_subject" and entry.subject_id == self.attacked_subject:
            raise AssertionError("cross_subject bank refuses attacked-subject entries")
        if entry.subject_id not in self.allowed_subjects:
            raise ValueError(f"entry subject {entry.subject_id!r} not allowed")
        if entry.split not in self.allowed_splits:
            raise ValueError(f"entry split {entry.split!r} not allowed")
        if self.allowed_trial_ids is not None and entry.trial_id not in self.allowed_trial_ids:
            raise ValueError(f"entry {entry.trial_id!r} not in the frozen allow-list")
        self.entries.append(entry)
        self._infer_shape()

    def symbols_available(self) -> list[int]:
        """Sorted symbol indices with at least one prototype."""
        return sorted({e.symbol_idx for e in self.entries})

    def entries_for(self, symbol_idx: int) -> list[BankEntry]:
        symbol_idx = int(symbol_idx)
        return [e for e in self.entries if e.symbol_idx == symbol_idx]

    def counts_per_symbol(self) -> np.ndarray:
        """Vector of length n_symbols with the prototype count per symbol."""
        if self.n_symbols <= 0:
            raise ValueError("bank has no symbol metadata")
        counts = np.zeros(self.n_symbols, dtype=np.int64)
        for entry in self.entries:
            counts[entry.symbol_idx] += 1
        return counts

    def __len__(self) -> int:
        return len(self.entries)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Re-check every invariant; run after build and after load."""
        if self.mode not in MODES:
            raise ValueError(f"unknown bank mode {self.mode!r}")
        if self.mode == "cross_subject" and self.attacked_subject in self.allowed_subjects:
            raise ValueError("cross_subject bank includes the attacked subject")
        if self.mode == "same_subject" and self.allowed_subjects != {self.attacked_subject}:
            raise ValueError("same_subject bank must name exactly the attacked subject")
        seen_ids = set()
        symbol_indices = set()
        shape = None
        for entry in self.entries:
            # Mode-specific exclusion checks come first so a tampered file
            # reports the leakage it attempted, not a generic filter error.
            if entry.subject_id == self.attacked_subject and self.mode == "cross_subject":
                raise AssertionError(
                    "cross_subject bank contains an attacked-subject entry"
                )
            if self.mode == "same_subject" and entry.subject_id != self.attacked_subject:
                raise AssertionError("same_subject bank contains another subject")
            if self.mode == "same_subject" and entry.split == "test":
                raise AssertionError("same_subject bank contains a test entry")
            if entry.subject_id not in self.allowed_subjects:
                raise ValueError(
                    f"entry subject {entry.subject_id!r} not in allowed subjects"
                )
            if entry.split not in self.allowed_splits:
                raise ValueError(f"entry split {entry.split!r} not allowed")
            if self.allowed_trial_ids is not None and entry.trial_id not in self.allowed_trial_ids:
                raise ValueError(f"entry {entry.trial_id!r} not in the frozen allow-list")
            if entry.dataset != self.dataset:
                raise ValueError(
                    f"entry dataset {entry.dataset!r} != bank dataset {self.dataset!r}"
                )
            if not (0 <= entry.symbol_idx < self.n_symbols):
                raise ValueError(f"entry symbol {entry.symbol_idx} out of range")
            if not np.isfinite(entry.feature).all():
                raise ValueError(f"entry {entry.trial_id!r} has non-finite values")
            if entry.trial_id in seen_ids:
                raise ValueError(f"duplicate trial identity {entry.trial_id!r}")
            seen_ids.add(entry.trial_id)
            symbol_indices.add(entry.symbol_idx)
            if shape is None:
                shape = entry.feature.shape
            elif entry.feature.shape != shape:
                raise ValueError("entry feature shapes are inconsistent")
            if entry.feature.dtype != np.float32:
                raise ValueError("entry feature must be float32")

    def _infer_shape(self) -> None:
        if not self.entries:
            return
        c, length = self.entries[0].feature.shape
        self.n_channels = int(c)
        self.epoch_len = int(length)

    # -- persistence -------------------------------------------------------

    def save(self, path) -> Path:
        """Write entries and mode metadata to HDF5."""
        import h5py

        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.entries)
        with h5py.File(path, "w") as f:
            f.attrs["bank_format"] = 1
            f.attrs["mode"] = self.mode
            f.attrs["attacked_subject"] = self.attacked_subject
            f.attrs["dataset"] = self.dataset
            f.attrs["layout_id"] = self.layout_id
            f.attrs["n_symbols"] = self.n_symbols
            f.attrs["n_channels"] = self.n_channels
            f.attrs["epoch_len"] = self.epoch_len
            f.attrs["train_split_only"] = self.mode == "same_subject"
            f.attrs["feature_kind"] = self.feature_kind
            f.attrs["allowed_subjects"] = np.array(sorted(self.allowed_subjects), dtype=h5py.string_dtype())
            f.attrs["allowed_splits"] = np.array(sorted(self.allowed_splits), dtype=h5py.string_dtype())
            if self.allowed_trial_ids is None:
                # h5py cannot store None; a sentinel empty array plus a flag.
                f.attrs["has_trial_allowlist"] = False
                f.attrs["allowed_trial_ids"] = np.array([], dtype=h5py.string_dtype())
            else:
                f.attrs["has_trial_allowlist"] = True
                f.attrs["allowed_trial_ids"] = np.array(
                    sorted(self.allowed_trial_ids), dtype=h5py.string_dtype()
                )
            f.create_dataset(
                "features", data=np.stack([e.feature for e in self.entries]) if n else np.zeros((0, 0, 0), np.float32)
            )
            f.create_dataset("symbol_idx", data=np.array([e.symbol_idx for e in self.entries], dtype=np.int64))
            f.create_dataset("subject_id", data=np.array([e.subject_id for e in self.entries], dtype=h5py.string_dtype()))
            f.create_dataset("dataset", data=np.array([e.dataset for e in self.entries], dtype=h5py.string_dtype()))
            f.create_dataset("trial_id", data=np.array([e.trial_id for e in self.entries], dtype=h5py.string_dtype()))
            f.create_dataset("split", data=np.array([e.split for e in self.entries], dtype=h5py.string_dtype()))
        return path

    @classmethod
    def load(cls, path) -> "PrototypeBank":
        """Rebuild and re-validate a bank from HDF5."""
        import h5py

        def _str(value):
            if isinstance(value, bytes):
                return value.decode()
            return str(value)

        with h5py.File(path, "r") as f:
            if int(f.attrs.get("bank_format", -1)) != 1:
                raise ValueError("unsupported bank file format")
            missing = [k for k in _META_KEYS if k not in f.attrs]
            if missing:
                raise ValueError(f"bank file missing metadata {missing}")
            mode = _str(f.attrs["mode"])
            attacked_subject = _str(f.attrs["attacked_subject"])
            dataset = _str(f.attrs["dataset"])
            allowed_subjects = frozenset(_str(s) for s in f.attrs["allowed_subjects"].tolist())
            allowed_splits = frozenset(_str(s) for s in f.attrs["allowed_splits"].tolist())
            has_allowlist = bool(f.attrs.get("has_trial_allowlist", False))
            allowed_trial_ids = None
            if has_allowlist:
                allowed_trial_ids = frozenset(
                    _str(s) for s in f.attrs["allowed_trial_ids"].tolist()
                )
            features = np.asarray(f["features"][()], dtype=np.float32)
            symbol_idx = np.asarray(f["symbol_idx"][()], dtype=np.int64)
            subject_id = [_str(s) for s in f["subject_id"][()].tolist()]
            datasets = [_str(s) for s in f["dataset"][()].tolist()]
            trial_id = [_str(s) for s in f["trial_id"][()].tolist()]
            split = [_str(s) for s in f["split"][()].tolist()]
            bank = cls(
                mode=mode,
                attacked_subject=attacked_subject,
                dataset=dataset,
                allowed_subjects=allowed_subjects,
                allowed_splits=allowed_splits,
                allowed_trial_ids=allowed_trial_ids,
                layout_id=_str(f.attrs["layout_id"]),
                n_symbols=int(f.attrs["n_symbols"]),
                n_channels=int(f.attrs["n_channels"]),
                epoch_len=int(f.attrs["epoch_len"]),
                feature_kind=_str(f.attrs["feature_kind"]),
            )
            n = len(symbol_idx)
            if not (features.shape[0] == n and len(subject_id) == n
                    and len(datasets) == n and len(trial_id) == n and len(split) == n):
                raise ValueError("bank fields have mismatched lengths")
            if n and (features.ndim != 3
                      or features.shape[1] != bank.n_channels
                      or features.shape[2] != bank.epoch_len):
                raise ValueError("bank feature array does not match stored shape metadata")
            bank.entries = [
                BankEntry(features[i], int(symbol_idx[i]), subject_id[i], datasets[i],
                          trial_id[i], split[i])
                for i in range(n)
            ]
        bank.validate()
        return bank
