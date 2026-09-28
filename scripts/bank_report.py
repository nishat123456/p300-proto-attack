"""Build same-subject and cross-subject prototype banks and report their stats.

For each Comp III subject S:

- same_subject bank: genuine target ERPs from the frozen *train* portion of S's
  official train split only. The val and test portions of S, and the whole test
  file, are never loaded or used.
- cross_subject bank: all trials (train and test) of the *other* subject(s) of
  the same dataset. The attacked subject is refused by the bank itself.

Outputs, next to the run: a CSV of per-symbol counts and nearest-neighbour
distance summaries, and a full config/provenance JSON sidecar. The script
supports a no-data `--dry-run` that validates the frozen splits and prints the
build plan without touching raw files; a real run on a machine missing B or the
test .mat files stops with a clear missing-data message instead of guessing.

Local machine note: only ``data/raw/comp3/Subject_A_Train.mat`` exists, so a
real full report needs the Kaggle data. Use ``--dry-run`` locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.data.loaders.comp3 import find_file, load_subject  # noqa: E402
from p3a.data.splits import load as load_split  # noqa: E402
from p3a.models.speller import Speller  # noqa: E402
from p3a.prototypes.bank import PrototypeBank  # noqa: E402
from p3a.prototypes.features import FeatureScaler  # noqa: E402
from p3a.utils.repro import environment, set_seed, write_config  # noqa: E402

DEFAULT_SUBJECTS = ("A", "B")


def _split_path(subject: str) -> Path:
    return REPO / f"splits/comp3_{subject}.json"


def _required_files(data_root: Path, subject: str, labels_dir: Path) -> list[Path]:
    """Raw files needed to build every bank for one attacked subject."""
    needed = [data_root / f"Subject_{subject}_Train.mat"]
    needed.append(labels_dir / f"true_labels_{subject.lower()}.txt")
    for other in DEFAULT_SUBJECTS:
        if other == subject:
            continue
        needed.append(data_root / f"Subject_{other}_Train.mat")
        needed.append(data_root / f"Subject_{other}_Test.mat")
        needed.append(labels_dir / f"true_labels_{other.lower()}.txt")
    return needed


def missing_data(data_root: Path, labels_dir: Path, subjects) -> list[str]:
    missing = []
    for subject in subjects:
        for path in _required_files(data_root, subject, labels_dir):
            try:
                find_file(path.parent, path.name)
            except FileNotFoundError:
                missing.append(str(path))
    return missing


def select_by_ids(trials: list, ids: set[str]) -> list:
    by_id = {t.trial_id: t for t in trials}
    missing = sorted(i for i in ids if i not in by_id)
    if missing:
        raise KeyError(f"frozen split ids not found among loaded trials: {missing[:3]}")
    return [by_id[i] for i in sorted(ids)]


def build_same_subject(subject: str, data_root: Path, labels_dir: Path,
                       speller: Speller, device: str) -> PrototypeBank:
    split = load_split(_split_path(subject))
    train_ids = set(split["train"])
    where = "cpu" if device == "cpu" else device
    train_trials = load_subject(data_root, subject, "train", labels_dir=labels_dir)
    trials = select_by_ids(train_trials, train_ids)
    return PrototypeBank.build(
        trials, speller, {subject}, {"train"},
        mode="same_subject", attacked_subject=subject,
        allowed_trial_ids=train_ids, device=where,
    )


def build_cross_subject(subject: str, data_root: Path, labels_dir: Path,
                        speller: Speller, device: str) -> PrototypeBank:
    others = [s for s in DEFAULT_SUBJECTS if s != subject]
    trials = []
    for other in others:
        for split in ("train", "test"):
            trials.extend(load_subject(data_root, other, split, labels_dir=labels_dir))
    where = "cpu" if device == "cpu" else device
    return PrototypeBank.build(
        trials, speller, set(others), {"train", "test"},
        mode="cross_subject", attacked_subject=subject, device=where,
    )


def nearest_distance_summary(bank: PrototypeBank, scaler: FeatureScaler) -> pd.DataFrame:
    """Per-symbol counts and per-entry nearest same-symbol distance stats.

    For each entry, the nearest distance is the smallest standardized Euclidean
    distance to a *different* entry of the same symbol. Symbols with fewer than
    two entries have no nearest distance and report NaN.
    """
    rows = []
    for symbol_idx in range(bank.n_symbols):
        entries = bank.entries_for(symbol_idx)
        if len(entries) >= 2:
            feats = np.stack([scaler.transform(e.feature) for e in entries], axis=0)
            diff = feats[:, None] - feats[None, :]
            sq = np.einsum("ijct,ijct->ij", diff, diff)
            np.fill_diagonal(sq, np.inf)
            nearest = np.sqrt(sq.min(axis=1))
            rows.append({
                "symbol_idx": symbol_idx,
                "count": len(entries),
                "nearest_mean": float(nearest.mean()),
                "nearest_median": float(np.median(nearest)),
                "nearest_min": float(nearest.min()),
                "nearest_max": float(nearest.max()),
            })
        else:
            rows.append({
                "symbol_idx": symbol_idx, "count": len(entries),
                "nearest_mean": float("nan"), "nearest_median": float("nan"),
                "nearest_min": float("nan"), "nearest_max": float("nan"),
            })
    frame = pd.DataFrame(rows)
    frame.insert(0, "n_symbols_total", bank.n_symbols)
    return frame


def print_distance_histogram(frame: pd.DataFrame, bins: int = 10) -> None:
    """Print a compact histogram of finite per-symbol nearest distances."""
    values = frame["nearest_mean"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        print("nearest-distance histogram: no symbols have two prototypes")
        return
    counts, edges = np.histogram(values, bins=min(bins, len(values)))
    print("nearest-distance histogram (per-symbol means):")
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        print(f"  [{left:.4g}, {right:.4g}): {'#' * int(count)} ({count})")


def report_one(subject: str, data_root: Path, labels_dir: Path, ckpt: Path,
               device: str) -> list[dict]:
    speller = Speller.load(ckpt).to(device).eval()
    records = []
    for mode, builder in (("same_subject", build_same_subject),
                          ("cross_subject", build_cross_subject)):
        bank = builder(subject, data_root, labels_dir, speller, device)
        scaler = FeatureScaler.fit([e.feature for e in bank.entries])
        frame = nearest_distance_summary(bank, scaler)
        frame.insert(0, "attacked_subject", subject)
        frame.insert(1, "mode", mode)
        frame["bank_size"] = len(bank)
        records.extend(frame.to_dict("records"))
        print(f"[{subject}/{mode}] {len(bank)} entries, "
              f"{len(bank.symbols_available())}/{bank.n_symbols} symbols covered")
        print_distance_histogram(frame)
    return records


def dry_run(subjects, data_root: Path, labels_dir: Path) -> dict:
    """Validate frozen splits and print the plan; never opens raw data."""
    plan = {}
    for subject in subjects:
        split = load_split(_split_path(subject))
        plan[subject] = {
            "split_path": str(_split_path(subject)),
            "n_train": len(split["train"]),
            "n_val": len(split["val"]),
            "n_test": len(split["test"]),
            "same_subject": {
                "mode": "same_subject",
                "allowed_subjects": [subject],
                "allowed_splits": ["train"],
                "uses": "frozen train trial ids only (val/test excluded)",
                "n_entries_expected": len(split["train"]),
            },
            "cross_subject": {
                "mode": "cross_subject",
                "allowed_subjects": [s for s in DEFAULT_SUBJECTS if s != subject],
                "allowed_splits": ["train", "test"],
                "attacked_subject_excluded": subject,
            },
        }
        print(f"[dry-run {subject}] train={plan[subject]['n_train']} "
              f"val={plan[subject]['n_val']} test={plan[subject]['n_test']}")
    return plan


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None,
                   help="directory holding true_labels_*.txt (default: data-root)")
    p.add_argument("--checkpoint-dir", default=str(REPO / "results/decoders"),
                   help="directory holding <subject>_<model>_s<seed>.pt checkpoints")
    p.add_argument("--model", default="eegnet")
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO / "results"))
    p.add_argument("--dry-run", action="store_true",
                   help="validate frozen splits and print the plan without raw data")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    data_root = Path(args.data_root)
    labels_dir = Path(args.labels_dir) if args.labels_dir else data_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        plan = dry_run(args.subjects, data_root, labels_dir)
        payload = {
            "mode": "dry_run",
            "seed": int(args.seed),
            "args": vars(args),
            "plan": plan,
            "data_root": str(data_root),
            "labels_dir": str(labels_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "provenance": environment(args.seed),
        }
        write_config(out_dir, "bank_report.dry_run.json", payload)
        print(f"wrote {out_dir / 'bank_report.dry_run.json'}")
        return payload

    missing = missing_data(data_root, labels_dir, args.subjects)
    if missing:
        raise FileNotFoundError(
            "missing raw data for a full bank report; run with --dry-run locally. "
            "First missing: " + ", ".join(missing[:5])
        )

    rows = []
    for subject in args.subjects:
        ckpt = Path(args.checkpoint_dir) / f"{subject}_{args.model}_s{args.model_seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing decoder checkpoint {ckpt}")
        rows.extend(report_one(subject, data_root, labels_dir, ckpt, args.device))

    frame = pd.DataFrame(rows)
    csv_path = out_dir / "bank_report.csv"
    frame.to_csv(csv_path, index=False)
    config = {
        "seed": int(args.seed),
        "args": vars(args),
        "n_rows": len(frame),
        "data_root": str(data_root),
        "labels_dir": str(labels_dir),
        "split_files": [str(_split_path(s)) for s in args.subjects],
        "provenance": environment(args.seed),
    }
    write_config(out_dir, "bank_report.config.json", config)
    print(f"wrote {csv_path}")
    print(frame.to_string(index=False))
    return config


if __name__ == "__main__":
    main()
