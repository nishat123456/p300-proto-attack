"""Train every subject x decoder x seed and write results/decoders_summary.csv.

Calls `train_decoder` from scripts/train_decoder.py in-process so a single
Kaggle job fits all runs. Each row carries that run's metrics plus a summary
row identifier; full per-run metrics live in the sibling .metrics.json files.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from train_decoder import ALL_MODELS, train_decoder  # noqa: E402

SUBJECTS = ("A", "B")
SEEDS = (0, 1, 2)

FIELDS = [
    "subject", "model", "seed", "val_auc", "test_flash_auc",
    "test_letter_acc_15reps", "test_letter_acc_5reps", "n_params",
    "n_train_trials", "n_val_trials", "n_test_trials", "best_epoch",
    "train_time_s", "smoke", "device", "epochs", "batch", "lr", "patience",
    "git_commit", "git_dirty", "git_state_hash", "repo_root", "checkpoint",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--out-dir", default=str(REPO / "results/decoders"))
    p.add_argument("--summary", default=str(REPO / "results/decoders_summary.csv"))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.subjects is None:
        # Local smoke has only subject A, so defaulting to A keeps `--smoke`
        # runnable without B data; full runs still cover both subjects.
        args.subjects = ["A"] if args.smoke else list(SUBJECTS)
    if args.smoke and args.device == "auto":
        args.device = "cpu"
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    rows = []
    for subject in args.subjects:
        for model in args.models:
            for seed in args.seeds:
                print(f"[train_all] subject={subject} model={model} seed={seed}", flush=True)
                metrics = train_decoder(
                    data_root=args.data_root, labels_dir=args.labels_dir,
                    subject=subject, model=model, seed=seed, out_dir=args.out_dir,
                    epochs=args.epochs, batch=args.batch, lr=args.lr,
                    patience=args.patience, device=args.device, smoke=args.smoke,
                )
                flat = {**metrics.get("args", {}), **metrics}
                rows.append({k: flat.get(k) for k in FIELDS})
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[train_all] wrote {len(rows)} rows to {summary}", flush=True)
    return summary


if __name__ == "__main__":
    main()
