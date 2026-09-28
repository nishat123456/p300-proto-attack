"""Evaluate every attack method on one Comp III subject (Prompt 10).

The heavy lifting lives in ``p3a.eval.evaluate.evaluate_attacks``; this script
parses arguments, resolves defaults, and prints a compact summary. It is
resumable: a (method, generator, bank, epsilon) combination whose NPZ and
parquet shard already exist is skipped, so a restarted run continues from where
it stopped. ``--dry-run`` validates the frozen splits, the pre-registered
targets and the expected asset paths without loading any model or raw EEG, which
is the safe local mode on a laptop that only has Subject A.

Focused smoke: ``--smoke`` uses the first few test trials and only the
``noise`` and ``pgd`` methods, so a full end-to-end pass runs on CPU in seconds
once a smoke surrogate exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.eval.evaluate import (  # noqa: E402
    DEFAULT_METHOD_SPECS,
    MethodSpec,
    evaluate_attacks,
    load_attack_targets,
)
from p3a.utils.repro import environment, set_seed  # noqa: E402

DEFAULT_SURROGATE_DIR = REPO / "results/decoders"
DEFAULT_SMOKE_SURROGATE = REPO / "results/smoke_prompt4/A_eegnet_s0.pt"
DEFAULT_SMOKE_DIRS = REPO / "results/smoke_prompt9"


def smoke_specs() -> list[MethodSpec]:
    """The two cheap methods, so a smoke run needs no bank or generator."""
    return [s for s in DEFAULT_METHOD_SPECS if s.kind in ("noise", "pgd")]


def dry_run(args) -> dict:
    """Validate splits, targets and asset paths; never open raw data or models."""
    from p3a.data.splits import load as load_split

    splits_dir = Path(args.splits_dir) if args.splits_dir else REPO / "splits"
    plan = {}
    for subject in _subjects(args):
        split = load_split(splits_dir / f"comp3_{subject}.json")
        targets = load_attack_targets(subject, splits_dir)
        n_test = len(split["test"])
        plan[subject] = {
            "n_train": len(split["train"]),
            "n_val": len(split["val"]),
            "n_test": n_test,
            "n_target_trials": len(targets),
            "targets_cover_test": all(t in targets for t in split["test"]),
            "n_targets_per_trial": len(next(iter(targets.values()))) if targets else 0,
            "generator_dir": str(args.generator_dir),
            "tiny_noise_dir": str(args.tiny_noise_dir),
            "victim_dir": str(args.victim_dir),
        }
        print(f"[dry-run {subject}] train={plan[subject]['n_train']} "
              f"val={plan[subject]['n_val']} test={n_test} "
              f"targets_cover_test={plan[subject]['targets_cover_test']} "
              f"n_targets_per_trial={plan[subject]['n_targets_per_trial']}")
    return plan


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None,
                   help="directory holding true_labels_*.txt (default: data-root)")
    p.add_argument("--subject", choices=("A", "B"), default="A")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="when given, run every listed subject")
    p.add_argument(
        "--surrogate", default=str(DEFAULT_SURROGATE_DIR),
        help=("EEGNet checkpoint, checkpoint directory, or path template containing "
              "{subject}; directories resolve to <subject>_eegnet_s<seed>.pt"),
    )
    p.add_argument("--generator-dir", default=str(REPO / "results/attacks"))
    p.add_argument("--tiny-noise-dir", default=str(REPO / "results/tiny_noise"))
    p.add_argument("--victim-dir", default=str(REPO / "results/decoders"))
    p.add_argument("--victim-seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "results/attacks_eval"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-trials", dest="batch_trials", type=int, default=8)
    p.add_argument("--max-flash-batch", dest="max_flash_batch", type=int, default=None)
    p.add_argument("--pgd-steps", dest="pgd_steps", type=int, default=40)
    p.add_argument("--limit-trials", dest="limit_trials", type=int, default=None,
                   help="cap the number of test trials (smoke/debug)")
    p.add_argument("--splits-dir", default=None)
    p.add_argument("--force", action="store_true",
                   help="recompute combinations whose artifacts already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="validate splits and targets without loading models or data")
    p.add_argument("--smoke", action="store_true",
                   help="noise + pgd on a few trials, CPU, smoke surrogate")
    return p.parse_args(argv)


def _subjects(args) -> list[str]:
    if args.subjects:
        return [str(s).upper() for s in args.subjects]
    return [str(args.subject).upper()]


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.device = "cpu"
        args.subjects = None
        args.subject = "A"
        if args.limit_trials is None:
            args.limit_trials = 3
        args.tiny_noise_dir = str(DEFAULT_SMOKE_DIRS)
        if Path(args.surrogate) == DEFAULT_SURROGATE_DIR:
            args.surrogate = str(DEFAULT_SMOKE_SURROGATE)

    if args.dry_run:
        set_seed(args.seed)
        plan = dry_run(args)
        payload = {"mode": "dry_run", "seed": int(args.seed), "args": vars(args),
                   "plan": plan, "provenance": environment(args.seed)}
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "evaluate_attacks.dry_run.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {path}")
        return payload

    results = {}
    for subject in _subjects(args):
        specs = smoke_specs() if args.smoke else list(DEFAULT_METHOD_SPECS)
        result = _run_subject(args, subject, specs)
        results[subject] = {
            "n_test_trials": result.n_trials,
            "n_targets_per_trial": result.n_targets,
            "n_combos": result.n_combos,
            "n_combos_done": result.n_combos_done,
            "n_combos_skipped": result.n_combos_skipped,
            "n_rows": result.n_rows,
            "clean_npz": result.clean_npz,
            "parquet": result.parquet,
        }
        print(json.dumps({subject: results[subject]}, indent=2))
    return results


def _run_subject(args, subject: str, specs):
    """Run one subject; smoke mode restricts methods and skips victim loading."""
    surrogate = _resolve_surrogate(args.surrogate, subject, args.seed)
    return evaluate_attacks(
        data_root=args.data_root, labels_dir=args.labels_dir, subject=subject,
        surrogate=str(surrogate), out_dir=args.out_dir,
        generator_dir=args.generator_dir, tiny_noise_dir=args.tiny_noise_dir,
        victim_dir=args.victim_dir, seed=args.seed, victim_seed=args.victim_seed,
        device=args.device, batch_trials=args.batch_trials,
        max_flash_batch=args.max_flash_batch, limit_trials=args.limit_trials,
        splits_dir=args.splits_dir, force=args.force, pgd_steps=args.pgd_steps,
        specs=specs, victims=not args.smoke,
    )


def _resolve_surrogate(value: str, subject: str, seed: int) -> Path:
    """Resolve a subject EEGNet from a file, directory, or format template."""
    if "{subject}" in value or "{seed}" in value:
        return Path(value.format(subject=subject, seed=seed))
    path = Path(value)
    if path.is_dir() or path.suffix == "":
        return path / f"{subject}_eegnet_s{seed}.pt"
    return path


if __name__ == "__main__":
    main()
