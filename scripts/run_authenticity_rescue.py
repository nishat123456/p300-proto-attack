"""Run the frozen post-hoc P300 authenticity-defense rescue evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.defenses import artifacts as art  # noqa: E402
from p3a.defenses.authenticity import (  # noqa: E402
    DEFAULT_FPRS,
    cross_validate_authenticity,
)
from p3a.defenses.genuine_vs_induced import DEFAULT_TRAIN_POOL  # noqa: E402
from p3a.utils.repro import environment, set_seed, write_config  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=("A", "B"), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fs-out", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--limit-combos", type=int, default=None)
    return parser.parse_args(argv)


def run(args):
    set_seed(args.seed)
    subject = args.subject.upper()
    root = Path(args.results_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = art.available_combos(root, subject)
    if args.limit_combos is not None:
        specs = specs[:args.limit_combos]
    combos = {spec: art.combo_to_inputs(art.load_combo(root, subject, spec))
              for spec in specs}
    pool = [inputs for spec, inputs in combos.items() if spec.method in DEFAULT_TRAIN_POOL]
    if not pool:
        raise ValueError("no registered training-pool attacks were found")
    by_label = {spec.label: inputs for spec, inputs in combos.items()}
    result = cross_validate_authenticity(
        pool, by_label, fs=args.fs_out, fprs=DEFAULT_FPRS,
        n_folds=args.folds, seed=args.seed, c=args.c,
    )
    rows = []
    for spec in specs:
        record = result[spec.label]
        for target_fpr in DEFAULT_FPRS:
            suffix = str(float(target_fpr)).replace(".", "p")
            rows.append({
                "subject": subject,
                "method": spec.method,
                "generator": spec.generator,
                "bank": spec.bank,
                "eps_multiplier": spec.eps_multiplier,
                "target_fpr": target_fpr,
                "auc": record["auc"],
                "calibrated_tpr": record.get(f"tpr_{suffix}"),
                "observed_far": record.get(f"far_{suffix}"),
                "n_folds": record["n_folds"],
            })
    path = out / "authenticity_rescue.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    config = {
        "status": "post-hoc exploratory rescue",
        "subject": subject,
        "results_root": str(root),
        "seed": args.seed,
        "folds": args.folds,
        "calibration": "next outer fold, genuine samples only",
        "fit_folds": args.folds - 2,
        "c": args.c,
        "target_fprs": list(DEFAULT_FPRS),
        "train_pool": list(DEFAULT_TRAIN_POOL),
        "n_combos": len(specs),
        "provenance": environment(args.seed),
    }
    write_config(out, "authenticity_rescue.config.json", config)
    return {"csv": str(path), "rows": len(rows), "combos": len(specs)}


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
