"""Run source-trained P300 authenticity detection on a target subject."""
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
    cross_subject_authenticity,
)
from p3a.defenses.genuine_vs_induced import DEFAULT_TRAIN_POOL  # noqa: E402
from p3a.utils.repro import set_seed, write_config  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-subject", choices=("A", "B"), required=True)
    parser.add_argument("--target-subject", choices=("A", "B"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fs-out", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    return parser.parse_args(argv)


def _load(root: Path, subject: str):
    specs = art.available_combos(root, subject)
    combos = {
        spec: art.combo_to_inputs(art.load_combo(root, subject, spec))
        for spec in specs
    }
    return specs, combos


def run(args):
    set_seed(args.seed)
    source_subject = args.source_subject.upper()
    target_subject = args.target_subject.upper()
    if source_subject == target_subject:
        raise ValueError("source and target subjects must differ")
    source_specs, source = _load(Path(args.source_root), source_subject)
    target_specs, target = _load(Path(args.target_root), target_subject)
    pool = [inputs for spec, inputs in source.items()
            if spec.method in DEFAULT_TRAIN_POOL]
    if not pool:
        raise ValueError("no registered source training-pool attacks were found")
    by_label = {spec.label: target[spec] for spec in target_specs}
    result = cross_subject_authenticity(
        pool, by_label, fs=args.fs_out, fprs=DEFAULT_FPRS,
        n_folds=args.folds, seed=args.seed, c=args.c,
    )

    rows = []
    for spec in target_specs:
        record = result[spec.label]
        for target_fpr in DEFAULT_FPRS:
            suffix = str(float(target_fpr)).replace(".", "p")
            rows.append({
                "source_subject": source_subject,
                "target_subject": target_subject,
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
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"cross_subject_authenticity_{source_subject}_to_{target_subject}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_config(out, f"cross_subject_authenticity_{source_subject}_to_{target_subject}.config.json", {
        "status": "exploratory Comp III subject-normalized repair",
        "source_subject": source_subject,
        "target_subject": target_subject,
        "seed": args.seed,
        "folds": args.folds,
        "c": args.c,
        "target_fprs": list(DEFAULT_FPRS),
        "train_pool": list(DEFAULT_TRAIN_POOL),
        "source_combos": len(source_specs),
        "target_combos": len(target_specs),
        "target_information_used": "clean calibration fold only",
    })
    return {"csv": str(csv_path), "rows": len(rows)}


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
