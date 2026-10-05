"""Export compact authenticity features without raw EEG or attack arrays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.defenses import artifacts as art  # noqa: E402
from p3a.defenses.authenticity import extract_authenticity_sets  # noqa: E402
from p3a.utils.repro import write_config  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fs-out", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--representation", choices=("summary", "channel_time"),
                        default="summary")
    return parser.parse_args(argv)


def _save(path: Path, feature_set):
    np.savez_compressed(
        path,
        values=feature_set.values.astype(np.float32),
        trial_index=np.asarray(feature_set.trial_index, dtype=np.int64),
        trial_id=np.asarray(feature_set.trial_id, dtype=str),
    )


def run(args):
    subject = str(args.subject).upper()
    root, out = Path(args.results_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = art.available_combos(root, subject)
    if not specs:
        raise ValueError(f"no attack artifacts found for subject {subject}")

    rows = []
    genuine_written = False
    for index, spec in enumerate(specs):
        inputs = art.combo_to_inputs(art.load_combo(root, subject, spec))
        representation = getattr(args, "representation", "summary")
        if representation == "channel_time":
            from p3a.defenses.spatiotemporal_features import extract_rich_sets
            induced, genuine = extract_rich_sets(inputs, args.fs_out)
        else:
            induced, genuine = extract_authenticity_sets(inputs, args.fs_out)
        if not genuine_written:
            _save(out / "genuine.npz", genuine)
            genuine_written = True
        filename = f"combo_{index:03d}.npz"
        _save(out / filename, induced)
        rows.append({
            "subject": subject,
            "method": spec.method,
            "generator": spec.generator,
            "bank": spec.bank,
            "eps_multiplier": spec.eps_multiplier,
            "label": spec.label,
            "file": filename,
            "n_induced": len(induced.values),
            "n_features": induced.values.shape[1],
            "representation": representation,
            "n_channels": inputs.attacked_erp.shape[1],
            "erp_samples": inputs.attacked_erp.shape[2],
        })
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    write_config(out, "feature_export.config.json", {
        "subject": subject,
        "seed": args.seed,
        "fs_out": args.fs_out,
        "representation": getattr(args, "representation", "summary"),
        "channel_order": "retained artifact channel index; no cross-montage mapping",
        "time_bins": 12 if getattr(args, "representation", "summary") == "channel_time" else 0,
        "n_combos": len(rows),
        "contents": "engineered features and identifiers only; no raw EEG",
    })
    return {"subject": subject, "combos": len(rows), "out_dir": str(out)}


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
