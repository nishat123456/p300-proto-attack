"""Run every detector and the plausibility metrics (Prompt 11).

Reads the NPZ and parquet artifacts written by ``scripts/evaluate_attacks.py``
and produces, for one subject:

- ``results/<subject>/detectors.csv``: one row per (detector, method, generator,
  bank, eps) with AUC, TPR at 5 percent FPR, and for the defense the false alarm
  rate on held-out genuine trials.
- ``results/<subject>/plausibility.csv``: one row per (method, generator, bank,
  eps, trial, target) with the waveform, P300, topography and PSD metrics.

Detectors:
1. the clean-vs-attacked window CNN, one per combination, trained and scored
   over the two trial-parity folds;
2. the genuine-vs-induced ERP defense, trained only on {pgd, target_only (both
   generators), noise} and evaluated on every method and epsilon;
3. the simple band-power and ERP-Mahalanobis detectors.

All splits are by test trial id parity. The script is read-only over the attack
artifacts; it never retrains an attack or reloads a victim.
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

from p3a.defenses import artifacts as art  # noqa: E402
from p3a.defenses.genuine_vs_induced import (  # noqa: E402
    DEFAULT_TRAIN_POOL,
    cross_validate_defense,
)
from p3a.defenses.metrics import DEFAULT_FPR, ChannelMap  # noqa: E402
from p3a.defenses.plausibility import (  # noqa: E402
    plausibility_metrics,
    target_genuine_templates,
)
from p3a.defenses.simple import evaluate_simple_detectors  # noqa: E402
from p3a.defenses.window_cnn import evaluate_window_detector  # noqa: E402
from p3a.utils.repro import environment, set_seed, write_config  # noqa: E402

DEFAULT_FS_OUT = 60.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", choices=("A", "B"), required=True)
    p.add_argument("--results-root", default=str(REPO / "results"),
                   help="root holding attacks_eval/<subject>/ from Prompt 10")
    p.add_argument("--out-dir", default=None,
                   help="default: <results-root>/<subject>, as Prompt 11 fixes")
    p.add_argument("--fs-out", type=float, default=DEFAULT_FS_OUT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fpr", type=float, default=DEFAULT_FPR)
    p.add_argument("--eps", nargs="+", type=float, default=None,
                   help="restrict to these epsilon multipliers")
    p.add_argument("--window-epochs", type=int, default=8)
    p.add_argument("--skip-window", action="store_true",
                   help="skip the window CNN (fast run)")
    p.add_argument("--skip-defense", action="store_true")
    p.add_argument("--skip-plausibility", action="store_true")
    p.add_argument("--limit-combos", type=int, default=None,
                   help="cap the number of combinations (smoke/debug)")
    p.add_argument("--channel-names", nargs="+", default=None,
                   help="channel labels; when absent Pz/Cz are autodetected")
    return p.parse_args(argv)


def _channel_map(args, clean) -> ChannelMap:
    if args.channel_names:
        return ChannelMap.from_names(args.channel_names)
    if clean.true_erp.shape[1] == 64:
        return ChannelMap.comp3(64)
    return ChannelMap.autodetect(clean.true_erp.astype(np.float64), args.fs_out)


def run_subject(args) -> dict:
    subject = str(args.subject).upper()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    specs = art.available_combos(results_root, subject, eps_multipliers=args.eps)
    if not specs:
        raise FileNotFoundError(
            f"no attack artifacts under {art.result_dir(results_root, subject)}; "
            "run scripts/evaluate_attacks.py first"
        )
    if args.limit_combos is not None:
        specs = specs[: int(args.limit_combos)]

    clean = art.load_clean(results_root, subject)
    channel_map = _channel_map(args, clean)
    detector_rows = []
    plausibility_rows = []

    # Load every combination once; reuse the arrays across detectors.
    combos = {}
    for spec in specs:
        combo = art.load_combo(results_root, subject, spec)
        combos[spec] = art.combo_to_inputs(combo)

    # 1. Window CNN per combination.
    if not args.skip_window:
        for spec in specs:
            inputs = combos[spec]
            summary, _ = evaluate_window_detector(
                inputs.windows, inputs.clean_windows, inputs.trial_index,
                fpr=args.fpr, seed=args.seed, epochs=args.window_epochs,
                device=args.device,
            )
            detector_rows.append(_detector_row(
                spec, "window_cnn", summary["auc"], summary["tpr_at_fpr"],
                far=float("nan"), folds=summary["folds"],
            ))

    # 2. Genuine-vs-induced defense: train on the held-in families only.
    if not args.skip_defense:
        pool = [combos[s] for s in specs if s.method in DEFAULT_TRAIN_POOL]
        if pool:
            eval_chunks = {spec.label: combos[spec] for spec in specs}
            defense = cross_validate_defense(
                pool, eval_chunks, fpr=args.fpr, seed=args.seed, device=args.device,
            )
            for spec in specs:
                summary = defense.get(spec.label)
                if summary is None:
                    continue
                detector_rows.append(_detector_row(
                    spec, "genuine_vs_induced", summary["auc"],
                    summary["tpr_at_fpr"], far=summary["far"],
                    folds=summary["folds"], model_kind=summary.get("model_kind"),
                ))

    # 3. Simple detectors per combination.
    for spec in specs:
        inputs = combos[spec]
        result = evaluate_simple_detectors(
            inputs.clean_duplicate_erp, inputs.attacked_erp,
            inputs.clean_windows, inputs.windows, args.fs_out,
            trial_index=inputs.trial_index, fpr=args.fpr,
        )
        detector_rows.append(_detector_row(
            spec, "erp_mahalanobis", result["mahalanobis_auc"],
            result["mahalanobis_tpr_at_fpr"], far=float("nan"),
            folds=result["folds"],
        ))
        detector_rows.append(_detector_row(
            spec, "band_power", result["band_power_auc"],
            result["band_power_tpr_at_fpr"], far=float("nan"),
            folds=result["folds"],
        ))

    # 4. Plausibility per combination.
    if not args.skip_plausibility:
        templates = target_genuine_templates(
            clean.true_erp, clean.true_idx
        )
        for spec in specs:
            inputs = combos[spec]
            records = plausibility_metrics(
                inputs.attacked_erp, inputs.true_idx, inputs.target_idx,
                templates, args.fs_out, channel_map,
            )
            for row, record in zip(_manifest_rows(combos[spec], spec), records):
                row.update(record)
                row["channel_map_source"] = channel_map.source
                row["pz_index"] = channel_map.pz_index
                row["cz_index"] = channel_map.cz_index
                plausibility_rows.append(row)
        # The clean genuine reference needs no plausibility row set; the CSV
        # holds one row per attacked sample which is what the tables consume.

    detectors_path = out_dir / "detectors.csv"
    plausibility_path = out_dir / "plausibility.csv"
    pd.DataFrame(detector_rows).to_csv(detectors_path, index=False)
    pd.DataFrame(plausibility_rows).to_csv(plausibility_path, index=False)
    config = {
        "subject": subject,
        "results_root": str(results_root),
        "out_dir": str(out_dir),
        "fs_out": float(args.fs_out),
        "seed": int(args.seed),
        "fpr": float(args.fpr),
        "device": args.device,
        "n_combos": len(specs),
        "eps": args.eps,
        "window_epochs": int(args.window_epochs),
        "skip_window": bool(args.skip_window),
        "skip_defense": bool(args.skip_defense),
        "skip_plausibility": bool(args.skip_plausibility),
        "channel_map_source": channel_map.source,
        "pz_index": int(channel_map.pz_index),
        "cz_index": int(channel_map.cz_index),
        "defense_train_pool": list(DEFAULT_TRAIN_POOL),
        "n_detector_rows": len(detector_rows),
        "n_plausibility_rows": len(plausibility_rows),
        "provenance": environment(args.seed),
    }
    write_config(out_dir, "run_detectors.config.json", config)
    return {"detectors": str(detectors_path), "plausibility": str(plausibility_path),
            "n_combos": len(specs), "n_detector_rows": len(detector_rows),
            "n_plausibility_rows": len(plausibility_rows),
            "channel_map_source": channel_map.source}


def _detector_row(spec, detector, auc_value, tpr, far, folds, model_kind=None):
    return {
        "subject": str(spec.subject).upper(),
        "detector": detector,
        "method": spec.method,
        "generator": spec.generator,
        "bank": spec.bank,
        "eps_multiplier": float(spec.eps_multiplier),
        "auc": float(auc_value),
        "tpr_at_fpr": float(tpr),
        "far": float(far),
        "model_kind": model_kind,
        "n_folds": len(folds) if folds else 0,
    }


def _manifest_rows(inputs, spec):
    """Per-sample identity rows aligned with the flattened attack arrays."""
    n = len(inputs.trial_index)
    return [{
        "subject": str(spec.subject).upper(),
        "method": spec.method,
        "generator": spec.generator,
        "bank": spec.bank,
        "trial_index": int(inputs.trial_index[i]),
        "trial_id": str(inputs.trial_id[i]),
        "true_idx": int(inputs.true_idx[i]),
        "target_symbol": str(inputs.target_symbol[i]),
        "target_idx": int(inputs.target_idx[i]),
        "attack_success": bool(inputs.success[i]),
        "label": inputs.meta.get("label"),
        "eps_multiplier": inputs.meta.get("eps_multiplier"),
    } for i in range(n)]


def main(argv=None):
    args = _parse_args(argv)
    result = run_subject(args)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
