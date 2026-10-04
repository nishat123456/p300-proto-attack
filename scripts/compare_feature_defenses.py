"""Kaggle-only fixed baseline comparison on inspected Comp III features."""
import argparse
import json
from pathlib import Path
import pandas as pd
from scripts.evaluate_auth_features import load_set
from p3a.defenses.authenticity import cross_subject_feature_sets
from p3a.defenses.genuine_vs_induced import DEFAULT_TRAIN_POOL
from p3a.utils.repro import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for subject in ("A", "B"):
        folder = root / subject
        manifest = pd.read_csv(folder / "manifest.csv")
        genuine = load_set(folder / "genuine.npz")
        pairs = {}
        for row in manifest.to_dict("records"):
            name = row["file"]
            if Path(name).name != name or name in pairs:
                raise ValueError("unsafe or duplicate feature identity")
            pairs[name] = (load_set(folder / name), genuine)
        sets[subject] = manifest, pairs
    rows, details = [], {}
    for source, target in (("A", "B"), ("B", "A")):
        sm, sp = sets[source]
        tm, tp = sets[target]
        pool = [sp[row.file] for row in sm.itertuples() if row.method in DEFAULT_TRAIN_POOL]
        for kind in ("logistic", "forest", "mahalanobis"):
            result = cross_subject_feature_sets(
                pool, tp, seed=args.seed, model_kind=kind,
                independent_normalization=True,
            )
            details[f"{source}_to_{target}_{kind}"] = result
            for row in tm.to_dict("records"):
                metrics = result[row["file"]]
                for far in (0.01, 0.05, 0.1):
                    suffix = str(far).replace(".", "p")
                    rows.append(dict(source=source, target=target, detector=kind,
                        method=row["method"], generator=row["generator"], bank=row["bank"],
                        eps_multiplier=row["eps_multiplier"], nominal_far=far,
                        tpr=metrics.get("tpr_"+suffix), far=metrics.get("far_"+suffix),
                        auc=metrics["auc"], n_folds=metrics["n_folds"]))
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    (out / "folds.json").write_text(json.dumps(details, indent=2))
    (out / "config.json").write_text(json.dumps(dict(
        seed=args.seed, logistic_c=0.1, forest_trees=200, forest_min_leaf=10,
        normalization="three genuine folds; distinct threshold and evaluation folds",
        status="exploratory fixed diagnostic; no novelty or confirmation claim",
        provenance=environment(args.seed)), indent=2))


if __name__ == "__main__":
    main()
