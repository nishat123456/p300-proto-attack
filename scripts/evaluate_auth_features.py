"""Evaluate compact A/B feature exports on Kaggle only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from p3a.defenses.authenticity import FeatureSet, cross_subject_feature_sets
from p3a.defenses.genuine_vs_induced import DEFAULT_TRAIN_POOL
from p3a.utils.repro import environment


def load_set(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"values", "trial_index", "trial_id"}:
            raise ValueError("unexpected feature payload")
        x = data["values"]
        ids, indices = data["trial_id"], data["trial_index"]
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("invalid engineered feature array")
        if len(x) != len(ids) or len(x) != len(indices):
            raise ValueError("unaligned feature identifiers")
        return FeatureSet(x, indices, ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20260929)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = {}
    for subject in ("A", "B"):
        folder = root / subject
        manifest = pd.read_csv(folder / "manifest.csv")
        genuine = load_set(folder / "genuine.npz")
        pairs = {}
        for row in manifest.to_dict("records"):
            file = row["file"]
            if Path(file).name != file:
                raise ValueError("unsafe feature filename")
            pairs[row["label"]] = (load_set(folder / file), genuine)
        subjects[subject] = (manifest, pairs)
    rows, folds = [], {}
    for source, target in (("A", "B"), ("B", "A")):
        sm, sp = subjects[source]
        tm, tp = subjects[target]
        pool = [sp[row.label] for row in sm.itertuples() if row.method in DEFAULT_TRAIN_POOL]
        result = cross_subject_feature_sets(pool, tp, seed=a.seed)
        folds[source + "_to_" + target] = result
        for row in tm.to_dict("records"):
            metrics = result[row["label"]]
            for far in (0.01, 0.05, 0.1):
                suffix = str(far).replace(".", "p")
                rows.append(dict(source=source, target=target, method=row["method"],
                    generator=row["generator"], bank=row["bank"], eps_multiplier=row["eps_multiplier"],
                    nominal_far=far, tpr=metrics.get("tpr_"+suffix),
                    far=metrics.get("far_"+suffix), n_folds=metrics["n_folds"]))
    pd.DataFrame(rows).to_csv(out / "cross_subject.csv", index=False)
    (out / "folds.json").write_text(json.dumps(folds, indent=2))
    (out / "config.json").write_text(json.dumps(dict(seed=a.seed, c=0.1,
        folds=5, status="exploratory development only", provenance=environment(a.seed)), indent=2))

if __name__ == "__main__":
    main()
