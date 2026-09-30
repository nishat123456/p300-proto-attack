"""Frozen clean-only FAR audit for BNCI 2014-008.

Fit the authenticity classifier on the registered Comp III attack-feature pool.
For each BNCI subject, use only the frozen validation trials for preprocessing,
domain normalization, and threshold calibration, then measure FAR on the frozen
test trials. This is a calibration gate, not an attack-detection result.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from p3a.data.epochs import downsample_onsets, epoch_window, gather_epochs
from p3a.data.loaders.bnci2014_008 import load_subject
from p3a.defenses.authenticity import FeatureSet, authenticity_features, _threshold
from p3a.defenses.genuine_vs_induced import DEFAULT_TRAIN_POOL
from p3a.eval.store import sample_window_starts, windows_from_preprocessed
from p3a.preproc.pipeline import Preprocessor
from p3a.prototypes.features import target_conditioned_erp
from p3a.utils.repro import environment, set_seed

SUBJECTS = tuple(f"A{i:02d}" for i in range(1, 9))
URL = "https://bnci-horizon-2020.eu/database/data-sets/008-2014/{subject}.mat"


def load_feature_set(path: Path) -> FeatureSet:
    with np.load(path, allow_pickle=False) as data:
        return FeatureSet(np.asarray(data["values"], dtype=np.float32),
                          np.asarray(data["trial_index"], dtype=np.int64),
                          np.asarray(data["trial_id"], dtype=str))


def source_training(root: Path):
    induced, genuine = [], []
    for subject in ("A", "B"):
        folder = root / subject
        manifest = pd.read_csv(folder / "manifest.csv")
        gen = load_feature_set(folder / "genuine.npz")
        genuine.append(gen.values)
        for row in manifest.to_dict("records"):
            if row["method"] in DEFAULT_TRAIN_POOL:
                induced.append(load_feature_set(folder / row["file"]).values)
    xg, xi = np.concatenate(genuine), np.concatenate(induced)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(xg)
    x = np.concatenate([scaler.transform(xi), scaler.transform(xg)])
    y = np.concatenate([np.ones(len(xi)), np.zeros(len(xg))])
    return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                              random_state=20260929).fit(x, y)


def clean_features(trials, fit_trials, seed: int) -> np.ndarray:
    pre = Preprocessor(len(trials[0].channel_names), trials[0].fs, target_fs=60.0)
    pre.fit_scaling([trial.eeg for trial in fit_trials])
    erps, windows = [], []
    length = epoch_window(pre.fs_out)
    for trial in trials:
        eeg = torch.as_tensor(trial.eeg[None], dtype=torch.float32)
        with torch.no_grad():
            signal = pre(eeg)
            onsets = torch.as_tensor(downsample_onsets(trial.onsets, pre.q)[None],
                                     dtype=torch.long)
            epochs = gather_epochs(signal, onsets, length)[0]
            groups = torch.as_tensor(trial.flash_groups, dtype=torch.bool)
            erp = target_conditioned_erp(
                epochs, groups, trial.layout.index(trial.target_symbol)
            ).numpy()
            starts = sample_window_starts(pre.fs_out, signal.shape[-1],
                                          seed=seed + trial.trial_index)
            win = windows_from_preprocessed(signal, starts[None], pre.fs_out)[0]
        erps.append(erp)
        windows.append(win)
    return authenticity_features(np.stack(erps), np.stack(windows), pre.fs_out)


def evaluate_subject(model, trials, split, nominal_far: float, seed: int) -> dict:
    by_id = {trial.trial_id: trial for trial in trials}
    calibration = [by_id[key] for key in split["val"]]
    test = [by_id[key] for key in split["test"]]
    values = clean_features(calibration + test, calibration, seed)
    cal, held = values[:len(calibration)], values[len(calibration):]
    target_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(cal)
    cal_scores = model.decision_function(target_scaler.transform(cal))
    test_scores = model.decision_function(target_scaler.transform(held))
    threshold = _threshold(cal_scores, nominal_far)
    return {"subject": trials[0].subject_id, "nominal_far": nominal_far,
            "observed_far": float(np.mean(test_scores >= threshold)),
            "n_calibration": len(calibration), "n_test": len(test),
            "threshold": threshold}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-root", required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--nominal-far", type=float, default=0.05)
    args = p.parse_args(argv)
    set_seed(args.seed)
    root, out = Path(args.data_root), Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    model = source_training(Path(args.features_root))
    rows = []
    for index, subject in enumerate(SUBJECTS):
        path = root / f"{subject}.mat"
        if not path.is_file():
            urllib.request.urlretrieve(URL.format(subject=subject), path)
        trials = load_subject(root, subject)
        payload = json.loads((Path(args.splits_dir) / f"bnci2014_008_{subject}.json").read_text())
        rows.append(evaluate_subject(model, trials, payload["split"],
                                     args.nominal_far, args.seed + index * 1000))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "bnci2014_008_clean_far.csv", index=False)
    passed = bool((frame.observed_far <= 0.05).all())
    report = {"stage": "clean-only external calibration gate",
              "attack_tpr_evaluated": False, "gate_far_le_0p05_all_subjects": passed,
              "subjects": rows, "provenance": environment(args.seed)}
    (out / "bnci2014_008_clean_far.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "bnci2014_008_clean_far.md").write_text(
        "# BNCI 2014-008 clean FAR gate\n\n" +
        f"All-subject FAR <=5% gate: **{'PASS' if passed else 'FAIL'}**.\n\n" +
        frame.to_markdown(index=False) +
        "\n\nThis clean-only gate does not measure attack TPR or establish Q1 readiness.\n"
    )
    print(json.dumps({"subjects": len(rows), "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
