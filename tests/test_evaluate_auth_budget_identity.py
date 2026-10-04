"""Distinct perturbation budgets must survive source fitting and reporting."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_budgets_remain_distinct_in_pool_and_metrics(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_auth_features.py"
    spec = importlib.util.spec_from_file_location("budget_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for subject in ("A", "B"):
        folder = tmp_path / subject
        folder.mkdir()
        records = []
        for index, eps in enumerate((0.05, 0.5)):
            filename = f"combo_{index}.npz"
            np.savez(folder / filename, values=np.full((2, 36), eps),
                     trial_index=np.arange(2), trial_id=np.array(["a", "b"]))
            records.append(dict(file=filename, label="pgd_none_none", method="pgd",
                                generator="none", bank="none", eps_multiplier=eps))
        np.savez(folder / "genuine.npz", values=np.zeros((2, 36)),
                 trial_index=np.arange(2), trial_id=np.array(["a", "b"]))
        pd.DataFrame(records).to_csv(folder / "manifest.csv", index=False)

    def evaluate(pool, targets, seed):
        assert [float(pair[0].values[0, 0]) for pair in pool] == [0.05, 0.5]
        return {key: dict(n_folds=5, tpr_0p05=float(pair[0].values[0, 0]),
                          far_0p05=0.0) for key, pair in targets.items()}

    monkeypatch.setattr(module, "cross_subject_feature_sets", evaluate)
    monkeypatch.setattr(module, "environment", lambda seed: {"seed": seed})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [str(path), "--root", str(tmp_path),
                                      "--out-dir", str(out)])
    module.main()
    rows = pd.read_csv(out / "cross_subject.csv")
    rows = rows[rows.nominal_far == 0.05]
    assert np.allclose(rows.tpr, rows.eps_multiplier)
    assert len(rows) == 4
