import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tests.defense_helpers import write_subject


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_authenticity_features.py"


def _module():
    spec = importlib.util.spec_from_file_location("export_auth_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_contains_compact_features_without_eeg_arrays(tmp_path):
    write_subject(tmp_path / "input", "A", n_trials=5, n_targets=2,
                  methods=[("pgd", "none", "none", 1.0)])
    out = tmp_path / "features"
    args = _module().parse_args([
        "--subject", "A", "--results-root", str(tmp_path / "input"),
        "--out-dir", str(out),
    ])
    result = _module().run(args)
    assert result["combos"] == 1
    manifest = pd.read_csv(out / "manifest.csv")
    assert len(manifest) == 1
    for name in ("genuine.npz", manifest.iloc[0]["file"]):
        with np.load(out / name) as data:
            assert set(data.files) == {"values", "trial_index", "trial_id"}
            assert data["values"].ndim == 2
