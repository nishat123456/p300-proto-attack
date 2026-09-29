import importlib.util
from pathlib import Path

import pandas as pd

from defense_helpers import write_subject


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cross_subject_authenticity.py"


def _module():
    spec = importlib.util.spec_from_file_location("cross_subject_authenticity_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_writes_both_metrics_and_provenance(tmp_path):
    methods = [
        ("pgd", "none", "none", 1.0),
        ("target_only", "signal", "cross_subject", 1.0),
        ("noise", "none", "none", 1.0),
        ("tiny_noise", "none", "none", 1.0),
    ]
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    write_subject(source_root, "A", n_trials=10, n_targets=1,
                  methods=methods, seed=1)
    write_subject(target_root, "B", n_trials=10, n_targets=1,
                  methods=methods, seed=2)
    out = tmp_path / "out"
    args = _module().parse_args([
        "--source-subject", "A", "--target-subject", "B",
        "--source-root", str(source_root), "--target-root", str(target_root),
        "--out-dir", str(out), "--seed", "7",
    ])
    result = _module().run(args)
    frame = pd.read_csv(result["csv"])
    assert len(frame) == 12
    assert set(frame["source_subject"]) == {"A"}
    assert set(frame["target_subject"]) == {"B"}
    assert frame["observed_far"].between(0, 1).all()
    assert (out / "cross_subject_authenticity_A_to_B.config.json").is_file()
