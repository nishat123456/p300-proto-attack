"""Fast CPU tests for scripts/run_detectors.py (Prompt 11).

The synthetic Prompt 10 artifact tree from ``defense_helpers`` stands in for a
real evaluation, so the full pipeline (window CNN, defense, simple detectors and
plausibility) can be run end to end in seconds. The tests check the CSV schemas,
the held-in defense pool, the parity split, and that a fast CPU smoke run
finishes with the expected row counts.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tests.defense_helpers import write_subject

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_detectors_script", REPO / "scripts" / "run_detectors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tree(tmp_path):
    methods = [
        ("pgd", "none", "none", 1.0),
        ("target_only", "signal", "cross_subject", 1.0),
        ("noise", "none", "none", 1.0),
        ("tiny_noise", "none", "none", 1.0),
        ("proto_nearest", "causal", "cross_subject", 1.0),
    ]
    write_subject(tmp_path, "A", n_trials=12, n_targets=3, methods=methods,
                  eps=(0.20,))


def test_run_detectors_writes_both_csvs(tmp_path):
    _write_tree(tmp_path)
    script = _load_script()
    result = script.run_subject(script._parse_args([
        "--subject", "A", "--results-root", str(tmp_path),
        "--out-dir", str(tmp_path / "det"), "--window-epochs", "3",
        "--limit-combos", "5",
    ]))
    detectors = pd.read_csv(result["detectors"])
    plausibility = pd.read_csv(result["plausibility"])
    assert set(detectors["detector"]) == {
        "window_cnn", "genuine_vs_induced", "erp_mahalanobis", "band_power"
    }
    assert {"auc", "tpr_at_fpr", "far", "method", "generator", "bank",
            "eps_multiplier", "model_kind"} <= set(detectors.columns)
    assert set(detectors["method"]) == {
        "pgd", "target_only", "noise", "tiny_noise", "proto_nearest"
    }
    # One defense row per combination; the held-out tiny_noise and proto_nearest
    # still receive a row because the defense is evaluated, not trained, on them.
    defense_rows = detectors[detectors["detector"] == "genuine_vs_induced"]
    assert set(defense_rows["method"]) >= {"tiny_noise", "proto_nearest"}
    # Plausibility has one row per attacked (trial, target) sample.
    n_samples = 5 * 12 * 3
    assert len(plausibility) == n_samples
    assert {"waveform_corr_pz", "waveform_corr_cz", "p300_latency_ms",
            "p300_amplitude", "topography_corr", "psd_distance_db",
            "channel_map_source", "target_symbol"} <= set(plausibility.columns)


def test_run_detectors_config_records_channel_map(tmp_path):
    _write_tree(tmp_path)
    script = _load_script()
    out_dir = tmp_path / "det"
    result = script.run_subject(script._parse_args([
        "--subject", "A", "--results-root", str(tmp_path), "--out-dir", str(out_dir),
        "--window-epochs", "2", "--limit-combos", "3",
    ]))
    import json
    config = json.loads((out_dir / "run_detectors.config.json").read_text())
    assert config["channel_map_source"] in ("provided", "autodetected")
    assert config["defense_train_pool"] == ["pgd", "target_only", "noise"]
    assert config["subject"] == "A"
    assert result["n_combos"] == 3


def test_run_detectors_channel_names_resolve(tmp_path):
    _write_tree(tmp_path)
    script = _load_script()
    out_dir = tmp_path / "det"
    script.run_subject(script._parse_args([
        "--subject", "A", "--results-root", str(tmp_path), "--out-dir", str(out_dir),
        "--channel-names", "Fz", "Cz", "Pz", "Oz", "--window-epochs", "2",
        "--limit-combos", "1",
    ]))
    plausibility = pd.read_csv(out_dir / "plausibility.csv")
    assert set(plausibility["channel_map_source"]) == {"provided"}
    assert set(plausibility["pz_index"]) == {2}
    assert set(plausibility["cz_index"]) == {1}


def test_run_detectors_default_out_dir_is_subject_dir(tmp_path):
    """Prompt 11 fixes results/<subject>/detectors.csv as the output path."""
    _write_tree(tmp_path)
    script = _load_script()
    result = script.run_subject(script._parse_args([
        "--subject", "A", "--results-root", str(tmp_path),
        "--skip-window", "--skip-defense", "--limit-combos", "1",
    ]))
    assert Path(result["detectors"]) == tmp_path / "A" / "detectors.csv"
    assert Path(result["plausibility"]) == tmp_path / "A" / "plausibility.csv"
    assert (tmp_path / "A" / "detectors.csv").is_file()


def test_run_detectors_missing_artifacts_raises(tmp_path):
    script = _load_script()
    with pytest.raises(FileNotFoundError, match="evaluate_attacks"):
        script.run_subject(script._parse_args([
            "--subject", "A", "--results-root", str(tmp_path / "empty"),
        ]))


def test_run_detectors_skip_flags(tmp_path):
    _write_tree(tmp_path)
    script = _load_script()
    result = script.run_subject(script._parse_args([
        "--subject", "A", "--results-root", str(tmp_path),
        "--out-dir", str(tmp_path / "det"), "--skip-window", "--skip-defense",
        "--skip-plausibility", "--limit-combos", "2",
    ]))
    detectors = pd.read_csv(result["detectors"])
    assert set(detectors["detector"]) == {"erp_mahalanobis", "band_power"}
    assert result["n_plausibility_rows"] == 0
