"""Fast CPU tests for the kill-test report and decision gate (Prompt 12).

Everything is built from small synthetic CSVs and JSON files, so the report
pipeline and the PREREG gate are exercised without any Kaggle output or model.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from p3a.report.kill_test import (
    DECISION_GATE_DEFENSE,
    DECISION_GATE_FULL,
    DECISION_GATE_STOP,
    build_report,
    decision_gate,
    detector_auc_at_success,
    interpolate_at_level,
    load_attacks_eval,
    load_decoder_accuracy,
    load_detectors,
)


def _attack_frame(rows):
    base = []
    for subject, method, generator, bank, eps, success, model in rows:
        base.append({
            "subject": subject, "method": method, "generator": generator, "bank": bank,
            "eps_multiplier": eps, "eps_raw": eps * 5.0, "trial_index": 0,
            "trial_id": "t", "true_symbol": "A", "true_idx": 0, "target_symbol": "B",
            "target_idx": 1, "retrieval_distance": float("nan"), "attack_ok": True,
            "success_surrogate_15": success, "success_surrogate_5": success,
            "success_deepconvnet_15": success, "success_deepconvnet_5": success,
            "linf": eps, "l2": 1.0, "rms": 0.1, "snr_db": 10.0,
        })
    return pd.DataFrame(base)


def _detector_frame(rows):
    return pd.DataFrame([{
        "subject": s, "detector": d, "method": m, "generator": g, "bank": b,
        "eps_multiplier": e, "auc": a, "tpr_at_fpr": t, "far": f,
        "model_kind": "logreg", "n_folds": 2,
    } for s, d, m, g, b, e, a, t, f in rows])


def test_interpolate_returns_none_when_level_unreached():
    assert interpolate_at_level([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 0.8) is None
    assert interpolate_at_level([0.1, 0.2, 0.3], [0.2, 0.6, 0.9], 0.8) == pytest.approx(0.2666, abs=1e-3)


def test_interpolate_handles_single_point_grid():
    assert interpolate_at_level([0.2], [0.9], 0.8) == pytest.approx(0.2)
    assert interpolate_at_level([0.2], [0.5], 0.8) is None


def test_interpolate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        interpolate_at_level([0.1, 0.2], [0.5], 0.8)


def test_detector_auc_at_success_interpolates_onto_success_epsilon():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.1, 0.3, "surrogate"),
        ("A", "proto_nearest", "signal", "cross_subject", 0.2, 0.6, "surrogate"),
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ])
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.1, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.2, 0.7, 0.9, float("nan")),
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.5, 0.9, float("nan")),
    ])
    value = detector_auc_at_success(attacks, detectors, level=0.8)
    # 0.8 success interpolates to eps 0.2667; detector AUC there is between 0.7 and 0.5.
    assert value == pytest.approx(0.5667, abs=0.02)


def test_detector_auc_at_success_none_when_unreached():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.1, 0.2, "surrogate"),
        ("A", "proto_nearest", "signal", "cross_subject", 0.2, 0.4, "surrogate"),
    ])
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.1, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.2, 0.7, 0.9, float("nan")),
    ])
    assert detector_auc_at_success(attacks, detectors, level=0.8) is None


def test_primary_matching_uses_surrogate_not_victim_success():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.1, 0.4, "surrogate"),
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ])
    attacks["success_deepconvnet_15"] = 0.0
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.1, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.5, 0.9, float("nan")),
    ])
    assert detector_auc_at_success(attacks, detectors, level=0.8) is not None


def test_decision_gate_full_paper_when_nearest_stealthiest_and_control_worse():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "target_only", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "tiny_noise", "none", "none", 0.3, 0.9, "surrogate"),
        ("A", "proto_wrong", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ])
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.5, 0.9, float("nan")),
        ("A", "window_cnn", "target_only", "signal", "cross_subject", 0.3, 0.8, 0.9, float("nan")),
        ("A", "window_cnn", "tiny_noise", "none", "none", 0.3, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_wrong", "signal", "cross_subject", 0.3, 0.95, 0.9, float("nan")),
        ("A", "genuine_vs_induced", "pgd", "none", "none", 0.3, 0.9, 0.85, 0.05),
        ("A", "genuine_vs_induced", "target_only", "signal", "cross_subject", 0.3, 0.9, 0.83, 0.05),
        ("A", "genuine_vs_induced", "noise", "none", "none", 0.3, 0.9, 0.81, 0.05),
    ])
    gate = decision_gate(attacks, detectors)
    assert gate["verdict"] == DECISION_GATE_FULL
    assert gate["h1_primary_favors_proto_nearest"] is True
    assert gate["defense_train_tpr"] == pytest.approx(0.83, abs=1e-9)


def test_decision_gate_defense_led_when_nearest_not_stealthiest():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "target_only", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "tiny_noise", "none", "none", 0.3, 0.9, "surrogate"),
        ("A", "proto_wrong", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ])
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "target_only", "signal", "cross_subject", 0.3, 0.6, 0.9, float("nan")),
        ("A", "window_cnn", "tiny_noise", "none", "none", 0.3, 0.5, 0.9, float("nan")),
        ("A", "window_cnn", "proto_wrong", "signal", "cross_subject", 0.3, 0.95, 0.9, float("nan")),
        ("A", "genuine_vs_induced", "pgd", "none", "none", 0.3, 0.9, 0.85, 0.05),
        ("A", "genuine_vs_induced", "target_only", "signal", "cross_subject", 0.3, 0.9, 0.83, 0.05),
        ("A", "genuine_vs_induced", "noise", "none", "none", 0.3, 0.9, 0.81, 0.05),
    ])
    gate = decision_gate(attacks, detectors)
    assert gate["verdict"] == DECISION_GATE_DEFENSE
    assert gate["h1_primary_favors_proto_nearest"] is False


def test_decision_gate_stop_when_defense_fails_on_training_families():
    attacks = _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "target_only", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
        ("A", "tiny_noise", "none", "none", 0.3, 0.9, "surrogate"),
        ("A", "proto_wrong", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ])
    detectors = _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.5, 0.9, float("nan")),
        ("A", "window_cnn", "target_only", "signal", "cross_subject", 0.3, 0.8, 0.9, float("nan")),
        ("A", "window_cnn", "tiny_noise", "none", "none", 0.3, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_wrong", "signal", "cross_subject", 0.3, 0.95, 0.9, float("nan")),
        ("A", "genuine_vs_induced", "pgd", "none", "none", 0.3, 0.6, 0.30, 0.05),
        ("A", "genuine_vs_induced", "target_only", "signal", "cross_subject", 0.3, 0.6, 0.35, 0.05),
        ("A", "genuine_vs_induced", "noise", "none", "none", 0.3, 0.6, 0.40, 0.05),
    ])
    gate = decision_gate(attacks, detectors)
    assert gate["verdict"] == DECISION_GATE_STOP
    assert gate["defense_train_tpr"] < 0.5


def _write_tree(root: Path):
    dec = root / "decoders"
    dec.mkdir(parents=True)
    for subject in ("A", "B"):
        for model in ("eegnet", "deepconvnet"):
            (dec / f"{subject}_{model}_s0.metrics.json").write_text(json.dumps({
                "subject": subject, "model": model, "seed": 0, "val_auc": 0.8,
                "test_flash_auc": 0.85, "test_letter_acc_15reps": 0.9,
                "test_letter_acc_5reps": 0.6, "n_test_trials": 100,
            }))
    subj = root / "attacks_eval" / "A"
    subj.mkdir(parents=True)
    _attack_frame([
        ("A", "proto_nearest", "signal", "cross_subject", 0.1, 0.3, "surrogate"),
        ("A", "proto_nearest", "signal", "cross_subject", 0.3, 0.9, "surrogate"),
    ]).to_parquet(subj / "attacks.parquet")
    _detector_frame([
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.1, 0.9, 0.9, float("nan")),
        ("A", "window_cnn", "proto_nearest", "signal", "cross_subject", 0.3, 0.5, 0.9, float("nan")),
        ("A", "genuine_vs_induced", "pgd", "none", "none", 0.3, 0.9, 0.85, 0.05),
    ]).to_csv(subj / "detectors.csv", index=False)
    pd.DataFrame([{
        "subject": "A", "method": "proto_nearest", "generator": "signal",
        "bank": "cross_subject", "eps_multiplier": 0.3, "waveform_corr_pz": 0.7,
        "p300_latency_ms": 330.0, "p300_amplitude": 1.2, "topography_corr": 0.6,
        "psd_distance_db": 2.0, "channel_map_source": "comp3_montage",
    }]).to_csv(subj / "plausibility.csv", index=False)


def test_loaders_find_nested_files(tmp_path):
    _write_tree(tmp_path)
    assert len(load_decoder_accuracy(tmp_path)) == 4
    assert len(load_attacks_eval(tmp_path)) == 2
    assert len(load_detectors(tmp_path)) == 3


def test_build_report_writes_markdown(tmp_path):
    _write_tree(tmp_path)
    out = tmp_path / "report"
    result = build_report(tmp_path, out)
    report = Path(result["report"])
    assert report == out / "kill_test_report.md"
    text = report.read_text()
    assert "# Kill-test report" in text
    assert "PREREG decision gate" in text
    assert "Verdict:" in text
    assert result["verdict"] in {
        DECISION_GATE_FULL, DECISION_GATE_DEFENSE, DECISION_GATE_STOP
    }
    assert (out / "kill_test_report.config.json").exists() is False  # CLI writes that


def test_build_report_empty_root_still_writes(tmp_path):
    out = tmp_path / "report"
    result = build_report(tmp_path / "missing", out)
    assert Path(result["report"]).is_file()
    text = (out / "kill_test_report.md").read_text()
    assert "no data" in text.lower()
    # With no evidence at all the gate must stop and reassess, never claim a win.
    assert result["verdict"] == DECISION_GATE_STOP


def test_loaders_return_empty_when_absent(tmp_path):
    assert load_decoder_accuracy(tmp_path).empty
    assert load_attacks_eval(tmp_path).empty
    assert load_detectors(tmp_path).empty
