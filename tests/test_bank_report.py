"""Fast CPU tests for scripts/bank_report.py planning and summaries."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_bank_report():
    spec = importlib.util.spec_from_file_location(
        "bank_report", REPO / "scripts" / "bank_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["bank_report"] = module
    spec.loader.exec_module(module)
    return module


bank_report = _load_bank_report()

from p3a.prototypes.bank import BankEntry, PrototypeBank  # noqa: E402
from p3a.prototypes.features import FeatureScaler  # noqa: E402


def _bank_with_features():
    bank = PrototypeBank(
        mode="same_subject", attacked_subject="A", dataset="synthetic",
        allowed_subjects={"A"}, allowed_splits={"train"},
        layout_id="rc6x6_comp3", n_symbols=36, n_channels=2, epoch_len=4,
    )
    feats = {
        0: [np.zeros((2, 4), np.float32), np.ones((2, 4), np.float32) * 2],
        3: [np.ones((2, 4), np.float32) * 5, np.ones((2, 4), np.float32) * 6],
        7: [np.ones((2, 4), np.float32) * 10],
    }
    for symbol, values in feats.items():
        for j, value in enumerate(values):
            bank.entries.append(BankEntry(value, symbol, "A", "synthetic",
                                          f"comp3:A:train:{symbol:03d}{j}", "train"))
    return bank


def test_nearest_summary_reports_counts_and_distances():
    bank = _bank_with_features()
    scaler = FeatureScaler.fit([e.feature for e in bank.entries])
    frame = bank_report.nearest_distance_summary(bank, scaler)
    by_symbol = frame.set_index("symbol_idx")
    assert set(frame["symbol_idx"]) == set(range(bank.n_symbols))
    assert int(by_symbol.loc[0, "count"]) == 2
    assert int(by_symbol.loc[7, "count"]) == 1
    assert int(by_symbol.loc[1, "count"]) == 0
    assert np.isnan(by_symbol.loc[7, "nearest_mean"])  # singleton has no neighbour
    assert np.isnan(by_symbol.loc[1, "nearest_mean"])  # absent symbol
    assert by_symbol.loc[0, "nearest_mean"] > 0


def test_distance_histogram_is_printed(capsys):
    bank = _bank_with_features()
    scaler = FeatureScaler.fit([e.feature for e in bank.entries])
    frame = bank_report.nearest_distance_summary(bank, scaler)
    bank_report.print_distance_histogram(frame)
    output = capsys.readouterr().out
    assert "nearest-distance histogram" in output
    assert "#" in output


def test_dry_run_uses_frozen_splits_and_excludes_attacked_subject(capsys):
    plan = bank_report.dry_run(["A", "B"], REPO / "data/raw/comp3", REPO / "data/raw/comp3")
    assert plan["A"]["n_train"] == 70
    assert plan["A"]["n_val"] == 15
    assert plan["A"]["n_test"] == 100
    assert plan["A"]["same_subject"]["allowed_subjects"] == ["A"]
    assert plan["A"]["same_subject"]["allowed_splits"] == ["train"]
    assert "val/test excluded" in plan["A"]["same_subject"]["uses"]
    # Cross-subject names only the other subject and declares the exclusion.
    assert plan["A"]["cross_subject"]["allowed_subjects"] == ["B"]
    assert plan["A"]["cross_subject"]["attacked_subject_excluded"] == "A"
    assert plan["B"]["cross_subject"]["allowed_subjects"] == ["A"]


def test_missing_data_reports_local_shortfall():
    missing = bank_report.missing_data(
        REPO / "data/raw/comp3", REPO / "data/raw/comp3", ["A"]
    )
    names = [Path(m).name for m in missing]
    # Subject A train and labels exist locally; attacked=A only needs B's train
    # and test files (A's own test is never used), which are not present.
    assert "Subject_A_Train.mat" not in names
    assert "true_labels_a.txt" not in names
    assert "Subject_B_Train.mat" in names
    assert "Subject_B_Test.mat" in names
    assert "Subject_A_Test.mat" not in names


def test_cli_dry_run_writes_config(tmp_path, capsys):
    out = tmp_path / "out"
    payload = bank_report.main([
        "--dry-run", "--subjects", "A", "--out-dir", str(out), "--seed", "3"
    ])
    assert payload["mode"] == "dry_run"
    assert payload["seed"] == 3
    assert payload["provenance"]["seed"] == 3
    assert (out / "bank_report.dry_run.json").is_file()


def test_cli_real_run_without_data_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing raw data"):
        bank_report.main(["--subjects", "A", "--out-dir", str(tmp_path), "--seed", "0"])


def test_complete_data_guard_reports_nothing_when_all_present(tmp_path):
    data = tmp_path / "data"
    labels = tmp_path / "labels"
    data.mkdir()
    labels.mkdir()
    for name in ("Subject_A_Train.mat", "Subject_A_Test.mat",
                 "Subject_B_Train.mat", "Subject_B_Test.mat"):
        (data / name).write_bytes(b"placeholder")
    for name in ("true_labels_a.txt", "true_labels_b.txt"):
        (labels / name).write_text("A" * 100)
    assert bank_report.missing_data(data, labels, ["A", "B"]) == []
