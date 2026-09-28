"""Fast CPU tests for the kill_test_report CLI and method filtering (Prompt 12)."""
import importlib.util
import json
from pathlib import Path

import pytest

from p3a.eval.evaluate import DEFAULT_METHOD_SPECS, select_method_specs

REPO = Path(__file__).resolve().parents[1]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "kill_test_report_cli", REPO / "scripts" / "kill_test_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_tree(root: Path):
    dec = root / "decoders"
    dec.mkdir(parents=True)
    (dec / "A_eegnet_s0.metrics.json").write_text(json.dumps({
        "subject": "A", "model": "eegnet", "seed": 0, "val_auc": 0.8,
        "test_flash_auc": 0.85, "test_letter_acc_15reps": 0.9,
        "test_letter_acc_5reps": 0.6, "n_test_trials": 100,
    }))


def test_select_method_specs_default_is_every_method():
    specs = select_method_specs()
    assert [s.label() for s in specs] == [s.label() for s in DEFAULT_METHOD_SPECS]


def test_select_method_specs_filters_by_name():
    specs = select_method_specs(["pgd", "proto_nearest"])
    names = {s.name for s in specs}
    assert names == {"pgd", "proto_nearest"}
    # Preserves registration order.
    labels = [s.label() for s in specs]
    assert labels == [s.label() for s in DEFAULT_METHOD_SPECS if s.name in names]


def test_select_method_specs_rejects_unknown():
    with pytest.raises(ValueError, match="unknown method"):
        select_method_specs(["not_a_method"])


def test_cli_writes_report_and_config(tmp_path, capsys):
    cli = _load_cli()
    out_root = tmp_path / "out"
    _write_minimal_tree(out_root)
    out_dir = tmp_path / "report"
    result = cli.main([
        "--output-root", str(out_root), "--out-dir", str(out_dir),
    ])
    assert Path(result["report"]).is_file()
    config = json.loads((out_dir / "kill_test_report.config.json").read_text())
    assert config["verdict"] in {
        "full attack + defense paper", "defense-led paper", "stop and reassess"
    }
    assert config["output_root"] == str(out_root)
    assert config["provenance"]["git_commit"]


def test_cli_default_paths_point_at_kaggle_output():
    cli = _load_cli()
    args = cli.parse_args([])
    assert args.output_root.endswith("kaggle/out/kill_test/out/kill_test")
    assert args.out_dir.endswith("results")


def test_cli_report_filename_is_fixed(tmp_path):
    cli = _load_cli()
    out_root = tmp_path / "out"
    _write_minimal_tree(out_root)
    out_dir = tmp_path / "somewhere"
    result = cli.main(["--output-root", str(out_root), "--out-dir", str(out_dir)])
    assert Path(result["report"]).name == "kill_test_report.md"
