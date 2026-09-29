"""CPU tests for the kill-test Kaggle kernel script (Prompt 12).

The script lives outside the installed package, so it is imported by path. Tests
use fake bundle and decoder directories and never call Kaggle, pip install, or
any real training.
"""
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "kaggle" / "jobs" / "kill_test" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("kaggle_kill_test_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = _load_run_module()


@pytest.fixture(autouse=True)
def _no_shard_file(monkeypatch, tmp_path):
    """Keep the developer's own shard.json from leaking into the default tests."""
    monkeypatch.setattr(run, "SHARD_FILE", tmp_path / "no_shard.json")


def _make_bundle(root: Path, label: str = "p300-proto-attack-code") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "p3a"\nversion = "0.1.0"\n')
    (root / label).mkdir(exist_ok=True)
    return root


def _make_decoders(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "A_eegnet_s0.pt").write_text("x")
    (root / "A_eegnet_s0.metrics.json").write_text("{}")
    return root


def test_find_code_bundle_locates_nested_p3a_bundle(tmp_path):
    bundle = _make_bundle(tmp_path / "Dataset p300" / "p300-proto-attack-code")
    decoy = _make_bundle(tmp_path / "other" / "not-p3a")
    (decoy / "pyproject.toml").write_text('[project]\nname = "other"\n')
    assert run.find_code_bundle(tmp_path) == bundle


def test_find_code_bundle_raises_when_nothing_attached(tmp_path):
    empty = tmp_path / "input"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no pyproject.toml"):
        run.find_code_bundle(empty)


def test_copy_bundle_excludes_caches_and_recopies(tmp_path):
    src = _make_bundle(tmp_path / "input" / "bundle")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_text("junk")
    dest = tmp_path / "work" / "code"
    run.copy_bundle(src, dest)
    assert (dest / "pyproject.toml").is_file()
    assert not (dest / "__pycache__").exists()
    run.copy_bundle(src, dest)
    assert not (dest / "code").exists()


def test_find_decoder_dir_prefers_named_decoders(tmp_path):
    dec = _make_decoders(tmp_path / "out" / "decoders")
    _make_decoders(tmp_path / "out" / "other_decoders")
    assert run.find_decoder_dir(tmp_path) == dec


def test_find_decoder_dir_raises_without_metrics(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError, match="decoder checkpoints"):
        run.find_decoder_dir(tmp_path)


def test_copy_decoders_only_copies_expected_suffixes(tmp_path):
    src = _make_decoders(tmp_path / "src")
    (src / "notes.txt").write_text("ignore me")
    dest = run.copy_decoders(src, tmp_path / "dst")
    names = {p.name for p in dest.iterdir()}
    assert names == {"A_eegnet_s0.pt", "A_eegnet_s0.metrics.json"}


def test_restore_prior_output_copies_checkpoints_and_artifacts(tmp_path):
    prior = tmp_path / "input" / "resume" / "out" / "kill_test"
    (prior / "attacks").mkdir(parents=True)
    (prior / "attacks_eval" / "A").mkdir(parents=True)
    (prior / "attacks" / "A_target_only_signal_cross_subject_s0.pt").write_text("ckpt")
    (prior / "attacks_eval" / "A" / "clean.npz").write_text("artifact")
    out = tmp_path / "working" / "out" / "kill_test"
    source = run.restore_prior_output(tmp_path / "input", out)
    assert source == prior
    assert (out / "attacks" / "A_target_only_signal_cross_subject_s0.pt").is_file()
    assert (out / "attacks_eval" / "A" / "clean.npz").is_file()


def test_install_editable_invokes_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))
    run.install_editable(tmp_path)
    assert calls and calls[0][1] is True
    cmd = calls[0][0]
    assert cmd[1:3] == ["-m", "pip"]
    assert cmd[3] == "install" and cmd[4] == "-e"


def test_shard_plan_covers_every_unit_exactly_once():
    for shard_count in (1, 2, 3, 5, 8):
        buckets = run.shard_plan(shard_count)
        assert len(buckets) == shard_count
        seen = []
        for bucket in buckets:
            assert bucket, "a shard must never be empty"
            for unit in bucket:
                seen.append((unit["subject"], unit["methods"]))
        all_units = [(u["subject"], u["methods"]) for u in run.plan_units()]
        assert sorted(seen) == sorted(all_units)


def test_shard_plan_rejects_bad_counts():
    with pytest.raises(ValueError, match="at least 1"):
        run.shard_plan(0)
    with pytest.raises(ValueError, match="exceeds"):
        run.shard_plan(99)


def test_shard_methods_are_coherent_per_subject():
    buckets = run.shard_plan(5)
    for bucket in buckets:
        for subject in run.shard_subjects(bucket):
            methods = run.shard_methods(bucket, subject)
            assert methods
            assert all(m in run.ALL_METHODS for m in methods)


def test_eval_methods_always_include_core_baselines():
    methods = run.eval_methods_for(["proto_nearest"])
    assert {"noise", "pgd"} <= set(methods)
    assert "proto_nearest" in methods
    # Order follows registration order, not set order.
    assert methods == [m for m in run.ALL_METHODS if m in set(methods)]


def test_resolve_plan_default_is_full_registered_job():
    plan, run_det = run.resolve_plan(run.parse_args([]))
    assert sorted(plan) == ["A", "B"]
    for methods in plan.values():
        assert set(methods) == set(run.ALL_METHODS)
    assert run_det is True


def test_resolve_plan_shard_skips_detectors_for_partial_set():
    args = run.parse_args(["--shard-index", "2", "--shard-count", "5"])
    plan, run_det = run.resolve_plan(args)
    assert run_det is False
    assert plan
    for methods in plan.values():
        assert set(methods) < set(run.ALL_METHODS)


def test_resolve_plan_explicit_subject_and_methods():
    args = run.parse_args(["--subject", "A", "--methods", "pgd", "noise",
                           "--detectors", "always"])
    plan, run_det = run.resolve_plan(args)
    assert plan == {"A": ["pgd", "noise"]}
    assert run_det is True


def test_resolve_plan_rejects_unknown_method():
    with pytest.raises(SystemExit, match="unknown method"):
        run.resolve_plan(run.parse_args(["--subject", "A", "--methods", "nope"]))


def test_env_shard_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "SHARD_FILE", tmp_path / "absent.json")
    monkeypatch.setenv("P3A_SHARD_INDEX", "0")
    monkeypatch.setenv("P3A_SHARD_COUNT", "5")
    args = run.parse_args([])
    assert args.shard_index == 0 and args.shard_count == 5
    plan, _ = run.resolve_plan(args)
    assert plan


def test_env_subjects_and_methods(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "SHARD_FILE", tmp_path / "absent.json")
    monkeypatch.setenv("P3A_SUBJECTS", "A")
    monkeypatch.setenv("P3A_METHODS", "pgd,noise")
    args = run.parse_args([])
    assert args.subject == ["A"] and args.methods == ["pgd", "noise"]


def test_shard_file_selection(tmp_path, monkeypatch):
    shard = tmp_path / "shard.json"
    shard.write_text('{"shard_index": 1, "shard_count": 5}')
    monkeypatch.setattr(run, "SHARD_FILE", shard)
    monkeypatch.delenv("P3A_SHARD_INDEX", raising=False)
    monkeypatch.delenv("P3A_SHARD_COUNT", raising=False)
    args = run.parse_args([])
    assert args.shard_index == 1 and args.shard_count == 5
    plan, _ = run.resolve_plan(args)
    assert plan


def test_shard_file_subjects_and_methods(tmp_path, monkeypatch):
    shard = tmp_path / "shard.json"
    shard.write_text('{"subjects": "A", "methods": ["pgd", "noise"]}')
    monkeypatch.setattr(run, "SHARD_FILE", shard)
    monkeypatch.delenv("P3A_SUBJECTS", raising=False)
    monkeypatch.delenv("P3A_METHODS", raising=False)
    args = run.parse_args([])
    assert args.subject == ["A"] and args.methods == ["pgd", "noise"]


def test_explicit_flags_win_over_shard_file(tmp_path, monkeypatch):
    shard = tmp_path / "shard.json"
    shard.write_text('{"shard_index": 1, "shard_count": 5}')
    monkeypatch.setattr(run, "SHARD_FILE", shard)
    args = run.parse_args(["--shard-index", "0", "--shard-count", "4"])
    assert args.shard_index == 0 and args.shard_count == 4


def test_load_shard_file_missing_returns_empty(tmp_path):
    assert run.load_shard_file(tmp_path / "absent.json") == {}


def test_load_shard_file_rejects_non_object(tmp_path):
    path = tmp_path / "shard.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        run.load_shard_file(path)


def test_shard_flags_must_pair():
    with pytest.raises(SystemExit, match="together"):
        run.parse_args(["--shard-index", "1"])


def test_run_streams_and_returns_exit_code(tmp_path, capsys):
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys\nprint('hello')\nprint('to stderr', file=sys.stderr)\nsys.exit(0)\n"
    )
    rc = run._run([os.fspath(__import__('sys').executable), script], cwd=tmp_path, handle=None)
    assert rc == 0
    captured = capsys.readouterr().out
    assert "hello" in captured and "to stderr" in captured


def test_finished_outputs_lists_files_only(tmp_path):
    out = tmp_path / "kill_test"
    (out / "attacks_eval").mkdir(parents=True)
    (out / "attacks_eval" / "attacks.parquet").write_text("x")
    (out / "kill_test.log").write_text("log")
    (out / "failure_traceback.txt").write_text("tb")
    names = run._finished_outputs(out)
    assert "attacks_eval/attacks.parquet" in names
    assert "kill_test.log" not in names
    assert "failure_traceback.txt" not in names


def test_generator_and_tiny_ckpt_names():
    assert run.generator_ckpt(Path("d"), "A", "proto_nearest", "signal",
                              "cross_subject", 0).name == \
        "A_proto_nearest_signal_cross_subject_s0.pt"
    assert run.tiny_ckpt(Path("d"), "A", 0.2, 0).name == "A_tiny_noise_eps0.20_s0.pt"


def test_main_preserves_outputs_and_writes_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run, "LOG_PATH", tmp_path / "out" / "kill_test.log")
    monkeypatch.setattr(run, "TRACEBACK_PATH", tmp_path / "out" / "failure_traceback.txt")
    run.OUT_DIR.mkdir(parents=True)
    (run.OUT_DIR / "attacks_eval").mkdir()
    (run.OUT_DIR / "attacks_eval" / "attacks.parquet").write_text("partial")

    monkeypatch.setattr(run, "find_code_bundle", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "copy_bundle", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "install_editable", lambda *a, **k: None)
    monkeypatch.setattr(run, "find_decoder_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "copy_decoders", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "build_banks", lambda *a, **k: 0)

    def boom(*a, **k):
        raise RuntimeError("kill test blew up")
    monkeypatch.setattr(run, "train_generators", boom)

    rc = run.main(["--subject", "A", "--methods", "proto_nearest", "--skip-banks"])
    assert rc == 1
    assert (run.OUT_DIR / "failure_traceback.txt").is_file()
    assert "kill test blew up" in (run.OUT_DIR / "failure_traceback.txt").read_text()
    assert (run.OUT_DIR / "attacks_eval" / "attacks.parquet").read_text() == "partial"
    log = (run.OUT_DIR / "kill_test.log").read_text()
    assert "preserved 1 finished output" in log


def test_main_never_prints_credentials(tmp_path, monkeypatch, capsys):
    """A sentinel credential in the environment must not reach the log/output."""
    monkeypatch.setenv("KAGGLE_KEY", "SUPERSECRETVALUE")
    monkeypatch.setenv("KAGGLE_USERNAME", "mustaqimnishat")
    monkeypatch.setattr(run, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run, "LOG_PATH", tmp_path / "out" / "kill_test.log")
    monkeypatch.setattr(run, "TRACEBACK_PATH", tmp_path / "out" / "failure_traceback.txt")
    run.OUT_DIR.mkdir(parents=True)
    monkeypatch.setattr(run, "find_code_bundle", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "copy_bundle", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "install_editable", lambda *a, **k: None)
    monkeypatch.setattr(run, "find_decoder_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "copy_decoders", lambda *a, **k: tmp_path)
    monkeypatch.setattr(run, "build_banks", lambda *a, **k: 0)
    monkeypatch.setattr(run, "train_generators", lambda *a, **k: 0)
    monkeypatch.setattr(run, "train_tiny_noise", lambda *a, **k: 0)
    monkeypatch.setattr(run, "run_evaluate", lambda *a, **k: 0)
    monkeypatch.setattr(run, "run_detectors", lambda *a, **k: 0)
    rc = run.main(["--subject", "A", "--methods", "pgd"])
    assert rc == 0
    combined = capsys.readouterr().out + (run.OUT_DIR / "kill_test.log").read_text()
    assert "SUPERSECRETVALUE" not in combined
    assert "mustaqimnishat" not in combined
