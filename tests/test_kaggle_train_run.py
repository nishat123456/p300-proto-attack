"""CPU tests for the Kaggle train_decoders kernel script.

The script lives outside the installed package (kaggle/jobs/...), so it is
imported by path. Tests use fake bundle/work directories and a fake training
script: no Kaggle, no pip install, and no real training run.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "kaggle" / "jobs" / "train_decoders" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("kaggle_train_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = _load_run_module()


def _make_bundle(root: Path, label: str = "p300-proto-attack-code") -> Path:
    """A minimal p3a-looking bundle: pyproject that mentions the package."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "p3a"\nversion = "0.1.0"\n'
    )
    (root / label).mkdir(exist_ok=True)
    return root


def test_find_code_bundle_locates_nested_p3a_bundle(tmp_path):
    # Kaggle nests attached datasets; the bundle is several levels deep.
    bundle = _make_bundle(tmp_path / "Dataset p300" / "p300-proto-attack-code")
    nested_decoy = _make_bundle(tmp_path / "other" / "not-p3a-bundle")
    (nested_decoy / "pyproject.toml").write_text('[project]\nname = "other"\n')
    found = run.find_code_bundle(tmp_path)
    assert found == bundle


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
    # A second copy replaces the old one rather than nesting.
    run.copy_bundle(src, dest)
    assert (dest / "pyproject.toml").is_file()
    assert not (dest / "code").exists()


def test_install_editable_invokes_pip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run.subprocess, "run", lambda cmd, check: calls.append((cmd, check))
    )
    run.install_editable(tmp_path)
    assert calls and calls[0][1] is True
    cmd = calls[0][0]
    assert cmd[1:3] == ["-m", "pip"]
    assert cmd[3] == "install" and cmd[4] == "-e"


def test_print_gpu_diagnostics_tolerates_missing_nvidia_smi(capsys, monkeypatch):
    def no_smi(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(run.subprocess, "run", no_smi)
    run.print_gpu_diagnostics()
    out = capsys.readouterr().out
    assert "nvidia-smi not found" in out
    assert "torch.cuda.is_available()" in out


def test_run_training_streams_and_returns_exit_code(tmp_path, capsys):
    bundle = _make_bundle(tmp_path / "bundle")
    scripts = bundle / "scripts"
    scripts.mkdir()
    (scripts / "train_all_decoders.py").write_text(
        "import sys\n"
        "print('hello from training')\n"
        "print('to stderr', file=sys.stderr)\n"
        "sys.exit(0)\n"
    )
    out_dir = tmp_path / "out"
    rc = run.run_training(
        code_dir=bundle, labels_dir=bundle, data_root=bundle,
        out_dir=out_dir, device="cpu",
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "hello from training" in captured
    assert "to stderr" in captured  # stderr is merged into the stream
    assert "exited" not in captured
    assert out_dir.is_dir()


def test_finished_outputs_lists_files_only(tmp_path):
    out = tmp_path / "out"
    (out / "decoders").mkdir(parents=True)
    (out / "decoders" / "A_eegnet_s0.pt").write_text("x")
    (out / "decoders_summary.csv").write_text("a,b\n")
    (out / "train_decoders.log").write_text("log")
    (out / "failure_traceback.txt").write_text("tb")
    names = run._finished_outputs(out)
    assert "decoders/A_eegnet_s0.pt" in names
    assert "decoders_summary.csv" in names
    assert "train_decoders.log" not in names
    assert "failure_traceback.txt" not in names


def test_main_preserves_outputs_and_writes_traceback(tmp_path, monkeypatch, capsys):
    bundle = _make_bundle(tmp_path / "bundle")
    work = tmp_path / "work"
    out = work / "out" / "decoders"
    out.mkdir(parents=True)
    (out / "A_eegnet_s0.pt").write_text("partial")  # finished before the failure

    monkeypatch.setattr(run, "find_code_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(run, "copy_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(run, "install_editable", lambda *a, **k: None)
    monkeypatch.setattr(run, "print_gpu_diagnostics", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("training blew up")
    monkeypatch.setattr(run, "run_training", boom)

    monkeypatch.setattr(run, "CODE_DIR", work / "code")
    monkeypatch.setattr(run, "OUT_DIR", out)
    monkeypatch.setattr(run, "LOG_PATH", out / "train_decoders.log")
    monkeypatch.setattr(run, "TRACEBACK_PATH", out / "failure_traceback.txt")

    rc = run.main()
    assert rc == 1
    assert (out / "failure_traceback.txt").is_file()
    assert "training blew up" in (out / "failure_traceback.txt").read_text()
    assert (out / "A_eegnet_s0.pt").read_text() == "partial"
    log = (out / "train_decoders.log").read_text()
    assert "training blew up" in log
    assert "preserved 1 finished output" in log
