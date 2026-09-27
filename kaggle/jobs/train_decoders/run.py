"""Kaggle kernel: train every Comp III decoder and stream the logs.

Flow:
1. locate the attached code bundle under /kaggle/input (rglob for the bundle
   `pyproject.toml` that declares the `p3a` package), copy it to
   /kaggle/working/code and `pip install -e` it,
2. print the GPU diagnostics (nvidia-smi and torch.cuda.is_available),
3. run `scripts/train_all_decoders.py` with the public data root and the bundle
   labels directory, streaming stdout/stderr line by line,
4. on any failure, print and persist the traceback and leave whatever finished
   (per-run checkpoints and metrics, plus the summary if it was written) in place.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("P3A_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("P3A_WORK_ROOT", "/kaggle/working"))
CODE_DIR = WORK_ROOT / "code"
OUT_DIR = WORK_ROOT / "out" / "decoders"
LOG_PATH = OUT_DIR / "train_decoders.log"
TRACEBACK_PATH = OUT_DIR / "failure_traceback.txt"
IGNORE_NAMES = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}


def _tee(message: str, handle=None) -> None:
    """Print a line and optionally append it to a log file."""
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


def find_code_bundle(input_root: Path = INPUT_ROOT) -> Path:
    """Find the attached bundle: a directory whose pyproject declares `p3a`.

    Kaggle nests attached datasets, so the bundle directory can be several
    levels below /kaggle/input. Prefer a path that also looks like the p3a
    bundle; fall back to the first pyproject.toml found, and fail clearly when
    there is none.
    """
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    candidates = []
    for meta in sorted(input_root.rglob("pyproject.toml")):
        text = meta.read_text(errors="replace")
        candidates.append((meta.parent, "[project]" in text and "p3a" in text))
    if not candidates:
        raise FileNotFoundError(
            f"no pyproject.toml found under {input_root}; "
            "is the p300-proto-attack-code dataset attached?"
        )
    preferred = [parent for parent, is_p3a in candidates if is_p3a]
    chosen = (preferred or [parent for parent, _ in candidates])[0]
    return chosen


def copy_bundle(src: Path, dest: Path = CODE_DIR) -> Path:
    """Copy the bundle to a writable location, replacing any previous copy."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest,
        ignore=shutil.ignore_patterns(*IGNORE_NAMES),
    )
    return dest


def install_editable(code_dir: Path = CODE_DIR) -> None:
    """pip install -e the copied bundle so `import p3a` resolves."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(code_dir)],
        check=True,
    )


def print_gpu_diagnostics(handle=None) -> None:
    """Print nvidia-smi output (if present) and torch's CUDA view."""
    _tee("=" * 60, handle)
    _tee("GPU diagnostics", handle)
    _tee("=" * 60, handle)
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        _tee(out.stdout.rstrip() if out.stdout else "(nvidia-smi produced no output)", handle)
    except FileNotFoundError:
        _tee("nvidia-smi not found", handle)
    try:
        import torch

        _tee(f"torch.__version__ = {torch.__version__}", handle)
        _tee(f"torch.cuda.is_available() = {torch.cuda.is_available()}", handle)
        if torch.cuda.is_available():
            _tee(f"device name = {torch.cuda.get_device_name(0)}", handle)
    except Exception:  # noqa: BLE001 - diagnostics must never abort the job
        _tee("torch import/cuda check failed:", handle)
        _tee(traceback.format_exc(), handle)


def run_training(code_dir: Path = CODE_DIR, labels_dir: Path | None = None,
                 data_root: Path = INPUT_ROOT, out_dir: Path = OUT_DIR,
                 device: str = "cuda", handle=None) -> int:
    """Run train_all_decoders.py, streaming combined stdout/stderr line by line."""
    if labels_dir is None:
        labels_dir = code_dir / "data" / "raw" / "comp3"
    script = code_dir / "scripts" / "train_all_decoders.py"
    if not script.is_file():
        raise FileNotFoundError(f"training script missing from bundle: {script}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--data-root", str(data_root),
        "--labels-dir", str(labels_dir),
        "--out-dir", str(out_dir),
        "--device", device,
    ]
    _tee("running: " + " ".join(cmd), handle)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(code_dir), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _tee(line.rstrip("\n"), handle)
    return proc.wait()


def _finished_outputs(out_dir: Path) -> list[str]:
    """Names of checkpoints/metrics that survived, so a failure is inspectable."""
    if not out_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and p.name not in {"train_decoders.log", "failure_traceback.txt"}
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with LOG_PATH.open("w") as handle:
        try:
            bundle = find_code_bundle()
            _tee(f"found code bundle at {bundle}", handle)
            copy_bundle(bundle, CODE_DIR)
            _tee(f"copied bundle to {CODE_DIR}", handle)
            install_editable(CODE_DIR)
            print_gpu_diagnostics(handle)
            rc = run_training(
                CODE_DIR, data_root=INPUT_ROOT, out_dir=OUT_DIR,
                device="cuda", handle=handle,
            )
            _tee(f"train_all_decoders.py exited with code {rc}", handle)
            if rc != 0:
                raise RuntimeError(
                    f"train_all_decoders.py failed with exit code {rc}"
                )
            _tee(f"elapsed {time.time() - started:.1f}s", handle)
            return rc
        except Exception:  # noqa: BLE001 - persist traceback, do not lose outputs
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            handle.write(tb + "\n")
            handle.flush()
            TRACEBACK_PATH.write_text(tb)
            finished = _finished_outputs(OUT_DIR)
            _tee(f"preserved {len(finished)} finished output file(s):", handle)
            for name in finished:
                _tee(f"  {name}", handle)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
