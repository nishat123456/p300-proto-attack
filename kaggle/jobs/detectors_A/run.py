"""Run the merged subject-A detector suite on a Kaggle GPU."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SUBJECT = "A"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_code() -> Path:
    for path in INPUT.rglob("pyproject.toml"):
        if "p3a" in path.read_text(errors="replace"):
            return path.parent
    raise FileNotFoundError("P3A code dataset is not attached")


def find_results_root() -> Path:
    for path in INPUT.rglob(f"attacks_eval/{SUBJECT}"):
        if (path / f"{SUBJECT}_clean.npz").is_file():
            return path.parent.parent
    raise FileNotFoundError(f"merged attack tree for subject {SUBJECT} is not attached")


code = WORK / "code"
shutil.copytree(find_code(), code, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code)], check=True)
out = WORK / "out" / "kill_test" / SUBJECT
subprocess.run([
    sys.executable, str(code / "scripts" / "run_detectors.py"),
    "--subject", SUBJECT,
    "--results-root", str(find_results_root()),
    "--out-dir", str(out),
    "--device", "cuda",
], check=True)
