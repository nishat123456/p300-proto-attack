"""Download and validate both frozen external P300 datasets."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_code() -> Path:
    for path in INPUT.rglob("scripts/prepare_bnci.py"):
        return path.parents[1]
    raise FileNotFoundError("P3A code dataset is not attached")


code = WORK / "code"
shutil.copytree(find_code(), code, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code)], check=True)
subprocess.run([
    sys.executable, str(code / "scripts" / "prepare_bnci.py"),
    "--data-root", "/kaggle/temp/p3a_bnci",
    "--out-dir", str(WORK / "out"),
    "--seed", "20260929",
], check=True)
