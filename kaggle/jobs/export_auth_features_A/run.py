"""Export compact subject-A authenticity features on Kaggle."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SUBJECT = "A"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_code() -> Path:
    for path in INPUT.rglob("scripts/export_authenticity_features.py"):
        return path.parents[1]
    raise FileNotFoundError("P3A feature-export code is not attached")


def prepare_results_root() -> Path:
    for clean in INPUT.rglob(f"{SUBJECT}_clean.npz"):
        root = WORK / "merged"
        attack_root = root / "attacks_eval"
        attack_root.mkdir(parents=True, exist_ok=True)
        (attack_root / SUBJECT).symlink_to(clean.parent, target_is_directory=True)
        return root
    raise FileNotFoundError(f"attack tree for subject {SUBJECT} is not attached")


code = WORK / "code"
shutil.copytree(find_code(), code, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code)], check=True)
subprocess.run([
    sys.executable, str(code / "scripts" / "export_authenticity_features.py"),
    "--subject", SUBJECT,
    "--results-root", str(prepare_results_root()),
    "--out-dir", str(WORK / "out" / "auth_features" / SUBJECT),
], check=True)
