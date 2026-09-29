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


def ensure_variable_trial_loader(repo: Path) -> None:
    """Patch a stale Kaggle attachment to the verified variable-count loader."""
    path = repo / "src" / "p3a" / "data" / "loaders" / "bnci2015_003.py"
    source = path.read_text()
    if "flash count must be a positive multiple" in source:
        return
    source = source.replace("N_TRIALS = 5\n", "")
    source = source.replace(
        '    if len(all_onsets) != N_TRIALS * N_FLASHES:\n'
        '        raise ValueError(f"expected {N_TRIALS * N_FLASHES} flashes, got {len(all_onsets)}")\n',
        '    if not len(all_onsets) or len(all_onsets) % N_FLASHES:\n'
        '        raise ValueError(\n'
        '            f"flash count must be a positive multiple of {N_FLASHES}, "\n'
        '            f"got {len(all_onsets)}"\n'
        '        )\n'
        '    n_trials = len(all_onsets) // N_FLASHES\n',
    )
    source = source.replace("    for index in range(N_TRIALS):\n",
                            "    for index in range(n_trials):\n")
    if "N_TRIALS" in source or "n_trials = len(all_onsets) // N_FLASHES" not in source:
        raise RuntimeError("could not apply the BNCI 2015-003 loader deployment fix")
    path.write_text(source)


code = WORK / "code"
shutil.copytree(find_code(), code, dirs_exist_ok=True)
ensure_variable_trial_loader(code)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code)], check=True)
subprocess.run([
    sys.executable, str(code / "scripts" / "prepare_bnci.py"),
    "--data-root", "/kaggle/temp/p3a_bnci",
    "--out-dir", str(WORK / "out"),
    "--seed", "20260929",
], check=True)
