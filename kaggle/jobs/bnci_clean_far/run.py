"""Run the frozen BNCI 2014-008 clean-FAR gate on Kaggle."""
import shutil
import subprocess
import sys
from pathlib import Path

inp, work = Path("/kaggle/input"), Path("/kaggle/working")
code_source = next(inp.rglob("scripts/evaluate_bnci_clean_far.py")).parents[1]
features = next(inp.rglob("A/manifest.csv")).parents[1]
code = work / "code"
shutil.copytree(code_source, code, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(code),
                "pytest", "tabulate"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_bnci_loaders.py", "tests/test_bnci_splits.py",
                "tests/test_bnci_clean_far.py"], cwd=code, check=True)
subprocess.run([
    sys.executable, str(code / "scripts/evaluate_bnci_clean_far.py"),
    "--features-root", str(features), "--splits-dir", str(code / "splits"),
    "--data-root", str(work / "bnci2014_008"),
    "--out-dir", str(work / "out"), "--seed", "20260929",
], check=True)
