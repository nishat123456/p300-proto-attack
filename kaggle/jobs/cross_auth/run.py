"""Run repository tests and cross-subject evaluation on Kaggle."""
import os
import subprocess
import sys
import shutil
from pathlib import Path
root = next(Path("/kaggle/input").rglob("evaluate_auth_features.py")).parents[1]
code = Path("/kaggle/working/code")
shutil.copytree(root, code, dirs_exist_ok=True)
(code / "tests/__init__.py").touch()
os.environ["PYTHONPATH"] = os.pathsep.join([str(code), str(code / "tests")])
subprocess.run([sys.executable,"-m","pip","install","-e",str(code),"pytest"],check=True)
out = Path("/kaggle/working/out")
out.mkdir(parents=True, exist_ok=True)
excluded = [
    "tests/test_comp3.py",
    "tests/test_letter_head.py",
    "tests/test_bank_report.py",
    "tests/test_kaggle_kill_test_run.py",
]
(out / "cloud_test_scope.txt").write_text(
    "Portable repository tests run on Kaggle. Raw-Comp-III and bank-report tests "
    "were excluded because this compact feature dataset intentionally contains no "
    "raw EEG. The credential-source-path test was excluded because Kaggle's immutable "
    "input path contains the account slug. Cross-subject authenticity and feature-export "
    "tests remain included.\n"
)
test_cmd = [sys.executable, "-m", "pytest", "-q"]
for path in excluded:
    test_cmd.append(f"--ignore={path}")
subprocess.run(test_cmd, cwd=code, check=True)
features = next(Path("/kaggle/input").rglob("A/manifest.csv")).parents[1]
subprocess.run([sys.executable,str(code/"scripts/evaluate_auth_features.py"),"--root",str(features),"--out-dir","/kaggle/working/out/cross_auth","--seed","20260929"],check=True)
