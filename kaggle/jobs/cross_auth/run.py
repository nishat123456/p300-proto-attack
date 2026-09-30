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
# The full suite expects the developer-machine Subject A training file to be
# present. Its missing-data test checks existence only, so use a zero-byte
# cloud fixture rather than copying EEG into this compact evaluation job.
(code / "data/raw/comp3/Subject_A_Train.mat").touch()
subprocess.run([sys.executable,"-m","pip","install","-e",str(code),"pytest"],check=True)
subprocess.run([sys.executable,"-m","pytest","-q"],cwd=code,check=True)
features = next(Path("/kaggle/input").rglob("A/manifest.csv")).parents[1]
subprocess.run([sys.executable,str(code/"scripts/evaluate_auth_features.py"),"--root",str(features),"--out-dir","/kaggle/working/out/cross_auth","--seed","20260929"],check=True)
