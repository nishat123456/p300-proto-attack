"""Run repository tests and cross-subject evaluation on Kaggle."""
import subprocess
import sys
import shutil
from pathlib import Path
root = next(Path("/kaggle/input").rglob("evaluate_auth_features.py")).parents[1]
code = Path("/kaggle/working/code")
shutil.copytree(root, code, dirs_exist_ok=True)
subprocess.run([sys.executable,"-m","pip","install","-e",str(code),"pytest"],check=True)
subprocess.run([sys.executable,"-m","pytest","-q"],cwd=code,check=True)
features = next(Path("/kaggle/input").rglob("A/manifest.csv")).parents[1]
subprocess.run([sys.executable,str(code/"scripts/evaluate_auth_features.py"),"--root",str(features),"--out-dir","/kaggle/working/out/cross_auth","--seed","20260929"],check=True)
