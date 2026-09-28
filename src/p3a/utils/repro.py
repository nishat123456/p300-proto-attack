"""Reproducibility helpers: seeding, git state, config sidecar files.

Every training script calls `set_seed`, then records `environment()` and a full
config JSON next to its outputs. The git record distinguishes a clean checkout
from a dirty working tree (uncommitted code) so results can be traced to code.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU and CUDA) and request deterministic cuDNN."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best effort: some ops have no deterministic kernel, so do not hard-fail.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git(*args: str, cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return out.stdout.strip()


def repo_root(cwd: str | Path | None = None) -> Path | None:
    """Resolve the git top level independent of the current directory."""
    top = _git("rev-parse", "--show-toplevel", cwd=cwd)
    return Path(top) if top else None


# Output and cache locations that must not be mistaken for dirty source code.
_IGNORED_PREFIXES = (
    "results/", "data/", "kaggle/out/", ".venv/", "checkpoints/",
    "prototype_banks/", "__pycache__/",
)


def _is_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return not any(normalized.startswith(p) for p in _IGNORED_PREFIXES)


def git_record(cwd: str | Path | None = None) -> dict:
    """Commit hash plus a dirty flag and a state hash of uncommitted source.

    Generated outputs (results, checkpoints, data) do not count as dirty; only
    changes to source keep the recorded commit hash from identifying the code,
    which is exactly the case while Prompt 1-4 modules are still untracked.
    Untracked files are hashed by their *contents* (not just their names), so
    editing an untracked module changes `git_state_hash` and the recorded git
    commit can no longer misrepresent which code produced a result.
    """
    import hashlib

    root = repo_root(cwd=cwd)
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if not commit and cwd is not None:
        bundle_commit = Path(cwd).resolve() / "GIT_COMMIT.txt"
        if bundle_commit.is_file():
            commit = bundle_commit.read_text().strip()
    def paths(*args):
        return set(_git(*args, "-z", cwd=root or cwd).split("\0")) - {""}
    changed = paths("diff", "HEAD", "--name-only")
    untracked = paths("ls-files", "--others", "--exclude-standard")
    source_paths = sorted(p for p in changed | untracked if _is_source_path(p))
    dirty = bool(source_paths)
    hasher = hashlib.sha256()
    for rel in source_paths:
        path = (root / rel) if root is not None else Path(rel)
        hasher.update(rel.encode() + b"\0")
        hasher.update(path.read_bytes() if path.is_file() else b"<deleted>")

    return {
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
        "git_state_hash": hasher.hexdigest()[:16],
    }


def environment(seed: int) -> dict:
    """Versions, seed, and git state to store with a run's outputs."""
    return {
        "seed": int(seed),
        "python": _python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pid": os.getpid(),
        **git_record(Path(__file__).resolve().parents[3]),
    }


def _python_version() -> str:
    import platform

    return platform.python_version()


def write_config(out_dir: str | Path, name: str, config: dict) -> Path:
    """Write a JSON sidecar next to outputs, creating the directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return path
