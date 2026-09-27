"""Build the Kaggle code bundle for the p3a project.

`build_bundle` cleans `kaggle/code_bundle/` and copies in only source and
metadata: `src/`, `scripts/`, `splits/`, `configs/`, `tests/`, `pyproject.toml`,
`AGENTS.md`, `PREREG.md`, and the Comp III `true_labels_*.txt` files. It writes
the dataset metadata and the current git commit.

Nothing else is copied: no virtualenv, no `.mat` data, no results, no
checkpoints, no caches, no agent logs or prompts, no credentials, and no nested
`code_bundle/` copy. `validate_bundle` re-checks a built bundle so a bad copy
fails loudly instead of being uploaded.

`--dry-run` builds into a temporary directory, validates and prints an
inspection summary, then removes the copy. It never touches the real bundle
directory and never calls Kaggle.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "kaggle" / "code_bundle"

DATASET_METADATA = {
    "title": "p300-proto-attack-code",
    "id": "mustaqimnishat/p300-proto-attack-code",
    "licenses": [{"name": "CC0-1.0"}],
}

# Directories copied wholesale (absent ones become empty dirs in the bundle).
COPY_DIRS = ("src", "scripts", "splits", "configs", "tests")
# Individual files copied from the repo root.
COPY_FILES = ("pyproject.toml", "AGENTS.md", "PREREG.md")

# Anything whose path component matches one of these is never copied.
EXCLUDED_DIR_NAMES = frozenset({
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints", ".git", ".kilo", ".github",
    "results", "checkpoints", "prototype_banks", "code_bundle", "agent_logs",
    "prompts", "out", "node_modules", "kaggle",
})
EXCLUDED_SUFFIXES = (
    ".mat", ".pyc", ".pyo", ".pt", ".pth", ".joblib", ".h5", ".ckpt", ".npy",
)
CREDENTIAL_NAMES = frozenset({
    ".server_pw", ".netrc", ".env", "kaggle.json", "credentials.json",
    "id_rsa", "id_ed25519", ".kaggle",
})

# A built bundle must contain these, or it is not usable by the Kaggle job.
REQUIRED_FILES = (
    "src/p3a/__init__.py",
    "src/p3a/models/speller.py",
    "scripts/train_all_decoders.py",
    "scripts/train_decoder.py",
    "splits/comp3_A.json",
    "splits/comp3_B.json",
    "pyproject.toml",
    "AGENTS.md",
    "PREREG.md",
    "GIT_COMMIT.txt",
    "dataset-metadata.json",
    "data/raw/comp3/true_labels_a.txt",
    "data/raw/comp3/true_labels_b.txt",
)
REQUIRED_DIRS = ("src", "scripts", "splits", "configs", "tests")


def git_commit(repo: str | Path = REPO) -> str:
    """Current HEAD commit hash, or "unknown" outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def is_excluded(rel: str | Path) -> bool:
    """True when a bundle-relative path must never appear in the bundle."""
    rel = Path(rel)
    if set(rel.parts) & EXCLUDED_DIR_NAMES:
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    name = rel.name
    if name in CREDENTIAL_NAMES:
        return True
    if name.startswith(".env"):
        return True
    if name.endswith(EXCLUDED_SUFFIXES):
        return True
    return False


def _ignore_factory(root: Path, dest: Path):
    root = Path(root).resolve()
    dest = Path(dest).resolve()

    def ignore(directory, names):
        dirpath = Path(directory)
        try:
            rel = dirpath.relative_to(root)
        except ValueError:
            rel = Path(".")
        excluded = []
        for name in names:
            if is_excluded(rel / name):
                excluded.append(name)
                continue
            # Guard against recursion if dest ever sits inside a source dir.
            candidate = (dirpath / name).resolve()
            if candidate == dest or dest in candidate.parents:
                excluded.append(name)
        return excluded

    return ignore


def build_bundle(repo: str | Path = REPO, dest: str | Path = DEFAULT_DEST,
                 clean: bool = True) -> Path:
    """Copy the selected source tree into `dest` and write bundle metadata."""
    repo = Path(repo).resolve()
    dest = Path(dest).resolve()
    if clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in COPY_DIRS:
        src = repo / name
        target = dest / name
        if src.is_dir():
            shutil.copytree(src, target, ignore=_ignore_factory(src, dest))
        else:
            target.mkdir(parents=True, exist_ok=True)

    for name in COPY_FILES:
        src = repo / name
        if not src.is_file():
            raise FileNotFoundError(f"required bundle file missing: {name}")
        shutil.copy2(src, dest / name)

    labels_dir = dest / "data" / "raw" / "comp3"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted((repo / "data" / "raw" / "comp3").glob("true_labels_*.txt"))
    if not label_files:
        raise FileNotFoundError("no true_labels_*.txt found under data/raw/comp3")
    for path in label_files:
        shutil.copy2(path, labels_dir / path.name)

    (dest / "dataset-metadata.json").write_text(
        json.dumps(DATASET_METADATA, indent=2) + "\n"
    )
    (dest / "GIT_COMMIT.txt").write_text(git_commit(repo) + "\n")
    return dest


def validate_bundle(dest: str | Path = DEFAULT_DEST) -> list[str]:
    """Return a list of problems; an empty list means the bundle is valid."""
    dest = Path(dest)
    problems: list[str] = []
    if not dest.is_dir():
        return [f"bundle directory does not exist: {dest}"]

    for rel in REQUIRED_FILES:
        if not (dest / rel).is_file():
            problems.append(f"missing required file: {rel}")
    for name in REQUIRED_DIRS:
        if not (dest / name).is_dir():
            problems.append(f"missing required directory: {name}")

    meta_path = dest / "dataset-metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"dataset-metadata.json is not valid JSON: {exc}")
        else:
            if meta != DATASET_METADATA:
                problems.append(
                    "dataset-metadata.json does not match the required contents"
                )

    commit_path = dest / "GIT_COMMIT.txt"
    if commit_path.is_file() and not commit_path.read_text().strip():
        problems.append("GIT_COMMIT.txt is empty")

    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dest)
        if is_excluded(rel):
            problems.append(f"forbidden file in bundle: {rel}")
    return problems


def _inspect(dest: Path) -> dict:
    """Summary of a built bundle for dry-run reporting."""
    files = [p for p in dest.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    top = []
    for child in sorted(dest.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            n = sum(1 for p in child.rglob("*") if p.is_file())
            top.append(f"  {child.name}/  ({n} files)")
        else:
            top.append(f"  {child.name}")
    return {
        "root": dest,
        "n_files": len(files),
        "total_bytes": total_bytes,
        "top_level": top,
        "commit": git_commit(REPO),
    }


def _report(dest: Path, problems: list[str]) -> int:
    info = _inspect(dest)
    print(f"bundle root: {info['root']}")
    print(f"files: {info['n_files']}  bytes: {info['total_bytes']}")
    print(f"git commit: {info['commit']}")
    print("top level:")
    for line in info["top_level"]:
        print(line)
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problem(s)):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("validation: OK")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--dest", default=str(DEFAULT_DEST))
    p.add_argument(
        "--dry-run", action="store_true",
        help="build into a temporary directory, validate, inspect, and remove it",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="p3a_bundle_") as tmp:
            dest = Path(tmp) / "code_bundle"
            build_bundle(repo=args.repo, dest=dest)
            problems = validate_bundle(dest)
            print("dry run: built a throwaway bundle, nothing uploaded, "
                  "kaggle/code_bundle untouched")
            return _report(dest, problems)
    dest = build_bundle(repo=args.repo, dest=args.dest)
    return _report(dest, validate_bundle(dest))


if __name__ == "__main__":
    raise SystemExit(main())
