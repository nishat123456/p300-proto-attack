"""CPU tests for reproducibility helpers (seed, git state, config sidecar)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from p3a.utils.repro import environment, git_record, set_seed, write_config


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (np.random.rand(), float(torch.rand(1)))
    set_seed(123)
    b = (np.random.rand(), float(torch.rand(1)))
    assert a == b


def test_environment_reports_versions_and_seed():
    env = environment(5)
    assert env["seed"] == 5
    assert env["torch"] and env["numpy"] and env["python"]
    assert isinstance(env["git_dirty"], bool)
    assert env["git_commit"]


def test_git_record_detects_dirty_state(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    clean = git_record(repo)
    assert clean["git_dirty"] is False
    assert len(clean["git_commit"]) >= 7

    (repo / "f.txt").write_text("changed")
    dirty = git_record(repo)
    assert dirty["git_dirty"] is True
    assert dirty["git_state_hash"] != clean["git_state_hash"]


def test_git_record_hashes_untracked_file_contents(tmp_path):
    """Regression: editing untracked source changes the hash, not just its name."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    module = repo / "nested" / "brand_new.py"
    module.parent.mkdir()
    module.write_text("def f():\n    return 1\n")
    first = git_record(repo)
    assert first["git_dirty"] is True

    # Same filename, same git status, different contents: hash must change.
    module.write_text("def f():\n    return 2\n")
    second = git_record(repo)
    assert second["git_state_hash"] != first["git_state_hash"]


def test_git_record_ignores_generated_outputs(tmp_path):
    """Editing results/ must not make source look dirty."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    out = repo / "results" / "run.json"
    out.parent.mkdir()
    out.write_text("{}")
    rec = git_record(repo)
    assert rec["git_dirty"] is False


def test_git_record_outside_repo_is_safe(tmp_path):
    rec = git_record(tmp_path)
    assert rec["git_commit"] == "unknown"
    assert rec["git_dirty"] is False


def test_git_record_uses_bundle_commit_outside_repo(tmp_path):
    commit = "52e967e44f2269bb302b55d511e11694605ddca9"
    (tmp_path / "GIT_COMMIT.txt").write_text(commit + "\n")
    rec = git_record(tmp_path)
    assert rec["git_commit"] == commit
    assert rec["git_dirty"] is False


def test_write_config_creates_json(tmp_path):
    path = write_config(tmp_path / "out", "run.json", {"seed": 3, "model": "eegnet"})
    assert json.loads(path.read_text()) == {"seed": 3, "model": "eegnet"}
