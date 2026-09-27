"""CPU tests for the Kaggle code bundle builder.

Each test builds a real bundle from this repo into a temporary directory, so no
Kaggle call, upload, or write into kaggle/code_bundle happens. The builder is
fast because it only copies small source/metadata files.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import make_code_bundle as mcb  # noqa: E402


@pytest.fixture
def built(tmp_path):
    dest = tmp_path / "code_bundle"
    mcb.build_bundle(repo=REPO, dest=dest)
    return dest


def test_bundle_validates_clean(built):
    assert mcb.validate_bundle(built) == []


def test_bundle_contains_required_payload(built):
    for rel in mcb.REQUIRED_FILES:
        assert (built / rel).is_file(), rel
    for name in mcb.REQUIRED_DIRS:
        assert (built / name).is_dir(), name
    # configs/ is copied even when empty so the Kaggle job's layout matches.
    assert (built / "configs").is_dir()
    assert (built / "data/raw/comp3/true_labels_a.txt").is_file()
    assert (built / "data/raw/comp3/true_labels_b.txt").is_file()


def test_dataset_metadata_is_exact(built):
    meta = json.loads((built / "dataset-metadata.json").read_text())
    assert meta == {
        "title": "p300-proto-attack-code",
        "id": "mustaqimnishat/p300-proto-attack-code",
        "licenses": [{"name": "CC0-1.0"}],
    }


def test_git_commit_written_and_matches_head(built):
    text = (built / "GIT_COMMIT.txt").read_text().strip()
    assert text
    assert text == mcb.git_commit(REPO)


def test_bundle_excludes_data_venv_results_and_nested_bundle(built):
    rels = [p.relative_to(built).as_posix() for p in built.rglob("*") if p.is_file()]
    forbidden_parts = {
        ".venv", "results", "checkpoints", "code_bundle", "__pycache__",
        "agent_logs", "prompts", ".git", ".kilo",
    }
    for rel in rels:
        assert not (set(Path(rel).parts) & forbidden_parts), rel
        assert not rel.endswith((".mat", ".pt", ".joblib", ".pyc"))
        assert Path(rel).name not in {"kaggle.json", ".server_pw", ".env"}
    # The source repo has a .mat and a results/ tree; neither may leak in.
    assert not any(rel.endswith(".mat") for rel in rels)


def test_build_twice_does_not_recurse(built, tmp_path):
    # Rebuilding over an existing bundle cleans it first, so a nested copy can
    # never accumulate.
    mcb.build_bundle(repo=REPO, dest=built)
    assert mcb.validate_bundle(built) == []
    assert not (built / "code_bundle").exists()


def test_is_excluded_rules():
    assert mcb.is_excluded("results/run.pt")
    assert mcb.is_excluded("src/p3a/__pycache__/x.pyc")
    assert mcb.is_excluded("data/raw/comp3/Subject_A_Train.mat")
    assert mcb.is_excluded("kaggle/code_bundle/pyproject.toml")
    assert mcb.is_excluded("src/p3a.egg-info/PKG-INFO")
    assert mcb.is_excluded(".venv/lib/x.py")
    assert mcb.is_excluded("kaggle/agent_logs/p01.log")
    assert mcb.is_excluded("kaggle/prompts/p05.txt")
    assert mcb.is_excluded(".env")
    assert mcb.is_excluded("kaggle.json")
    assert not mcb.is_excluded("src/p3a/models/speller.py")
    assert not mcb.is_excluded("tests/test_preproc.py")


def test_validate_detects_planted_forbidden_file(built):
    (built / "checkpoints").mkdir()
    (built / "checkpoints" / "leak.pt").write_text("x")
    problems = mcb.validate_bundle(built)
    assert any("forbidden file" in p for p in problems)
    assert any("leak.pt" in p for p in problems)


def test_validate_detects_missing_required_file(built):
    (built / "GIT_COMMIT.txt").unlink()
    problems = mcb.validate_bundle(built)
    assert any("GIT_COMMIT.txt" in p for p in problems)


def test_validate_detects_wrong_metadata(built):
    (built / "dataset-metadata.json").write_text(json.dumps({"id": "someone/else"}))
    problems = mcb.validate_bundle(built)
    assert any("dataset-metadata.json" in p for p in problems)


def test_dry_run_does_not_touch_real_bundle(tmp_path, capsys):
    dest = tmp_path / "never_created"
    rc = mcb.main(["--repo", str(REPO), "--dest", str(dest), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "validation: OK" in out
    assert "nothing uploaded" in out
    assert not dest.exists()
