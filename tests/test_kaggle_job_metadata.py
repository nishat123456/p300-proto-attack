"""Validate the Kaggle job metadata against Prompt 5 exactly.

These are pure file checks: no Kaggle CLI call is made.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOB_DIR = REPO / "kaggle" / "jobs" / "train_decoders"
META_PATH = JOB_DIR / "kernel-metadata.json"

EXPECTED = {
    "id": "mustaqimnishat/p3a-train-decoders",
    "title": "p3a-train-decoders",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "mustaqimnishat/p300-proto-attack-code",
        "manthandixit726262/bci-competition-iii-dataset-ii",
    ],
}


def _meta():
    return json.loads(META_PATH.read_text())


def test_metadata_file_exists():
    assert META_PATH.is_file()


def test_metadata_matches_prompt_exactly():
    # Ignore any extra keys Kaggle tolerates; every required key must match.
    meta = _meta()
    for key, value in EXPECTED.items():
        assert meta.get(key) == value, f"{key}: {meta.get(key)!r} != {value!r}"


def test_dataset_sources_order_and_contents():
    assert _meta()["dataset_sources"] == EXPECTED["dataset_sources"]


def test_code_file_present_next_to_metadata():
    assert (JOB_DIR / _meta()["code_file"]).is_file()


def test_job_files_have_no_credentials():
    for path in JOB_DIR.rglob("*"):
        if path.is_file():
            name = path.name
            assert name not in {"kaggle.json", ".netrc", ".server_pw", ".env"}


def test_kaggle_sh_has_required_subcommands():
    script = (REPO / "scripts" / "kaggle.sh").read_text()
    for sub in ("bundle", "push", "status", "output", "dry-run"):
        assert sub in script
    # Prompt 5 fixes the accelerator name; the script must not guess others.
    assert "NvidiaTeslaT4" in script
