"""Validate the kill-test Kaggle job metadata against Prompt 12 (no CLI calls)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOB_DIR = REPO / "kaggle" / "jobs" / "kill_test"
META_PATH = JOB_DIR / "kernel-metadata.json"

EXPECTED = {
    "id": "mustaqimnishat/p3a-kill-test",
    "title": "p3a-kill-test",
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
    "kernel_sources": [
        "mustaqimnishat/p3a-train-decoders",
    ],
}


def _meta():
    return json.loads(META_PATH.read_text())


def test_metadata_file_exists():
    assert META_PATH.is_file()


def test_metadata_matches_prompt():
    meta = _meta()
    for key, value in EXPECTED.items():
        assert meta.get(key) == value, f"{key}: {meta.get(key)!r} != {value!r}"


def test_kernel_sources_attach_the_decoders_kernel():
    assert _meta()["kernel_sources"] == ["mustaqimnishat/p3a-train-decoders"]


def test_code_file_present_next_to_metadata():
    assert (JOB_DIR / _meta()["code_file"]).is_file()


def test_job_files_have_no_credentials():
    for path in JOB_DIR.rglob("*"):
        if path.is_file():
            assert path.name not in {"kaggle.json", ".netrc", ".server_pw", ".env"}


def test_run_py_mentions_shard_env_vars_and_no_credentials():
    text = (JOB_DIR / "run.py").read_text()
    assert "P3A_SHARD_INDEX" in text and "P3A_SHARD_COUNT" in text
    assert "shard.json" in text
    for needle in ("KAGGLE_KEY", "KAGGLE_USERNAME", "kaggle.json"):
        assert needle not in text


def test_kaggle_sh_supports_shard_and_output():
    script = (REPO / "scripts" / "kaggle.sh").read_text()
    for sub in ("shard)", "output)"):
        assert sub in script
    assert "shard.json" in script
