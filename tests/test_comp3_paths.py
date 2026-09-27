"""CPU tests for the Kaggle-safe Comp III loader path resolution."""
from pathlib import Path

import numpy as np
import pytest

from p3a.data.loaders.comp3 import find_file


def _write(path: Path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_file_prefers_direct_child(tmp_path):
    _write(tmp_path / "Subject_A_Train.mat", "flat")
    assert find_file(tmp_path, "Subject_A_Train.mat").parent == tmp_path


def test_find_file_skips_direct_same_name_directory(tmp_path):
    # Kaggle can attach a *directory* named like the .mat; exists() would return
    # that folder, so find_file must require an actual file and keep searching.
    (tmp_path / "Subject_A_Train.mat").mkdir()
    _write(tmp_path / "Subject_A_Train.mat" / "nested" / "Subject_A_Train.mat", "real")
    found = find_file(tmp_path, "Subject_A_Train.mat")
    assert found.is_file()
    assert found.read_text() == "real"


def test_find_file_recurses_into_nested_kaggle_layout(tmp_path):
    nested = tmp_path / "Dataset p300" / "Subject_A_Train.mat"
    _write(nested / "Subject_A_Train.mat", "nested")
    found = find_file(tmp_path, "Subject_A_Train.mat")
    assert found == nested / "Subject_A_Train.mat"
    assert found.read_text() == "nested"


def test_find_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        find_file(tmp_path, "Subject_B_Train.mat")


def test_find_file_does_not_need_numpy_data(tmp_path):
    # Only name resolution is tested; no .mat parsing or real data required.
    _write(tmp_path / "nested" / "true_labels_a.txt", "ABC")
    assert find_file(tmp_path, "true_labels_a.txt").read_text() == "ABC"
