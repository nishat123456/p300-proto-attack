import pytest

from p3a.data import splits


def test_write_once(tmp_path):
    ids = [f"t{i}" for i in range(20)]
    s = splits.make_train_val(ids, n_val=5, seed=0)
    assert len(s["val"]) == 5 and not set(s["val"]) & set(s["train"])
    p = tmp_path / "s.json"
    h = splits.write_once(p, s)
    assert splits.write_once(p, s) == h
    with pytest.raises(RuntimeError):
        splits.write_once(p, splits.make_train_val(ids, n_val=5, seed=1))
    assert splits.load(p) == s
