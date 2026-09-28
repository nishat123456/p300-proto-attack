"""Integration test for the high-level evaluate_attacks entry point (Prompt 10).

The raw loader, the split loader and the surrogate path are replaced by
synthetic equivalents, so the wiring is exercised on CPU in seconds without any
.mat file. The test checks the properties that matter at the seam: only the
train split is read to compute the epsilon scale, only the test split is
evaluated, and the pre-registered target subsets are matched to test trials by
id.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from p3a.eval import evaluate as evaluate_mod
from p3a.eval.evaluate import evaluate_attacks

from tests.eval_helpers import make_loadable_speller, make_trial


def _write_surrogate(path: Path, trials) -> Path:
    make_loadable_speller(trials).save(path, training_config={"smoke": True}, seed=0)
    return path


def test_evaluate_attacks_smoke_uses_train_scale_and_test_trials(monkeypatch, tmp_path):
    # 4 train trials and 3 test trials, C = 4, all with distinct ids.
    train = [make_trial(i, split="train", target_symbol=None) for i in range(4)]
    test = [make_trial(100 + i, split="test", target_symbol=None) for i in range(3)]
    all_trials = train + test
    split = {"train": [t.trial_id for t in train], "val": [],
             "test": [t.trial_id for t in test]}
    targets = {t.trial_id: [s for s in "ABCDEFGHIJ" if s != t.target_symbol][:10]
               for t in test}
    loaded = []

    def fake_load_subject(root, subject, split_name, labels_dir=None):
        loaded.append(split_name)
        return all_trials if split_name == "train" else test

    monkeypatch.setattr("p3a.data.loaders.comp3.load_subject", fake_load_subject)
    monkeypatch.setattr(evaluate_mod, "load_attack_targets", lambda subject, splits_dir=None: targets)
    monkeypatch.setattr(evaluate_mod, "repo_root", lambda: tmp_path)
    # The high-level function imports load_split locally; patch the module it uses.
    import p3a.data.splits as splits_mod
    monkeypatch.setattr(splits_mod, "load", lambda path: split)
    monkeypatch.setattr(evaluate_mod, "discover_subjects", lambda splits_dir: ["A"])

    ckpt = _write_surrogate(tmp_path / "surrogate.pt", all_trials)
    result = evaluate_attacks(
        data_root=str(tmp_path), labels_dir=None, subject="A", surrogate=str(ckpt),
        out_dir=str(tmp_path / "out"), generator_dir=str(tmp_path),
        tiny_noise_dir=str(tmp_path), victim_dir=str(tmp_path), device="cpu",
        specs=[evaluate_mod.MethodSpec("noise", "noise")],
        eps_multipliers=(0.20,), victims=False,
    )
    # Train is read once for the epsilon scale; test once for evaluation.
    assert loaded == ["test", "train"] or sorted(loaded) == ["test", "train"]
    assert result.n_trials == 3
    assert result.n_targets == 10
    assert result.n_rows == 3 * 10
    frame = pd.read_parquet(result.parquet)
    assert set(frame["eps_multiplier"]) == {0.20}
    # Every stored target symbol belongs to the pre-registered subset.
    for trial_id, group in frame.groupby("trial_id"):
        assert set(group["target_symbol"]) <= set(targets[trial_id])
    # The config sidecar carries the eval config and provenance.
    config = json.loads((tmp_path / "out" / "A" / "evaluate_attacks.config.json").read_text())
    assert config["config"]["subject"] == "A"
    assert config["config"]["victims"] is False
    assert config["provenance"]["git_commit"]


def test_evaluate_attacks_missing_victim_raises(monkeypatch, tmp_path):
    trials = [make_trial(i, split="train") for i in range(2)]
    test = [make_trial(50 + i, split="test") for i in range(2)]
    split = {"train": [t.trial_id for t in trials], "val": [],
             "test": [t.trial_id for t in test]}
    targets = {t.trial_id: ["A", "B", "C"] for t in test}

    monkeypatch.setattr("p3a.data.loaders.comp3.load_subject",
                        lambda root, subject, s, labels_dir=None: trials if s == "train" else test)
    monkeypatch.setattr(evaluate_mod, "load_attack_targets", lambda subject, splits_dir=None: targets)
    monkeypatch.setattr(evaluate_mod, "repo_root", lambda: tmp_path)
    import p3a.data.splits as splits_mod
    monkeypatch.setattr(splits_mod, "load", lambda path: split)
    ckpt = _write_surrogate(tmp_path / "surrogate.pt", trials + test)
    with pytest.raises(FileNotFoundError, match="victim"):
        evaluate_attacks(
            data_root=str(tmp_path), labels_dir=None, subject="A",
            surrogate=str(ckpt), out_dir=str(tmp_path / "out"),
            generator_dir=str(tmp_path), tiny_noise_dir=str(tmp_path),
            victim_dir=str(tmp_path), device="cpu",
            specs=[evaluate_mod.MethodSpec("noise", "noise")],
            eps_multipliers=(0.20,), victims=True, limit_trials=1,
        )


def test_target_only_does_not_build_prototype_bank(monkeypatch, tmp_path):
    train = [make_trial(i, split="train") for i in range(2)]
    test = [make_trial(20 + i, split="test") for i in range(2)]
    split = {"train": [t.trial_id for t in train], "val": [],
             "test": [t.trial_id for t in test]}
    targets = {t.trial_id: ["A", "B", "C"] for t in test}

    monkeypatch.setattr("p3a.data.loaders.comp3.load_subject",
                        lambda root, subject, s, labels_dir=None: train if s == "train" else test)
    monkeypatch.setattr(evaluate_mod, "load_attack_targets", lambda subject, splits_dir=None: targets)
    monkeypatch.setattr(evaluate_mod, "repo_root", lambda: tmp_path)
    import p3a.data.splits as splits_mod
    monkeypatch.setattr(splits_mod, "load", lambda path: split)
    monkeypatch.setattr(evaluate_mod, "build_bank",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bank built")))

    surrogate = _write_surrogate(tmp_path / "surrogate.pt", train + test)
    from p3a.attacks.train import build_generator, save_generator
    generator = build_generator("causal", 36, 4, False,
                                {"hidden": 4, "template_len": 8})
    save_generator(
        tmp_path / "A_target_only_causal_cross_subject_s0.pt", generator,
        "causal", "target_only", "cross_subject", {}, {},
        {"hidden": 4, "template_len": 8}, 1.0, 36, 4, False,
    )
    result = evaluate_attacks(
        data_root=str(tmp_path), labels_dir=None, subject="A", surrogate=str(surrogate),
        out_dir=str(tmp_path / "out"), generator_dir=str(tmp_path),
        tiny_noise_dir=str(tmp_path), victim_dir=str(tmp_path), device="cpu",
        specs=[evaluate_mod.MethodSpec("target_only", "generator", "causal",
                                       "cross_subject")],
        eps_multipliers=(0.20,), victims=False,
    )
    assert result.n_rows == 6
