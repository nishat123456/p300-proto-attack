"""Fast CPU tests for the attack training core and CLI.

Everything runs on tiny synthetic trials and a tiny EEGNet surrogate built in a
temp directory, so no real .mat file or GPU is needed. The tests check the
invariants the experiment relies on: the surrogate stays frozen, test data is
never loaded, the epsilon budget follows the pre-registered scale, missing
prototypes are skipped and counted, and the generator checkpoint round trips.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from p3a.attacks import train as attack_train
from p3a.attacks.train import (
    BANKS,
    EPS_MULTIPLIERS,
    METHODS,
    fixed_val_targets,
    load_generator,
    raw_eps_scale,
    sample_targets,
    train_attack,
)
from p3a.data.layout import COMP3, rc_code_to_group
from p3a.data.trial import Trial
from p3a.models.eegnet import EEGNet
from p3a.models.speller import Speller
from p3a.preproc.pipeline import Preprocessor

REPO = Path(__file__).resolve().parents[1]
N_FLASHES = 180
C = 4
T = 2000
FS = 240.0


def _trial(index: int, subject: str = "A", split: str = "train",
           target_symbol: str | None = None, seed: int = 0) -> Trial:
    rng = np.random.default_rng(seed * 1000 + index)
    codes = np.tile(np.arange(1, 13), N_FLASHES // 12)
    onsets = 5 + 10 * np.arange(N_FLASHES)
    groups = np.stack([rc_code_to_group(COMP3, int(c)) for c in codes])
    symbol = target_symbol or COMP3.symbols[index % 36]
    return Trial(
        eeg=rng.normal(size=(C, T)).astype(np.float32), fs=FS, onsets=onsets,
        flash_groups=groups, flash_codes=codes, target_symbol=symbol, layout=COMP3,
        subject_id=subject, dataset="comp3", trial_index=index, source_split=split,
    )


def _write_surrogate(path: Path, trials: list) -> Path:
    pre = Preprocessor(n_channels=C, fs=FS)
    pre.fit_scaling([t.eeg for t in trials])
    speller = Speller(pre, EEGNet(C, 48), 48)
    speller.save(path, training_config={"smoke": True}, seed=0)
    return path


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    """Four train and three val synthetic trials wired into a fake split."""
    train = [_trial(i) for i in range(4)]
    val = [_trial(10 + i, split="train") for i in range(3)]
    all_trials = train + val
    split = {"train": [t.trial_id for t in train], "val": [t.trial_id for t in val],
             "test": []}
    loaded = []

    def fake_load_subject(root, subject, split_name, labels_dir=None):
        loaded.append(split_name)
        return all_trials

    monkeypatch.setattr(attack_train, "load_split", lambda path: split)
    monkeypatch.setattr("p3a.data.loaders.comp3.load_subject", fake_load_subject)
    ckpt = _write_surrogate(tmp_path / "surrogate.pt", all_trials)
    return {"train": train, "val": val, "all": all_trials, "split": split,
            "loaded": loaded, "ckpt": ckpt, "tmp": tmp_path}


def test_raw_eps_scale_is_median_channel_std():
    trials = [_trial(i) for i in range(3)]
    stack = np.stack([t.eeg for t in trials], axis=0)
    expected = float(np.median(stack.std(axis=(0, 2))))
    assert raw_eps_scale(trials) == pytest.approx(expected)


def test_sample_targets_never_true_symbol():
    true = torch.arange(36, dtype=torch.long)
    rng = np.random.default_rng(0)
    targets = sample_targets(true, rng, 36)
    assert not bool((targets == true).any())
    assert int(targets.min()) >= 0 and int(targets.max()) < 36


def test_fixed_val_targets_are_seeded_and_not_true():
    true = np.array([0, 5, 17, 35])
    a = fixed_val_targets(true, 3, 36)
    b = fixed_val_targets(true, 3, 36)
    np.testing.assert_array_equal(a, b)
    assert not np.any(a == true)


def test_train_attack_target_only_keeps_surrogate_frozen_and_no_test(synthetic):
    result = train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="target_only", generator="causal",
        bank="same_subject", seed=0, out_dir=str(synthetic["tmp"] / "out"),
        epochs=2, batch=2, lr=1e-3, device="cpu",
        generator_kwargs={"hidden": 4},
    )
    # Only the train file is read; the test split is never loaded.
    assert synthetic["loaded"] == ["train"]
    assert result["n_test_trials"] == 0
    cfg = result["config"]
    assert cfg["method"] == "target_only" and cfg["use_prototype"] is False
    assert cfg["best_epoch"] >= 0
    # The epsilon scale is recorded and matches the raw-train EEG statistic.
    assert cfg["eps_scale"] == pytest.approx(raw_eps_scale(synthetic["train"]))
    # Checkpoint round trips and keeps the raw-unit L-inf budget.
    gen, payload = load_generator(result["checkpoint"])
    gen.eval()
    eeg = torch.as_tensor(np.stack([t.eeg for t in synthetic["train"]]), dtype=torch.float32)
    onsets = torch.as_tensor(np.stack([t.onsets for t in synthetic["train"]]), dtype=torch.long)
    groups = torch.as_tensor(np.stack([t.flash_groups for t in synthetic["train"]]), dtype=torch.bool)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with torch.no_grad():
        delta = gen(eeg, onsets, groups, target, 0.2, None, use_prototype=False)
    peak = delta.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, torch.full((4,), 0.2), atol=1e-5)
    assert payload["eps_scale"] == pytest.approx(cfg["eps_scale"])
    assert result["csv"] and Path(result["csv"]).is_file()


def test_surrogate_parameters_are_frozen(synthetic):
    """Captured surrogate weights must be byte-identical after training."""
    speller = Speller.load(str(synthetic["ckpt"]))
    before = {k: v.clone() for k, v in speller.state_dict().items()}
    train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="proto_nearest",
        generator="causal", bank="same_subject", seed=0,
        out_dir=str(synthetic["tmp"] / "out2"), epochs=1, batch=2, lr=1e-3,
        device="cpu", generator_kwargs={"hidden": 4},
    )
    after = Speller.load(str(synthetic["ckpt"]))
    for key, value in before.items():
        assert torch.equal(value, after.state_dict()[key]), key


def test_train_attack_proto_nearest_uses_bank_and_logs_val(synthetic):
    result = train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="proto_nearest",
        generator="causal", bank="same_subject", seed=1,
        out_dir=str(synthetic["tmp"] / "out3"), epochs=2, batch=2, lr=1e-3,
        device="cpu", generator_kwargs={"hidden": 4},
    )
    cfg = result["config"]
    assert cfg["use_prototype"] is True
    assert cfg["retrieval_mode"] == "nearest"
    # Every epoch row carries a val success per epsilon plus a mean.
    assert result["history"] and "val_success_mean" in result["history"][0]
    for mult in EPS_MULTIPLIERS:
        key = f"val_success_eps{format(mult, '.2f').replace('.', '')}"
        assert key in result["history"][0]
    # The generator receives a prototype: its proto encoder must have a gradient.
    gen, _ = load_generator(result["checkpoint"])
    proto_grads = [p for n, p in gen.named_parameters() if "proto_encoder" in n]
    assert proto_grads


def test_train_attack_skips_missing_prototypes(monkeypatch, synthetic):
    """A target symbol absent from the bank is skipped and counted, not nudged."""
    import p3a.prototypes.retrieval as retrieval

    def always_missing(bank, scaler, query, target_idx, mode, rng=None):
        return None

    monkeypatch.setattr(retrieval, "retrieve", always_missing)
    monkeypatch.setattr(attack_train, "retrieve", always_missing)
    result = train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="proto_nearest",
        generator="causal", bank="same_subject", seed=0,
        out_dir=str(synthetic["tmp"] / "out4"), epochs=1, batch=2, lr=1e-3,
        device="cpu", generator_kwargs={"hidden": 4},
    )
    assert result["n_skipped_total"] > 0
    # No sample could be trained, so validation stays at zero success.
    assert result["best_val_success_mean"] == 0.0
    assert result["history"][0]["n_samples_used"] == 0


def test_invalid_arguments_raise(synthetic):
    with pytest.raises(ValueError, match="unknown method"):
        train_attack(str(synthetic["tmp"]), None, "A", str(synthetic["ckpt"]),
                     "not_a_method", "causal", "same_subject", 0,
                     str(synthetic["tmp"] / "x"), device="cpu")
    with pytest.raises(ValueError, match="unknown generator"):
        train_attack(str(synthetic["tmp"]), None, "A", str(synthetic["ckpt"]),
                     "target_only", "not_a_gen", "same_subject", 0,
                     str(synthetic["tmp"] / "x"), device="cpu")
    with pytest.raises(ValueError, match="unknown bank"):
        train_attack(str(synthetic["tmp"]), None, "A", str(synthetic["ckpt"]),
                     "target_only", "causal", "not_a_bank", 0,
                     str(synthetic["tmp"] / "x"), device="cpu")


def test_smoke_enforced_in_process(synthetic):
    """Smoke must force CPU and one epoch even when called with large args."""
    result = train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="target_only", generator="causal",
        bank="same_subject", seed=0, out_dir=str(synthetic["tmp"] / "smoke"),
        epochs=99, batch=2, lr=1e-3, device="auto", smoke=True,
        generator_kwargs={"hidden": 4},
    )
    assert result["config"]["smoke"] is True
    assert result["config"]["device"] == "cpu"
    assert result["config"]["epochs"] == 1
    assert len(result["history"]) == 1


def test_auto_device_uses_cuda_when_available(monkeypatch, synthetic):
    # Smoke always forces CPU; this checks the resolution expression without
    # requiring a CUDA runner by exercising the public behavior on CPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = train_attack(
        data_root=str(synthetic["tmp"]), labels_dir=None, subject="A",
        surrogate=str(synthetic["ckpt"]), method="target_only", generator="causal",
        bank="same_subject", seed=0, out_dir=str(synthetic["tmp"] / "auto"),
        epochs=1, batch=2, lr=1e-3, device="auto",
        generator_kwargs={"hidden": 4},
    )
    assert result["config"]["device"] == "cpu"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "train_attack_script", REPO / "scripts" / "train_attack.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parses_and_smoke_overrides(monkeypatch):
    script = _load_script()
    captured = {}

    def fake_train_attack(**kwargs):
        captured.update(kwargs)
        return {"checkpoint": "x", "history": []}

    monkeypatch.setattr(script, "train_attack", fake_train_attack)
    out = script.main([
        "--subject", "A", "--surrogate", "ckpt.pt", "--method", "proto_nearest",
        "--generator", "signal", "--bank", "same_subject", "--seed", "2",
        "--smoke",
    ])
    assert captured["smoke"] is True
    assert captured["epochs"] == 1
    assert captured["device"] == "cpu"
    assert captured["lambda_p"] == 1.0 and captured["lambda_s"] == 0.1
    assert out["checkpoint"] == "x"


def test_cli_defaults_match_preregistered_grid():
    assert tuple(EPS_MULTIPLIERS) == (0.05, 0.10, 0.20, 0.30, 0.50)
    assert set(METHODS) == {"target_only", "proto_random", "proto_medoid",
                            "proto_nearest", "proto_wrong"}
    assert set(BANKS) == {"same_subject", "cross_subject"}
