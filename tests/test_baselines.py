"""Fast CPU tests for the raw-unit attack baselines (Prompt 9).

The tests build a tiny random speller and tiny synthetic trials, so the whole
module runs in seconds on CPU. They check the properties the experiment depends
on: exact per-sample L-infinity budgets, that PGD actually raises targeted
success, that the Tiny Noise template uses the same causal placement as the
generators and stays inside its box, that checkpoints round trip, and that the
training entry point only ever reads the frozen train split.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from p3a.attacks import baselines as baseline_mod
from p3a.attacks.baselines import (
    TinyNoiseTemplate,
    random_noise,
    targeted_pgd,
    train_tiny_noise,
)
from p3a.attacks.generators import linf_rescale, place_template_at_onsets
from p3a.data.layout import COMP3, rc_code_to_group
from p3a.data.trial import Trial
from p3a.models.eegnet import EEGNet
from p3a.models.speller import Speller
from p3a.preproc.pipeline import Preprocessor

REPO = Path(__file__).resolve().parents[1]
C = 4
T = 2000
FS = 240.0
N_FLASHES = 180
L = 48


class _TinyDecoder(nn.Module):
    """Affine flash decoder so the symbol logits are an affine map of raw EEG."""

    def __init__(self, n_channels: int, L: int):
        super().__init__()
        self.fc = nn.Linear(n_channels * L, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1)).squeeze(1)


def _trial(index: int, *, split: str = "train", target_symbol: str | None = None,
           seed: int = 0, subject: str = "A") -> Trial:
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


def _toy_speller(trials: list, seed: int = 0) -> Speller:
    """Random affine decoder: cheap, and its logits are affine in raw EEG."""
    torch.manual_seed(seed)
    pre = Preprocessor(n_channels=C, fs=FS)
    pre.fit_scaling([t.eeg for t in trials])
    return Speller(pre, _TinyDecoder(C, L), L).eval()


def _loadable_speller(trials: list, seed: int = 0) -> Speller:
    """Registry-backed EEGNet speller so the checkpoint can be reloaded."""
    torch.manual_seed(seed)
    pre = Preprocessor(n_channels=C, fs=FS)
    pre.fit_scaling([t.eeg for t in trials])
    return Speller(pre, EEGNet(C, L), L).eval()


def _batch(trials: list, device="cpu"):
    eeg = torch.as_tensor(np.stack([t.eeg for t in trials]), dtype=torch.float32, device=device)
    onsets = torch.as_tensor(np.stack([t.onsets for t in trials]), dtype=torch.long, device=device)
    groups = torch.as_tensor(np.stack([t.flash_groups for t in trials]), dtype=torch.bool, device=device)
    return eeg, onsets, groups


def test_targeted_pgd_respects_linf_budget():
    trials = [_trial(i) for i in range(3)]
    speller = _toy_speller(trials)
    eeg, onsets, groups = _batch(trials)
    speller.requires_grad_(False)
    target = torch.tensor([3, 7, 30], dtype=torch.long)
    eps = torch.tensor([0.05, 0.20, 0.5])
    delta = targeted_pgd(speller, eeg, onsets, groups, target, eps, steps=5)
    assert delta.shape == eeg.shape and delta.dtype == torch.float32
    peak = delta.flatten(1).abs().amax(dim=1)
    assert torch.all(peak <= eps + 1e-6)
    assert torch.all(peak > 0)


def test_targeted_pgd_zero_budget_is_zero():
    trials = [_trial(0)]
    speller = _toy_speller(trials)
    eeg, onsets, groups = _batch(trials)
    speller.requires_grad_(False)
    delta = targeted_pgd(speller, eeg, onsets, groups, torch.tensor([1]), 0.0, steps=3)
    assert torch.count_nonzero(delta) == 0


def test_targeted_pgd_raises_targeted_success_and_lowers_loss():
    trials = [_trial(i) for i in range(6)]
    speller = _toy_speller(trials)
    eeg, onsets, groups = _batch(trials)
    speller.requires_grad_(False)
    target = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    def score(x):
        with torch.no_grad():
            logits = speller(x, onsets, groups)["logits"]
            ce = torch.nn.functional.cross_entropy(logits, target).item()
        return int((logits.argmax(dim=1) == target).sum()), ce

    # Budget is generously larger than the EEG scale so the affine toy decoder
    # can be moved onto the target from any starting point.
    eps = 20.0
    before_ce = score(eeg)[1]
    delta = targeted_pgd(speller, eeg, onsets, groups, target, eps, steps=40)
    after_correct, after_ce = score(eeg + delta)
    assert after_ce < before_ce
    assert after_correct == len(trials)


def test_random_noise_shapes_budget_and_reproducibility():
    eps = torch.tensor([0.1, 0.2, 0.3])
    a = random_noise((3, C, T), eps, rng=np.random.default_rng(0))
    b = random_noise((3, C, T), eps, rng=np.random.default_rng(0))
    assert a.shape == (3, C, T) and a.dtype == torch.float32
    torch.testing.assert_close(a, b)
    peak = a.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, eps, atol=1e-5)
    zero = random_noise((2, C, T), 0.0, rng=np.random.default_rng(1))
    assert torch.count_nonzero(zero) == 0


def test_tiny_noise_placement_matches_shared_causal_semantics():
    trials = [_trial(i) for i in range(2)]
    eeg, onsets, groups = _batch(trials)
    target = torch.tensor([4, 20], dtype=torch.long)
    eps = 0.3
    tmpl = TinyNoiseTemplate(C, template_len=192)
    tmpl.template = torch.randn(C, 192)

    delta = tmpl.delta(onsets, groups, target, T, eps)
    expected = place_template_at_onsets(
        tmpl.template[None].expand(2, C, 192), onsets, groups, target, T
    )
    expected = linf_rescale(expected, eps)
    torch.testing.assert_close(delta, expected)
    peak = delta.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, torch.full((2,), eps), atol=1e-5)
    # Causality: the template depends on the schedule and template only, never
    # on source EEG content, so placement is deterministic and reusable.
    delta_again = tmpl.delta(onsets, groups, target, T, eps)
    torch.testing.assert_close(delta, delta_again)


def test_tiny_noise_fit_reduces_ce_and_keeps_budget():
    trials = [_trial(i) for i in range(6)]
    speller = _toy_speller(trials)
    eps = float(np.median(np.stack([t.eeg for t in trials]).std(axis=(0, 2)))) * 0.3
    tmpl = TinyNoiseTemplate(C, template_len=192)
    history = tmpl.fit(speller, trials, eps, epochs=15, lr=0.05, seed=0, device="cpu")
    assert history["ce_history"][-1] < history["ce_history"][0]
    assert float(tmpl.template.abs().max()) <= eps + 1e-6


def test_tiny_noise_save_load_roundtrip(tmp_path):
    tmpl = TinyNoiseTemplate(C, template_len=192)
    tmpl.template = torch.randn(C, 192)
    path = tmpl.save(tmp_path / "tn.pt")
    loaded = TinyNoiseTemplate.load(path)
    assert loaded.n_channels == C and loaded.template_len == 192
    torch.testing.assert_close(loaded.template, tmpl.template)


def test_train_tiny_noise_writes_one_checkpoint_per_eps(monkeypatch, tmp_path):
    train = [_trial(i) for i in range(4)]
    split = {"train": [t.trial_id for t in train], "val": [], "test": []}
    loaded_splits = []

    def fake_load_subject(root, subject, split_name, labels_dir=None):
        loaded_splits.append(split_name)
        return train

    monkeypatch.setattr("p3a.data.loaders.comp3.load_subject", fake_load_subject)
    monkeypatch.setattr("p3a.data.splits.load", lambda path: split)
    ckpt = tmp_path / "surrogate.pt"
    _loadable_speller(train).save(ckpt, training_config={"smoke": True}, seed=0)

    result = train_tiny_noise(
        data_root=str(tmp_path), labels_dir=None, subject="A", surrogate=str(ckpt),
        seed=0, out_dir=str(tmp_path / "out"), epochs=1, lr=0.05, device="cpu",
        eps_multipliers=(0.05, 0.20),
    )
    # Only the frozen train file is read; the test split is never loaded.
    assert loaded_splits == ["train"]
    assert result["n_test_trials"] == 0
    assert set(result["checkpoints"]) == {"0.05", "0.20"}
    for key, path in result["checkpoints"].items():
        assert Path(path).is_file()
        # The config sidecar sits next to the checkpoint with the full settings.
        config_path = Path(str(path).replace(".pt", ".config.json"))
        assert config_path.is_file()
        loaded = TinyNoiseTemplate.load(path)
        assert float(loaded.template.abs().max()) <= float(result["eps_scale"]) * float(key) + 1e-6


def test_random_noise_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        random_noise((2, C), 0.1)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "train_tiny_noise_script", REPO / "scripts" / "train_tiny_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parses_and_smoke_overrides(monkeypatch):
    script = _load_script()
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return {"checkpoints": {}}

    monkeypatch.setattr(script, "train_tiny_noise", fake_train)
    out = script.main([
        "--subject", "A", "--surrogate", "ckpt.pt", "--seed", "2", "--smoke",
    ])
    assert captured["smoke"] is True
    assert captured["epochs"] == 1
    assert captured["device"] == "cpu"
    assert captured["subject"] == "A"
    assert out == {"checkpoints": {}}
