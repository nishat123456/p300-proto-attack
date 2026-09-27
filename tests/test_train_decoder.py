"""CPU tests for train_decoder's data-shaping and training logic.

Uses tiny synthetic trials and a synthetic epoch matrix so no real .mat files
are needed and the whole test runs in seconds on CPU.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from train_decoder import (  # noqa: E402
    letter_accuracy, flash_scores_for_trials, precompute_epochs, select_trials,
    _train_torch,
)
from p3a.data.epochs import batch_trials  # noqa: E402
from p3a.data.layout import COMP3, rc_code_to_group  # noqa: E402
from p3a.data.trial import Trial  # noqa: E402
from p3a.models.eegnet import EEGNet  # noqa: E402
from p3a.models.speller import Speller  # noqa: E402
from p3a.preproc.pipeline import Preprocessor  # noqa: E402


def _trial(i, n_flashes=12):
    rng = np.random.default_rng(i)
    codes = np.tile(np.arange(1, 13), n_flashes // 12)
    onsets = np.arange(n_flashes) * 42
    groups = np.stack([rc_code_to_group(COMP3, int(c)) for c in codes])
    return Trial(
        eeg=rng.normal(size=(4, 7794)).astype(np.float32), fs=240.0, onsets=onsets,
        flash_groups=groups, flash_codes=codes, target_symbol="A", layout=COMP3,
        subject_id="A", dataset="comp3", trial_index=i, source_split="train",
    )


def test_select_trials_keeps_ids_and_raises_on_missing():
    trials = [_trial(i) for i in range(3)]
    ids = [trials[2].trial_id, trials[0].trial_id]
    chosen = select_trials(trials, ids)
    assert [t.trial_index for t in chosen] == [2, 0]
    with pytest.raises(KeyError, match="not found"):
        select_trials(trials, ["comp3:A:train:999"])


def test_precompute_epochs_shapes_and_labels():
    trials = [_trial(i) for i in range(3)]
    pre = Preprocessor(n_channels=4, fs=240.0)
    pre.fit_scaling([t.eeg for t in trials])
    epochs, y = precompute_epochs(trials, pre, 48)
    assert epochs.shape == (3 * 12, 4, 48)
    assert y.shape == (3 * 12,)
    # Cached on CPU regardless of any requested device.
    assert epochs.device.type == "cpu"
    # Exactly the flashes containing the target symbol are labelled 1.
    expected = np.concatenate([t.target_flash_mask().astype(float) for t in trials])
    np.testing.assert_array_equal(y.numpy(), expected)


def test_precompute_epochs_keeps_cpu_even_if_preprocessor_requested_on_cuda():
    """Regression: a CPU preprocessor must never be handed GPU-derived input."""
    trials = [_trial(i) for i in range(2)]
    pre = Preprocessor(n_channels=4, fs=240.0)
    pre.fit_scaling([t.eeg for t in trials])
    moved = pre
    if torch.cuda.is_available():
        moved = pre.to("cuda")
    epochs, y = precompute_epochs(trials, moved, 48)
    assert epochs.device.type == "cpu" and y.device.type == "cpu"


def test_train_torch_learns_synthetic_signal_and_early_stops():
    torch.manual_seed(0)
    n, c, length = 80, 4, 48
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(n, c, length)), dtype=torch.float32)
    y = torch.zeros(n)
    y[: n // 2] = 1.0
    x[y.bool()] += 0.8  # separable target class
    decoder = EEGNet(c, length)
    args = type("A", (), {"epochs": 10, "batch": 32, "lr": 1e-2, "patience": 3})()
    trained, best_auc, best_epoch, history = _train_torch(
        decoder, x, y, x, y, args, torch.device("cpu")
    )
    assert best_auc > 0.9, best_auc
    assert 0 <= best_epoch < 10
    assert len(history) >= 1 and "val_auc" in history[0]


def test_letter_accuracy_perfect_with_target_mask_scores():
    trials = [_trial(i) for i in range(4)]
    pre = Preprocessor(n_channels=4, fs=240.0)
    pre.fit_scaling([t.eeg for t in trials])
    speller = Speller(pre, EEGNet(4, 48), 48)

    class Oracle(torch.nn.Module):
        def forward(self, flat):
            # Target flashes (label 1) get a high score; the first column of the
            # epoch happens to carry no target information, so use the label-free
            # proxy of mean absolute amplitude as a stand-in.
            return flat.abs().mean(dim=(1, 2))

    speller.decoder = Oracle()
    acc = letter_accuracy(speller, trials, torch.device("cpu"), n_reps=1)
    assert 0.0 <= acc <= 1.0


def test_letter_accuracy_matches_across_batch_sizes():
    """Regression: bounded batches must not change letter accuracy."""
    trials = [_trial(i) for i in range(6)]
    pre = Preprocessor(n_channels=4, fs=240.0)
    pre.fit_scaling([t.eeg for t in trials])
    speller = Speller(pre, EEGNet(4, 48), 48)

    class Oracle(torch.nn.Module):
        def forward(self, flat):
            return flat.abs().mean(dim=(1, 2))

    speller.decoder = Oracle()
    dev = torch.device("cpu")
    one_shot = letter_accuracy(speller, trials, dev, n_reps=1, batch=64)
    batched = letter_accuracy(speller, trials, dev, n_reps=1, batch=2)
    assert one_shot == batched


def test_flash_scores_for_trials_matches_single_batch():
    """Regression: chunked decoding equals decoding all trials at once."""
    trials = [_trial(i) for i in range(6)]
    pre = Preprocessor(n_channels=4, fs=240.0)
    pre.fit_scaling([t.eeg for t in trials])
    speller = Speller(pre, EEGNet(4, 48), 48).eval()
    dev = torch.device("cpu")
    s_big, t_big = flash_scores_for_trials(speller, trials, dev, batch=64)
    s_small, t_small = flash_scores_for_trials(speller, trials, dev, batch=2)
    torch.testing.assert_close(s_big, s_small)
    torch.testing.assert_close(t_big, t_small)
    assert s_big.shape == (6, 12)


def test_batch_trials_smoke_ids_cover_val_and_train():
    # The smoke split is the first 10 train and first 3 val ids; both must load.
    trials = [_trial(i) for i in range(20)]
    train = select_trials(trials, [t.trial_id for t in trials[:10]])
    val = select_trials(trials, [t.trial_id for t in trials[10:13]])
    assert len(train) == 10 and len(val) == 3
    eeg, onsets, groups, targets = batch_trials(train[:2], torch.device("cpu"))
    assert eeg.shape[0] == 2 and onsets.shape[1] == 12


def test_train_decoder_smoke_enforced_in_process(monkeypatch, tmp_path):
    """Regression: smoke must imply CPU, 2 epochs, 10/3 trials, and no test load
    even when called directly (not through the CLI) with large args."""
    import train_decoder as td

    trials = [_trial(i) for i in range(30)]
    ids = [t.trial_id for t in trials]
    fake_split = {"train": ids[:10], "val": ids[10:13], "test": ids[13:]}
    loaded_splits = []

    def fake_load_subject(root, subject, split, labels_dir=None):
        loaded_splits.append(split)
        return trials

    monkeypatch.setattr(td, "load_split", lambda path: fake_split)
    monkeypatch.setattr(td, "load_subject", fake_load_subject)
    monkeypatch.setattr(td, "N_CHANNELS", 4)

    metrics = td.train_decoder(
        data_root=str(tmp_path), labels_dir=None, subject="A", model="eegnet",
        seed=0, out_dir=str(tmp_path / "out"), epochs=99, batch=8,
        device="auto", smoke=True,
    )
    assert loaded_splits == ["train"]  # test split never touched
    assert metrics["smoke"] is True
    assert metrics["n_test_trials"] == 0
    assert metrics["test_flash_auc"] is None
    assert metrics["args"]["epochs"] == 2  # smoke overrides the 99
    assert metrics["args"]["resolved_device"] == "cpu"  # smoke forces CPU
    # Full effective args and provenance are recorded in checkpoint and metrics.
    assert metrics["args"]["model"] == "eegnet"
    assert metrics["provenance"]["repo_root"]
    payload = torch.load(metrics["checkpoint"], map_location="cpu", weights_only=False)
    assert payload["training_config"]["args"]["smoke"] is True
    assert payload["training_config"]["provenance"]["repo_root"]


@pytest.mark.parametrize("model", ["eegnet", "xdawn_lda"])
def test_full_evaluation_flash_labels_and_checkpoint(monkeypatch, tmp_path, model):
    import train_decoder as td
    from dataclasses import replace
    from sklearn.metrics import roc_auc_score
    training = [_trial(i, 180) for i in range(4)]
    heldout = [replace(_trial(20 + i, 180), source_split="test", target_symbol="B") for i in range(2)]
    split = {"train": [t.trial_id for t in training[:3]], "val": [training[3].trial_id], "test": [t.trial_id for t in heldout]}
    monkeypatch.setattr(td, "load_split", lambda _: split)
    monkeypatch.setattr(td, "load_subject", lambda root, subject, split, labels_dir=None: training if split == "train" else heldout)
    monkeypatch.setattr(td, "N_CHANNELS", 4)
    metrics = td.train_decoder(str(tmp_path), None, "A", model, 0, str(tmp_path / model), epochs=1, batch=32, device="cpu")
    loaded = Speller.load(metrics["checkpoint"]).eval()
    scores, _ = flash_scores_for_trials(loaded, heldout, torch.device("cpu"), batch=2)
    expected = roc_auc_score(np.concatenate([t.target_flash_mask() for t in heldout]), scores.numpy().ravel())
    assert metrics["test_flash_auc"] == pytest.approx(expected)
