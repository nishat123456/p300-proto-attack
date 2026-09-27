"""Train one flash decoder for one Comp III subject.

The heavy lifting lives in `train_decoder`, which `train_all_decoders.py`
calls in-process. Local CPU smoke mode (`--smoke`) uses 10 train trials, 3 val
trials, no test data, and 2 epochs, so it must never touch the test split.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from p3a.data.epochs import batch_trials, epoch_window  # noqa: E402
from p3a.data.loaders.comp3 import load_subject  # noqa: E402
from p3a.data.splits import load as load_split  # noqa: E402
from p3a.models.letter_head import first_reps, symbol_logits  # noqa: E402
from p3a.models.registry import build_model  # noqa: E402
from p3a.models.speller import Speller  # noqa: E402
from p3a.models.xdawn_lda import XdawnLDA  # noqa: E402
from p3a.preproc.pipeline import Preprocessor  # noqa: E402
from p3a.utils.repro import environment, set_seed, write_config  # noqa: E402

FLASHES_PER_REP = 12
FS = 240.0
N_CHANNELS = 64
XDAWN = "xdawn_lda"
ALL_MODELS = ("eegnet", "deepconvnet", "shallowconvnet", XDAWN)


def repo_root() -> Path:
    """Repository root resolved from this file, never from the cwd."""
    return Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def select_trials(trials: list, ids: list[str]) -> list:
    """Keep trials whose frozen trial_id is in `ids`, preserving id order."""
    by_id = {t.trial_id: t for t in trials}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"split ids not found among loaded trials: {missing[:3]}")
    return [by_id[i] for i in ids]


@torch.no_grad()
def precompute_epochs(trials: list, preprocessor: Preprocessor, L: int):
    """Stack preprocessed flash epochs and per-flash labels, cached on CPU.

    Preprocessing always runs on CPU (the fitted Preprocessor is a CPU module),
    so no raw trial or FIR state is ever handed to a CPU preprocessor from a GPU
    tensor. Results are cached on CPU; callers move batches to their device.
    """
    preprocessor = preprocessor.cpu()
    eegs, labels = [], []
    for t in trials:
        eeg = torch.as_tensor(t.eeg, dtype=torch.float32).unsqueeze(0)  # CPU
        y = preprocessor(eeg)[0]  # [C, T']
        q = preprocessor.q
        onsets = torch.as_tensor(t.onsets // q, dtype=torch.long)
        idx = onsets[:, None] + torch.arange(L)
        ep = y[:, idx]  # [C, F, L]
        eegs.append(ep.permute(1, 0, 2).contiguous())  # [F, C, L]
        labels.append(torch.as_tensor(t.target_flash_mask(), dtype=torch.float32))
    epochs = torch.cat(eegs, dim=0)
    y = torch.cat(labels, dim=0)
    return epochs, y


@torch.no_grad()
def flash_scores_for_trials(speller: Speller, trials: list, device: torch.device, batch: int = 16):
    """Compute [B, F] flash scores and targets [B] over bounded trial batches.

    Never stacks all raw EEG at once: each chunk is loaded and moved to the
    device, decoded, and released, so memory stays bounded by `batch`.
    """
    speller.eval()
    scores, targets = [], []
    trial_batch = min(batch, 8)
    for start in range(0, len(trials), trial_batch):
        chunk = trials[start:start + trial_batch]
        eeg, onsets, groups, target = batch_trials(chunk, device)
        # Keep the decoder's own batch bounded too: a trial batch of B trials
        # expands to B*F flash epochs, so cap the forward pass at `batch` epochs.
        out = speller(eeg, onsets, groups, max_flash_batch=batch)
        scores.append(out["flash_scores"].cpu())
        targets.append(target.cpu())
    return torch.cat(scores), torch.cat(targets)


@torch.no_grad()
def letter_accuracy(speller: Speller, trials: list, device: torch.device,
                    n_reps: int, flashes_per_rep: int = FLASHES_PER_REP,
                    batch: int = 16) -> float:
    """Fraction of trials whose argmax symbol is correct using the first n_reps.

    Iterates in bounded trial batches and only ever holds one batch of raw EEG
    plus its per-flash scores and symbol groups, instead of stacking the whole
    test set of raw trials.
    """
    if not trials:
        return float("nan")
    speller.eval()
    n_ok = 0
    trial_batch = min(batch, 8)
    for start in range(0, len(trials), trial_batch):
        chunk = trials[start:start + trial_batch]
        eeg, onsets, groups, target = batch_trials(chunk, device)
        out = speller(eeg, onsets, groups, max_flash_batch=batch)
        scores = out["flash_scores"]
        onsets_b, groups_b, scores_b = first_reps(
            onsets, groups, scores, n_reps, flashes_per_rep
        )
        logits = symbol_logits(scores_b, groups_b)
        n_ok += int((logits.argmax(dim=1) == target).sum())
    return n_ok / len(trials)


def _val_scores_batched(decoder, val_ep, device, batch):
    """Decoder scores for validation epochs, decoded in bounded batches."""
    scores = []
    with torch.no_grad():
        for start in range(0, len(val_ep), batch):
            chunk = val_ep[start:start + batch].to(device)
            scores.append(decoder(chunk).cpu())
    return torch.cat(scores).numpy()


def _train_torch(decoder, train_ep, train_y, val_ep, val_y, args, device):
    """Train a torch decoder with BCE(pos_weight) and early stop on val AUC.

    Epochs stay cached on CPU; only the current minibatch is moved to the
    device, and validation is decoded in bounded batches.
    """
    decoder.to(device)
    train_y_cpu = train_y.cpu()
    val_y_np = val_y.cpu().numpy()

    n_pos = float(train_y_cpu.sum())
    n_neg = float(len(train_y_cpu) - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1.0), 1e-6), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    n = len(train_y_cpu)
    best_auc, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []
    for epoch in range(args.epochs):
        decoder.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            xb = train_ep[idx].to(device)
            yb = train_y_cpu[idx].to(device)
            optimizer.zero_grad()
            logits = decoder(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / n

        decoder.eval()
        val_scores = _val_scores_batched(decoder, val_ep, device, args.batch)
        val_auc = float(roc_auc_score(val_y_np, val_scores)) if len(np.unique(val_y_np)) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if math.isfinite(val_auc) and val_auc > best_auc + 1e-6:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        decoder.load_state_dict(best_state)
    decoder.eval()
    return decoder, best_auc, best_epoch, history


def train_decoder(
    data_root: str,
    labels_dir: str | None,
    subject: str,
    model: str,
    seed: int,
    out_dir: str,
    epochs: int = 100,
    batch: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "auto",
    smoke: bool = False,
) -> dict:
    """Train one (subject, model, seed); write checkpoint + metrics; return metrics."""
    subject = str(subject).upper()
    model = str(model).lower()
    if model not in ALL_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {ALL_MODELS}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    # Smoke constraints are enforced here, not only in the CLI, so an in-process
    # caller (train_all_decoders) cannot accidentally train on the test split,
    # run on GPU, or train many epochs. Smoke is CPU, 2 epochs, 10 train / 3 val
    # trials, and loads no test data at all.
    if smoke:
        epochs = 2
        device = "cpu"

    if epochs < 1 or batch < 1 or patience < 1 or lr <= 0:
        raise ValueError("epochs, batch, patience and lr must be positive")
    set_seed(seed)
    dev = resolve_device("cpu" if model == XDAWN else device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    splits_path = repo_root() / f"splits/comp3_{subject}.json"

    # 1. Frozen splits; smoke never loads test data at all.
    split = load_split(splits_path)
    train_ids, val_ids = list(split["train"]), list(split["val"])
    test_ids = [] if smoke else list(split["test"])
    if smoke:
        train_ids, val_ids = sorted(train_ids)[:10], sorted(val_ids)[:3]

    train_all = load_subject(data_root, subject, "train", labels_dir=labels_dir)
    train_trials = select_trials(train_all, train_ids)
    val_trials = select_trials(train_all, val_ids)
    test_trials = [] if smoke else select_trials(
        load_subject(data_root, subject, "test", labels_dir=labels_dir), test_ids
    )

    # 2. Fit preprocessing on TRAIN only, explicitly on CPU.
    pre = Preprocessor(n_channels=N_CHANNELS, fs=FS).cpu()
    pre.fit_scaling([t.eeg for t in train_trials])
    L = epoch_window(pre.fs_out)

    # 3. Precompute epochs (no grad), cached on CPU; batches move to `dev` later.
    train_ep, train_y = precompute_epochs(train_trials, pre, L)
    val_ep, val_y = precompute_epochs(val_trials, pre, L)

    start = time.time()
    if model == XDAWN:
        decoder = XdawnLDA(n_filters=4).fit(
            train_ep.numpy(), train_y.numpy().astype(bool)
        )
        val_scores = decoder.decision_function(val_ep.numpy())
        val_auc = float(roc_auc_score(val_y.numpy(), val_scores)) if len(np.unique(val_y.numpy())) > 1 else float("nan")
        best_epoch, history = 0, []
        n_params = int(decoder.filters_.size + decoder.clf_.coef_.size + decoder.clf_.intercept_.size)
    else:
        decoder = build_model(model, N_CHANNELS, L)
        decoder, val_auc, best_epoch, history = _train_torch(
            decoder, train_ep, train_y, val_ep, val_y,
            argparse.Namespace(epochs=epochs, batch=batch, lr=lr, patience=patience), dev,
        )
        n_params = int(sum(p.numel() for p in decoder.parameters()))
    train_time = time.time() - start

    # Full effective args + provenance, stored in both checkpoint and metrics.
    effective = {
        "subject": subject,
        "model": model,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch": int(batch),
        "lr": float(lr),
        "patience": int(patience),
        "device": device,
        "resolved_device": str(dev),
        "smoke": bool(smoke),
        "data_root": str(data_root),
        "labels_dir": None if labels_dir is None else str(labels_dir),
        "out_dir": str(out_path),
        "n_channels": N_CHANNELS,
        "fs": FS,
        "L": L,
        "fs_out": pre.fs_out,
        "n_train_trials": len(train_trials),
        "n_val_trials": len(val_trials),
        "n_test_trials": len(test_trials),
        "splits_path": str(splits_path),
    }
    provenance = {**environment(seed), "repo_root": str(repo_root())}

    speller = Speller(pre, decoder, L)
    # Inference runs with the preprocessor on the training device.
    speller.to(dev)
    suffix = ".joblib" if model == XDAWN else ".pt"
    ckpt_path = out_path / f"{subject}_{model}_s{seed}{suffix}"
    speller.save(
        ckpt_path,
        training_config={"args": effective, "provenance": provenance},
        seed=seed,
    )

    # 4. Metrics. Test is decoded in bounded flash batches, never the whole set.
    if test_trials:
        test_scores, _ = flash_scores_for_trials(speller, test_trials, dev, batch=batch)
        test_y = np.concatenate([t.target_flash_mask() for t in test_trials])
        test_flash_auc = float(roc_auc_score(test_y, test_scores.numpy().reshape(-1)))
        acc15 = letter_accuracy(speller, test_trials, dev, 15, batch=batch)
        acc5 = letter_accuracy(speller, test_trials, dev, 5, batch=batch)
    else:
        test_flash_auc, acc15, acc5 = None, None, None

    metrics = {
        "subject": subject,
        "model": model,
        "val_auc": val_auc,
        "test_flash_auc": test_flash_auc,
        "test_letter_acc_15reps": acc15,
        "test_letter_acc_5reps": acc5,
        "n_params": n_params,
        "n_train_trials": len(train_trials),
        "n_val_trials": len(val_trials),
        "n_test_trials": len(test_trials),
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "smoke": bool(smoke),
        "L": L,
        "fs_out": pre.fs_out,
        "history": history,
        "checkpoint": str(ckpt_path),
        "args": effective,
        "provenance": provenance,
        **provenance,
    }
    write_config(out_path, f"{subject}_{model}_s{seed}.metrics.json", metrics)
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(REPO / "data/raw/comp3"))
    p.add_argument("--labels-dir", default=None)
    p.add_argument("--subject", required=True, choices=("A", "B"))
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "results/decoders"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true",
                   help="10 train / 3 val / 0 test trials, 2 epochs (CPU, local)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        args.epochs = 2
        if args.device == "auto":
            args.device = "cpu"
    metrics = train_decoder(
        data_root=args.data_root, labels_dir=args.labels_dir, subject=args.subject,
        model=args.model, seed=args.seed, out_dir=args.out_dir, epochs=args.epochs,
        batch=args.batch, lr=args.lr, patience=args.patience, device=args.device,
        smoke=args.smoke,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
