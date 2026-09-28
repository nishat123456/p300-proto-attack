"""Attack training core for prototype-guided GPN perturbations (Prompt 8).

This module trains one perturbation generator for one (subject, method,
generator, bank) combination. It is the in-process API that
``scripts/train_attack.py`` calls; the script only parses arguments and prints
the result.

Design notes
------------
- The surrogate is a ``Speller`` loaded from an EEGNet checkpoint. It is put in
  ``eval`` mode and every parameter is frozen, so gradients flow only into the
  generator. The attacker's preprocessing is exactly the surrogate's fitted
  preprocessing, applied to the raw signal that carries the perturbation.
- Victim models (DeepConvNet, ShallowConvNet, xDAWN+LDA) are never imported or
  loaded here. Selection uses only the frozen train/val splits; test trials are
  never read.
- The epsilon budget is ``multiplier * s`` where ``s`` is the median over
  channels of the per-channel standard deviation of the subject's raw TRAIN EEG
  and the multiplier is drawn uniformly from the pre-registered grid. The
  multiplier is also fed to the generator as conditioning.
- Prototypes come from a ``PrototypeBank`` through the shared ``retrieve``
  helper and a ``FeatureScaler`` fit on bank members only. A sample whose
  attacker target has no prototype is skipped and counted, never nudged.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p3a.attacks.generators import CausalGPN, SignalGPN
from p3a.data.epochs import batch_trials
from p3a.data.splits import load as load_split
from p3a.models.speller import Speller
from p3a.prototypes.bank import PrototypeBank
from p3a.prototypes.features import FeatureScaler, target_conditioned_erp
from p3a.prototypes.retrieval import retrieve
from p3a.utils.repro import environment, set_seed, write_config

# Pre-registered epsilon multipliers (PREREG.md, fixed settings).
EPS_MULTIPLIERS = (0.05, 0.10, 0.20, 0.30, 0.50)

METHODS = ("target_only", "proto_random", "proto_medoid", "proto_nearest", "proto_wrong")
GENERATORS = ("signal", "causal")
BANKS = ("same_subject", "cross_subject")

# method -> (retrieval mode, use_prototype). target_only conditions on zeros.
METHOD_SPECS = {
    "target_only": (None, False),
    "proto_random": ("random", True),
    "proto_medoid": ("medoid", True),
    "proto_nearest": ("nearest", True),
    "proto_wrong": ("wrong_letter", True),
}

ATTACK_FORMAT = 1
FLASHES_PER_REP = 12
DEFAULT_SUBJECTS = ("A", "B")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def select_trials(trials: list, ids) -> list:
    """Keep trials whose frozen trial_id is in `ids`, preserving id order."""
    ids = list(ids)
    by_id = {t.trial_id: t for t in trials}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"split ids not found among loaded trials: {missing[:3]}")
    return [by_id[i] for i in ids]


def discover_subjects(splits_dir: Path) -> list[str]:
    """Subject codes with a frozen comp3 split, excluding attack-target files."""
    subjects = []
    for path in sorted(Path(splits_dir).glob("comp3_*.json")):
        stem = path.stem  # comp3_A or comp3_A_attack_targets
        parts = stem.split("_")
        if len(parts) == 2:
            subjects.append(parts[1])
    return subjects


def raw_eps_scale(trials: list) -> float:
    """Median over channels of the per-channel std of raw TRAIN EEG.

    All train trials are pooled per channel before the standard deviation, so
    the scale reflects the subject's overall raw amplitude rather than one
    trial. The median over channels keeps a few noisy channels from inflating
    the budget.
    """
    if not trials:
        raise ValueError("raw_eps_scale needs at least one trial")
    stack = np.stack([np.asarray(t.eeg, dtype=np.float64) for t in trials], axis=0)  # [N,C,T]
    per_channel_std = stack.std(axis=(0, 2))  # [C]
    return float(np.median(per_channel_std))


def build_generator(generator_kind: str, n_symbols: int, n_channels: int,
                    use_prototype: bool, kwargs: dict | None = None) -> nn.Module:
    """Construct a SignalGPN or CausalGPN with the shared prototype flag."""
    kwargs = dict(kwargs or {})
    if generator_kind == "signal":
        return SignalGPN(n_symbols, n_channels, use_prototype=use_prototype, **kwargs)
    if generator_kind == "causal":
        return CausalGPN(n_symbols, n_channels, use_prototype=use_prototype, **kwargs)
    raise ValueError(f"unknown generator {generator_kind!r}; expected {GENERATORS}")


def build_bank(subject: str, bank_mode: str, data_root, labels_dir, speller: Speller,
               train_ids, prototype_subjects=(), device: str = "cpu") -> PrototypeBank:
    """Build the prototype bank for the requested leakage policy.

    same_subject uses only the attacked subject's frozen train portion.
    cross_subject uses every trial (train and test) of the other subjects and
    refuses the attacked subject itself, which the bank enforces.
    """
    from p3a.data.loaders.comp3 import load_subject

    if bank_mode == "same_subject":
        trials = load_subject(data_root, subject, "train", labels_dir=labels_dir)
        kept = select_trials(trials, sorted(train_ids))
        return PrototypeBank.build(
            kept, speller, {subject}, {"train"},
            mode="same_subject", attacked_subject=subject,
            allowed_trial_ids=set(train_ids), device=device,
        )
    if bank_mode == "cross_subject":
        others = [str(s) for s in prototype_subjects if str(s) != subject]
        if not others:
            raise ValueError("cross_subject bank needs at least one other subject")
        trials = []
        for other in others:
            for split in ("train", "test"):
                trials.extend(load_subject(data_root, other, split, labels_dir=labels_dir))
        return PrototypeBank.build(
            trials, speller, set(others), {"train", "test"},
            mode="cross_subject", attacked_subject=subject, device=device,
        )
    raise ValueError(f"unknown bank mode {bank_mode!r}; expected {BANKS}")


class _Standardizer(nn.Module):
    """Torch mirror of a ``FeatureScaler`` so the loss can standardize features."""

    def __init__(self, scaler: FeatureScaler):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(scaler.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(scaler.std, dtype=torch.float32))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.mean[None]) / self.std[None]


def sample_targets(true_symbols: torch.Tensor, rng: np.random.Generator,
                   n_symbols: int) -> torch.Tensor:
    """Draw one random attacker target per sample, never equal to the true one."""
    out = np.empty(len(true_symbols), dtype=np.int64)
    for i in range(len(true_symbols)):
        true = int(true_symbols[i])
        candidates = np.arange(n_symbols)
        out[i] = int(rng.choice(candidates[candidates != true]))
    return torch.as_tensor(out, dtype=torch.long, device=true_symbols.device)


def fixed_val_targets(true_symbols: np.ndarray, seed: int, n_symbols: int) -> np.ndarray:
    """A seeded, fixed attacker target per validation trial, never the true one."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(true_symbols), dtype=np.int64)
    for i, true in enumerate(true_symbols):
        candidates = np.arange(n_symbols)
        out[i] = int(rng.choice(candidates[candidates != int(true)]))
    return out


def _query_features(speller: Speller, eeg: torch.Tensor, onsets: torch.Tensor,
                    groups: torch.Tensor, target: torch.Tensor) -> list[np.ndarray]:
    """Target-conditioned ERP of each source trial at the attacker target flashes."""
    with torch.no_grad():
        epochs = speller.epochs(eeg, onsets)  # [B, F, C, L]
        feats = []
        for i in range(eeg.shape[0]):
            f = target_conditioned_erp(epochs[i], groups[i], int(target[i]))
            feats.append(f.detach().cpu().numpy())
    return feats


def _gather_prototypes(bank, scaler, queries, target: torch.Tensor, mode,
                       rng: np.random.Generator):
    """Return per-sample standardized prototype features and the kept indices.

    ``mode is None`` means target_only: every sample is kept with no prototype.
    """
    keep, protos = [], []
    for i in range(len(queries)):
        if mode is None:
            keep.append(i)
            protos.append(None)
            continue
        result = retrieve(bank, scaler, queries[i], int(target[i]), mode, rng)
        if result is None:
            continue  # attacker target has no prototype: skip and count
        keep.append(i)
        protos.append(scaler.transform(bank.entries[result.index].feature))
    return keep, protos


def _subset(tensor: torch.Tensor, index) -> torch.Tensor:
    return tensor[index] if len(index) != tensor.shape[0] else tensor


def _generator_forward(generator, generator_kind, eeg, onsets, groups, target,
                       eps_raw, proto_feat, use_prototype, eps_mult):
    return generator(
        eeg, onsets, groups, target, eps_raw, proto_feat,
        use_prototype=use_prototype, eps_mult=eps_mult,
    )


def evaluate_success(speller: Speller, generator: nn.Module, generator_kind: str,
                     trials: list, targets: np.ndarray, eps_scale: float,
                     eps_multipliers, bank, scaler, retrieval_mode, use_prototype,
                     device, batch: int, n_symbols: int) -> dict:
    """Targeted success per epsilon multiplier on a fixed target per trial.

    Returns a dict keyed by the multiplier string (for example "0.20") plus a
    ``mean`` entry. A trial whose target has no prototype counts as a failure,
    so a missing prototype can never look like a success.
    """
    generator.eval()
    speller.eval()
    per_mult = {f"{m:.2f}": [] for m in eps_multipliers}
    fixed = torch.as_tensor(targets, dtype=torch.long, device=device)

    for start in range(0, len(trials), batch):
        chunk = trials[start:start + batch]
        eeg, onsets, groups, _true = batch_trials(chunk, device)
        target = fixed[start:start + batch]
        queries = _query_features(speller, eeg, onsets, groups, target)
        if retrieval_mode is None:
            keep = list(range(eeg.shape[0]))
            protos = [None] * eeg.shape[0]
        else:
            keep, protos = _gather_prototypes(
                bank, scaler, queries, target, retrieval_mode,
                np.random.default_rng(0),
            )
        idx = torch.as_tensor(keep, dtype=torch.long, device=device)
        proto_feat = None
        if use_prototype and keep:
            proto_feat = torch.as_tensor(
                np.stack(protos), dtype=torch.float32, device=device
            )
        for mult in eps_multipliers:
            key = f"{mult:.2f}"
            if not keep:
                per_mult[key].extend([0.0] * eeg.shape[0])
                continue
            eps_raw = torch.full((len(keep),), float(mult) * eps_scale, device=device)
            eps_mult_t = torch.full((len(keep),), float(mult), device=device)
            with torch.no_grad():
                delta = _generator_forward(
                    generator, generator_kind, _subset(eeg, idx), _subset(onsets, idx),
                    _subset(groups, idx), _subset(target, idx), eps_raw,
                    proto_feat, use_prototype, eps_mult_t,
                )
                logits = speller(_subset(eeg, idx) + delta, _subset(onsets, idx),
                                 _subset(groups, idx), max_flash_batch=batch)["logits"]
                ok = (logits.argmax(dim=1) == _subset(target, idx)).float().cpu().numpy()
            full = np.zeros(eeg.shape[0], dtype=np.float32)
            full[np.asarray(keep)] = ok
            per_mult[key].extend(full.tolist())
    summary = {key: float(np.mean(values)) if values else float("nan")
               for key, values in per_mult.items()}
    summary["mean"] = float(np.mean([v for v in summary.values() if np.isfinite(v)])) \
        if any(np.isfinite(v) for v in summary.values()) else float("nan")
    return summary


def save_generator(path, generator: nn.Module, generator_kind: str, method: str,
                   bank_mode: str, config: dict, provenance: dict,
                   gen_config: dict, eps_scale: float, n_symbols: int,
                   n_channels: int, use_prototype: bool) -> Path:
    """Save a generator checkpoint plus everything needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attack_format": ATTACK_FORMAT,
        "generator_kind": generator_kind,
        "generator_state": generator.state_dict(),
        "generator_config": dict(gen_config),
        "method": method,
        "bank_mode": bank_mode,
        "use_prototype": bool(use_prototype),
        "eps_scale": float(eps_scale),
        "n_symbols": int(n_symbols),
        "n_channels": int(n_channels),
        "config": dict(config),
        "provenance": dict(provenance),
    }
    torch.save(payload, path)
    return path


def load_generator(path) -> tuple[nn.Module, dict]:
    """Rebuild a generator from a checkpoint written by ``save_generator``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("attack_format") != ATTACK_FORMAT:
        raise ValueError("unsupported attack checkpoint format")
    generator = build_generator(
        payload["generator_kind"], payload["n_symbols"], payload["n_channels"],
        payload["use_prototype"], payload["generator_config"],
    )
    generator.load_state_dict(payload["generator_state"])
    return generator, payload


def train_attack(
    data_root: str,
    labels_dir: str | None,
    subject: str,
    surrogate: str,
    method: str,
    generator: str,
    bank: str,
    seed: int,
    out_dir: str,
    lambda_p: float = 1.0,
    lambda_s: float = 0.1,
    epochs: int = 60,
    batch: int = 8,
    lr: float = 1e-4,
    device: str = "auto",
    eps_multipliers=EPS_MULTIPLIERS,
    generator_kwargs: dict | None = None,
    prototypes_per_subject=None,
    smoke: bool = False,
    smoke_train: int = 4,
    smoke_val: int = 3,
) -> dict:
    """Train one attack generator and write checkpoint, config and CSV.

    ``smoke`` is a local CPU path: it keeps the real loss and real data flow but
    uses the first few frozen train/val trials and one epoch. It never loads the
    test split and never changes the training math.
    """
    subject = str(subject).upper()
    method = str(method).lower()
    generator = str(generator).lower()
    bank = str(bank).lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected {METHODS}")
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; expected {GENERATORS}")
    if bank not in BANKS:
        raise ValueError(f"unknown bank mode {bank!r}; expected {BANKS}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if epochs < 1 or batch < 1 or lr <= 0:
        raise ValueError("epochs, batch and lr must be positive")
    if lambda_p < 0 or lambda_s < 0:
        raise ValueError("lambda_p and lambda_s must be nonnegative")

    if smoke:
        epochs = 1
        device = "cpu"

    resolved = ("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else device
    dev = torch.device(resolved)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    retrieval_mode, use_prototype = METHOD_SPECS[method]
    split_path = repo_root() / f"splits/comp3_{subject}.json"
    split = load_split(split_path)
    train_ids = sorted(split["train"])
    val_ids = sorted(split["val"])
    if smoke:
        train_ids = train_ids[:smoke_train]
        val_ids = val_ids[:smoke_val]

    from p3a.data.loaders.comp3 import load_subject

    train_all = load_subject(data_root, subject, "train", labels_dir=labels_dir)
    train_trials = select_trials(train_all, train_ids)
    val_trials = select_trials(train_all, val_ids)
    # Test trials are never loaded: only the frozen train file is read above.

    eps_scale = raw_eps_scale(train_trials)

    speller = Speller.load(surrogate).to(dev)
    speller.eval()
    for param in speller.parameters():
        param.requires_grad_(False)
    n_symbols = train_trials[0].layout.n_symbols
    n_channels = train_trials[0].eeg.shape[0]

    if prototypes_per_subject is None:
        if bank == "cross_subject":
            prototype_subjects = [s for s in discover_subjects(split_path.parent)
                                  if s != subject]
            if not prototype_subjects:
                prototype_subjects = [s for s in DEFAULT_SUBJECTS if s != subject]
        else:
            prototype_subjects = [subject]
    else:
        prototype_subjects = list(prototypes_per_subject)

    need_bank = retrieval_mode is not None
    prototype_bank = None
    scaler = None
    standardizer = None
    if need_bank:
        prototype_bank = build_bank(
            subject, bank, data_root, labels_dir, speller, train_ids,
            prototype_subjects, device=resolved,
        )
        scaler = FeatureScaler.fit([e.feature for e in prototype_bank.entries])
        standardizer = _Standardizer(scaler).to(dev)

    gen_config = dict(generator_kwargs or {})
    net = build_generator(generator, n_symbols, n_channels, use_prototype, gen_config).to(dev)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    val_true = np.array([t.layout.index(t.target_symbol) for t in val_trials], dtype=np.int64)
    val_targets = fixed_val_targets(val_true, seed, n_symbols)

    csv_rows = []
    best_mean, best_epoch, best_state = -1.0, -1, None
    eps_keys = [f"{m:.2f}" for m in eps_multipliers]
    n_skipped = 0
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(train_trials))
        running = {"loss": 0.0, "ce": 0.0, "proto": 0.0, "reg": 0.0}
        n_samples = 0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            chunk = [train_trials[i] for i in idx]
            eeg, onsets, groups, true_target = batch_trials(chunk, dev)
            rng = np.random.default_rng(seed * 1000003 + epoch * 1009 + start)
            target = sample_targets(true_target, rng, n_symbols)
            queries = _query_features(speller, eeg, onsets, groups, target)
            keep, protos = _gather_prototypes(
                prototype_bank, scaler, queries, target, retrieval_mode, rng
            )
            n_skipped += int(eeg.shape[0]) - len(keep)
            if not keep:
                continue
            gen_eeg = _subset(eeg, keep)
            gen_onsets = _subset(onsets, keep)
            gen_groups = _subset(groups, keep)
            gen_target = _subset(target, keep)
            mults = rng.choice(np.asarray(eps_multipliers, dtype=np.float32), size=len(keep))
            eps_mult_t = torch.as_tensor(mults, dtype=torch.float32, device=dev)
            eps_raw = eps_mult_t * float(eps_scale)
            proto_feat = None
            if use_prototype:
                proto_feat = torch.as_tensor(
                    np.stack(protos), dtype=torch.float32, device=dev
                )

            delta = _generator_forward(
                net, generator, gen_eeg, gen_onsets, gen_groups, gen_target,
                eps_raw, proto_feat, use_prototype, eps_mult_t,
            )
            x_adv = gen_eeg + delta
            logits = speller(x_adv, gen_onsets, gen_groups, max_flash_batch=batch)["logits"]
            ce = F.cross_entropy(logits, gen_target)
            reg = (((delta ** 2).flatten(1).mean(dim=1)) / (eps_raw ** 2)).mean()
            loss = ce + lambda_s * reg
            proto_loss = torch.zeros((), device=dev)
            if use_prototype:
                adv_epochs = speller.epochs(x_adv, gen_onsets)  # [B, F, C, L]
                adv_feats = torch.stack([
                    target_conditioned_erp(adv_epochs[i], gen_groups[i], int(gen_target[i]))
                    for i in range(len(keep))
                ])
                scaled_adv = standardizer(adv_feats)
                # _gather_prototypes already returns scaler-transformed bank
                # features because the same standardized tensor conditions the
                # generator. Do not standardize those features a second time.
                proto_loss = ((scaled_adv - proto_feat) ** 2).mean()
                loss = loss + lambda_p * proto_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.detach()) * len(keep)
            running["ce"] += float(ce.detach()) * len(keep)
            running["proto"] += float(proto_loss.detach()) * len(keep)
            running["reg"] += float(reg.detach()) * len(keep)
            n_samples += len(keep)

        denom = max(n_samples, 1)
        train_parts = {k: v / denom for k, v in running.items()}
        val_summary = evaluate_success(
            speller, net, generator, val_trials, val_targets, eps_scale,
            eps_multipliers, prototype_bank, scaler, retrieval_mode, use_prototype,
            dev, batch, n_symbols,
        )
        net.train()
        row = {
            "epoch": epoch,
            "train_loss": train_parts["loss"],
            "train_ce": train_parts["ce"],
            "train_proto": train_parts["proto"],
            "train_reg": train_parts["reg"],
            "n_samples_used": n_samples,
            "n_skipped": n_skipped,
            "val_success_mean": val_summary["mean"],
        }
        for key in eps_keys:
            row[f"val_success_eps{key.replace('.', '')}"] = val_summary[key]
        csv_rows.append(row)

        if np.isfinite(val_summary["mean"]) and val_summary["mean"] > best_mean + 1e-9:
            best_mean, best_epoch = val_summary["mean"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    config = {
        "subject": subject,
        "surrogate": str(surrogate),
        "method": method,
        "generator": generator,
        "bank": bank,
        "seed": int(seed),
        "lambda_p": float(lambda_p),
        "lambda_s": float(lambda_s),
        "epochs": int(epochs),
        "batch": int(batch),
        "lr": float(lr),
        "device": resolved,
        "eps_multipliers": [float(m) for m in eps_multipliers],
        "eps_scale": eps_scale,
        "generator_kwargs": gen_config,
        "use_prototype": bool(use_prototype),
        "retrieval_mode": retrieval_mode,
        "n_symbols": int(n_symbols),
        "n_channels": int(n_channels),
        "n_train_trials": len(train_trials),
        "n_val_trials": len(val_trials),
        "n_test_trials": 0,
        "n_skipped_total": n_skipped,
        "prototype_subjects": list(prototype_subjects),
        "split_path": str(split_path),
        "smoke": bool(smoke),
        "best_epoch": best_epoch,
        "best_val_success_mean": None if best_epoch < 0 else best_mean,
        "data_root": str(data_root),
        "labels_dir": None if labels_dir is None else str(labels_dir),
    }
    provenance = {**environment(seed), "repo_root": str(repo_root())}
    suffix = f"{subject}_{method}_{generator}_{bank}_s{seed}"
    ckpt_path = save_generator(
        out_path / f"{suffix}.pt", net, generator, method, bank, config,
        provenance, gen_config, eps_scale, n_symbols, n_channels, use_prototype,
    )
    csv_path = out_path / f"{suffix}.csv"
    fieldnames = list(csv_rows[0].keys()) if csv_rows else ["epoch"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_config(out_path, f"{suffix}.config.json", {
        "config": config, "provenance": provenance, "history": csv_rows,
    })

    return {
        "checkpoint": str(ckpt_path),
        "csv": str(csv_path),
        "config_path": str(out_path / f"{suffix}.config.json"),
        "best_epoch": best_epoch,
        "best_val_success_mean": None if best_epoch < 0 else best_mean,
        "eps_scale": eps_scale,
        "n_train_trials": len(train_trials),
        "n_val_trials": len(val_trials),
        "n_test_trials": 0,
        "n_skipped_total": n_skipped,
        "history": csv_rows,
        "config": config,
        "provenance": provenance,
    }
