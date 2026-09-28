"""Resumable attack evaluation core (Prompt 10).

This module computes perturbations for every attack method and epsilon
multiplier, evaluates targeted success for the surrogate and every victim at 15
and 5 repetitions, records the raw-unit budget, and stores float16 decision ERPs
and preprocessed windows for the downstream detector stage.

Two layers exist so the logic can be tested without raw data or a GPU:

- ``run_evaluation`` is the core resumable loop. It takes already-loaded trials,
  models and delta providers and writes one NPZ plus one parquet shard per
  (method, generator, bank, epsilon) combination, then rebuilds a single
  ``attacks.parquet`` from the shards. A combination whose NPZ and shard already
  exist is skipped, so an interrupted run resumes where it stopped.
- ``evaluate_attacks`` is the high-level entry point the CLI calls. It loads the
  frozen test split and pre-registered targets, the surrogate checkpoint, the
  seed-0 victims, the prototype banks, the trained generators and the Tiny Noise
  templates, then delegates to ``run_evaluation``.

Victims are only ever evaluated; they are never loaded inside an attack or used
for selection. Nothing here reads the train or validation split except to build
the prototype banks and to compute the raw epsilon scale, exactly as the
pre-registration fixes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p3a.attacks.baselines import TinyNoiseTemplate, random_noise, targeted_pgd
from p3a.attacks.train import (
    EPS_MULTIPLIERS,
    METHOD_SPECS as ATTACK_METHOD_SPECS,
    build_bank,
    discover_subjects,
    load_generator,
    raw_eps_scale,
    repo_root,
    select_trials,
)
from p3a.eval import success as success_mod
from p3a.eval.budget import budget_metrics
from p3a.eval.store import (
    N_WINDOWS,
    ArtifactSpec,
    AttackArtifacts,
    CleanArtifacts,
    attack_npz_path,
    clean_npz_path,
    sample_window_starts,
    windows_from_preprocessed,
)
from p3a.models.speller import Speller
from p3a.prototypes.features import FeatureScaler, target_conditioned_erp
from p3a.prototypes.retrieval import retrieve
from p3a.utils.repro import environment, set_seed, write_config

__all__ = [
    "DEFAULT_METHOD_SPECS",
    "EvalResult",
    "GeneratorProvider",
    "MethodSpec",
    "NoiseProvider",
    "PGDProvider",
    "TinyNoiseProvider",
    "combo_key",
    "evaluate_attacks",
    "load_attack_targets",
    "run_evaluation",
    "select_method_specs",
]

NONE = "none"


@dataclass(frozen=True)
class MethodSpec:
    """One evaluated attack configuration.

    ``kind`` is one of ``noise``, ``pgd``, ``tiny_noise`` or ``generator``.
    ``name`` is the retrieval method for generators (or the baseline name),
    ``generator`` is the generator family (or ``none``), and ``bank`` is the
    prototype bank policy (or ``none`` for baselines).
    """

    name: str
    kind: str
    generator: str = NONE
    bank: str = NONE

    def label(self) -> str:
        return f"{self.name}_{self.generator}_{self.bank}"


def combo_key(spec: MethodSpec) -> str:
    return spec.label()


def default_method_specs() -> list[MethodSpec]:
    """The full method list fixed by Prompt 10, in reporting order."""
    specs = [
        MethodSpec("noise", "noise"),
        MethodSpec("pgd", "pgd"),
        MethodSpec("tiny_noise", "tiny_noise"),
    ]
    for generator in ("signal", "causal"):
        specs.append(MethodSpec("target_only", "generator", generator, "cross_subject"))
    for method in ("proto_random", "proto_medoid", "proto_nearest", "proto_wrong"):
        for generator in ("signal", "causal"):
            specs.append(MethodSpec(method, "generator", generator, "cross_subject"))
    for generator in ("signal", "causal"):
        specs.append(MethodSpec("proto_nearest", "generator", generator, "same_subject"))
    return specs


DEFAULT_METHOD_SPECS = tuple(default_method_specs())


def select_method_specs(methods=None) -> list[MethodSpec]:
    """The default spec list, optionally restricted to a set of method names.

    ``methods`` is a collection of method names (for example ``{"pgd",
    "proto_nearest"}``) or None for every method. Restricting the list lets a
    Kaggle shard evaluate a subset of methods without changing the default,
    which still covers the full registered job.
    """
    if methods is None:
        return list(DEFAULT_METHOD_SPECS)
    wanted = {str(m) for m in methods}
    known = {spec.name for spec in DEFAULT_METHOD_SPECS}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown method name(s): {unknown}; expected {sorted(known)}")
    return [spec for spec in DEFAULT_METHOD_SPECS if spec.name in wanted]


# --------------------------------------------------------------------------
# Delta providers
# --------------------------------------------------------------------------

class _DeltaProvider:
    """Base class: every provider returns (delta, distance, ok) for a batch."""

    def compute(self, speller, eeg, onsets, groups, target, eps_raw, eps_mult,
                rng):
        raise NotImplementedError


class NoiseProvider(_DeltaProvider):
    """Amplitude-matched Gaussian noise over the whole trial."""

    def compute(self, speller, eeg, onsets, groups, target, eps_raw, eps_mult, rng):
        delta = random_noise(tuple(eeg.shape), eps_raw, rng=rng)
        delta = delta.to(device=eeg.device, dtype=eeg.dtype)
        return delta, np.full(eeg.shape[0], np.nan), np.ones(eeg.shape[0], dtype=bool)


class PGDProvider(_DeltaProvider):
    """Targeted L-infinity PGD through the surrogate's differentiable pipeline."""

    def __init__(self, steps: int = 40, random_start: bool = True):
        self.steps = int(steps)
        self.random_start = bool(random_start)

    def compute(self, speller, eeg, onsets, groups, target, eps_raw, eps_mult, rng):
        delta = targeted_pgd(
            speller, eeg, onsets, groups, target, eps_raw,
            steps=self.steps, random_start=self.random_start,
        )
        return delta, np.full(eeg.shape[0], np.nan), np.ones(eeg.shape[0], dtype=bool)


class TinyNoiseProvider(_DeltaProvider):
    """A pre-trained universal template selected by the epsilon multiplier."""

    def __init__(self, templates: dict[str, TinyNoiseTemplate]):
        if not templates:
            raise ValueError("TinyNoiseProvider needs at least one template")
        self.templates = dict(templates)

    def _template_for(self, eps_mult) -> TinyNoiseTemplate:
        if torch.is_tensor(eps_mult):
            values = eps_mult.detach().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(eps_mult).reshape(-1)
        key = f"{float(values[0]):.2f}"
        # Fall back to the nearest trained multiplier so rounding never misses.
        if key not in self.templates:
            keys = sorted(self.templates, key=lambda k: abs(float(k) - float(key)))
            key = keys[0]
        return self.templates[key]

    def compute(self, speller, eeg, onsets, groups, target, eps_raw, eps_mult, rng):
        template = self._template_for(eps_mult)
        delta = template.delta(onsets, groups, target, eeg.shape[-1], eps_raw)
        return delta.to(eeg.device), np.full(eeg.shape[0], np.nan), np.ones(eeg.shape[0], dtype=bool)


class GeneratorProvider(_DeltaProvider):
    """A trained GPN generator plus the prototype retrieval it was conditioned on."""

    def __init__(self, generator, bank, scaler, mode: str, use_prototype: bool,
                 device: str = "cpu"):
        self.generator = generator
        self.bank = bank
        self.scaler = scaler
        self.mode = mode
        self.use_prototype = bool(use_prototype)
        self.device = device

    def compute(self, speller, eeg, onsets, groups, target, eps_raw, eps_mult, rng):
        batch = eeg.shape[0]
        keep, protos, distances = [], [], []
        if self.use_prototype:
            with torch.no_grad():
                epochs = speller.epochs(eeg, onsets)  # [B, F, C, L]
                for i in range(batch):
                    query = target_conditioned_erp(
                        epochs[i], groups[i], int(target[i].detach())
                    ).detach().cpu().numpy()
                    result = retrieve(
                        self.bank, self.scaler, query, int(target[i].detach()),
                        self.mode, rng,
                    )
                    if result is None:
                        continue
                    keep.append(i)
                    protos.append(self.scaler.transform(self.bank.entries[result.index].feature))
                    distances.append(result.distance)
        else:
            keep = list(range(batch))
            protos = [None] * batch
            distances = [np.nan] * batch

        delta = torch.zeros_like(eeg)
        ok = np.zeros(batch, dtype=bool)
        distance_out = np.full(batch, np.nan, dtype=np.float64)
        if not keep:
            return delta, distance_out, ok

        index = torch.as_tensor(keep, dtype=torch.long, device=eeg.device)
        proto_feat = None
        if self.use_prototype:
            proto_feat = torch.as_tensor(
                np.stack(protos), dtype=torch.float32, device=eeg.device
            )
        eps_raw_sub = eps_raw[index] if torch.is_tensor(eps_raw) else eps_raw
        eps_mult_sub = eps_mult[index] if torch.is_tensor(eps_mult) else eps_mult
        with torch.no_grad():
            sub = self.generator(
                eeg[index], onsets[index], groups[index], target[index], eps_raw_sub,
                proto_feat, use_prototype=self.use_prototype, eps_mult=eps_mult_sub,
            )
        delta[index] = sub
        ok[index] = True
        distance_out[np.asarray(keep)] = np.asarray(distances, dtype=np.float64)
        return delta, distance_out, ok


# --------------------------------------------------------------------------
# Target loading and clean artifacts
# --------------------------------------------------------------------------

def load_attack_targets(subject: str, splits_dir=None) -> dict[str, list[str]]:
    """Pre-registered attacker targets per test trial id from the frozen split."""
    splits_dir = Path(splits_dir) if splits_dir is not None else repo_root() / "splits"
    path = splits_dir / f"comp3_{str(subject).upper()}_attack_targets.json"
    payload = json.loads(Path(path).read_text())
    return {str(k): list(str(v)) for k, v in payload["split"].items()}


def _targets_for_trials(trials, attack_targets: dict[str, list[str]]) -> list[list[str]]:
    out = []
    for trial in trials:
        if trial.trial_id not in attack_targets:
            raise KeyError(f"no pre-registered targets for {trial.trial_id!r}")
        out.append(list(attack_targets[trial.trial_id]))
    return out


def _build_clean_artifacts(speller: Speller, trials, targets_per_trial, device,
                           window_seed: int) -> CleanArtifacts:
    """Clean genuine ERPs, clean target ERPs and clean windows for one subject."""
    true_erp, target_erp, starts_all, windows = [], [], [], []
    true_idx, target_idx = [], []
    for ti, trial in enumerate(trials):
        eeg = torch.as_tensor(trial.eeg[None], dtype=torch.float32, device=device)
        onsets = torch.as_tensor(trial.onsets[None], dtype=torch.long, device=device)
        groups = torch.as_tensor(trial.flash_groups[None], dtype=torch.bool, device=device)
        t_idx = [trial.layout.index(s) for s in targets_per_trial[ti]]
        with torch.no_grad():
            epochs = speller.epochs(eeg, onsets)[0]
            t_true = trial.layout.index(trial.target_symbol)
            true_erp.append(
                target_conditioned_erp(epochs, groups[0], t_true)
                .to(torch.float32).cpu().numpy()
            )
            target_erp.append(np.stack([
                target_conditioned_erp(epochs, groups[0], k)
                .to(torch.float32).cpu().numpy()
                for k in t_idx
            ]))
            x_pre = speller.preprocessor(eeg)
        fs_out = speller.preprocessor.fs_out
        starts = sample_window_starts(fs_out, x_pre.shape[-1], seed=window_seed + ti)
        starts_all.append(starts)
        windows.append(windows_from_preprocessed(x_pre, starts[None], fs_out)[0])
        true_idx.append(t_true)
        target_idx.append(t_idx)
    return CleanArtifacts(
        subject=trials[0].subject_id,
        trial_index=np.array([t.trial_index for t in trials], dtype=np.int64),
        trial_id=np.array([t.trial_id for t in trials], dtype=object),
        true_idx=np.array(true_idx, dtype=np.int64),
        true_erp=np.stack(true_erp).astype(np.float16),
        target_idx=np.array(target_idx, dtype=np.int64),
        target_erp=np.stack(target_erp).astype(np.float16),
        window_starts=np.stack(starts_all).astype(np.int64),
        windows=np.stack(windows).astype(np.float16),
        meta={"n_windows": N_WINDOWS, "window_seed": int(window_seed)},
    )


# --------------------------------------------------------------------------
# Core evaluation loop
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    subject: str
    n_trials: int
    n_targets: int
    n_combos: int
    n_combos_done: int
    n_combos_skipped: int
    n_rows: int
    clean_npz: str
    parquet: str
    spec_rows: list = field(default_factory=list)


def _success_row(flash_scores: torch.Tensor, groups: torch.Tensor,
                 target: torch.Tensor, n_reps: int) -> np.ndarray:
    """Per-pair correctness at ``n_reps`` repetitions, as a float array.

    Delegates to the shared success module so the runner and the success tests
    can never disagree on how a reduced-repetition decision is taken.
    """
    choice = success_mod.reduced_rep_choice(flash_scores, groups, None, n_reps)
    return (choice == target).float().cpu().numpy()


def _shard_path(out_dir: Path, spec: MethodSpec, mult: float) -> Path:
    eps = f"{mult:.2f}".replace(".", "p")
    return out_dir / "shards" / f"{spec.label()}_eps{eps}.parquet"


def _rebuild_parquet(out_dir: Path) -> str:
    """Concatenate every shard into a single attacks.parquet."""
    shard_dir = out_dir / "shards"
    frames = []
    if shard_dir.is_dir():
        for path in sorted(shard_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parquet = out_dir / "attacks.parquet"
    frame.to_parquet(parquet, index=False)
    return str(parquet)


def run_evaluation(
    *,
    trials: list,
    targets_per_trial: list[list[str]],
    models: dict[str, Speller],
    providers: dict[str, "_DeltaProvider"],
    eps_scale: float,
    out_dir,
    seed: int = 0,
    eps_multipliers=EPS_MULTIPLIERS,
    specs=None,
    device: str = "cpu",
    max_flash_batch: int | None = None,
    force: bool = False,
    batch_trials: int = 8,
) -> EvalResult:
    """Evaluate every (spec, eps) combination and write artifacts and rows.

    ``models`` must contain a ``surrogate`` entry and may contain any number of
    victim entries. ``providers`` maps a spec's combo key to its delta provider.
    A combination is skipped when its NPZ and shard already exist and ``force``
    is False, which makes the run resumable.
    """
    if "surrogate" not in models:
        raise ValueError("models must include a 'surrogate' speller")
    if not trials:
        raise ValueError("run_evaluation needs at least one trial")
    if float(eps_scale) <= 0:
        raise ValueError("eps_scale must be positive")
    specs = list(DEFAULT_METHOD_SPECS if specs is None else specs)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    surrogate = models["surrogate"]

    clean_path = clean_npz_path(out_path, trials[0].subject_id)
    rebuild_clean = force or not clean_path.exists()
    if not rebuild_clean:
        clean = CleanArtifacts.load(clean_path)
        expected_ids = np.array([t.trial_id for t in trials], dtype=object)
        expected_targets = np.array([
            [t.layout.index(symbol) for symbol in symbols]
            for t, symbols in zip(trials, targets_per_trial)
        ], dtype=np.int64)
        rebuild_clean = (
            not np.array_equal(clean.trial_id, expected_ids)
            or not np.array_equal(clean.target_idx, expected_targets)
        )
    if rebuild_clean:
        clean = _build_clean_artifacts(
            surrogate, trials, targets_per_trial, device, window_seed=seed
        )
        clean.save(clean_path)
        # Existing attack artifacts were computed against a different trial or
        # target set, so they cannot be resumed safely.
        force = True

    eps_list = [float(m) for m in eps_multipliers]
    n_done, n_skipped, n_rows = 0, 0, 0
    spec_rows = []
    for spec in specs:
        for mult in eps_list:
            spec_art = ArtifactSpec(
                subject=trials[0].subject_id, method=spec.name,
                generator=spec.generator, bank=spec.bank, eps_multiplier=mult,
            )
            npz_path = attack_npz_path(out_path, spec_art)
            shard = _shard_path(out_path, spec, mult)
            if not force and npz_path.exists() and shard.exists():
                n_skipped += 1
                continue

            rows, erp, wins = _evaluate_combo(
                spec=spec, mult=mult, trials=trials, targets_per_trial=targets_per_trial,
                models=models, provider=providers[spec.label()],
                clean=clean, eps_scale=float(eps_scale), device=device,
                seed=seed, max_flash_batch=max_flash_batch, batch_trials=batch_trials,
            )
            AttackArtifacts(
                subject=trials[0].subject_id,
                trial_index=clean.trial_index,
                trial_id=clean.trial_id,
                true_idx=clean.true_idx,
                target_idx=clean.target_idx,
                target_symbol=np.array(targets_per_trial, dtype=object),
                decision_erp=erp,
                window_starts=clean.window_starts,
                windows=wins,
                meta={"method": spec.name, "generator": spec.generator,
                      "bank": spec.bank, "eps_multiplier": mult},
            ).save(npz_path)
            pd.DataFrame(rows).to_parquet(shard, index=False)
            n_done += 1
            n_rows += len(rows)
            spec_rows.append({"method": spec.name, "generator": spec.generator,
                              "bank": spec.bank, "eps_multiplier": mult,
                              "n_rows": len(rows)})

    parquet = _rebuild_parquet(out_path)
    total_rows = 0
    if Path(parquet).exists():
        total_rows = int(len(pd.read_parquet(parquet)))
    return EvalResult(
        subject=trials[0].subject_id,
        n_trials=len(trials),
        n_targets=len(targets_per_trial[0]) if targets_per_trial else 0,
        n_combos=len(specs) * len(eps_list),
        n_combos_done=n_done,
        n_combos_skipped=n_skipped,
        n_rows=total_rows,
        clean_npz=str(clean_path),
        parquet=parquet,
        spec_rows=spec_rows,
    )


def _evaluate_combo(*, spec, mult, trials, targets_per_trial, models,
                    provider, clean, eps_scale, device, seed, max_flash_batch,
                    batch_trials=8):
    """One (spec, eps) pass: build attacked inputs and collect rows and arrays.

    Every (trial, attacker target) pair is flattened into one sample list and
    processed in chunks of ``batch_trials`` samples, so a GPU sees a real batch
    instead of one trial at a time. All trials of a subject share C, T, F and K,
    so any chunk stacks cleanly.
    """
    surrogate = models["surrogate"]
    rng = np.random.default_rng(seed)
    eps_raw_value = float(mult) * float(eps_scale)
    n_trials = len(trials)
    n_targets = len(targets_per_trial[0])
    c_len, l_len = clean.true_erp.shape[1], clean.true_erp.shape[2]
    n_windows, c_ch, w_len = clean.windows.shape[1], clean.windows.shape[2], clean.windows.shape[3]
    erp_out = np.empty((n_trials, n_targets, c_len, l_len), dtype=np.float16)
    wins_out = np.empty((n_trials, n_targets, n_windows, c_ch, w_len), dtype=np.float16)
    rows = []

    # Flatten to (trial, target) pairs, keeping the mapping for the rows.
    pairs = [(ti, k) for ti in range(n_trials) for k in range(n_targets)]
    stack_eeg = np.stack([t.eeg for t in trials])                 # [T, C, T_raw]
    stack_onsets = np.stack([t.onsets for t in trials])           # [T, F]
    stack_groups = np.stack([t.flash_groups for t in trials])     # [T, F, K]
    stack_starts = np.asarray(clean.window_starts)                # [T, n_windows]
    stack_target = np.asarray(clean.target_idx)                   # [T, K]
    batch_trials = max(int(batch_trials), 1)
    fs_out = surrogate.preprocessor.fs_out

    for start in range(0, len(pairs), batch_trials):
        chunk = pairs[start:start + batch_trials]
        ti_idx = np.array([p[0] for p in chunk])
        k_idx = np.array([p[1] for p in chunk])
        eeg = torch.as_tensor(stack_eeg[ti_idx], dtype=torch.float32, device=device)
        onsets = torch.as_tensor(stack_onsets[ti_idx], dtype=torch.long, device=device)
        groups = torch.as_tensor(stack_groups[ti_idx], dtype=torch.bool, device=device)
        target = torch.as_tensor(stack_target[ti_idx, k_idx], dtype=torch.long, device=device)
        eps_raw = torch.full((len(chunk),), eps_raw_value, dtype=torch.float32, device=device)
        eps_mult = torch.full((len(chunk),), float(mult), dtype=torch.float32, device=device)

        delta, distance, ok = provider.compute(
            surrogate, eeg, onsets, groups, target, eps_raw, eps_mult, rng
        )
        delta = delta.to(dtype=torch.float32, device=device)
        x_adv = eeg + delta
        metrics = budget_metrics(eeg, delta)

        per_model_success = {}
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                scores = model(x_adv, onsets, groups,
                               max_flash_batch=max_flash_batch)["flash_scores"]
                for reps in (15, 5):
                    per_model_success[(name, reps)] = _success_row(
                        scores, groups, target, reps)

        with torch.no_grad():
            adv_epochs = surrogate.epochs(x_adv, onsets)          # [m, F, C, L]
            x_pre = surrogate.preprocessor(x_adv)                 # [m, C, T']
            erp_chunk = torch.stack([
                target_conditioned_erp(adv_epochs[i], groups[i], int(target[i]))
                for i in range(len(chunk))
            ]).to(torch.float32).cpu().numpy()
            starts = stack_starts[ti_idx]
            win_chunk = windows_from_preprocessed(x_pre, starts, fs_out)

        for m, (ti, k) in enumerate(chunk):
            erp_out[ti, k] = erp_chunk[m].astype(np.float16)
            wins_out[ti, k] = win_chunk[m]
            trial = trials[ti]
            row = {
                "subject": trial.subject_id,
                "method": spec.name,
                "generator": spec.generator,
                "bank": spec.bank,
                "eps_multiplier": float(mult),
                "eps_raw": eps_raw_value,
                "trial_index": int(trial.trial_index),
                "trial_id": trial.trial_id,
                "true_symbol": trial.target_symbol,
                "true_idx": int(clean.true_idx[ti]),
                "target_symbol": targets_per_trial[ti][k],
                "target_idx": int(stack_target[ti, k]),
                "retrieval_distance": float(distance[m]),
                "attack_ok": bool(ok[m]),
            }
            for name in models:
                row[f"success_{name}_15"] = float(per_model_success[(name, 15)][m])
                row[f"success_{name}_5"] = float(per_model_success[(name, 5)][m])
            for key in ("linf", "l2", "rms", "snr_db"):
                row[key] = float(metrics[key][m].item())
            rows.append(row)

    return rows, erp_out, wins_out


# --------------------------------------------------------------------------
# High-level entry point
# --------------------------------------------------------------------------

def evaluate_attacks(
    data_root: str,
    labels_dir: str | None,
    subject: str,
    surrogate: str,
    out_dir: str,
    generator_dir: str,
    tiny_noise_dir: str,
    victim_dir: str,
    *,
    seed: int = 0,
    victim_seed: int = 0,
    eps_multipliers=EPS_MULTIPLIERS,
    device: str = "auto",
    batch_trials: int = 8,
    max_flash_batch: int | None = None,
    limit_trials: int | None = None,
    splits_dir=None,
    force: bool = False,
    pgd_steps: int = 40,
    specs=None,
    victims: bool = True,
) -> EvalResult:
    """Load every asset for one subject and run the resumable evaluation.

    ``specs`` restricts the evaluated method list (used by the smoke mode to run
    only ``noise`` and ``pgd``); assets are loaded lazily, so a run that needs no
    generator never opens a bank. ``victims=False`` evaluates the surrogate only,
    which is what the local smoke path uses when victim checkpoints are absent.
    """
    from p3a.data.loaders.comp3 import load_subject
    from p3a.data.splits import load as load_split

    subject = str(subject).upper()
    if seed < 0 or victim_seed < 0:
        raise ValueError("seeds must be nonnegative")
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else device
    dev = torch.device(resolved)
    # Keep resumability shards and artifacts isolated by attacked subject.
    out_path = Path(out_dir) / subject
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    splits_dir = Path(splits_dir) if splits_dir is not None else repo_root() / "splits"
    split = load_split(splits_dir / f"comp3_{subject}.json")
    test_ids = list(split["test"])
    if limit_trials is not None:
        test_ids = sorted(test_ids)[:int(limit_trials)]

    attack_targets = load_attack_targets(subject, splits_dir)
    test_trials_all = load_subject(data_root, subject, "test", labels_dir=labels_dir)
    test_trials = select_trials(test_trials_all, test_ids)
    targets_per_trial = _targets_for_trials(test_trials, attack_targets)

    surrogate_speller = Speller.load(surrogate).to(dev).eval()

    # Raw epsilon scale uses TRAIN trials only, matching the attack training.
    train_ids = sorted(split["train"])
    train_all = load_subject(data_root, subject, "train", labels_dir=labels_dir)
    train_trials = select_trials(train_all, train_ids)
    eps_scale = raw_eps_scale(train_trials)

    specs = list(DEFAULT_METHOD_SPECS if specs is None else specs)
    kinds = {s.kind for s in specs}

    models = {"surrogate": surrogate_speller}
    if victims:
        for name in ("deepconvnet", "shallowconvnet", "xdawn_lda"):
            suffix = ".joblib" if name == "xdawn_lda" else ".pt"
            ckpt = Path(victim_dir) / f"{subject}_{name}_s{victim_seed}{suffix}"
            if not ckpt.is_file():
                raise FileNotFoundError(f"missing victim checkpoint {ckpt}")
            models[name] = Speller.load(ckpt).to(dev).eval()

    # Build one provider per generator spec, reusing the trained generators.
    prototype_subjects = [s for s in discover_subjects(splits_dir) if s != subject]
    if not prototype_subjects:
        prototype_subjects = ["B" if subject == "A" else "A"]
    providers: dict[str, _DeltaProvider] = {}
    bank_cache = {}

    def bank_for(mode: str):
        if mode in bank_cache:
            return bank_cache[mode]
        bank = build_bank(
            subject, mode, data_root, labels_dir, surrogate_speller, train_ids,
            prototype_subjects, device=resolved,
        )
        scaler = FeatureScaler.fit([e.feature for e in bank.entries])
        bank_cache[mode] = (bank, scaler)
        return bank_cache[mode]

    tiny_templates = {}
    if "tiny_noise" in kinds:
        for mult in eps_multipliers:
            ckpt = Path(tiny_noise_dir) / f"{subject}_tiny_noise_eps{float(mult):.2f}_s{seed}.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(f"missing tiny noise checkpoint {ckpt}")
            tiny_templates[f"{float(mult):.2f}"] = TinyNoiseTemplate.load(ckpt)

    for spec in specs:
        if spec.kind == "noise":
            providers[spec.label()] = NoiseProvider()
        elif spec.kind == "pgd":
            providers[spec.label()] = PGDProvider(steps=pgd_steps)
        elif spec.kind == "tiny_noise":
            providers[spec.label()] = TinyNoiseProvider(tiny_templates)
        elif spec.kind == "generator":
            ckpt = Path(generator_dir) / (
                f"{subject}_{spec.name}_{spec.generator}_{spec.bank}_s{seed}.pt"
            )
            if not ckpt.is_file():
                raise FileNotFoundError(f"missing generator checkpoint {ckpt}")
            generator, payload = load_generator(ckpt)
            generator.to(dev).eval()
            mode, use_prototype = ATTACK_METHOD_SPECS[spec.name]
            if use_prototype:
                bank, scaler = bank_for(spec.bank)
            else:
                bank, scaler = None, None
            providers[spec.label()] = GeneratorProvider(
                generator, bank, scaler, mode, use_prototype, device=resolved
            )
        else:
            raise ValueError(f"unknown method kind {spec.kind!r}")

    result = run_evaluation(
        trials=test_trials, targets_per_trial=targets_per_trial, models=models,
        providers=providers, eps_scale=float(eps_scale), out_dir=out_path,
        seed=seed, eps_multipliers=eps_multipliers, specs=specs, device=resolved,
        max_flash_batch=max_flash_batch, force=force, batch_trials=batch_trials,
    )
    config = {
        "subject": subject,
        "surrogate": str(surrogate),
        "generator_dir": str(generator_dir),
        "tiny_noise_dir": str(tiny_noise_dir),
        "victim_dir": str(victim_dir),
        "seed": int(seed),
        "victim_seed": int(victim_seed),
        "eps_multipliers": [float(m) for m in eps_multipliers],
        "eps_scale": float(eps_scale),
        "device": resolved,
        "n_test_trials": len(test_trials),
        "n_targets": len(targets_per_trial[0]) if targets_per_trial else 0,
        "batch_trials": int(batch_trials),
        "max_flash_batch": max_flash_batch,
        "limit_trials": limit_trials,
        "force": bool(force),
        "pgd_steps": int(pgd_steps),
        "victims": bool(victims),
        "n_combos": result.n_combos,
        "n_combos_done": result.n_combos_done,
        "n_combos_skipped": result.n_combos_skipped,
        "n_rows": result.n_rows,
        "specs": [{"name": s.name, "kind": s.kind, "generator": s.generator,
                   "bank": s.bank} for s in specs],
        "data_root": str(data_root),
        "labels_dir": None if labels_dir is None else str(labels_dir),
    }
    provenance = {**environment(seed), "repo_root": str(repo_root())}
    write_config(out_path, "evaluate_attacks.config.json",
                 {"config": config, "provenance": provenance})
    return result
