"""Attack baselines that produce perturbations in RAW EEG units (Prompt 9).

Every baseline here builds a perturbation `delta` with the same shape as the
raw trial and the same per-sample L-infinity budget `eps_raw`, then hands it
back so the caller can evaluate it through the shared `Speller`. Nothing in
this module touches the victims, the test split, or the prototype bank.

Implemented baselines
---------------------
- ``targeted_pgd``: a non-generative, signal-specific L-infinity PGD attack on
  the symbol cross entropy. It is the strongest but least causal reference and
  the placeholder that shows the letter loss is attackable at all.
- ``TinyNoiseTemplate``: a re-implementation of the *idea* of Zhang et al.,
  "Tiny noise, big mistakes", NSR 2021, optimized by gradients rather than by
  their closed form. One universal raw template per (subject, eps) is placed at
  every flash containing the attacker target, using the same causal placement
  semantics as ``CausalGPN``. It carries no conditioning and no prototype, so it
  is the fair universal baseline for the prototype-guided generators.
- ``random_noise``: amplitude-matched Gaussian noise over the whole trial.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p3a.attacks.generators import (
    DEFAULT_TEMPLATE_LEN,
    linf_rescale,
    place_template_at_onsets,
    target_flash_mask,
)
from p3a.utils.repro import environment, set_seed, write_config

__all__ = [
    "BASELINE_FORMAT",
    "DEFAULT_TEMPLATE_LEN",
    "TinyNoiseTemplate",
    "random_noise",
    "targeted_pgd",
]

BASELINE_FORMAT = 1


def _as_eps_raw(eps_raw, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalize `eps_raw` to a finite, non-negative [B] tensor."""
    if not torch.is_tensor(eps_raw):
        eps_raw = torch.as_tensor(eps_raw, dtype=dtype, device=device)
    eps_raw = eps_raw.to(dtype=dtype, device=device)
    if eps_raw.ndim == 0:
        eps_raw = eps_raw.expand(batch).clone()
    if eps_raw.shape != (batch,):
        raise ValueError("eps_raw must be a scalar or one value per sample")
    if not bool(torch.isfinite(eps_raw).all()) or bool((eps_raw < 0).any()):
        raise ValueError("eps_raw must contain finite, non-negative values")
    return eps_raw


def _symbol_logits(speller, x_adv: torch.Tensor, onsets: torch.Tensor,
                   groups: torch.Tensor, max_flash_batch: int | None = None) -> torch.Tensor:
    return speller(x_adv, onsets, groups, max_flash_batch=max_flash_batch)["logits"]


def targeted_pgd(speller, eeg: torch.Tensor, onsets: torch.Tensor, groups: torch.Tensor,
                 target: torch.Tensor, eps_raw, steps: int = 40, step=None,
                 random_start: bool = True, max_flash_batch: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-sample L-infinity PGD on the symbol cross entropy.

    Parameters
    ----------
    speller : the (frozen) surrogate `Speller`; its parameters must have
        ``requires_grad=False`` so gradients flow only into the perturbation.
    eeg : raw trial [B, C, T].
    onsets, groups : the flash schedule [B, F] and membership [B, F, K].
    target : attacker target symbol index [B].
    eps_raw : scalar or [B] raw-unit L-infinity budget.
    steps : number of descent steps; the default step is ``eps_raw / 10``.
    step : optional scalar step size; defaults to ``eps_raw / 10``.
    random_start : initialize inside the box with uniform noise.
    generator : optional torch Generator so the random start is reproducible.

    Returns
    -------
    delta [B, C, T] float32 with ``||delta||_inf <= eps_raw`` per sample. A
    sample with a zero budget gets an exactly zero perturbation.
    """
    if eeg.ndim != 3 or not eeg.is_floating_point():
        raise ValueError("eeg must be a floating point [B, C, T] tensor")
    batch = eeg.shape[0]
    if target.ndim != 1 or target.shape[0] != batch:
        raise ValueError("target must be [B] aligned with eeg")
    if steps < 1:
        raise ValueError("steps must be positive")
    eps = _as_eps_raw(eps_raw, batch, eeg.dtype, eeg.device)
    if step is None:
        step_size = (eps / 10.0).clamp_min(1e-12)
    else:
        step_size = torch.as_tensor(step, dtype=eeg.dtype, device=eeg.device)
        if step_size.ndim == 0:
            step_size = step_size.expand(batch)
        step_size = step_size.to(eeg.dtype)

    box = eps[:, None, None]
    if random_start:
        uniform = torch.rand(eeg.shape, dtype=eeg.dtype, device=eeg.device,
                             generator=generator)
        delta = (uniform * 2.0 - 1.0) * box
    else:
        delta = torch.zeros_like(eeg)

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        logits = _symbol_logits(speller, eeg + delta, onsets, groups, max_flash_batch)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            # The sign direction is steepest descent under an L-infinity
            # constraint. Move against it to lower the target cross entropy.
            delta = delta - step_size[:, None, None] * grad.sign()
            delta = delta.clamp(min=-box, max=box)
    return delta.detach()


def random_noise(shape, eps_raw, rng=None) -> torch.Tensor:
    """Amplitude-matched Gaussian noise with per-sample L-infinity norm `eps_raw`.

    `shape` is (B, C, T); `eps_raw` is a scalar or one value per sample. When
    `rng` is a numpy Generator the draw is reproducible. A zero budget yields an
    exactly zero perturbation rather than a rescale of pure noise.
    """
    if len(shape) != 3:
        raise ValueError("shape must be (B, C, T)")
    batch = int(shape[0])
    if rng is not None:
        noise_np = rng.standard_normal(size=tuple(shape))
        noise = torch.as_tensor(noise_np, dtype=torch.float32)
    else:
        noise = torch.randn(*shape)
    eps = _as_eps_raw(eps_raw, batch, noise.dtype, noise.device)
    # linf_rescale keeps an all-zero sample at zero and makes the peak exactly eps.
    return linf_rescale(noise, eps)


class TinyNoiseTemplate:
    """A universal raw template placed at target flashes (Tiny Noise baseline).

    The template is ``[C, template_len]`` (800 ms at 240 Hz by default) and is
    the same for every trial of a subject at a given eps. It is optimized with
    Adam to minimize the symbol cross entropy of the targeted speller, then
    projected back into the L-infinity box after every step. Placement uses the
    shared causal semantics, so the baseline is directly comparable to
    ``CausalGPN``. No prototype and no conditioning are used.
    """

    def __init__(self, n_channels: int, template_len: int = DEFAULT_TEMPLATE_LEN):
        self.n_channels = int(n_channels)
        self.template_len = int(template_len)
        if self.n_channels < 1 or self.template_len < 1:
            raise ValueError("n_channels and template_len must be positive")
        self.template = torch.zeros(self.n_channels, self.template_len)

    def delta(self, onsets: torch.Tensor, groups: torch.Tensor, target: torch.Tensor,
              length: int, eps_raw) -> torch.Tensor:
        """Place the template at each target flash and rescale to `eps_raw`.

        Arguments mirror `CausalGPN`: `onsets [B,F]`, `groups [B,F,K]`,
        `target [B]`, `length` the raw trial length. The per-sample L-infinity
        norm of the returned delta equals `eps_raw` (zero budget stays zero).
        """
        batch = onsets.shape[0]
        template = self.template.to(device=onsets.device, dtype=torch.float32)
        batched = template[None].expand(batch, self.n_channels, self.template_len)
        delta = place_template_at_onsets(batched, onsets, groups, target, length)
        eps = _as_eps_raw(eps_raw, batch, delta.dtype, delta.device)
        return linf_rescale(delta, eps)

    def fit(self, speller, trials, eps_raw, epochs: int = 30, lr: float = 0.05,
            seed: int = 0, targets=None, batch: int | None = None,
            device: str = "cpu", max_flash_batch: int | None = None,
            log_every: int = 0) -> dict:
        """Train the template on TRAIN trials with a random attacker target.

        `trials` are `Trial` objects (typically the frozen train split). Each
        epoch draws a fresh random attacker target per trial, never equal to the
        trial's true symbol, unless a fixed `targets` array is supplied. The
        optimizer is Adam and the template is clipped to ``+/- eps_raw`` after
        each step, so the budget always holds exactly.

        Returns a history dict with the eps budget, optimizer settings and the
        per-epoch mean symbol cross entropy.
        """
        from p3a.data.epochs import batch_trials

        if not trials:
            raise ValueError("TinyNoiseTemplate.fit needs at least one trial")
        if epochs < 1 or lr <= 0:
            raise ValueError("epochs and lr must be positive")
        batch = int(batch or len(trials))
        device = torch.device(device)
        set_seed(seed)
        n_symbols = trials[0].layout.n_symbols
        true_symbols = np.array(
            [t.layout.index(t.target_symbol) for t in trials], dtype=np.int64
        )
        eps = float(eps_raw)
        self.template = torch.zeros(self.n_channels, self.template_len)

        for param in speller.parameters():
            param.requires_grad_(False)
        speller.eval()

        template = torch.zeros(self.n_channels, self.template_len, device=device,
                               requires_grad=True)
        optimizer = torch.optim.Adam([template], lr=lr)
        rng = np.random.default_rng(seed)
        history = []
        if targets is not None:
            fixed_targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        for epoch in range(epochs):
            order = rng.permutation(len(trials))
            losses, n_seen = 0.0, 0
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                chunk = [trials[i] for i in idx]
                eeg, onsets, groups, _true = batch_trials(chunk, device)
                if targets is not None:
                    target = fixed_targets[torch.as_tensor(idx, dtype=torch.long,
                                                           device=device)]
                else:
                    target = _sample_targets(true_symbols[idx], rng, n_symbols, device)
                batched = template[None].expand(len(idx), self.n_channels, self.template_len)
                delta = place_template_at_onsets(batched, onsets, groups, target,
                                                 eeg.shape[-1])
                # Overlapping target flashes can make the placed perturbation
                # exceed the template box, so enforce the raw trial budget in
                # the same way used at evaluation time.
                delta = linf_rescale(delta, eps)
                x_adv = eeg + delta
                logits = _symbol_logits(speller, x_adv, onsets, groups, max_flash_batch)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    template.clamp_(min=-eps, max=eps)
                losses += float(loss.detach()) * len(idx)
                n_seen += len(idx)
            history.append(losses / max(n_seen, 1))
            if log_every and (epoch % log_every == 0):
                print(f"epoch {epoch}: ce {history[-1]:.4f}")

        self.template = template.detach().cpu().clone()
        return {"eps_raw": eps, "epochs": epochs, "lr": lr, "seed": seed,
                "n_trials": len(trials), "targets": None if targets is None else list(targets),
                "ce_history": history, "final_ce": history[-1] if history else float("nan")}

    # -- checkpointing -----------------------------------------------------

    def checkpoint(self) -> dict:
        return {
            "baseline_format": BASELINE_FORMAT,
            "kind": "tiny_noise",
            "n_channels": self.n_channels,
            "template_len": self.template_len,
            "template": self.template.clone(),
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint()
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path) -> "TinyNoiseTemplate":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("baseline_format") != BASELINE_FORMAT or payload.get("kind") != "tiny_noise":
            raise ValueError("unsupported Tiny Noise checkpoint format")
        obj = cls(payload["n_channels"], payload["template_len"])
        obj.template = payload["template"].clone()
        return obj


def _sample_targets(true_symbols: np.ndarray, rng: np.random.Generator, n_symbols: int,
                    device) -> torch.Tensor:
    """One random attacker target per sample, never the true symbol."""
    out = np.empty(len(true_symbols), dtype=np.int64)
    for i, true in enumerate(true_symbols):
        candidates = np.arange(n_symbols)
        out[i] = int(rng.choice(candidates[candidates != int(true)]))
    return torch.as_tensor(out, dtype=torch.long, device=device)


def train_tiny_noise(
    data_root: str,
    labels_dir: str | None,
    subject: str,
    surrogate: str,
    seed: int,
    out_dir: str,
    epochs: int = 30,
    lr: float = 0.05,
    device: str = "auto",
    eps_multipliers=(0.05, 0.10, 0.20, 0.30, 0.50),
    eps_scale: float | None = None,
    template_len: int = DEFAULT_TEMPLATE_LEN,
    smoke: bool = False,
    smoke_train: int = 4,
    max_flash_batch: int | None = None,
) -> dict:
    """Train one Tiny Noise template per epsilon multiplier for one subject.

    The training data are the frozen TRAIN trials only; the test split is never
    loaded. The epsilon scale is the same statistic the generators use: the
    median over channels of the per-channel standard deviation of the subject's
    raw train EEG. Each trained template is written to ``out_dir`` with a config
    sidecar containing the full settings and the git state.
    """
    from p3a.attacks.train import raw_eps_scale, repo_root, select_trials
    from p3a.data.loaders.comp3 import load_subject
    from p3a.data.splits import load as load_split
    from p3a.models.speller import Speller

    subject = str(subject).upper()
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if epochs < 1 or lr <= 0:
        raise ValueError("epochs and lr must be positive")
    if smoke:
        epochs = 1
        device = "cpu"

    resolved = ("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else device
    dev = torch.device(resolved)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    split_path = repo_root() / f"splits/comp3_{subject}.json"
    split = load_split(split_path)
    train_ids = sorted(split["train"])
    if smoke:
        train_ids = train_ids[:smoke_train]

    train_all = load_subject(data_root, subject, "train", labels_dir=labels_dir)
    train_trials = select_trials(train_all, train_ids)

    scale = float(eps_scale) if eps_scale is not None else raw_eps_scale(train_trials)
    speller = Speller.load(surrogate).to(dev)
    n_channels = train_trials[0].eeg.shape[0]

    checkpoints = {}
    histories = {}
    for mult in eps_multipliers:
        eps_raw = float(mult) * scale
        template = TinyNoiseTemplate(n_channels, template_len=template_len)
        history = template.fit(
            speller, train_trials, eps_raw, epochs=epochs, lr=lr, seed=seed,
            device=resolved, max_flash_batch=max_flash_batch,
        )
        suffix = f"{subject}_tiny_noise_eps{mult:.2f}_s{seed}"
        ckpt_path = template.save(out_path / f"{suffix}.pt")
        config = {
            "subject": subject,
            "method": "tiny_noise",
            "surrogate": str(surrogate),
            "seed": int(seed),
            "epochs": int(epochs),
            "lr": float(lr),
            "device": resolved,
            "eps_multiplier": float(mult),
            "eps_raw": eps_raw,
            "eps_scale": scale,
            "template_len": int(template_len),
            "n_channels": int(n_channels),
            "n_train_trials": len(train_trials),
            "n_test_trials": 0,
            "split_path": str(split_path),
            "smoke": bool(smoke),
            "data_root": str(data_root),
            "labels_dir": None if labels_dir is None else str(labels_dir),
        }
        provenance = {**environment(seed), "repo_root": str(repo_root())}
        write_config(out_path, f"{suffix}.config.json",
                     {"config": config, "provenance": provenance,
                      "history": history})
        checkpoints[f"{mult:.2f}"] = str(ckpt_path)
        histories[f"{mult:.2f}"] = history

    return {
        "subject": subject,
        "eps_scale": scale,
        "checkpoints": checkpoints,
        "histories": histories,
        "n_train_trials": len(train_trials),
        "n_test_trials": 0,
    }
