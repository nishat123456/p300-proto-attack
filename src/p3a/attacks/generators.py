"""GPN-style perturbation generators with optional prototype conditioning.

Implemented from the description in PROMPTS.md (Prompt 7); nothing is ported
from any existing repository. Two generators share a `Conditioner` that turns
the attacker target, an optional prototype ERP and the current epsilon
multiplier into a conditioning vector `z`:

- `SignalGPN` reads the source trial and predicts a dense perturbation over the
  raw sampling grid. It is signal-specific and not causal, so it is the upper
  bound in the experiments.
- `CausalGPN` never reads the source EEG content. It emits one fixed-length
  template that is placed at every flash containing the attacker target, so the
  whole perturbation is precomputable from the flash schedule alone (the fair
  comparison to Tiny Noise).

Both return a perturbation in RAW units whose per-sample L-infinity norm equals
the requested epsilon, so the caller only has to add it to the raw signal. The
GPN body is a small ResNet-style encoder-decoder over the EEG viewed as a 2D
image with (channels, time) as its spatial axes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Conditioner",
    "SignalGPN",
    "CausalGPN",
    "target_flash_mask",
    "linf_rescale",
    "place_template_at_onsets",
]

# The causal template is 800 ms at 240 Hz, matching the epoch window.
DEFAULT_TEMPLATE_LEN = 192


def target_flash_mask(groups: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """Boolean [B, F] mask of flashes that light the attacker target.

    Uses only boolean membership, so the same helper serves the signal-specific
    target map and the causal template placement.
    """
    if groups.ndim != 3:
        raise ValueError("groups must be [B, F, K]")
    if target_idx.ndim != 1 or target_idx.shape[0] != groups.shape[0]:
        raise ValueError("target_idx must be [B] aligned with groups")
    if groups.dtype != torch.bool:
        raise ValueError("groups must be boolean")
    b = torch.arange(groups.shape[0], device=groups.device)[:, None]
    f = torch.arange(groups.shape[1], device=groups.device)[None, :]
    return groups[b, f, target_idx[:, None]]


def linf_rescale(delta: torch.Tensor, eps) -> torch.Tensor:
    """Rescale every sample of `delta` [B, C, T] so its L-infinity norm is eps.

    `eps` is a scalar or [B] tensor in raw units. A sample whose perturbation is
    exactly zero stays zero: rescaling cannot invent a direction.
    """
    if not torch.is_tensor(eps):
        eps = torch.as_tensor(eps, dtype=delta.dtype, device=delta.device)
    eps = eps.to(dtype=delta.dtype, device=delta.device)
    if eps.ndim == 0:
        eps = eps.expand(delta.shape[0])
    if eps.shape != (delta.shape[0],):
        raise ValueError("eps must be a scalar or one value per sample")
    if not bool(torch.isfinite(eps).all()) or bool((eps < 0).any()):
        raise ValueError("eps must contain finite, non-negative values")
    peak = delta.flatten(1).abs().amax(dim=1).clamp_min(1e-12)
    scale = eps / peak
    return delta * scale[:, None, None]


def place_template_at_onsets(template: torch.Tensor, onsets: torch.Tensor,
                             groups: torch.Tensor, target_idx: torch.Tensor,
                             length: int) -> torch.Tensor:
    """Place one template [B, C, W] at every flash containing the attacker target.

    This is the shared causal placement semantics: the (template_len) window
    starting at each selected flash onset is added into the trial, overlaps add,
    and contributions past sample `length` are dropped. Only boolean membership
    (`groups`) selects flashes, so the result is identical for any source EEG
    with the same schedule, target and template.
    """
    batch, n_channels, template_len = template.shape
    if onsets.ndim != 2 or onsets.shape[1] != groups.shape[1]:
        raise ValueError("onsets and groups must align on (B, F)")
    if onsets.shape[0] != batch:
        raise ValueError("template batch must match onsets")
    if groups.ndim != 3 or groups.shape[0] != batch:
        raise ValueError("groups must be [B, F, K] aligned with the template")
    mask = target_flash_mask(groups, target_idx)  # [B, F]
    delta = template.new_zeros(batch, n_channels, length)
    offsets = torch.arange(template_len, device=template.device)
    for f in range(onsets.shape[1]):
        selected = mask[:, f]
        if not bool(selected.any()):
            continue
        index = onsets[:, f][:, None] + offsets[None, :]  # [B, template_len]
        valid = index < length
        index = index.clamp(max=length - 1)
        contrib = template * selected.to(template.dtype)[:, None, None]
        contrib = contrib * valid.to(template.dtype)[:, None, :]
        delta = delta.scatter_add(
            2, index[:, None, :].expand(batch, n_channels, template_len).contiguous(),
            contrib,
        )
    return delta


def _one_hot(target_idx: torch.Tensor, n_symbols: int) -> torch.Tensor:
    if target_idx.dtype != torch.long:
        raise ValueError("target_idx must be an int64 tensor")
    if bool((target_idx < 0).any()) or bool((target_idx >= n_symbols).any()):
        raise ValueError(f"target_idx outside [0, {n_symbols})")
    return F.one_hot(target_idx, n_symbols).to(torch.float32)


class Conditioner(nn.Module):
    """Build the conditioning vector `z` shared by both generators.

    Three branches are concatenated and mixed by a small MLP:
    target one-hot [K] -> 64, prototype ERP [C, L] -> 64 (a zero vector when no
    prototype is used), and the epsilon multiplier scalar -> 16.
    """

    def __init__(self, n_symbols: int, n_channels: int, z_dim: int = 128,
                 proto_hidden: int = 64, eps_dim: int = 16):
        super().__init__()
        self.n_symbols = int(n_symbols)
        self.n_channels = int(n_channels)
        self.z_dim = int(z_dim)
        self.target_encoder = nn.Linear(self.n_symbols, 64)
        self.proto_encoder = nn.Sequential(
            nn.Conv1d(self.n_channels, proto_hidden, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(proto_hidden, 64, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.eps_encoder = nn.Linear(1, eps_dim)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + eps_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
        )

    def forward(self, target_onehot: torch.Tensor, proto_feat: torch.Tensor | None,
                eps_mult, use_prototype: bool = True) -> torch.Tensor:
        """Return z [B, z_dim].

        `target_onehot` is [B, K] float, `proto_feat` is [B, C, L] float (or
        None), `eps_mult` is a scalar or [B] tensor. `use_prototype=False` (the
        target-only GPN) replaces the prototype branch with zeros.
        """
        if target_onehot.ndim != 2 or target_onehot.shape[1] != self.n_symbols:
            raise ValueError("target_onehot must be [B, K]")
        batch = target_onehot.shape[0]
        target = self.target_encoder(target_onehot.to(torch.float32))

        if use_prototype and proto_feat is not None:
            if proto_feat.ndim != 3 or proto_feat.shape[0] != batch:
                raise ValueError("proto_feat must be [B, C, L]")
            if proto_feat.shape[1] != self.n_channels:
                raise ValueError("proto_feat channel count does not match")
            proto = self.proto_encoder(proto_feat.to(torch.float32)).mean(dim=2)
        else:
            proto = target.new_zeros(batch, 64)

        if not torch.is_tensor(eps_mult):
            eps_mult = torch.as_tensor(eps_mult, dtype=torch.float32,
                                       device=target.device)
        eps_mult = eps_mult.to(dtype=torch.float32, device=target.device)
        if eps_mult.numel() == 1:
            eps_mult = eps_mult.reshape(1).expand(batch)
        if eps_mult.numel() != batch:
            raise ValueError("eps_mult must be a scalar or one value per sample")
        eps_mult = eps_mult.reshape(batch, 1)
        eps = self.eps_encoder(eps_mult)

        return self.mlp(torch.cat([target, proto, eps], dim=1))


class _ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus an identity skip, as in the GPN encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(x + h)


class _FiLM(nn.Module):
    """Per-channel feature-wise affine modulation from the conditioning vector.

    Standard initialization is used (not a zero-init) so the prototype branch
    actually influences the decoded perturbation from the first forward pass;
    the experiments require the conditioning path to be active because the
     redundancy check compares prototype guidance against target-only guidance.
    """

    def __init__(self, z_dim: int, n_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(z_dim, n_channels)
        self.to_beta = nn.Linear(z_dim, n_channels)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(z)[:, :, None, None]
        beta = self.to_beta(z)[:, :, None, None]
        return x * (1.0 + gamma) + beta


class _GPNBody(nn.Module):
    """ResNet-style encoder-decoder over an EEG image [B, in_ch, C, T'].

    reflect-pad + 7x7 conv (base) + BN + ReLU; 3x3 stride-2 conv (mid) + BN +
    ReLU; two residual blocks; transposed 3x3 stride-2 conv (base) + BN + ReLU;
    reflect-pad + 7x7 conv to one channel + tanh. Output spatial size equals the
    input, cropped if the stride arithmetic gained a pixel.
    """

    def __init__(self, in_channels: int = 2, base: int = 64, mid: int = 128):
        super().__init__()
        self.base = int(base)
        self.mid = int(mid)
        self.stem_conv = nn.Conv2d(in_channels, self.base, kernel_size=7)
        self.stem_bn = nn.BatchNorm2d(self.base)
        self.down_conv = nn.Conv2d(self.base, self.mid, kernel_size=3, stride=2, padding=1)
        self.down_bn = nn.BatchNorm2d(self.mid)
        self.res1 = _ResidualBlock(self.mid)
        self.res2 = _ResidualBlock(self.mid)
        self.up_conv = nn.ConvTranspose2d(self.mid, self.base, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.up_bn = nn.BatchNorm2d(self.base)
        self.head_conv = nn.Conv2d(self.base, 1, kernel_size=7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, films: nn.ModuleList | None = None,
                z: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        y = self.relu(self.stem_bn(self.stem_conv(F.pad(x, (3, 3, 3, 3), mode="reflect"))))
        y = self.relu(self.down_bn(self.down_conv(y)))
        y = self.res1(y)
        if films is not None:
            y = films[0](y, z)
        y = self.res2(y)
        if films is not None:
            y = films[1](y, z)
        y = self.relu(self.up_bn(self.up_conv(y)))
        y = y[..., :height, :width]
        return self.tanh(self.head_conv(F.pad(y, (3, 3, 3, 3), mode="reflect")))


def _validate_generator_inputs(eeg, onsets, groups, target_idx):
    if eeg.ndim != 3 or not eeg.is_floating_point():
        raise ValueError("eeg must be a floating point [B, C, T] tensor")
    batch = eeg.shape[0]
    if onsets.ndim != 2 or onsets.shape[0] != batch or onsets.dtype != torch.long:
        raise ValueError("onsets must be an int64 [B, F] tensor")
    if groups.ndim != 3 or groups.shape[:2] != onsets.shape or groups.dtype != torch.bool:
        raise ValueError("groups must be a boolean [B, F, K] tensor")
    if target_idx.ndim != 1 or target_idx.shape[0] != batch:
        raise ValueError("target_idx must be [B]")
    if not (eeg.device == onsets.device == groups.device == target_idx.device):
        raise ValueError("eeg, onsets, groups, and target_idx must share a device")
    if bool((onsets < 0).any()) or bool((onsets >= eeg.shape[-1]).any()):
        raise ValueError("onsets must fall inside the raw trial")


class SignalGPN(nn.Module):
    """Signal-specific generator: sees the source trial and the target schedule.

    The raw trial is average-pooled by `down` along time; a second image channel
    marks the downsampled onsets of flashes that contain the attacker target.
    The decoded [B, C, T/down] output is linearly interpolated back to the raw
    length and rescaled to the per-sample epsilon.
    """

    def __init__(self, n_symbols: int, n_channels: int, z_dim: int = 128,
                 down: int = 4, base: int = 64, mid: int = 128,
                 use_prototype: bool = True):
        super().__init__()
        self.conditioner = Conditioner(n_symbols, n_channels, z_dim=z_dim)
        self.body = _GPNBody(in_channels=2, base=base, mid=mid)
        self.films = nn.ModuleList([_FiLM(z_dim, mid), _FiLM(z_dim, mid)])
        self.down = int(down)
        self.use_prototype = bool(use_prototype)

    def forward(self, eeg: torch.Tensor, onsets: torch.Tensor, groups: torch.Tensor,
                target_idx: torch.Tensor, eps, proto_feat: torch.Tensor | None = None,
                use_prototype: bool | None = None,
                eps_mult: torch.Tensor | None = None) -> torch.Tensor:
        _validate_generator_inputs(eeg, onsets, groups, target_idx)
        batch, n_channels, length = eeg.shape
        if n_channels != self.conditioner.n_channels:
            raise ValueError("eeg channel count does not match the generator")
        proto_flag = self.use_prototype if use_prototype is None else bool(use_prototype)
        # The epsilon *multiplier* conditions the network; the raw epsilon only
        # sets the rescale budget. Callers that do not pass a multiplier keep the
        # original behavior of conditioning on the raw epsilon.
        cond_eps = eps if eps_mult is None else eps_mult
        z = self.conditioner(_one_hot(target_idx, self.conditioner.n_symbols),
                             proto_feat, cond_eps, proto_flag)

        pooled_len = math.ceil(length / self.down)
        pad = pooled_len * self.down - length
        if pad:
            eeg = F.pad(eeg, (0, pad))
        pooled = F.avg_pool1d(eeg, kernel_size=self.down, stride=self.down)  # [B, C, T/down]

        mask = target_flash_mask(groups, target_idx)  # [B, F]
        down_onsets = onsets // self.down  # [B, F]
        acc = pooled.new_zeros(batch, 1, pooled_len)
        acc.scatter_add_(2, down_onsets[:, None, :], mask.to(pooled.dtype)[:, None, :])
        target_map = (acc > 0).to(pooled.dtype)  # [B, 1, T/down]
        target_map = target_map.unsqueeze(2).expand(batch, 1, n_channels, pooled_len)

        image = torch.cat([pooled[:, None, :, :], target_map], dim=1)  # [B, 2, C, T/down]
        out = self.body(image, self.films, z).squeeze(1)  # [B, C, T/down]
        out = F.interpolate(out, size=length, mode="linear", align_corners=False)
        return linf_rescale(out, eps)


class CausalGPN(nn.Module):
    """Causal generator: precomputable from the flash schedule alone.

    The conditioning vector is expanded by an MLP and three transposed 1D
    convolutions into one template [C, template_len] (800 ms at 240 Hz). The
    perturbation for a trial is the sum of the template placed at every target
    flash onset, with overlaps adding and anything past the trial end cropped.
    Only the source trial's shape is used, never its content, so two different
    trials with the same schedule, target and prototype get identical output.
    """

    def __init__(self, n_symbols: int, n_channels: int, z_dim: int = 128,
                 template_len: int = DEFAULT_TEMPLATE_LEN, hidden: int = 32,
                 use_prototype: bool = True):
        super().__init__()
        self.template_len = int(template_len)
        if self.template_len % 8 != 0:
            raise ValueError("template_len must be divisible by 8 for three x2 upsamples")
        self.seed_len = self.template_len // 8
        self.hidden = int(hidden)
        self.conditioner = Conditioner(n_symbols, n_channels, z_dim=z_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, self.hidden * self.seed_len),
            nn.ReLU(),
        )
        self.deconv1 = nn.ConvTranspose1d(self.hidden, self.hidden, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(self.hidden)
        self.deconv2 = nn.ConvTranspose1d(self.hidden, self.hidden, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden)
        self.deconv3 = nn.ConvTranspose1d(self.hidden, self.hidden, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(self.hidden)
        self.to_channels = nn.Conv1d(self.hidden, n_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.use_prototype = bool(use_prototype)

    def _template(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        h = self.mlp(z).reshape(batch, self.hidden, self.seed_len)
        h = self.relu(self.bn1(self.deconv1(h)))
        h = self.relu(self.bn2(self.deconv2(h)))
        h = self.relu(self.bn3(self.deconv3(h)))
        return self.tanh(self.to_channels(h))  # [B, C, template_len]

    def place_template(self, template: torch.Tensor, onsets: torch.Tensor,
                       groups: torch.Tensor, target_idx: torch.Tensor,
                       length: int) -> torch.Tensor:
        """Scatter the template at each target flash onset, overlaps adding.

        Thin wrapper over the module level `place_template_at_onsets` so the
        generator and the Tiny Noise baseline share exactly one placement
        implementation.
        """
        return place_template_at_onsets(template, onsets, groups, target_idx, length)

    def forward(self, eeg: torch.Tensor, onsets: torch.Tensor, groups: torch.Tensor,
                target_idx: torch.Tensor, eps, proto_feat: torch.Tensor | None = None,
                use_prototype: bool | None = None,
                eps_mult: torch.Tensor | None = None) -> torch.Tensor:
        _validate_generator_inputs(eeg, onsets, groups, target_idx)
        if eeg.shape[1] != self.conditioner.n_channels:
            raise ValueError("eeg channel count does not match the generator")
        length = eeg.shape[-1]
        proto_flag = self.use_prototype if use_prototype is None else bool(use_prototype)
        # The epsilon *multiplier* conditions the network; the raw epsilon only
        # sets the rescale budget. Callers that do not pass a multiplier keep the
        # original behavior of conditioning on the raw epsilon.
        cond_eps = eps if eps_mult is None else eps_mult
        z = self.conditioner(_one_hot(target_idx, self.conditioner.n_symbols),
                             proto_feat, cond_eps, proto_flag)
        template = self._template(z)
        delta = self.place_template(template, onsets, groups, target_idx, length)
        return linf_rescale(delta, eps)
