"""Differentiable P300 preprocessing pipeline.

Raw EEG [B, C, T] -> linear-phase FIR bandpass -> optional common average
reference -> decimation -> per-channel standardization. Every step is a
torch op so gradients can flow back to the raw signal for adversarial
attacks. Scaling statistics are fitted separately on raw training trials.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin


class Preprocessor(nn.Module):
    """Differentiable bandpass, re-reference, decimate, scale pipeline.

    Parameters
    ----------
    n_channels : number of EEG channels C.
    fs : raw sampling rate in Hz.
    band : (low, high) passband edges in Hz.
    target_fs : desired output sampling rate; the decimation factor is
        q = round(fs / target_fs) and fs_out = fs / q.
    car : apply common average reference (subtract channel mean).
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        band: tuple[float, float] = (0.5, 20.0),
        target_fs: float = 60.0,
        car: bool = True,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.band = (float(band[0]), float(band[1]))
        self.target_fs = float(target_fs)
        self.car = bool(car)

        numtaps = 2 * int(fs) + 1
        if numtaps % 2 == 0:
            numtaps += 1
        taps = firwin(numtaps, self.band, pass_zero=False, fs=fs)
        self.numtaps = int(numtaps)
        self.pad = self.numtaps // 2

        q = int(round(fs / target_fs))
        if q < 1:
            q = 1
        self.q = q
        self.fs_out = fs / q

        self.register_buffer("taps", torch.as_tensor(taps, dtype=torch.float32))
        self.register_buffer("ch_mean", torch.zeros(self.n_channels, dtype=torch.float32))
        self.register_buffer("ch_std", torch.ones(self.n_channels, dtype=torch.float32))

    def bandpass(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise linear-phase FIR bandpass, length preserved, no delay."""
        b, c, _ = x.shape
        weight = self.taps.view(1, 1, -1).expand(c, 1, -1)
        xp = F.pad(x, (self.pad, self.pad), mode="reflect")
        return F.conv1d(xp, weight, groups=c)

    def _filter_car_decimate(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bandpass(x)
        if self.car:
            y = y - y.mean(dim=1, keepdim=True)
        return y[..., :: self.q]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._filter_car_decimate(x)
        return (y - self.ch_mean[:, None]) / self.ch_std[:, None]

    @torch.no_grad()
    def fit_scaling(self, list_of_raw_arrays) -> "Preprocessor":
        """Fit per-channel mean/std over steps 1-3 of the given raw trials.

        Each array is [C, T] raw EEG (numpy or torch) at `self.fs`.
        """
        count = 0
        total = torch.zeros(self.n_channels, dtype=torch.float64)
        total_sq = torch.zeros(self.n_channels, dtype=torch.float64)
        for arr in list_of_raw_arrays:
            x = torch.as_tensor(np.asarray(arr), dtype=torch.float32).unsqueeze(0)
            y = self._filter_car_decimate(x)[0].double()
            total += y.sum(dim=1)
            total_sq += (y * y).sum(dim=1)
            count += y.shape[1]
        if count == 0:
            raise ValueError("fit_scaling received no samples")
        mean = total / count
        var = torch.clamp(total_sq / count - mean * mean, min=1e-12)
        self.ch_mean.copy_(mean.float())
        self.ch_std.copy_(torch.sqrt(var).float())
        return self

    def to_config(self) -> dict:
        """Hyperparameters needed to rebuild the module (buffers go in state_dict)."""
        return {
            "n_channels": self.n_channels,
            "fs": self.fs,
            "band": list(self.band),
            "target_fs": self.target_fs,
            "car": self.car,
            "numtaps": self.numtaps,
            "q": self.q,
            "fs_out": self.fs_out,
        }

    @classmethod
    def from_config(cls, config: dict) -> "Preprocessor":
        """Rebuild from `to_config()`; load a state_dict afterwards for buffers."""
        return cls(
            n_channels=config["n_channels"],
            fs=config["fs"],
            band=tuple(config["band"]),
            target_fs=config["target_fs"],
            car=config["car"],
        )
