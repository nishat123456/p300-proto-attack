"""EEGNet (Lawhern et al. 2018) as a flash-level P300 decoder.

Input epochs [N, C, L] (preprocessed, 60 Hz, L = 48 for Comp III). Output is
one logit per epoch [N]: the evidence that a P300 is present. The flatten size
is measured from the convolution stack so the same code works for any C, L.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_dim(encoder: nn.Module, n_channels: int, length: int) -> int:
    """Run one dummy epoch in eval mode to measure the flattened feature size."""
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.zeros(1, 1, n_channels, length))
    encoder.train(was_training)
    return int(out.shape[1])


class EEGNet(nn.Module):
    """Compact depthwise-separable CNN for short EEG epochs.

    F1 temporal filters, a depthwise spatial filter of depth D over channels,
    then a separable temporal convolution and a linear head.
    """

    def __init__(
        self,
        n_channels: int,
        length: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        temporal_kernel: int = 32,
        separable_kernel: int = 16,
        pool1: int = 4,
        pool2: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.length = int(length)
        if self.n_channels < 1 or self.length < 1:
            raise ValueError("n_channels and length must be positive")

        self.encoder = nn.Sequential(
            nn.Conv2d(1, F1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (self.n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, separable_kernel),
                      padding=(0, separable_kernel // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(_flatten_dim(self.encoder, self.n_channels, self.length), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected epochs [N, C, L]")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        features = self.encoder(x.unsqueeze(1))
        return self.classifier(features).squeeze(1)
