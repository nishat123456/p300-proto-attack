"""DeepConvNet (Schirrmeister et al. 2017) adapted to short P300 epochs.

Blocks of temporal convolution with batch norm, ELU, max pooling and dropout.
Only the first block applies a spatial convolution over the channels. Output
is one logit per epoch [N].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _flatten_dim(encoder: nn.Module, n_channels: int, length: int) -> int:
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.zeros(1, 1, n_channels, length))
    encoder.train(was_training)
    return int(out.shape[1])


class DeepConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        length: int,
        filters: tuple[int, ...] = (25, 50, 100),
        kernel_length: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.length = int(length)
        if self.n_channels < 1 or self.length < 1:
            raise ValueError("n_channels and length must be positive")

        blocks: list[nn.Module] = []
        in_channels = 1
        for i, n_filters in enumerate(filters):
            blocks.append(nn.Conv2d(in_channels, n_filters, (1, kernel_length),
                                    padding=(0, kernel_length // 2), bias=False))
            if i == 0:
                blocks.append(nn.Conv2d(n_filters, n_filters, (self.n_channels, 1), bias=False))
            blocks.extend([
                nn.BatchNorm2d(n_filters),
                nn.ELU(),
                nn.MaxPool2d((1, 2)),
                nn.Dropout(dropout),
            ])
            in_channels = n_filters
        blocks.append(nn.Flatten())
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(_flatten_dim(self.encoder, self.n_channels, self.length), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected epochs [N, C, L]")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        return self.classifier(self.encoder(x.unsqueeze(1))).squeeze(1)
