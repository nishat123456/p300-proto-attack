"""ShallowConvNet for flash-level P300 detection.

Temporal convolution, spatial convolution over channels, batch norm, squaring,
average pooling with optional padding so short epochs still work, log
compression, dropout and a linear head. Output is one logit per epoch [N].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_dim(n_filters: int, n_channels: int, length: int, kernel_length: int,
                 pool_kernel: int, pool_stride: int) -> int:
    """Feature size after temporal conv, spatial conv, pooling and flattening."""
    after_temporal = length + 2 * (kernel_length // 2) - kernel_length + 1
    pad = 0
    if after_temporal < pool_kernel:
        pad = math.ceil((pool_kernel - after_temporal) / 2)
    after_pool = (after_temporal + 2 * pad - pool_kernel) // pool_stride + 1
    if after_pool < 1:
        raise ValueError("epoch is too short for the pooling kernel")
    return n_filters * after_pool


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        length: int,
        n_filters: int = 40,
        kernel_length: int = 13,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.length = int(length)
        self.pool_kernel = int(pool_kernel)
        self.pool_stride = int(pool_stride)
        if self.n_channels < 1 or self.length < 1:
            raise ValueError("n_channels and length must be positive")

        self.temporal = nn.Conv2d(1, n_filters, (1, kernel_length),
                                  padding=(0, kernel_length // 2), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, (self.n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.dropout = nn.Dropout(dropout)

        after_temporal = self.length + 2 * (kernel_length // 2) - kernel_length + 1
        self.pool_pad = 0
        if after_temporal < self.pool_kernel:
            self.pool_pad = math.ceil((self.pool_kernel - after_temporal) / 2)

        self.classifier = nn.Linear(
            _flatten_dim(n_filters, self.n_channels, self.length, kernel_length,
                         self.pool_kernel, self.pool_stride),
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected epochs [N, C, L]")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        y = self.temporal(x.unsqueeze(1))
        y = self.spatial(y)
        y = self.bn(y)
        y = y.square()
        y = F.avg_pool2d(y, (1, self.pool_kernel), (1, self.pool_stride),
                         padding=(0, self.pool_pad))
        y = torch.log(torch.clamp(y, min=1e-6))
        y = self.dropout(y)
        return self.classifier(y.flatten(1)).squeeze(1)
