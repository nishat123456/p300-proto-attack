"""Clean-vs-attacked window CNN, the strongest defender (Prompt 11).

For one (method, generator, bank, epsilon) combination, this detector decides
whether a one-second preprocessed window came from the clean trial or from the
attacked trial. It is trained on one parity fold and scored on the other, then
the two directions are averaged.

The model is small and signal-shaped: a temporal convolution over the window,
then a depthwise spatial convolution over channels, batch norm, a pool, and a
linear map to one logit. Windows of one sample are scored independently and
their logits averaged, so the detector does not depend on a particular window
order. Everything runs on CPU in seconds for the toy sizes used in tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from p3a.defenses.metrics import DEFAULT_FPR, auc, folds_by_parity, tpr_at_fpr
from p3a.utils.repro import set_seed

__all__ = [
    "WindowCNN",
    "WindowDetectorReport",
    "evaluate_window_detector",
    "score_window_model",
    "train_window_model",
]


class WindowCNN(nn.Module):
    """Temporal then spatial convolution over a [1, C, W] window."""

    def __init__(self, n_channels: int, width: int, temporal: int = 8,
                 spatial: int = 8, pool: int = 4, dropout: float = 0.25):
        super().__init__()
        if n_channels < 1 or width < 1:
            raise ValueError("n_channels and width must be positive")
        self.n_channels = int(n_channels)
        self.width = int(width)
        self.temporal = nn.Conv2d(1, temporal, kernel_size=(1, 15), padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(temporal)
        self.spatial = nn.Conv2d(temporal, temporal * spatial,
                                 kernel_size=(n_channels, 1), groups=temporal)
        self.bn2 = nn.BatchNorm2d(temporal * spatial)
        self.pool = nn.AvgPool2d((1, pool))
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, width)
            flat = self._features(probe).shape[1]
        self.head = nn.Linear(flat, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.temporal(x)))
        h = self.act(self.bn2(self.spatial(h)))
        h = self.pool(h)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [B] for a batch of single windows [B, 1, C, W]."""
        return self.head(self.drop(self._features(x))).squeeze(1)


def _windows_to_tensor(windows: np.ndarray) -> torch.Tensor:
    """[N, n_win, C, W] float window array -> [N*n_win, 1, C, W] float32 tensor."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("windows must be [N, n_windows, C, W]")
    n, w, c, width = arr.shape
    return torch.as_tensor(arr.reshape(n * w, 1, c, width))


@dataclass
class DetectorScores:
    """Per-sample scores plus the model that produced them."""

    scores: np.ndarray
    n_windows: int
    model: "WindowCNN"


def score_window_model(model: WindowCNN, windows: np.ndarray,
                       batch: int = 256) -> np.ndarray:
    """Mean window logit per sample, shape [N].

    Windows of a sample are scored individually and averaged, so the score does
    not depend on their order and a missing window would only weaken it.
    """
    tensor = _windows_to_tensor(windows)
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch):
            out.append(model(tensor[start:start + batch].to(dev)).cpu())
    per_window = torch.cat(out).reshape(np.asarray(windows).shape[0], -1)
    return per_window.mean(dim=1).numpy()


def train_window_model(attacked: np.ndarray, clean: np.ndarray,
                       train_mask: np.ndarray, n_channels: int, width: int,
                       seed: int = 0, epochs: int = 8, lr: float = 1e-3,
                       batch: int = 128, temporal: int = 8, spatial: int = 8,
                       pool: int = 4, device: str = "cpu") -> DetectorScores:
    """Train the window CNN on the training-fold clean and attacked windows.

    ``attacked`` and ``clean`` are [N, n_windows, C, W]; ``train_mask`` is the
    boolean [N] selection of the training fold. Each attacked window is a
    positive and each clean window a negative, so both classes see every sample.
    """
    attacked = np.asarray(attacked, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    if attacked.shape != clean.shape:
        raise ValueError("attacked and clean windows must share a shape")
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if train_mask.shape[0] != attacked.shape[0]:
        raise ValueError("train_mask must have one entry per sample")
    if not train_mask.any():
        raise ValueError("training fold is empty")
    set_seed(seed)
    dev = torch.device(device)

    neg = _windows_to_tensor(clean[train_mask]).to(dev)
    pos = _windows_to_tensor(attacked[train_mask]).to(dev)
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    x = torch.cat([pos, neg], dim=0)
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)]).to(dev)

    model = WindowCNN(n_channels, width, temporal=temporal, spatial=spatial,
                      pool=pool).to(dev)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(max(n_neg / max(n_pos, 1), 1e-6), device=dev)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch):
            idx = order[start:start + batch].to(dev)
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
    return DetectorScores(model=model, n_windows=attacked.shape[1],
                          scores=np.zeros(0))


def evaluate_window_detector(attacked: np.ndarray, clean: np.ndarray,
                             trial_index, fpr: float = DEFAULT_FPR, seed: int = 0,
                             epochs: int = 8, model_kwargs: dict | None = None,
                             device: str = "cpu") -> tuple[dict, np.ndarray]:
    """Two-fold cross-validated AUC and TPR at ``fpr`` for the window CNN.

    Returns a summary dict with the mean AUC, mean TPR, per-fold values and the
    out-of-fold scores, and the out-of-fold scores for any downstream use. The
    negative class is the clean window of the same sample, so the detector can
    only succeed by reading the perturbation, never by reading identity.
    """
    attacked = np.asarray(attacked, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    trial_index = np.asarray(trial_index, dtype=np.int64).reshape(-1)
    folds = folds_by_parity(trial_index)
    n, n_windows, c, width = attacked.shape

    oof = np.full(n, np.nan, dtype=np.float64)
    fold_rows = []
    for fold, test_mask in folds.items():
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        det = train_window_model(
            attacked, clean, train_mask, c, width, seed=seed, epochs=epochs,
            device=device, **(model_kwargs or {}),
        )
        scores = score_window_model(det.model, np.concatenate(
            [attacked[test_mask], clean[test_mask]], axis=0
        ))
        pos = scores[: int(test_mask.sum())]
        neg = scores[int(test_mask.sum()):]
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        all_scores = np.concatenate([pos, neg])
        # Place out-of-fold scores at the positive sample positions only.
        oof[np.flatnonzero(test_mask)] = pos
        fold_auc = auc(all_scores, labels)
        fold_tpr, threshold = tpr_at_fpr(all_scores, labels, fpr)
        fold_rows.append({
            "fold": int(fold), "auc": fold_auc, "tpr_at_fpr": fold_tpr,
            "threshold": threshold, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
        })

    if not fold_rows:
        return {"auc": float("nan"), "tpr_at_fpr": float("nan"), "folds": []}, oof
    summary = {
        "auc": float(np.nanmean([r["auc"] for r in fold_rows])),
        "tpr_at_fpr": float(np.nanmean([r["tpr_at_fpr"] for r in fold_rows])),
        "folds": fold_rows,
    }
    return summary, oof


@dataclass
class WindowDetectorReport:
    """Flattened rows for one combination, ready for the results CSV."""

    rows: list

    def to_records(self) -> list:
        return list(self.rows)
