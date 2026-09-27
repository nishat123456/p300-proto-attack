"""xDAWN spatial filtering followed by linear discriminant analysis.

Classical P300 decoder: the target-class average ERP gives the xDAWN spatial
filters, and an LDA is fitted on the flattened filtered epochs. This is the
pipeline family used by the Tiny Noise baseline and serves as a victim model.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    import joblib
except ImportError:  # pragma: no cover - joblib ships with scikit-learn
    joblib = None


class XdawnLDA:
    """xDAWN + LDA flash decoder with a scikit-learn style interface."""

    def __init__(self, n_filters: int = 4, reg: float = 1e-6):
        self.n_filters = int(n_filters)
        self.reg = float(reg)
        self.filters_: np.ndarray | None = None
        self.clf_: LinearDiscriminantAnalysis | None = None

    @staticmethod
    def _signal_covariance(epochs: np.ndarray, reg: float) -> np.ndarray:
        """Covariance of all epochs concatenated in time: mean over time of X X^T."""
        n, c, length = epochs.shape
        flat = epochs.transpose(1, 0, 2).reshape(c, n * length)
        cov = flat @ flat.T / (n * length)
        cov = cov + reg * np.trace(cov) / c * np.eye(c)
        return cov

    def fit(self, epochs_np, y) -> "XdawnLDA":
        x = np.asarray(epochs_np, dtype=np.float64)
        y = np.asarray(y).astype(bool).reshape(-1)
        if x.ndim != 3:
            raise ValueError("epochs must have shape [N, C, L]")
        if x.shape[0] != y.shape[0]:
            raise ValueError("epochs and labels must have the same length")
        if not y.any():
            raise ValueError("xDAWN needs at least one target epoch")
        n, c, length = x.shape
        p = x[y].mean(axis=0)  # [C, L]
        if c < self.n_filters:
            raise ValueError("n_filters cannot exceed the number of channels")

        outer = p @ p.T
        s = self._signal_covariance(x, self.reg)
        # Generalized eigendecomposition of (P P^T, S); largest eigenvalues first.
        eigvals, eigvecs = eigh(outer, s)
        order = np.argsort(eigvals)[::-1]
        self.filters_ = eigvecs[:, order[: self.n_filters]].T  # [n_filters, C]

        features = np.einsum("fc,ncl->nfl", self.filters_, x).reshape(n, -1)
        self.clf_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.clf_.fit(features, y.astype(int))
        return self

    def _features(self, epochs_np) -> np.ndarray:
        if self.filters_ is None or self.clf_ is None:
            raise RuntimeError("XdawnLDA must be fitted before use")
        x = np.asarray(epochs_np, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != self.filters_.shape[1]:
            raise ValueError("epochs have the wrong shape for the fitted filters")
        return np.einsum("fc,ncl->nfl", self.filters_, x).reshape(x.shape[0], -1)

    def decision_function(self, epochs_np) -> np.ndarray:
        return self.clf_.decision_function(self._features(epochs_np)).reshape(-1)

    def predict(self, epochs_np) -> np.ndarray:
        """Boolean P300-present prediction, matching decision_function sign."""
        return (self.decision_function(epochs_np) > 0).astype(bool)

    def save(self, path) -> None:
        if joblib is None:
            raise RuntimeError("joblib is required to save XdawnLDA")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "XdawnLDA":
        if joblib is None:
            raise RuntimeError("joblib is required to load XdawnLDA")
        return joblib.load(path)
