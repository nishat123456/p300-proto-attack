"""Speller: raw EEG trial in, flash scores and letter logits out.

A Speller bundles the differentiable preprocessor, a flash decoder (any torch
model from the registry, or the classical XdawnLDA), and the epoch length L.
It is the single object the attack, prototype and evaluation stages share, so
that preprocessing and the decoder travel together in one checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from p3a.data.epochs import downsample_onsets, gather_epochs
from p3a.models.letter_head import symbol_logits
from p3a.models.registry import MODEL_NAMES, build_model
from p3a.models.xdawn_lda import XdawnLDA

XDAWN_NAMES = ("xdawn_lda", "xdawnlda")


def decoder_name(decoder) -> str:
    """Canonical name for a decoder instance, used to rebuild it on load."""
    if isinstance(decoder, XdawnLDA):
        return "xdawn_lda"
    return {
        "EEGNet": "eegnet",
        "DeepConvNet": "deepconvnet",
        "ShallowConvNet": "shallowconvnet",
    }.get(type(decoder).__name__, type(decoder).__name__.lower())


class Speller(nn.Module):
    """Pipe a batch of trials through preprocessing and the flash decoder.

    Parameters
    ----------
    preprocessor : fitted `Preprocessor` (owns the FIR, CAR, scaling).
    decoder : torch flash decoder, or a fitted `XdawnLDA`.
    L : epoch length in preprocessed samples (48 for Comp III at 60 Hz).
    """

    def __init__(self, preprocessor, decoder, L: int):
        super().__init__()
        if not isinstance(preprocessor, nn.Module):
            raise TypeError("preprocessor must be an nn.Module")
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.L = int(L)
        if self.L < 1:
            raise ValueError("L must be positive")

    @property
    def n_channels(self) -> int:
        return self.preprocessor.n_channels

    @property
    def decoder_name(self) -> str:
        return decoder_name(self.decoder)

    def epochs(self, eeg: torch.Tensor, onsets: torch.Tensor) -> torch.Tensor:
        """Preprocessed flash epochs [B, F, C, L], differentiable in eeg."""
        if eeg.ndim != 3:
            raise ValueError("expected raw eeg [B, C, T]")
        if onsets.ndim != 2 or onsets.shape[0] != eeg.shape[0]:
            raise ValueError("expected onsets [B, F] aligned with eeg")
        x = self.preprocessor(eeg)
        onsets_ds = downsample_onsets(onsets, self.preprocessor.q)
        return gather_epochs(x, onsets_ds, self.L)

    def _decode(self, epochs_flat: torch.Tensor, batch: int | None = None) -> torch.Tensor:
        """Flash logits [N] for epochs [N, C, L]; numpy path for XdawnLDA.

        When `batch` is set, epochs are decoded in bounded chunks so a large
        evaluation set is never pushed through the decoder as one giant tensor.
        """
        if batch is None or batch <= 0 or batch >= epochs_flat.shape[0]:
            if isinstance(self.decoder, XdawnLDA):
                scores = self.decoder.decision_function(
                    epochs_flat.detach().cpu().numpy()
                )
                return torch.as_tensor(scores, dtype=torch.float32, device=epochs_flat.device)
            return self.decoder(epochs_flat)

        out = []
        for start in range(0, epochs_flat.shape[0], batch):
            out.append(self._decode(epochs_flat[start:start + batch], batch=None))
        return torch.cat(out, dim=0)

    def forward(self, eeg: torch.Tensor, onsets: torch.Tensor, groups: torch.Tensor,
                max_flash_batch: int | None = None) -> dict:
        """Return dict(flash_scores [B,F], logits [B,K])."""
        ep = self.epochs(eeg, onsets)  # [B, F, C, L]
        b, f = ep.shape[0], ep.shape[1]
        flat = ep.reshape(b * f, ep.shape[2], ep.shape[3])
        flash_scores = self._decode(flat, batch=max_flash_batch).reshape(b, f)
        logits = symbol_logits(flash_scores, groups)
        return {"flash_scores": flash_scores, "logits": logits}

    # -- checkpointing -----------------------------------------------------

    def checkpoint(self, training_config: dict | None = None, seed: int | None = None) -> dict:
        """Portable payload: preprocessor config + state, decoder, L, fs, seed."""
        payload = {
            "speller_format": 1,
            "preprocessor_config": self.preprocessor.to_config(),
            "preprocessor_state": self.preprocessor.state_dict(),
            "decoder_name": self.decoder_name,
            "L": self.L,
            "fs": self.preprocessor.fs,
            "fs_out": self.preprocessor.fs_out,
            "training_config": dict(training_config or {}),
            "seed": seed,
        }
        if isinstance(self.decoder, XdawnLDA):
            payload["decoder_kind"] = "xdawn_lda"
            payload["decoder_state"] = {
                "n_filters": self.decoder.n_filters,
                "reg": self.decoder.reg,
                "filters_": self.decoder.filters_,
                "clf_": self.decoder.clf_,
            }
        else:
            payload["decoder_kind"] = "torch"
            payload["decoder_state"] = self.decoder.state_dict()
        return payload

    def save(self, path, training_config: dict | None = None, seed: int | None = None) -> Path:
        """Save via torch (.pt) or joblib (.joblib, for the classical decoder)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint(training_config=training_config, seed=seed)
        if path.suffix == ".joblib":
            import joblib

            joblib.dump(payload, path)
        else:
            torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path) -> "Speller":
        """Rebuild a Speller from a checkpoint written by `save`."""
        path = Path(path)
        if path.suffix == ".joblib":
            import joblib

            payload = joblib.load(path)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_checkpoint(payload)

    @classmethod
    def from_checkpoint(cls, payload: dict) -> "Speller":
        from p3a.preproc.pipeline import Preprocessor

        if payload.get("speller_format") != 1:
            raise ValueError("unsupported speller checkpoint format")
        pre = Preprocessor.from_config(payload["preprocessor_config"])
        pre.load_state_dict(payload["preprocessor_state"])

        name = payload["decoder_name"]
        if payload.get("decoder_kind") == "xdawn_lda" or name in XDAWN_NAMES:
            decoder = XdawnLDA(
                n_filters=payload["decoder_state"]["n_filters"],
                reg=payload["decoder_state"]["reg"],
            )
            decoder.filters_ = payload["decoder_state"]["filters_"]
            decoder.clf_ = payload["decoder_state"]["clf_"]
        else:
            if name not in MODEL_NAMES:
                raise ValueError(f"unknown decoder {name!r} in checkpoint")
            decoder = build_model(name, pre.n_channels, payload["L"])
            decoder.load_state_dict(payload["decoder_state"])
        return cls(pre, decoder, payload["L"])
