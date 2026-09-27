"""Registry of torch flash decoders, keyed by name."""
from __future__ import annotations

from p3a.models.deepconvnet import DeepConvNet
from p3a.models.eegnet import EEGNet
from p3a.models.shallowconvnet import ShallowConvNet

MODEL_NAMES = ("eegnet", "deepconvnet", "shallowconvnet")


def build_model(name: str, n_channels: int, L: int):
    """Build a torch flash decoder by name for the given channel count and epoch length.

    The classical XdawnLDA decoder is fitted rather than built, so it is not
    part of this registry; scripts handle it separately.
    """
    key = str(name).lower()
    if key == "eegnet":
        return EEGNet(n_channels, L)
    if key == "deepconvnet":
        return DeepConvNet(n_channels, L)
    if key == "shallowconvnet":
        return ShallowConvNet(n_channels, L)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
