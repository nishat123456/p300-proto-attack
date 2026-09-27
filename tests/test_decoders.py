"""CPU tests for the flash-level P300 decoders and the model registry."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from p3a.models.deepconvnet import DeepConvNet
from p3a.models.eegnet import EEGNet
from p3a.models.registry import MODEL_NAMES, build_model
from p3a.models.shallowconvnet import ShallowConvNet
from p3a.models.xdawn_lda import XdawnLDA

TORCH_MODELS = [EEGNet, DeepConvNet, ShallowConvNet]


@pytest.mark.parametrize("model_cls", TORCH_MODELS)
@pytest.mark.parametrize("n_channels,L", [(64, 48), (8, 51), (16, 51)])
def test_torch_models_forward_backward(model_cls, n_channels, L):
    torch.manual_seed(0)
    model = model_cls(n_channels, L)
    x = torch.randn(7, n_channels, L, requires_grad=True)
    logits = model(x)
    assert logits.shape == (7,)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("model_cls", TORCH_MODELS)
def test_torch_models_eval_deterministic(model_cls):
    torch.manual_seed(0)
    model = model_cls(64, 48).eval()
    x = torch.randn(4, 64, 48)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    torch.testing.assert_close(a, b)
    # Dropout must be active in train mode and change at least one output.
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        c = model(x)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("name,expected", [
    ("eegnet", EEGNet),
    ("deepconvnet", DeepConvNet),
    ("shallowconvnet", ShallowConvNet),
])
def test_registry_builds_and_runs(name, expected):
    model = build_model(name, 64, 48)
    assert isinstance(model, expected)
    logits = model(torch.randn(3, 64, 48))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer", 64, 48)


def _synthetic_erp_dataset(n=120, c=8, length=48, erp_scale=1.0, seed=0):
    """Target epochs carry a fixed ERP template plus noise; non-targets are noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 60.0
    template = np.exp(-((t - 0.3) ** 2) / (2 * 0.05 ** 2))
    template = np.sin(2 * np.pi * 5 * t) * template
    unit = (template - template.mean()) / template.std()
    base = rng.normal(size=(n, c, length))
    y = np.zeros(n, dtype=bool)
    y[rng.permutation(n)[: n // 2]] = True
    channels = np.linspace(-1.0, 1.0, c)
    base[y] += erp_scale * channels[None, :, None] * unit[None, None, :]
    return base, y


def test_xdawn_lda_separates_synthetic_erp():
    x, y = _synthetic_erp_dataset(seed=0)
    split = 80
    clf = XdawnLDA(n_filters=4)
    clf.fit(x[:split], y[:split])
    scores = clf.decision_function(x[split:])
    assert scores.shape == (len(x) - split,)
    auc = roc_auc_score(y[split:], scores)
    assert auc > 0.9, auc


def test_xdawn_lda_requires_target_class():
    x, _ = _synthetic_erp_dataset(n=10)
    with pytest.raises(ValueError, match="target"):
        XdawnLDA().fit(x, np.zeros(len(x), dtype=bool))


def test_xdawn_lda_save_load_roundtrip(tmp_path):
    x, y = _synthetic_erp_dataset(seed=1)
    clf = XdawnLDA().fit(x[:60], y[:60])
    before = clf.decision_function(x[60:])
    path = tmp_path / "xdawn.joblib"
    clf.save(path)
    loaded = XdawnLDA.load(path)
    np.testing.assert_allclose(loaded.decision_function(x[60:]), before)
    assert loaded.filters_.shape[-1] == x.shape[1]
