"""Fast CPU tests for the GPN-style perturbation generators.

The generators are exercised with tiny synthetic schedules so the suite stays
in seconds while still checking the properties the experiment relies on: output
shapes, exact per-sample L-infinity budget, gradient flow to every generator
parameter, and the causality of `CausalGPN` (no dependence on source EEG).
"""
import numpy as np
import pytest
import torch

from p3a.attacks.generators import (
    CausalGPN,
    Conditioner,
    SignalGPN,
    linf_rescale,
    target_flash_mask,
)


def _schedule(batch=2, n_channels=64, length=7794, n_flashes=12, n_symbols=36,
              seed=0, device="cpu"):
    """Synthetic batch with a full row/column sweep so every symbol appears."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eeg = torch.randn(batch, n_channels, length, generator=g).to(device)
    step = length // (n_flashes + 1)
    onsets = torch.stack([
        torch.arange(n_flashes, device=device) * step + 5 + b * 3
        for b in range(batch)
    ]).to(torch.long)
    codes = list(range(1, 13))
    from p3a.data.layout import COMP3, rc_code_to_group

    groups = torch.as_tensor(
        np.stack([np.stack([rc_code_to_group(COMP3, c) for c in codes])
                  for _ in range(batch)]), dtype=torch.bool, device=device,
    )
    assert groups.shape == (batch, n_flashes, n_symbols)
    target = torch.tensor([3, 17][:batch], dtype=torch.long, device=device)
    proto = torch.randn(batch, n_channels, 48, generator=g).to(device)
    return eeg, onsets, groups, target, proto


def test_target_flash_mask_selects_target_flashes():
    _, _, groups, target, _ = _schedule(batch=2)
    mask = target_flash_mask(groups, target)
    assert mask.shape == (2, groups.shape[1]) and mask.dtype == torch.bool
    for b in range(2):
        expected = groups[b, :, target[b]]
        assert torch.equal(mask[b], expected)


def test_signal_gpn_shapes_and_linf_budget():
    eeg, onsets, groups, target, proto = _schedule()
    eps = torch.tensor([0.05, 0.20])
    gen = SignalGPN(n_symbols=36, n_channels=64).eval()
    delta = gen(eeg, onsets, groups, target, eps, proto)
    assert delta.shape == eeg.shape and delta.dtype == torch.float32
    peak = delta.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, eps, atol=1e-5)


def test_causal_gpn_shapes_and_linf_budget():
    eeg, onsets, groups, target, proto = _schedule()
    eps = 0.30
    gen = CausalGPN(n_symbols=36, n_channels=64)
    delta = gen(eeg, onsets, groups, target, eps, proto)
    assert delta.shape == eeg.shape and delta.dtype == torch.float32
    peak = delta.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, torch.full((2,), eps), atol=1e-5)


def test_linf_rescale_is_exact_and_keeps_zero():
    delta = torch.randn(3, 4, 50)
    out = linf_rescale(delta, torch.tensor([0.1, 0.2, 0.3]))
    peak = out.flatten(1).abs().amax(dim=1)
    assert torch.allclose(peak, torch.tensor([0.1, 0.2, 0.3], dtype=out.dtype), atol=1e-6)
    zeros = linf_rescale(torch.zeros(2, 4, 50), 0.5)
    assert torch.count_nonzero(zeros) == 0
    with pytest.raises(ValueError, match="non-negative"):
        linf_rescale(delta, -0.1)


def test_gradients_reach_all_signal_generator_params():
    eeg, onsets, groups, target, proto = _schedule(batch=2, n_channels=8, length=800,
                                                    n_flashes=12)
    gen = SignalGPN(n_symbols=36, n_channels=8, base=8, mid=16)
    delta = gen(eeg, onsets, groups, target, 0.2, proto)
    delta.sum().backward()
    missing = [name for name, p in gen.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert missing == []


def test_gradients_reach_all_causal_generator_params():
    eeg, onsets, groups, target, proto = _schedule(batch=2, n_channels=8, length=800)
    gen = CausalGPN(n_symbols=36, n_channels=8, hidden=8)
    delta = gen(eeg, onsets, groups, target, 0.2, proto)
    delta.sum().backward()
    missing = [name for name, p in gen.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert missing == []


def test_causal_output_is_independent_of_source_eeg():
    eeg, onsets, groups, target, proto = _schedule(batch=2)
    other = torch.randn_like(eeg) * 100.0
    gen = CausalGPN(n_symbols=36, n_channels=64).eval()
    with torch.no_grad():
        a = gen(eeg, onsets, groups, target, 0.25, proto)
        b = gen(other, onsets, groups, target, 0.25, proto)
    torch.testing.assert_close(a, b)


def test_signal_output_depends_on_source_eeg():
    eeg, onsets, groups, target, proto = _schedule(batch=2)
    other = torch.randn_like(eeg)
    gen = SignalGPN(n_symbols=36, n_channels=64).eval()
    with torch.no_grad():
        a = gen(eeg, onsets, groups, target, 0.25, proto)
        b = gen(other, onsets, groups, target, 0.25, proto)
    assert not torch.allclose(a, b)


def test_use_prototype_flag_changes_conditioning():
    _, onsets, groups, target, proto = _schedule(batch=2, n_channels=8, length=800)
    cond = Conditioner(n_symbols=36, n_channels=8).eval()
    onehot = torch.nn.functional.one_hot(target, 36).float()
    with torch.no_grad():
        with_proto = cond(onehot, proto, 0.2, use_prototype=True)
        without = cond(onehot, None, 0.2, use_prototype=False)
    assert with_proto.shape == (2, 128)
    assert not torch.allclose(with_proto, without)


def test_prototype_presence_changes_generator_output():
    eeg, onsets, groups, target, proto = _schedule(batch=2, n_channels=8, length=800)
    gen = SignalGPN(n_symbols=36, n_channels=8, base=8, mid=16).eval()
    with torch.no_grad():
        a = gen(eeg, onsets, groups, target, 0.2, proto, use_prototype=True)
        b = gen(eeg, onsets, groups, target, 0.2, proto, use_prototype=False)
    assert not torch.allclose(a, b)


def test_input_validation():
    eeg, onsets, groups, target, proto = _schedule(batch=2, n_channels=8, length=800)
    gen = SignalGPN(n_symbols=36, n_channels=8, base=8, mid=16)
    bad_target = torch.tensor([0, 36], dtype=torch.long)
    try:
        gen(eeg, onsets, groups, bad_target, 0.2, proto)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range target must raise")
    with pytest.raises(ValueError, match="channel count"):
        gen(eeg[:, :-1], onsets, groups, target, 0.2, proto)
    bad_onsets = onsets.clone()
    bad_onsets[0, 0] = eeg.shape[-1]
    with pytest.raises(ValueError, match="inside the raw trial"):
        gen(eeg, bad_onsets, groups, target, 0.2, proto)
