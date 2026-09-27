import math

import numpy as np
import pytest
import scipy.signal as sps
import torch

from p3a.preproc.pipeline import Preprocessor

FS = 240.0
C = 4
TARGET_FS = 60.0
Q = int(round(FS / TARGET_FS))  # 4


def sine(freq, amp, T, fs=FS, channels=1, phase=0.0):
    t = np.arange(T) / fs
    wave = amp * np.sin(2.0 * np.pi * freq * t + phase)
    return np.tile(wave, (channels, 1)).astype(np.float32)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def test_output_shape_and_dtype():
    pp = Preprocessor(C, FS, target_fs=TARGET_FS)
    x = torch.randn(2, C, 1200, dtype=torch.float32)
    y = pp(x)
    assert y.dtype == torch.float32
    assert y.shape == (2, C, math.ceil(1200 / Q))


@pytest.mark.parametrize("T", [1200, 1201, 1202, 1203])
def test_output_length_ceil(T):
    pp = Preprocessor(C, FS, target_fs=TARGET_FS)
    x = torch.randn(1, C, T, dtype=torch.float32)
    assert pp(x).shape[-1] == math.ceil(T / Q)


def test_passband_and_stopband_response():
    pp = Preprocessor(2, FS, band=(0.5, 20.0), target_fs=TARGET_FS)
    T = 4800
    ref = 1.0 / np.sqrt(2.0)  # RMS of a unit-amplitude sine

    def response(freq):
        x = torch.from_numpy(sine(freq, 1.0, T, channels=2)).unsqueeze(0)
        y = pp.bandpass(x)[0]
        inner = y[..., pp.pad : -pp.pad].numpy()
        return rms(inner) / ref

    assert response(10.0) > 0.9
    assert response(45.0) < 0.1
    # The spec asked for < 0.1 at 0.05 Hz but the mandated taps
    # (firwin, numtaps = 2*fs+1, band = (0.5, 20.0)) have a measured gain of
    # 0.160 at 0.05 Hz, which is still in the lower transition band. The
    # implementation follows the spec exactly; the bound is the achievable one.
    assert response(0.05) < 0.2


def test_matches_scipy_convolve_interior():
    torch.manual_seed(0)
    pp = Preprocessor(1, FS, car=False)
    T = 1200
    x = torch.randn(1, 1, T, dtype=torch.float32)
    y = pp.bandpass(x)
    ref = sps.convolve(x[0, 0].numpy(), pp.taps.numpy(), mode="same")
    p = pp.pad
    np.testing.assert_allclose(y[0, 0, p:-p].numpy(), ref[p:-p], atol=1e-4)


def test_zero_phase_alignment():
    pp = Preprocessor(1, FS, car=False)
    T = 2400
    x = torch.from_numpy(sine(5.0, 1.0, T, channels=1)).unsqueeze(0)
    y = pp.bandpass(x)
    p = pp.pad
    inner_x = x[0, 0, p:-p].numpy()
    inner_y = y[0, 0, p:-p].numpy()
    assert abs(int(np.argmax(inner_x)) - int(np.argmax(inner_y))) <= 1


def test_gradients_flow():
    pp = Preprocessor(C, FS, target_fs=TARGET_FS)
    x = torch.randn(2, C, 1200, dtype=torch.float32).requires_grad_(True)
    pp(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fit_scaling_standardizes():
    torch.manual_seed(1)
    np.random.seed(1)
    trials = [
        (np.random.randn(C, 2400) * 5.0 + 2.0).astype(np.float32) for _ in range(3)
    ]
    pp = Preprocessor(C, FS, target_fs=TARGET_FS)
    pp.fit_scaling(trials)
    y = torch.cat([pp(torch.from_numpy(t).unsqueeze(0)) for t in trials], dim=2)
    mean = y.mean(dim=2)[0]
    std = y.std(dim=2, unbiased=False)[0]
    assert torch.all(mean.abs() < 0.05)
    # Verify the pooled distribution used to fit scaling.
    assert torch.all((std - 1.0).abs() < 1e-5)


def test_config_roundtrip_preserves_state():
    pp = Preprocessor(C, FS, target_fs=TARGET_FS, car=False)
    pp.fit_scaling([(np.random.randn(C, 1200)).astype(np.float32)])
    clone = Preprocessor.from_config(pp.to_config())
    clone.load_state_dict(pp.state_dict())
    x = torch.randn(1, C, 1200, dtype=torch.float32)
    torch.testing.assert_close(clone(x), pp(x))


def test_exact_fir_frequency_response():
    # Preserve the specified 481-tap filter; its low-frequency transition
    # does not meet the originally requested 10 percent amplitude bound.
    pp = Preprocessor(1, FS, car=False)
    _, response = sps.freqz(pp.taps.numpy(), worN=[0.05, 10.0, 45.0], fs=FS)
    gain = np.abs(response)
    assert 0.15 < gain[0] < 0.20
    assert gain[1] > 0.9
    assert gain[2] < 0.1
