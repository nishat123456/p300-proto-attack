"""ERP plausibility metrics for attacked trials (Prompt 11).

Each attacked decision ERP is compared against the mean genuine ERP of the same
target symbol, built from the clean trials of the subject, on four axes:

- waveform correlation at Pz and Cz,
- P300 peak latency (ms) and amplitude at Pz, searched in 250-500 ms,
- topography correlation over all channels at the genuine peak latency,
- PSD distance (Welch, 1-20 Hz) between the attacked and genuine ERP.

The Pz and Cz indices come from a ``ChannelMap``. Comp III ships no channel
labels, so when names are unavailable the map autodetects a proxy and the output
records ``channel_map_source`` next to the numbers; a reader must never take an
autodetected Pz for a measured one.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from p3a.defenses.metrics import ChannelMap

__all__ = [
    "P300_WINDOW",
    "PSD_BAND",
    "plausibility_metrics",
    "target_genuine_templates",
]

# P300 search window in seconds, fixed by PREREG/PLAN sections on plausibility.
P300_WINDOW = (0.25, 0.5)
PSD_BAND = (1.0, 20.0)


def target_genuine_templates(clean_true_erp: np.ndarray,
                             true_idx: np.ndarray) -> dict[int, np.ndarray]:
    """Mean genuine true-symbol ERP per symbol, keyed by symbol index."""
    flat_erp = np.asarray(clean_true_erp, dtype=np.float64)
    flat_idx = np.asarray(true_idx, dtype=np.int64).reshape(-1)
    if flat_erp.ndim != 3 or flat_erp.shape[0] != flat_idx.size:
        raise ValueError("clean_true_erp must be [T, C, L] aligned with true_idx")
    templates: dict[int, np.ndarray] = {}
    for symbol in np.unique(flat_idx):
        templates[int(symbol)] = flat_erp[flat_idx == symbol].mean(axis=0)
    return templates


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _peak_latency_amplitude(erp: np.ndarray, pz: int, fs_out: float,
                            window=P300_WINDOW) -> tuple[float, float]:
    """P300 peak latency in ms and signed amplitude at the Pz channel."""
    lo = int(round(window[0] * fs_out))
    hi = min(int(round(window[1] * fs_out)), erp.shape[1])
    hi = max(hi, lo + 1)
    segment = erp[pz, lo:hi]
    idx = int(np.argmax(segment))
    latency_ms = 1000.0 * (lo + idx) / float(fs_out)
    return latency_ms, float(segment[idx])


def _psd_distance(a: np.ndarray, b: np.ndarray, fs_out: float,
                  band=PSD_BAND) -> float:
    """RMS difference of Welch PSDs (dB) between two ERPs over ``band`` Hz."""
    nperseg = min(a.shape[-1], max(int(fs_out), 8))
    freqs, pa = signal.welch(a, fs=float(fs_out), nperseg=nperseg, axis=-1)
    _, pb = signal.welch(b, fs=float(fs_out), nperseg=nperseg, axis=-1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return float("nan")
    eps = 1e-12
    pa_db = 10.0 * np.log10(np.maximum(pa[..., mask], eps))
    pb_db = 10.0 * np.log10(np.maximum(pb[..., mask], eps))
    return float(np.sqrt(np.mean((pa_db - pb_db) ** 2)))


def plausibility_metrics(attacked_erp: np.ndarray, true_idx, target_idx: np.ndarray,
                         templates: dict[int, np.ndarray], fs_out: float,
                         channel_map: ChannelMap) -> list[dict]:
    """One plausibility record per attacked ERP sample.

    ``attacked_erp`` is [N, C, L], ``true_idx`` is [N] the trial's true symbol
    (used only to pick the matching recovered template when the target symbol is
    absent), and ``target_idx`` is [N] the attacker target whose genuine
    template is the reference. Genuine templates come from clean trials, so an
    attacked sample is always compared against an independent genuine reference.
    """
    attacked = np.asarray(attacked_erp, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=np.int64).reshape(-1)
    target_idx = np.asarray(target_idx, dtype=np.int64).reshape(-1)
    if attacked.shape[0] != target_idx.size or true_idx.size != target_idx.size:
        raise ValueError("attacked_erp, true_idx and target_idx must be aligned")

    records = []
    for i in range(attacked.shape[0]):
        target_symbol = int(target_idx[i])
        reference_idx = target_symbol
        if reference_idx not in templates:
            reference_idx = int(true_idx[i])
        reference = templates.get(reference_idx)
        if reference is None:
            records.append({"waveform_corr_pz": float("nan"),
                            "waveform_corr_cz": float("nan"),
                            "p300_latency_ms": float("nan"),
                            "p300_amplitude": float("nan"),
                            "topography_corr": float("nan"),
                            "psd_distance_db": float("nan"),
                            "reference_symbol_idx": target_symbol,
                            "channel_map_source": channel_map.source})
            continue
        erp = attacked[i]
        gen_latency, _ = _peak_latency_amplitude(reference, channel_map.pz_index, fs_out)
        gen_peak = int(round(gen_latency / 1000.0 * fs_out))
        gen_peak = min(max(gen_peak, 0), reference.shape[1] - 1)
        lat, amp = _peak_latency_amplitude(erp, channel_map.pz_index, fs_out)
        records.append({
            "waveform_corr_pz": _corr(erp[channel_map.pz_index], reference[channel_map.pz_index]),
            "waveform_corr_cz": _corr(erp[channel_map.cz_index], reference[channel_map.cz_index]),
            "p300_latency_ms": lat,
            "p300_amplitude": amp,
            "topography_corr": _corr(erp[:, gen_peak], reference[:, gen_peak]),
            "psd_distance_db": _psd_distance(erp, reference, fs_out),
            "reference_symbol_idx": reference_idx,
            "channel_map_source": channel_map.source,
        })
    return records
