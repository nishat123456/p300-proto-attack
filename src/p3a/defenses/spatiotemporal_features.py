"""Fixed channel/time-bin ablation; conventional representation, not novelty."""
import numpy as np
from p3a.defenses.authenticity import authenticity_features, FeatureSet


def rich_features(erp, windows, fs=60.0, bins=12):
    erp = np.asarray(erp, dtype=np.float32)
    if erp.ndim != 3 or erp.shape[-1] < bins or bins < 1:
        raise ValueError("ERP must have at least one sample per time bin")
    if not np.isfinite(erp).all():
        raise ValueError("nonfinite ERP")
    temporal = np.stack([part.mean(axis=-1) for part in
                         np.array_split(erp, bins, axis=-1)], axis=-1)
    # Channel-major, chronological bins; no fitting, labels or selected electrodes.
    return np.concatenate([authenticity_features(erp, windows, fs),
                           temporal.reshape(len(erp), -1)], axis=1).astype(np.float32)


def extract_rich_sets(chunk, fs=60.0):
    success = np.asarray(chunk.success, dtype=bool)
    indices = np.asarray(chunk.trial_index)
    ids = np.asarray(chunk.trial_id)
    induced = FeatureSet(rich_features(chunk.attacked_erp[success],
                        chunk.windows[success], fs), indices[success], ids[success])
    _, keep = np.unique(ids, return_index=True)
    keep = np.sort(keep)
    genuine = FeatureSet(rich_features(chunk.genuine_erp[keep],
                        chunk.clean_windows[keep], fs), indices[keep], ids[keep])
    return induced, genuine
