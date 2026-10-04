"""Interpretable P300 authenticity detector for the post-hoc rescue study.

The detector deliberately uses a small feature vector instead of flattened ERP
samples.  Features summarize temporal shape, scalp dispersion, spectrum, and
consistency across the stored windows.  Evaluation is nested by trial: one
outer fold is evaluated, one different fold supplies only genuine calibration
scores, and the remaining folds fit a regularized logistic regression.  The
reported operating-point TPR therefore uses a threshold chosen without seeing
the evaluation fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from p3a.defenses.artifacts import AttackInputs

DEFAULT_FPRS = (0.01, 0.05, 0.10)
DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    trial_index: np.ndarray
    trial_id: np.ndarray


def _moments(x: np.ndarray, axes: tuple[int, ...]) -> list[np.ndarray]:
    return [
        np.mean(x, axis=axes),
        np.std(x, axis=axes),
        np.sqrt(np.mean(np.square(x), axis=axes)),
        np.max(np.abs(x), axis=axes),
        np.ptp(x, axis=axes),
    ]


def _relative_band_power(x: np.ndarray, fs: float, reduce_axes: tuple[int, ...],
                         bands=((0.0, 2.0), (2.0, 5.0), (5.0, 10.0),
                                (10.0, 20.0), (20.0, 30.0))) -> list[np.ndarray]:
    power = np.abs(np.fft.rfft(x.astype(np.float32), axis=-1)) ** 2
    freq = np.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs))
    total = np.mean(power[..., (freq >= 0.0) & (freq <= min(30.0, fs / 2.0))],
                    axis=-1)
    total = np.mean(total, axis=reduce_axes) + 1e-12
    out = []
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < min(hi, fs / 2.0 + 1e-9))
        if not mask.any():
            out.append(np.zeros(x.shape[0], dtype=np.float64))
            continue
        band = np.mean(power[..., mask], axis=-1)
        out.append(np.mean(band, axis=reduce_axes) / total)
    return out


def authenticity_features(erp: np.ndarray, windows: np.ndarray,
                          fs: float = 60.0) -> np.ndarray:
    """Return low-dimensional physiological features for aligned samples.

    ``erp`` is ``[N,C,L]`` and ``windows`` is ``[N,R,C,W]``.  No labels or
    subject identities enter feature construction.
    """
    erp = np.asarray(erp, dtype=np.float32)
    windows = np.asarray(windows, dtype=np.float32)
    if erp.ndim != 3 or windows.ndim != 4 or len(erp) != len(windows):
        raise ValueError("erp [N,C,L] and windows [N,R,C,W] must align")
    if erp.shape[1] != windows.shape[2]:
        raise ValueError("erp and windows must have the same channel count")

    columns: list[np.ndarray] = []
    columns.extend(_moments(erp, (1, 2)))
    columns.append(np.sqrt(np.mean(np.diff(erp, axis=-1) ** 2, axis=(1, 2))))
    columns.append(np.sqrt(np.mean(np.diff(erp, n=2, axis=-1) ** 2, axis=(1, 2))))

    lo = min(max(int(round(0.25 * fs)), 0), erp.shape[-1] - 1)
    hi = min(max(int(round(0.50 * fs)), lo + 1), erp.shape[-1])
    p300 = erp[..., lo:hi]
    columns.extend(_moments(p300, (1, 2)))
    topography = np.mean(p300, axis=-1)
    columns.append(np.std(topography, axis=1))
    columns.append(np.max(topography, axis=1) - np.min(topography, axis=1))
    peak_index = np.argmax(p300, axis=-1).astype(np.float64) + lo
    columns.append(np.mean(peak_index, axis=1) / fs)
    columns.append(np.std(peak_index, axis=1) / fs)
    columns.extend(_relative_band_power(erp, fs, reduce_axes=(1,)))

    columns.extend(_moments(windows, (1, 2, 3)))
    columns.append(np.sqrt(np.mean(np.diff(windows, axis=-1) ** 2,
                                   axis=(1, 2, 3))))
    # Genuine EEG windows should retain natural repeat-to-repeat variability.
    repeat_mean = windows.mean(axis=(2, 3))
    repeat_rms = np.sqrt(np.mean(windows ** 2, axis=(2, 3)))
    columns.append(np.std(repeat_mean, axis=1))
    columns.append(np.std(repeat_rms, axis=1))
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    flat = flat - flat.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(flat ** 2, axis=-1, keepdims=True)) + 1e-12
    unit = flat / norm
    centroid = unit.mean(axis=1, keepdims=True)
    consistency = np.sum(unit * centroid, axis=-1)
    columns.append(np.mean(consistency, axis=1))
    columns.append(np.std(consistency, axis=1))
    columns.extend(_relative_band_power(windows, fs, reduce_axes=(1, 2)))

    features = np.column_stack(columns).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def _sets(chunk: AttackInputs, fs: float) -> tuple[FeatureSet, FeatureSet]:
    success = np.asarray(chunk.success, dtype=bool)
    induced = FeatureSet(
        authenticity_features(chunk.attacked_erp[success], chunk.windows[success], fs),
        np.asarray(chunk.trial_index)[success], np.asarray(chunk.trial_id)[success],
    )
    seen: set[str] = set()
    keep = []
    for i, trial_id in enumerate(chunk.trial_id):
        key = str(trial_id)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    genuine = FeatureSet(
        authenticity_features(chunk.genuine_erp[keep], chunk.clean_windows[keep], fs),
        np.asarray(chunk.trial_index)[keep], np.asarray(chunk.trial_id)[keep],
    )
    return induced, genuine


def extract_authenticity_sets(chunk: AttackInputs,
                              fs: float = 60.0) -> tuple[FeatureSet, FeatureSet]:
    """Extract induced and deduplicated genuine features for serialization."""
    return _sets(chunk, fs)


def _concat(parts: list[FeatureSet], deduplicate: bool = False) -> FeatureSet:
    values = np.concatenate([p.values for p in parts], axis=0)
    trial_index = np.concatenate([p.trial_index for p in parts])
    trial_id = np.concatenate([p.trial_id for p in parts])
    if deduplicate:
        seen, keep = set(), []
        for i, value in enumerate(trial_id):
            key = str(value)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        keep = np.asarray(keep, dtype=np.int64)
        values, trial_index, trial_id = values[keep], trial_index[keep], trial_id[keep]
    return FeatureSet(values, trial_index, trial_id)


def _threshold(scores: np.ndarray, fpr: float) -> float:
    try:
        return float(np.quantile(scores, 1.0 - fpr, method="higher"))
    except TypeError:  # NumPy before 1.22
        return float(np.quantile(scores, 1.0 - fpr, interpolation="higher"))


def cross_validate_authenticity(pool: list[AttackInputs],
                                eval_chunks: dict[str, AttackInputs],
                                fs: float = 60.0,
                                fprs=DEFAULT_FPRS,
                                n_folds: int = DEFAULT_FOLDS,
                                seed: int = 0,
                                c: float = 0.1) -> dict[str, dict]:
    """Nested trial-fold evaluation with clean-only threshold calibration."""
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    if not pool or not eval_chunks:
        raise ValueError("pool and eval_chunks must be non-empty")

    pool_pairs = [_sets(chunk, fs) for chunk in pool]
    train_induced = _concat([pair[0] for pair in pool_pairs])
    train_genuine = _concat([pair[1] for pair in pool_pairs], deduplicate=True)
    eval_sets = {label: _sets(chunk, fs) for label, chunk in eval_chunks.items()}
    records: dict[str, list[dict]] = {label: [] for label in eval_chunks}

    for outer in range(n_folds):
        calibration = (outer + 1) % n_folds
        fit_ind = (train_induced.trial_index % n_folds != outer) & \
                  (train_induced.trial_index % n_folds != calibration)
        fit_gen = (train_genuine.trial_index % n_folds != outer) & \
                  (train_genuine.trial_index % n_folds != calibration)
        cal_gen = train_genuine.trial_index % n_folds == calibration
        if not fit_ind.any() or not fit_gen.any() or not cal_gen.any():
            continue

        x = np.concatenate([train_induced.values[fit_ind],
                            train_genuine.values[fit_gen]])
        y = np.concatenate([np.ones(int(fit_ind.sum())),
                            np.zeros(int(fit_gen.sum()))])
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(x)
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                   random_state=seed).fit(scaler.transform(x), y)
        calibration_scores = model.decision_function(
            scaler.transform(train_genuine.values[cal_gen])
        )
        thresholds = {float(fpr): _threshold(calibration_scores, float(fpr))
                      for fpr in fprs}

        for label, (induced, genuine) in eval_sets.items():
            ind_test = induced.trial_index % n_folds == outer
            gen_test = genuine.trial_index % n_folds == outer
            if not ind_test.any() or not gen_test.any():
                continue
            ind_scores = model.decision_function(scaler.transform(induced.values[ind_test]))
            gen_scores = model.decision_function(scaler.transform(genuine.values[gen_test]))
            scores = np.concatenate([ind_scores, gen_scores])
            labels = np.concatenate([np.ones(len(ind_scores)), np.zeros(len(gen_scores))])
            row = {
                "fold": outer,
                "calibration_fold": calibration,
                "auc": float(roc_auc_score(labels, scores)),
                "n_induced": int(len(ind_scores)),
                "n_genuine": int(len(gen_scores)),
            }
            for fpr, threshold in thresholds.items():
                suffix = str(fpr).replace(".", "p")
                row[f"threshold_{suffix}"] = threshold
                row[f"tpr_{suffix}"] = float(np.mean(ind_scores >= threshold))
                row[f"far_{suffix}"] = float(np.mean(gen_scores >= threshold))
            records[label].append(row)

    summary = {}
    for label, rows in records.items():
        item = {"folds": rows, "auc": float("nan"), "n_folds": len(rows)}
        if rows:
            item["auc"] = float(np.mean([row["auc"] for row in rows]))
            for fpr in fprs:
                suffix = str(float(fpr)).replace(".", "p")
                item[f"tpr_{suffix}"] = float(np.mean([r[f"tpr_{suffix}"] for r in rows]))
                item[f"far_{suffix}"] = float(np.mean([r[f"far_{suffix}"] for r in rows]))
        summary[label] = item
    return summary


def cross_subject_authenticity(source_pool: list[AttackInputs],
                               target_eval_chunks: dict[str, AttackInputs],
                               fs: float = 60.0,
                               fprs=DEFAULT_FPRS,
                               n_folds: int = DEFAULT_FOLDS,
                               seed: int = 0,
                               c: float = 0.1) -> dict[str, dict]:
    """Fit on one subject and evaluate another with clean-only calibration.

    The classifier sees induced and genuine samples from the source subject
    only. Source features are normalized using source genuine samples. For each
    target outer fold, a different target fold supplies genuine samples to fit
    the target normalization and operating thresholds. No target attacks or
    target evaluation samples enter fitting or calibration.
    """
    return cross_subject_feature_sets(
        [_sets(chunk, fs) for chunk in source_pool],
        {label: _sets(chunk, fs) for label, chunk in target_eval_chunks.items()},
        fprs=fprs, n_folds=n_folds, seed=seed, c=c,
    )


def cross_subject_feature_sets(source_pairs, target_sets, fprs=DEFAULT_FPRS,
                               n_folds=DEFAULT_FOLDS, seed=0, c=0.1,
                               model_kind="logistic", independent_normalization=False):
    """Evaluate exported features with the same clean-only fitting protocol."""
    if n_folds < 2 or not source_pairs or not target_sets:
        raise ValueError("need source and target sets and at least two folds")
    source_induced = _concat([pair[0] for pair in source_pairs])
    source_genuine = _concat([pair[1] for pair in source_pairs], deduplicate=True)
    if not len(source_induced.values) or not len(source_genuine.values):
        raise ValueError("source subject needs induced and genuine samples")

    # Domain normalization is defined by clean physiology only. This prevents
    # target attacks from influencing either the feature transform or model.
    source_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
        source_genuine.values
    )
    source_x = np.concatenate([
        source_scaler.transform(source_induced.values),
        source_scaler.transform(source_genuine.values),
    ])
    source_y = np.concatenate([
        np.ones(len(source_induced.values)),
        np.zeros(len(source_genuine.values)),
    ])
    from p3a.defenses.feature_baselines import fit_detector
    model = fit_detector(model_kind, source_x, source_y, seed, c)

    target_genuine = _concat(
        [pair[1] for pair in target_sets.values()], deduplicate=True
    )
    records: dict[str, list[dict]] = {label: [] for label in target_sets}

    for outer in range(n_folds):
        calibration = (outer + 1) % n_folds
        cal_gen = target_genuine.trial_index % n_folds == calibration
        if not cal_gen.any():
            continue
        norm_gen = cal_gen
        if independent_normalization:
            if n_folds < 3:
                raise ValueError("independent normalization needs at least three folds")
            norm_gen = ((target_genuine.trial_index % n_folds != outer) &
                        (target_genuine.trial_index % n_folds != calibration))
        target_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
            target_genuine.values[norm_gen]
        )
        calibration_scores = model.decision_function(
            target_scaler.transform(target_genuine.values[cal_gen])
        )
        thresholds = {
            float(fpr): _threshold(calibration_scores, float(fpr)) for fpr in fprs
        }

        for label, (induced, genuine) in target_sets.items():
            ind_test = induced.trial_index % n_folds == outer
            gen_test = genuine.trial_index % n_folds == outer
            if not ind_test.any() or not gen_test.any():
                continue
            ind_scores = model.decision_function(
                target_scaler.transform(induced.values[ind_test])
            )
            gen_scores = model.decision_function(
                target_scaler.transform(genuine.values[gen_test])
            )
            scores = np.concatenate([ind_scores, gen_scores])
            labels = np.concatenate([
                np.ones(len(ind_scores)), np.zeros(len(gen_scores)),
            ])
            row = {
                "fold": outer,
                "calibration_fold": calibration,
                "auc": float(roc_auc_score(labels, scores)),
                "n_induced": int(len(ind_scores)),
                "n_genuine": int(len(gen_scores)),
                "n_calibration_genuine": int(cal_gen.sum()),
                "n_normalization_genuine": int(norm_gen.sum()),
            }
            for fpr, threshold in thresholds.items():
                suffix = str(fpr).replace(".", "p")
                row[f"threshold_{suffix}"] = threshold
                row[f"tpr_{suffix}"] = float(np.mean(ind_scores >= threshold))
                row[f"far_{suffix}"] = float(np.mean(gen_scores >= threshold))
            records[label].append(row)

    summary = {}
    for label, rows in records.items():
        item = {"folds": rows, "auc": float("nan"), "n_folds": len(rows)}
        if rows:
            item["auc"] = float(np.mean([row["auc"] for row in rows]))
            for fpr in fprs:
                suffix = str(float(fpr)).replace(".", "p")
                item[f"tpr_{suffix}"] = float(
                    np.mean([row[f"tpr_{suffix}"] for row in rows])
                )
                item[f"far_{suffix}"] = float(
                    np.mean([row[f"far_{suffix}"] for row in rows])
                )
        summary[label] = item
    return summary
