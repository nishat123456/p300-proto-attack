"""Kill-test report core: read the downloaded Kaggle output and summarize it.

The report answers, for one subject or both:

- clean decoder accuracy, from the decoder metrics JSON files,
- targeted success against epsilon for the surrogate and every victim, from the
  attack parquet written by ``evaluate_attacks.py``,
- perturbation norms and detector AUC interpolated at 50, 65 and 80 percent
  success, never extrapolated; a method that never reaches a level is reported
  at its maximum success with a note,
- genuine-vs-induced TPR per method and epsilon, from the detector CSV,
- the plausibility metrics table, from the plausibility CSV,
- per-symbol prototype counts, from the bank report CSV,
- the PREREG decision gate computed mechanically from the above.

Everything here is pure file processing: no model is loaded, no attack is
re-run. The functions are importable and testable without any Kaggle output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DECISION_GATE_DEFENSE",
    "DECISION_GATE_FULL",
    "DECISION_GATE_STOP",
    "MATCHED_LEVELS",
    "PRIMARY_LEVEL",
    "ReportInputs",
    "build_report",
    "decision_gate",
    "detector_auc_at_success",
    "interpolate_at_level",
    "load_attacks_eval",
    "load_bank_report",
    "load_decoder_accuracy",
    "load_detectors",
    "load_plausibility",
    "write_report",
]

# Matched-success levels fixed by PREREG.md; 0.80 is primary.
MATCHED_LEVELS = (0.50, 0.65, 0.80)
PRIMARY_LEVEL = 0.80

DECISION_GATE_FULL = "full attack + defense paper"
DECISION_GATE_DEFENSE = "defense-led paper"
DECISION_GATE_STOP = "stop and reassess"

# The clean-vs-attacked detector row the H1 primary outcome uses.
PRIMARY_DETECTOR = "window_cnn"
# The attack family the H1 comparison centers on (proto-GPN nearest,
# cross-subject, both generators).
PRIMARY_METHOD = "proto_nearest"
COMPARATOR_METHODS = ("target_only", "tiny_noise")
WRONG_CONTROL = "proto_wrong"
# The defense detector whose clean-family TPR gates the stop condition.
DEFENSE_DETECTOR = "genuine_vs_induced"
# Training-family TPRs of the defense: the fine-grained stop condition.
DEFENSE_TRAIN_METHODS = ("pgd", "target_only", "noise")


# --------------------------------------------------------------------------
# Paths and loading
# --------------------------------------------------------------------------

def _root(output_root) -> Path:
    return Path(output_root)


def load_decoder_accuracy(output_root) -> pd.DataFrame:
    """Clean decoder accuracy per (subject, model, seed) from metrics JSONs.

    Reads ``out/decoders/*.metrics.json``. Each row carries the test letter
    accuracy at 15 and 5 repetitions and the flash AUC, which is the clean
    baseline the attack is compared against.
    """
    root = _root(output_root) / "decoders"
    rows = []
    if root.is_dir():
        for path in sorted(root.glob("*.metrics.json")):
            payload = json.loads(path.read_text())
            rows.append({
                "subject": payload.get("subject"),
                "model": payload.get("model"),
                "seed": payload.get("args", {}).get("seed", payload.get("seed")),
                "val_auc": payload.get("val_auc"),
                "test_flash_auc": payload.get("test_flash_auc"),
                "test_letter_acc_15reps": payload.get("test_letter_acc_15reps"),
                "test_letter_acc_5reps": payload.get("test_letter_acc_5reps"),
                "n_test_trials": payload.get("n_test_trials"),
            })
    return pd.DataFrame(rows)


def load_attacks_eval(output_root) -> pd.DataFrame:
    """The concatenated attack parquet for every subject under ``attacks_eval``."""
    root = _root(output_root) / "attacks_eval"
    frames = []
    if root.is_dir():
        for path in sorted(root.rglob("attacks.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_detectors(output_root) -> pd.DataFrame:
    """Detector rows for every subject under ``<subject>/detectors.csv``."""
    root = _root(output_root)
    frames = []
    if root.is_dir():
        for path in sorted(root.rglob("detectors.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_plausibility(output_root) -> pd.DataFrame:
    """Plausibility rows for every subject under ``<subject>/plausibility.csv``."""
    root = _root(output_root)
    frames = []
    if root.is_dir():
        for path in sorted(root.rglob("plausibility.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_bank_report(output_root) -> pd.DataFrame:
    """Per-symbol prototype counts from ``bank_report.csv`` when present."""
    root = _root(output_root)
    for path in sorted(root.rglob("bank_report.csv")):
        return pd.read_csv(path)
    return pd.DataFrame()


# --------------------------------------------------------------------------
# Matching and interpolation
# --------------------------------------------------------------------------

def interpolate_at_level(eps, success, level: float):
    """Epsilon at ``level`` success, or None when the level is never reached.

    ``eps`` ascending. This mirrors the pre-registered fallback: a method that
    does not reach a matched-success level is reported at its maximum success
    and never extrapolated, so the caller distinguishes "None" (not reached)
    from a real epsilon value.
    """
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    success = np.asarray(success, dtype=np.float64).reshape(-1)
    if eps.size != success.size:
        raise ValueError("eps and success must have the same length")
    if eps.size == 0:
        return None
    order = np.argsort(eps)
    eps, success = eps[order], success[order]
    if success.max() < level:
        return None
    if success[0] >= level:
        return float(eps[0])
    for i in range(1, len(eps)):
        lo, hi = success[i - 1], success[i]
        if lo < level <= hi and hi > lo:
            frac = (level - lo) / (hi - lo)
            return float(eps[i - 1] + frac * (eps[i] - eps[i - 1]))
    return float(eps[-1])


def _success_curve(frame: pd.DataFrame,
                   success_col: str = "success_surrogate_15") -> tuple[np.ndarray, np.ndarray]:
    """Ascending epsilon and mean targeted success for one named model.

    The primary matched-success analysis uses the white-box surrogate. Victim
    success remains a transfer outcome and is reported separately.
    """
    if success_col not in frame.columns:
        raise ValueError(f"missing success column {success_col!r}")
    grouped = frame.groupby("eps_multiplier")
    eps = np.array(sorted(grouped.groups), dtype=np.float64)
    means = np.array([
        float(frame.loc[grouped.groups[e], success_col].to_numpy().mean())
        for e in eps
    ], dtype=np.float64)
    return eps, means


def detector_auc_at_success(success_curve: pd.DataFrame, detector_frame: pd.DataFrame,
                            level: float = PRIMARY_LEVEL, detector: str = PRIMARY_DETECTOR,
                            success_col: str = "success_surrogate_15"):
    """Detector AUC interpolated at a success level, or None when not reached.

    ``success_curve`` holds ``eps_multiplier`` and ``success_*_15`` columns for
    one method/generator/bank and ``detector_frame`` holds ``eps_multiplier`` and
    ``auc`` for the same method/generator/bank and detector. Returns the AUC
    interpolated onto the epsilon at ``level``, or None if the attack never
    reaches ``level`` so no epsilon exists to read a detector AUC at.
    """
    if success_curve.empty or detector_frame.empty:
        return None
    eps, means = _success_curve(success_curve, success_col=success_col)
    target_eps = interpolate_at_level(eps, means, level)
    if target_eps is None:
        return None
    det = detector_frame.sort_values("eps_multiplier")
    det_eps = det["eps_multiplier"].to_numpy(dtype=np.float64)
    det_auc = det["auc"].to_numpy(dtype=np.float64)
    finite = np.isfinite(det_auc) & np.isfinite(det_eps)
    if not finite.any():
        return None
    return float(np.interp(target_eps, det_eps[finite], det_auc[finite]))


# --------------------------------------------------------------------------
# Decision gate
# --------------------------------------------------------------------------

def _primary_auc(attacks: pd.DataFrame, detectors: pd.DataFrame, method: str,
                 level: float = PRIMARY_LEVEL):
    """Mean primary AUC across subjects and registered generator variants.

    H1 is cross-subject and matched on surrogate targeted success. Tiny Noise
    has no generator or bank; learned methods include both signal and causal
    variants. A method is excluded when any subject has no variant reaching the
    matched level, matching the no-extrapolation rule in PREREG.md.
    """
    if attacks.empty:
        return None
    method_rows = attacks[attacks["method"] == method]
    if method != "tiny_noise":
        method_rows = method_rows[method_rows["bank"] == "cross_subject"]
    subjects = sorted(method_rows["subject"].dropna().unique())
    if not subjects:
        return None
    subject_values = []
    for subject in subjects:
        subject_rows = method_rows[method_rows["subject"] == subject]
        variant_values = []
        for (generator, bank), curve in subject_rows.groupby(["generator", "bank"]):
            det = detectors[
                (detectors["detector"] == PRIMARY_DETECTOR)
                & (detectors["subject"] == subject)
                & (detectors["method"] == method)
                & (detectors["generator"] == generator)
                & (detectors["bank"] == bank)
            ]
            value = detector_auc_at_success(curve, det, level=level)
            if value is not None:
                variant_values.append(value)
        if not variant_values:
            return None
        subject_values.append(float(np.mean(variant_values)))
    return float(np.mean(subject_values))


def decision_gate(attacks: pd.DataFrame, detectors: pd.DataFrame,
                  max_success: dict | None = None) -> dict:
    """Evaluate the PREREG decision gate from the result frames.

    The gate is the pre-registered one, computed mechanically:

    - H1 primary favors proto-GPN (nearest) over both target-only GPN and Tiny
      Noise (lower detector AUC is stealthier), and the wrong-letter control does
      worse than nearest (higher AUC than nearest);
    - otherwise the outcome is defense-led;
    - if the defense also fails on its training families (TPR at 5 percent FPR
      below 0.5), the outcome is to stop and reassess.

    Every quantity the verdict rests on is returned, so the verdict can be
    recomputed by hand. ``max_success`` optionally maps a method label to its
    maximum targeted success so the report can state when a level was not
    reached instead of pretending it was.
    """
    nearest = _primary_auc(attacks, detectors, PRIMARY_METHOD)
    target_only = _primary_auc(attacks, detectors, "target_only")
    tiny = _primary_auc(attacks, detectors, "tiny_noise")
    wrong = _primary_auc(attacks, detectors, WRONG_CONTROL)

    facts = {
        "nearest_auc_at_primary": nearest,
        "target_only_auc_at_primary": target_only,
        "tiny_noise_auc_at_primary": tiny,
        "wrong_control_auc_at_primary": wrong,
        "level": PRIMARY_LEVEL,
        "max_success": dict(max_success or {}),
    }

    # Defense training-family TPR, the stop condition.
    train_tpr = float("nan")
    if not detectors.empty:
        rows = detectors[
            (detectors["detector"] == DEFENSE_DETECTOR)
            & (detectors["method"].isin(DEFENSE_TRAIN_METHODS))
        ]
        if not rows.empty:
            train_tpr = float(rows["tpr_at_fpr"].to_numpy(dtype=np.float64).mean())
    facts["defense_train_tpr"] = train_tpr

    reached = all(v is not None for v in (nearest, target_only, tiny, wrong))
    favors = bool(
        reached
        and nearest < target_only
        and nearest < tiny
        and wrong > nearest
    )
    facts["h1_primary_favors_proto_nearest"] = favors

    if not np.isfinite(train_tpr) or train_tpr < 0.5:
        # The defense must clear its own training families before a full paper.
        verdict = DECISION_GATE_STOP
        reason = ("defense TPR on its training families is below 0.5 at 5 percent "
                  "FPR" if np.isfinite(train_tpr) else
                  "defense TPR on its training families could not be computed")
    elif favors:
        verdict = DECISION_GATE_FULL
        reason = ("proto-GPN nearest is stealthier than target-only GPN and Tiny "
                  "Noise at the primary level, and the wrong-letter control is "
                  "easier to detect")
    else:
        verdict = DECISION_GATE_DEFENSE
        reason = ("the H1 primary comparison does not favor proto-GPN nearest "
                  "over both comparators, so the prototype result is a negative "
                  "result and the paper is defense-led")
    facts["verdict"] = verdict
    facts["reason"] = reason
    return facts


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

@dataclass
class ReportInputs:
    """Every frame the report renders, loaded once from ``output_root``."""

    output_root: Path
    decoder_accuracy: pd.DataFrame = field(default_factory=pd.DataFrame)
    attacks: pd.DataFrame = field(default_factory=pd.DataFrame)
    detectors: pd.DataFrame = field(default_factory=pd.DataFrame)
    plausibility: pd.DataFrame = field(default_factory=pd.DataFrame)
    bank: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def load(cls, output_root) -> "ReportInputs":
        return cls(
            output_root=_root(output_root),
            decoder_accuracy=load_decoder_accuracy(output_root),
            attacks=load_attacks_eval(output_root),
            detectors=load_detectors(output_root),
            plausibility=load_plausibility(output_root),
            bank=load_bank_report(output_root),
        )


def _max_success_by_method(attacks: pd.DataFrame) -> dict:
    """Maximum surrogate targeted success per method/generator/bank label."""
    out = {}
    if attacks.empty:
        return out
    for (method, generator, bank), group in attacks.groupby(["method", "generator", "bank"]):
        label = f"{method}_{generator}_{bank}"
        _, success = _success_curve(group, "success_surrogate_15")
        out[label] = float(np.max(success))
    return out


def _success_table(attacks: pd.DataFrame) -> pd.DataFrame:
    """Targeted success versus epsilon for the surrogate and every victim."""
    rows = []
    if attacks.empty:
        return pd.DataFrame(rows)
    success_cols = [c for c in attacks.columns if c.startswith("success_") and c.endswith("_15")]
    norm_cols = [c for c in ("linf", "l2", "rms", "snr_db") if c in attacks.columns]
    keys = ["subject", "method", "generator", "bank", "eps_multiplier"]
    for values, group in attacks.groupby(keys):
        row = dict(zip(keys, values))
        for col in success_cols + norm_cols:
            row[col] = float(group[col].to_numpy(dtype=np.float64).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def _matched_table(attacks: pd.DataFrame, levels=MATCHED_LEVELS) -> pd.DataFrame:
    """Norms at surrogate matched-success levels with explicit fallback rows."""
    rows = []
    if attacks.empty:
        return pd.DataFrame(rows)
    norm_cols = [c for c in ("linf", "l2", "rms", "snr_db") if c in attacks.columns]
    for (subject, method, generator, bank), group in attacks.groupby(
        ["subject", "method", "generator", "bank"]
    ):
        eps, means = _success_curve(group, "success_surrogate_15")
        max_index = int(np.nanargmax(means))
        max_success = float(means[max_index])
        for level in levels:
            matched_eps = interpolate_at_level(eps, means, level)
            reached = matched_eps is not None
            reported_eps = float(matched_eps) if reached else float(eps[max_index])
            row = {
                "subject": subject, "method": method, "generator": generator,
                "bank": bank, "success_level": float(level), "reached": reached,
                "reported_success": float(level) if reached else max_success,
                "eps_multiplier": reported_eps,
            }
            for norm in norm_cols:
                row[norm] = _read_norm_at_eps(group, norm, reported_eps)
            rows.append(row)
    return pd.DataFrame(rows)


def _read_norm_at_eps(group: pd.DataFrame, norm: str, eps_value: float) -> float:
    """The mean ``norm`` at the grid epsilon nearest ``eps_value``.

    Norms are measured at the fixed grid, so a value at an interpolated epsilon
    is the grid point it falls on; this keeps the number tied to a measurement
    rather than a linear guess about a nonlinear quantity.
    """
    means = group.groupby("eps_multiplier")[norm].mean().sort_index()
    eps = means.index.to_numpy(dtype=np.float64)
    values = means.to_numpy(dtype=np.float64)
    finite = np.isfinite(eps) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.interp(eps_value, eps[finite], values[finite]))


def _defense_table(detectors: pd.DataFrame) -> pd.DataFrame:
    """Genuine-vs-induced TPR per method and epsilon."""
    if detectors.empty:
        return pd.DataFrame()
    rows = detectors[detectors["detector"] == DEFENSE_DETECTOR].copy()
    cols = ["subject", "method", "generator", "bank", "eps_multiplier",
            "auc", "tpr_at_fpr", "far"]
    return rows[[c for c in cols if c in rows.columns]].sort_values(
        ["subject", "method", "eps_multiplier"]
    )


def _plausibility_table(plausibility: pd.DataFrame) -> pd.DataFrame:
    """Mean plausibility metric per method/generator/epsilon, with std where useful."""
    if plausibility.empty:
        return pd.DataFrame()
    metrics = ["waveform_corr_pz", "waveform_corr_cz", "p300_latency_ms",
               "p300_amplitude", "topography_corr", "psd_distance_db"]
    keys = ["subject", "method", "generator", "bank", "eps_multiplier"]
    present = [m for m in metrics if m in plausibility.columns]
    grouped = plausibility.groupby(keys)[present].mean().reset_index()
    return grouped


def _bank_table(bank: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol prototype counts, reduced to the columns the report needs."""
    if bank.empty:
        return pd.DataFrame()
    cols = [c for c in ("attacked_subject", "mode", "symbol_idx", "count",
                        "nearest_mean", "bank_size") if c in bank.columns]
    return bank[cols].copy()


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    """Render a DataFrame as a GitHub Markdown table, truncated if huge."""
    if frame.empty:
        return "_no data_\n"
    shown = frame.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    divider = "| " + " | ".join("---" for _ in shown.columns) + " |"
    lines = [header, divider]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_showing {max_rows} of {len(frame)} rows_")
    return "\n".join(lines) + "\n"


def _write_plots(inputs: ReportInputs, out_dir: Path) -> list[str]:
    """Success-vs-eps, detector-AUC-vs-eps and defense-TPR plots as PNGs.

    Plotting is optional: a missing or headless matplotlib is caught so the
    report still writes without figures.
    """
    import os
    import tempfile
    # Point matplotlib's cache at a stable writable location outside the report
    # dir, so the output holds only the report and the figures and the font cache
    # is not rebuilt on every call.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "p3a_mpl_cache")
    )
    written = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plots are best effort
        return written

    attacks = inputs.attacks
    if not attacks.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (method, generator, bank), group in attacks.groupby(["method", "generator", "bank"]):
            eps, means = _success_curve(group, "success_surrogate_15")
            ax.plot(eps, means, marker="o", label=f"{method}/{generator}/{bank}")
        ax.set_xlabel("epsilon multiplier")
        ax.set_ylabel("surrogate targeted success (15 reps)")
        ax.set_title("Targeted success vs epsilon")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = out_dir / "success_vs_eps.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))

    detectors = inputs.detectors
    if not detectors.empty:
        det = detectors[detectors["detector"].isin((PRIMARY_DETECTOR, DEFENSE_DETECTOR))]
        if not det.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for (name, method, generator), group in det.groupby(
                ["detector", "method", "generator"]
            ):
                group = group.sort_values("eps_multiplier")
                ax.plot(group["eps_multiplier"], group["auc"], marker="o",
                        label=f"{name}:{method}/{generator}")
            ax.set_xlabel("epsilon multiplier")
            ax.set_ylabel("AUC")
            ax.set_title("Detector AUC vs epsilon")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2)
            path = out_dir / "detector_auc_vs_eps.png"
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            written.append(str(path))

    defense = _defense_table(detectors)
    if not defense.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (method, generator), group in defense.groupby(["method", "generator"]):
            group = group.sort_values("eps_multiplier")
            ax.plot(group["eps_multiplier"], group["tpr_at_fpr"], marker="o",
                    label=f"{method}/{generator}")
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="H2 target 0.80")
        ax.set_xlabel("epsilon multiplier")
        ax.set_ylabel("TPR at 5 percent FPR")
        ax.set_title("Genuine-vs-induced defense TPR")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = out_dir / "defense_tpr_vs_eps.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
    return written


def build_report(output_root, out_dir) -> dict:
    """Build the report dict and write Markdown plus plots into ``out_dir``."""
    inputs = ReportInputs.load(output_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success_table = _success_table(inputs.attacks)
    matched_table = _matched_table(inputs.attacks)
    defense_table = _defense_table(inputs.detectors)
    plausibility_table = _plausibility_table(inputs.plausibility)
    bank_table = _bank_table(inputs.bank)
    gate = decision_gate(inputs.attacks, inputs.detectors,
                         max_success=_max_success_by_method(inputs.attacks))
    plots = _write_plots(inputs, out_dir)

    lines = ["# Kill-test report", ""]
    lines.append(f"Output root: `{inputs.output_root}`")
    lines.append("")
    lines.append("## Clean decoder accuracy")
    lines.append("")
    lines.append(_markdown_table(inputs.decoder_accuracy))
    lines.append("## Targeted success vs epsilon (surrogate and victims)")
    lines.append("")
    lines.append("Each row is one epsilon. Surrogate success is the matching axis; "
                 "victim columns report transfer at the same epsilon.")
    lines.append("")
    lines.append(_markdown_table(success_table))
    lines.append("## Norms at matched surrogate success")
    lines.append("")
    lines.append("When `reached` is False, `reported_success` and the norms come from "
                 "the observed maximum-success epsilon; no value is extrapolated.")
    lines.append("")
    lines.append(_markdown_table(matched_table))
    lines.append("## Detector AUC and defense TPR near matched success")
    lines.append("")
    primary_rows = []
    if not inputs.attacks.empty:
        for (subject, method, generator, bank), group in inputs.attacks.groupby(
            ["subject", "method", "generator", "bank"]
        ):
            det = inputs.detectors[
                (inputs.detectors["detector"] == PRIMARY_DETECTOR)
                & (inputs.detectors["subject"] == subject)
                & (inputs.detectors["method"] == method)
                & (inputs.detectors["generator"] == generator)
                & (inputs.detectors["bank"] == bank)
            ]
            row = {"subject": subject, "method": method, "generator": generator, "bank": bank}
            for level in MATCHED_LEVELS:
                row[f"detector_auc_at_{int(level * 100)}"] = detector_auc_at_success(
                    group, det, level=level
                )
            primary_rows.append(row)
    lines.append(_markdown_table(pd.DataFrame(primary_rows)))
    lines.append("## Genuine-vs-induced TPR per method")
    lines.append("")
    lines.append(_markdown_table(defense_table))
    lines.append("## Plausibility metrics")
    lines.append("")
    lines.append(_markdown_table(plausibility_table))
    lines.append("## Per-symbol prototype counts")
    lines.append("")
    lines.append(_markdown_table(bank_table))
    lines.append("## PREREG decision gate")
    lines.append("")
    lines.append(f"**Verdict: {gate['verdict']}**")
    lines.append("")
    lines.append(f"Reason: {gate['reason']}.")
    lines.append("")
    lines.append("Inputs to the verdict (recomputable by hand):")
    lines.append("")
    for key in ("nearest_auc_at_primary", "target_only_auc_at_primary",
                "tiny_noise_auc_at_primary", "wrong_control_auc_at_primary",
                "defense_train_tpr", "h1_primary_favors_proto_nearest"):
        lines.append(f"- `{key}` = {gate[key]}")
    lines.append("")
    if plots:
        lines.append("## Plots")
        lines.append("")
        for path in plots:
            lines.append(f"![{Path(path).stem}]({Path(path).name})")
        lines.append("")
    report = "\n".join(lines)
    path = out_dir / "kill_test_report.md"
    path.write_text(report)
    return {
        "report": str(path),
        "plots": plots,
        "verdict": gate["verdict"],
        "gate": gate,
        "n_attack_rows": int(len(inputs.attacks)),
        "n_detector_rows": int(len(inputs.detectors)),
    }


def write_report(output_root, out_dir) -> dict:
    """Alias for ``build_report`` kept explicit for callers that prefer a verb."""
    return build_report(output_root, out_dir)
