"""Local kill-test report from the downloaded Kaggle output (Prompt 12).

This package reads the artifacts a kill-test Kaggle run produced (clean decoder
metrics, attack artifacts from Prompt 10, detector and plausibility CSVs from
Prompt 11, and prototype bank counts) and turns them into a Markdown report plus
PNG plots. It does no training and no attack evaluation: everything is computed
from files, so it runs on a laptop after the output is downloaded.

The PREREG decision gate is evaluated mechanically from those files, with every
input to the verdict written down next to it so the verdict can be recomputed by
hand.
"""
from __future__ import annotations

from p3a.report.kill_test import (  # noqa: F401
    DECISION_GATE_FULL,
    DECISION_GATE_DEFENSE,
    DECISION_GATE_STOP,
    build_report,
    decision_gate,
    detector_auc_at_success,
    interpolate_at_level,
    load_attacks_eval,
    load_bank_report,
    load_detectors,
    load_plausibility,
    write_report,
)

__all__ = [
    "DECISION_GATE_FULL",
    "DECISION_GATE_DEFENSE",
    "DECISION_GATE_STOP",
    "build_report",
    "decision_gate",
    "detector_auc_at_success",
    "interpolate_at_level",
    "load_attacks_eval",
    "load_bank_report",
    "load_detectors",
    "load_plausibility",
    "write_report",
]
