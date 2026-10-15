# Source-stage execution decisions,14 October2026

The raw pipeline is now exercised on real development EEG. Actual sampling rate
58.8235Hz and47-sample0.8s epochs must be recorded, not mislabeled60Hz/48.
Machine-readable source-stage settings are configs/openbmi_reliability_stage_a.json.
This makes the source-training implementation concrete while independent
performance remains sealed pending complete attack/statistical specification.

Source trains on official development train characters, not official test.
Target normalization is fitted on its separate13 clean raw trials;20 threshold
characters and36 evaluation characters remain disjoint. This is an empirical
operating-point evaluation, not a guaranteed conformal procedure. Per-user FAR
intervals and subject-clustered summaries remain necessary. Calibration ties
are retained, score strictly greater than threshold rejects; do not switch
comparison after seeing attacks.

Three source seeds are specified in advance; all reported, no best-seed result.
Fixed final-epoch training avoids checkpoint selection from target subjects.
The selected final encoder layer and LedoitWolf covariance are disclosed
adaptations rather than exact published reproduction. The0/.001 input adjustment
variants are both reported rather than selected by evaluation performance.
Dropout draws30 and LIDk20 are fixed source-stage defaults, not inferred from
successful target attacks. Full LID batch/reference and self-exclusion behavior
must be implemented consistently, including source reference sampling metadata.

All-subject structural audit is currently running. It does not train decoders or
expose attack/detection outcomes. This stageA specification does not regenerate
old splits, alter the prior failure gates, unlock BNCI2015-003 or assert Q1
readiness. Remaining deliverables: source execution runner, complete threat/
attack/cost/statistical freeze, immutable cloud manifests and independent data.
