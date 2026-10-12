# OpenBMI centered epoch alignment

Kaggle p3a-openbmi-centered v1 completed12 October2026, runner74938a7.
Five focused schema/arithmetic assertions passed, not full repository or loader
validation. All processing cloud-only; compact summaries retrieved, no raw EEG.

At fixed documented MATLAB one-based conversion shift-1, per-channel temporal
mean-centered raw slices match provided smt epochs to numerical precision.
Relative L2 errors for session1 train/test, session2 train/test are
6.8786e-16,2.6906e-16,4.5753e-16,1.5766e-16. At shift0 errors are
0.4048,0.2457,0.2874,0.4425. Same fixed event subset as prior audit;
297600 scalar comparison points per phase, not independent observations.

Thus the prior mismatch is consistent with a constant per-channel temporal
offset for inspected epochs, rather than waveform misalignment at shift-1.
This does not identify the exact baseline interval/algorithm or prove all
recording epochs align. The offset was specified from original MATLAB source,
not selected for detector success. No defense model evaluated.

Combined with stored-order schedule-label agreement, this supports building a
bounded loader candidate preserving event order and subtracting1 from MATLAB
indices. Before performance work: validate all epoch boundary/index checks,
explain first-pair reversal using recording/stimulus provenance, add synthetic
regressions, and test additional predeclared subjects on Kaggle. Do not silently
sort timestamps. No all-subject or deployed compatibility claim yet.

Raw EEG stays Kaggle; gates/splits unchanged and BNCI2015-003 locked. This is
data validity progress, not Q1/Q2-ready detection evidence. Completed monitor
retired; no training job launched by this report.
