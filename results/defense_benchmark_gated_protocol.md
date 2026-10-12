# Paper 4: gated defense-reliability benchmark protocol

12 October2026. Status: **reviewable design, not a completed or fully frozen
performance preregistration**. Current evidence supports a bounded OpenBMI
loader candidate. No large experiment is launched by this document.

## Trigger provenance findings

Original public [random_speller.m](https://github.com/PatternRecognition/OpenBMI/blob/master/GUI_module/new_p300/random_speller.m)
selects symbols through cell_order. Its trigger branch emits1/2 according to
whether the target belongs to the selected group in one mode; another emits
n_run, a flash-position value. This is source evidence for different encoding
modes, not proof which branch/version generated the released MAT files.
[ERP_performance.m](https://github.com/PatternRecognition/OpenBMI/blob/master/GigaScience/function_ERP/ERP_performance.m)
uses random_cell_order.mat and known copy-spelling text.
[prep_segmentation.m](https://github.com/PatternRecognition/OpenBMI/blob/master/BMI_modules/PreProcessing/prep_segmentation.m)
uses MATLAB one-based sample indexing and preserves supplied event order.
Sources inspected12 October. No external MATLAB code copied into our software.

Three-subject metadata/schedule evidence and sampled waveform alignment are
recorded in results/openbmi_loader_validation_report.md and earlier reports.
First-pair timestamp reversal physical cause remains unverified. No repository
source inspected here establishes a photodiode/physical onset correction, an
exact acquisition commit or the release conversion that produced the reversal.
Therefore keep stored order as the candidate's semantic schedule order, but
label timestamps as recorded markers, not verified visual presentation times.
Do not invent a correction or claim timing/adaptive security robustness yet.

## Question and intended estimand

Candidate: how well do source-trained P300 manipulation detectors retain
character-level detection after clean-user calibration across subjects/sessions
and unseen attack families, and what clean rejection/spelling cost results?
Existing EEG detector and P300 authentication studies precede this project.
Novelty is a distinct reproducible operating-point evaluation only if the closest
literature matrix establishes a substantive gap. No first-detector claim.

Primary reporting unit is the independent character decision within a person
and session. Report subject-level observed clean FAR and held-out-family TPR,
with uncertainty and clean character accuracy/rejection. A group mean is a
separate estimand, not a per-user5% guarantee. Attacked targets/repetitions are
clustered under their original character; do not count them as new participants.

## Stage gates and executable next actions

1. **Provenance gate:** establish correspondence between release events and
   schedule, exact marker/index semantics, and first-pair reversal cause. Seek
   original release conversion/stimulus logs or author documentation; no message
   to authors is sent without explicit approval. If unavailable, narrow to an
   explicitly recorded-marker offline benchmark or choose a better documented
   dataset. Do not quietly exclude the first flash or reorder events.
2. **Loader gate:** run repository tests on Kaggle, validate schedule coverage,
   character identities, all boundaries and session-unique IDs on a predeclared
   broader metadata subset. Preserve failures. Current5 focused tests are not
   the full suite. Check integer/finite values, labels restricted1/2, positive
   sampling rate, and channels before production use; candidate needs these
   additional validations. No frozen splits are modified.
3. **Design gate:** execute cloud-only sample/uncertainty design for explicitly
   separate normalization, calibration and evaluation decisions. Public33/36
   per-session characters cannot automatically certify per-user low FAR. Specify
   acceptable interval width, session dependence, pooling and multiplicity
   before choosing a claim. Select subjects by predeclared ID/seed, not results.
4. **Literature-baseline gate:** complete BU/LID/network-feature Mahalanobis
   reproduction specifications and the2026 detector comparison details.
   Current engineered logistic/forest/Mahalanobis are controls, not exact
   reproductions. Freeze all source-only tuning ranges/settings before evaluation.
5. **Manifest gate:** after gates1-4, append a complete performance protocol and
   write new immutable OpenBMI subject/session/character partitions and targets.
   Keep CompIII and inspectedBNCI development-only. A new dataset is not magically
   untouched if its performance is later inspected during design. BNCI2015-003
   remains locked under existing gates. No splits written in this planning step.
6. **Execution gate:** cloud-only source-decoder/detector training, held-out
   subject/session/family evaluation, negative controls and independent clean
   thresholds. Evaluate genuine decoder mistakes, not only correct characters.
   Use predicted-character evidence for deployment claims. Report conditional
   TPR plus defended targeted success across all attempts; failed attacks remain
   in end-to-end denominators. No threshold retuning on target attack outcomes.
7. **Security/practical gate:** only after validity gates, freeze a detector-aware
   evasion objective and optimization/convergence checks, timing assumptions,
   perturbation budgets before preprocessing, retry policy and spelling-delay
   analysis. These are defense stress tests, not a new attack contribution.

No arbitrary numerical detection threshold makes a paper Q1/Q2. Existing
registered gates remain visible and unchanged; failure requires narrowed claims,
not moving gates. Seeds and subject partition IDs are intentionally not invented
for a supposedly frozen benchmark before the design gates are met.

## Concrete pursue-or-stop decision

Proceed with provenance and loader-hardening work. **Do not launch the larger
performance benchmark yet.** If trustworthy trigger provenance cannot be obtained,
prepare a recorded-marker feasibility/negative-result study or change dataset
with a fresh literature/compatibility assessment. If the larger study is feasible,
freeze it before performance outcomes, then test honestly. Q1/Q2 readiness still
requires independent evidence, contribution assessment and a manuscript.
