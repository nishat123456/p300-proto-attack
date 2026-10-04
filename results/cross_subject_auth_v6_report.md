# Corrected cross-subject authenticity evaluation

Kaggle `mustaqimnishat/p3a-cross-auth` v6 completed on 4 October 2026.
All computation and 288 portable tests ran on Kaggle; no local analysis ran.
The budget-identity regression passed. Test exclusions are recorded in
`cross_subject_auth_v6/cloud_test_scope.txt`.

This report supersedes the epsilon-specific TPR claims in
`cross_subject_auth_report.md`. Historical outputs are preserved. Each feature
file is now a distinct training/evaluation identity instead of overwriting
different budgets under one method label. Source training therefore also changes.

## Directly inspected rows: budget 0.2, nominal FAR 5%

| Attack | A to B TPR | B to A TPR |
| --- | ---: | ---: |
| PGD | 99.3% | 100.0% |
| Target-only causal | 44.5% | 74.8% |
| Target-only signal-specific | 91.4% | 99.8% |
| Tiny Noise | 69.6% | 99.6% |
| Prototype medoid, causal, cross-subject | 12.2% | 27.8% |
| Prototype nearest, causal, cross-subject | 12.7% | 31.8% |

Observed clean FAR on these five-fold rows is 4% A to B and 3% B to A.
These are development measurements on two previously inspected subjects, not
external validation or a guaranteed population FAR. All budget rows remain in
the CSV. This table is descriptive, not a replacement primary outcome or a
new gate. Random noise has too few evaluable folds for reliable conclusions.

## Scientific decision

The corrected detector remains weak against unseen causal prototype attacks.
The previously reported strong B-to-A prototype detection was not valid at the
labeled epsilon. The current architecture is not ready for broad expansion or
a positive robust-defense claim. Preserve the defense-only scope, but require
prior-work comparison and a calibration/feature feasibility decision before a
new method is selected. BNCI2015-003 remains locked; existing gates are unchanged.

## Provenance qualification

Correction and test source: private Git commit `1430322`.
Kaggle's immutable feature bundle contains base commit
`97ef266ab027dcaef9718a6d075e6d2abf4c3867`; `config.json` records this base,
not the overlay. The v6 runner overlays the corrected evaluator and test before
execution. Its exact submitted source is retained in
`cross_subject_auth_v6/executed_runner.py`. Do not treat the runtime clean Git
flag as proof that no overlay executed. Raw outputs/config and cloud log are
retained unchanged. No raw EEG or working code tree was downloaded.

## Calibration/compatibility feasibility inspection

Both the A/B evaluator and the BNCI clean audit fit target normalization on the
same clean samples used for threshold calibration. Separate held-out testing
does not make these two fitting stages independent. A distribution-free FAR
guarantee must not be claimed from this design.

BNCI2014-008 has 24/5/6 frozen characters per subject. Fitting normalization on
the 24 training characters could separate it from the five validation threshold
samples, but five calibration and six test characters still cannot resolve a
precise 5% operating point. More flashes/windows cannot be counted as independent
characters. No revised protocol or data split has been executed.

The 36 engineered features aggregate channel statistics and therefore accept
both 64- and 8-channel arrays, but identical feature dimension does not establish
equivalent distributions. Montage, referencing, repetition count, stimulus
timing and ERP-conditioning semantics require an explicit compatibility audit.
The BNCI clean audit conditions on the true character, so it is not a complete
deployment evaluation on decoded-character evidence and supplies no attack TPR.

Chen and Wu's existing detector study evaluates uncertainty, local intrinsic
dimensionality, and Mahalanobis features with leave-one-subject-out evaluation
on ERN and motor imagery, reporting detector AUC. That is relevant precedent,
not an exact P300 calibrated-operating-point comparison:
https://arxiv.org/pdf/2212.00727
The prospective gap still needs a fuller current-literature comparison before
committing to a replacement detector or claiming novelty.
