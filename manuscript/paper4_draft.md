# When P300 attack detection does not transfer: operating-point reliability and clean spelling cost

Working manuscript,13 October2026. **Not submission-ready.** Authors/order,
affiliations and target journal require confirmation. No acceptance claim.

## Abstract draft

P300 speller security depends on both detecting manipulations and avoiding
false rejection of genuine character decisions. We investigate whether attack
ranking performance translates into useful independently clean-calibrated
operating points. A CompetitionIII development study compares conventional
classifiers, engineered authenticity features and a paired channel/time
representation. Prototype-guided targeted attacks remain difficult to detect
under subject transfer, and richer features produce direction-dependent
trade-offs rather than consistent improvement. A cloud-reproduced software and
data audit identifies attack-budget identity defects, calibration sample
limitations and event-order assumptions that affect defensible conclusions.
An independent recorded-marker OpenBMI evaluation is planned but not completed.
Consequently this draft establishes development failures and methodological
limits, not a validated general-purpose defense or population low-FAR guarantee.

## Candidate contribution, subject to independent validation

A character-level evaluation separating clean-user normalization, threshold
calibration and evaluation, with observed FAR and uncertainty, unseen attack
families, defended targeted success and explicit rejection cost. This is not a
new detector or first EEG/P300 defense benchmark. The independent study must
establish the magnitude and practical consequences of failures beyond prior work.

## Related work

- Jung etal., GPN, JBHI2023: initial generator motivation; no unlicensed code copied.
- Zhong etal., AUAP, ESWA2025: structural attack motivation; sparse mask not evaluated.
- Zhang etal., TinyNoise, NSR2021: targeted P300 attacks already established.
- Mezzina etal., Sensors2021: synchronized channel-mixing protection under a
  trusted transmission architecture; not directly comparable to signal-only detection.
- Chen/Wu, JNE2024: BU/LID/network-feature Mahalanobis and subject-held-out EEG detection.
- Zhang/Gu, JPCS2024, DOI10.1088/1742-6596/2761/1/012037: detector comparison
  includes P300/MI per indexed abstract; full-method verification outstanding.
- Chen/Jia/Wu, NeuralNetworks2025, DOI10.1016/j.neunet.2025.107516: nine defenses,
  five datasets, three networks and four scenarios; generic benchmarking not novel.
- Conceicao daCosta etal., ICEIS2026, DOI10.5220/0014819600004018:
  conventional detector/ANFIS comparison; full protocol extraction incomplete.

## Methods: completed development work

See immutable splits and append-only PREREG. CompIII official test outcomes are
already inspected; report them as development evidence. Historical budget-label
overwrite claims remain superseded by corrected evaluations. Target attacks do
not fit clean thresholds; later exploratory repairs must not become confirmation.

## Verified results to retain

Original prototype stealth hypothesis failed. Independent-normalization external
clean audit failed its strict gate. At budget0.2, fixed logistic nearest-causal
cross-subject TPR is12.4% A-to-B and39.2% B-to-A, observed FAR5%/6%. Adding
channel/time bins changesTPR to30.6%/3.6%, FAR3%/3%, while TinyNoise detection
drops68.7% to3.9% A-to-B and100% to50.5% B-to-A. No general feature improvement.
Cloud portable suite314 passed with explicit raw/path exclusions. These are
engineering verification, not independent statistical validation.

## Independent results: pending

No independent OpenBMI decoder, attack or detector performance has been measured.
Do not fill tables or conclusions from expected results. Recorded-marker indexing
and schedule correspondence are bounded-subset validated; physical timing cause
remains unresolved. No precise per-user5% assurance from36 test characters.

## Discussion and limitations

AUC reflects ranking and can coexist with weak upper-tail detection. The
contribution must demonstrate where this matters across users and attack types,
not rediscover the mathematical distinction. Known-label ERP analysis is not a
deployed unknown-intent authenticity system. Rejection-cost simulations require
an explicit retry model and are not live-user latency measurements. IID bounds
require independent decisions and do not erase session dependence. Negative
results limit tested methods/settings, not all possible defenses.

## Required before submission

Independent benchmark, closest-paper fulltext matrix, reproduced baselines,
uncertainty/cluster-aware analysis, clean-error/rejection cost, defense-aware
stress test, source-audited results tables/figures, complete methods/supplement,
author/professor review, journal scope/current category-quartile verification.
