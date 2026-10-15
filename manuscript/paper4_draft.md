# When P300 attack detection does not transfer: operating-point reliability and clean spelling cost

Working manuscript,15 October2026. **Not submission-ready.** Authors/order,
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
  full text reviewed: evaluates confidence, predictive entropy, Bayesian uncertainty
  and KDE on CompIII P300 and MI, including detector-aware CW against KDE.
  Adaptive P300 detection and KDE therefore precede this study. The reviewed
  protocol emphasizes flash-level ranking/balanced accuracy rather than the
  independently calibrated character-level reliability/cost estimand here.
  Conventional negative-Gaussian KDE is an explicit adaptation: the printed
  equation has a positive exponent inconsistent with its low-density rationale.
  See results/zhang_gu_2024_fulltext_review.md; no exact reproduction claim.
- Chen/Jia/Wu, NeuralNetworks2025, DOI10.1016/j.neunet.2025.107516: nine defenses,
  five datasets, three networks and four scenarios; generic benchmarking not novel.
- Conceicao daCosta etal., ICEIS2026, DOI10.5220/0014819600004018:
  conventional detector/ANFIS precedent; primary protocol review is now recorded
  in results/iceis2026_primary_review.md. No first-detector claim.

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
Earlier cloud portable suite314 passed with explicit raw/path exclusions.
New focused integration tests progressed to28 passing cases, including raw
adaptive gradients/budget, reference sampling, calibration leakage and variable-
length source template guards. A broader current suite initially failed test
collection because of namespace/payload packaging; the corrected Kagglev14 run passed351
tests with41 warnings in45.98s, preserving the same four raw/path exclusions.
These are software verification scopes, not independent statistical validation.

## OpenBMI development clean-validation result

Fixed validation subject19 had36/36 clean characters decoded correctly for each
of the three source seeds, with no seed selected. At nominal5%, rejection counts
for seeds20261013/14/15 were confidence0/0/0, entropy0/0/0, BU0/2/1, LID1/0/2,
MD0/2/3, adjusted MD0/2/3 and KDE1/1/0, each out of36. Preserve all1/5/10%
operating-point records in results/validation_clean_v1. These are repeated
measurements on one development subject, not108 independent characters or3
people. There are no genuine decoder errors here to establish rejection
behavior on mistaken clean decisions. Zero false alarms has a two-sided95%
IID upper bound9.74%; serial dependence and user shift remain unresolved.
No detector is selected as a confirmed winner and no threshold is retuned.
Attack detection and sealed-user outcomes remain unmeasured.

## Independent results: pending

No independent OpenBMI decoder, attack or detector performance has been measured.
Do not fill tables or conclusions from expected results. Recorded-marker indexing
and schedule correspondence were inspected across all54 session1 participants.
Subject52 official train fails identity, with528 schedule/known-label mismatches
and no compatible symbol in any33 blocks under the fixed schedule. Its test
passes; subjects53/54 pass both phases. Full54-person compatibility is false.
Physical timing cause remains unresolved. No precise per-user5% assurance from36
test characters. Fixed three-seed source training completed on Kaggle at the recorded30-epoch
final checkpoints, with no independent performance outcome. The source-reference/template/calibration pipeline and fixed validation19
clean audit have completed across all3seeds. Attack-path and independent
evaluation verification remain pending.

## Independent methods specification: prospective, not executed results

The exact candidate settings are in configs/openbmi_independent_protocol_candidate.json.
They remain gated until source verification, end-to-end integration, final
protocol review and immutable cloud manifests are complete. Existing historical
splits and failed gates are preserved; BNCI2015-003 remains locked.

### Participants, identity and missingness

Development subjects1-18, validation19-36 and evaluation37-54 are assigned by ID,
with model seeds20261013/14/15 and no best-seed selection. Of18 intended evaluation
participants,52 lacks a usable official training identity under the fixed loader.
Keep this person in feasibility/participant flow, with calibrated performance
unavailable. If final cloud verification agrees, performance is estimable for17
eligible people only. This is pre-outcome metadata eligibility, not exclusion by
accuracy or detection rate. Available-case results do not imply missing-at-random
or reliability for all18 people. No substitution, inferred relabeling or reuse
of test characters for calibration is permitted.

### Decoder, references and clean operating points

The source-only EEGNet uses0.5-20Hz filtering/CAR, raw1000Hz decimated by17 to
58.8235Hz, and0.8second epochs. Adam.001, batch64,30 epochs and final checkpoints
are fixed. Source checkpoints/EEG files, reference features and schedule are
bound by hashes; models and raw data remain on Kaggle. The source reference set
uses at most5000 flash activations with deterministic subject/class quotas.
Seven controls/adaptations are confidence, binary predictive entropy, dropout
uncertainty, LID(k20), shared-shrinkage network Mahalanobis, confidence-adjusted
Mahalanobis(epsilon.001 normalized units) and KDE(bandwidth.5). Shared LedoitWolf
covariance/final-layer aggregation are adaptations, not exact paper reproductions.

For each eligible user, split33 official train decisions into13 normalization
and20 calibration decisions with fixed seed20261013; keep all36 official test
decisions for evaluation. Refit channel normalization only on the13 decisions.
Compute clean calibration scores without true-character selection: average flash
anomaly over groups supporting the decoder-predicted symbol. The higher empirical
quantile at nominal1/5/10% defines rejection strictly above the threshold, retaining
ties. At20 decisions, nominal1% and5% both select the maximum; nominal values are
not per-user guarantees. Dropout calibration and every reported clean/attacked
score use100 draws; only adaptive gradient estimation uses30 draws. Recalibrating
a different draw-count score at evaluation is forbidden.

### Manipulation and adaptive-evasion scope

This is an offline recorded-marker signal-access benchmark, not verified physical
onset or real-time hardware security. Perturbations act before preprocessing on
all raw channels/samples. Preserve budget ratios.05/.10/.20/.30/.50, scaled from
the median channel standard deviation of concatenated normalization-only raw
EEG. Freeze one wrong-symbol target per evaluation character, shared across
methods/budgets/seeds. Never fit thresholds or choose a model from target attacks.

Compare targeted FGSM,200-step/3-restart PGD, Gaussian controls and a gradient-
trained source-only template idea adaptation. The latter uses800 samples at
1000Hz(0.8s),200Adam updates/.01, eight individually accumulated variable-length
source-trial gradients, and primary.20 source raw scale; rescale the same learned
template across budgets. No raw padding, target-user template fitting, exact NSR
reproduction or physical causality claim. Actual training remains pending.

Detector-aware PGD combines targeted cross-entropy and a softplus detector penalty
at weights0/.1/1/10, with200steps/3restarts and400steps/5restarts convergence checks
at budget.20. Thresholds/MAD scales come only from clean calibration. Report every
setting and the adversary's strongest successful nonrejected attempt; this selects
an attack, not a winning defense. Preserve all failures and inspect gradient/
convergence evidence before interpreting failed evasion as robustness.

### Denominators, uncertainty and cost

Primary summaries are per-user observed clean FAR at nominal5% and defended
targeted success over every original attack attempt at budget.20. Conditional
TPR among successful attacks is secondary and must disclose its denominator.
Include genuine decoder mistakes among clean decisions; report accepted-correct
and accepted-wrong rates over all original clean characters and conditional
accepted accuracy with its denominator. No successful-attack-only denominator
may replace end-to-end attack success.

Describe per-user two-sided95% exact binomial intervals under IID assumptions,
not serial-dependence-proof assurance. Average all three seeds within each user,
then use equal-user paired2000-resample bootstrap(seed20261013). Report all
methods/budgets and pointwise differences versus confidence, without selected
winners or familywise significance claims. Cost includes observed rejection and
accepted-wrong counts, recorded-marker duration proxies and synchronized GPU
inference timing. The retry expression1/(1-FAR)-1 is ideal IID sensitivity only,
not measured clinical latency or backspace-corrected throughput.

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
