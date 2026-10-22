# Clean rejection and adaptive manipulation in P300 spellers: character-level reliability of existing detectors

Working manuscript,21 October2026 Central. **Not submission-ready.** Authors/order,
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
A prespecified recorded-marker OpenBMI evaluation is running, with independent results pending.
Consequently this draft establishes development failures and methodological
limits, not a validated general-purpose defense or population low-FAR guarantee.

## Candidate contribution, subject to independent validation

A character-level evaluation separating clean-user normalization, threshold
calibration and evaluation, with observed FAR and uncertainty, multiple fixed manipulation families, defended
targeted success and explicit rejection cost. This is not a
new detector or first EEG/P300 defense benchmark. The independent study must
establish the magnitude and practical consequences of failures beyond prior work.

## Related work

- Jung etal., GPN, JBHI2023: initial generator motivation; no unlicensed code copied.
- Zhong etal., AUAP, ESWA2025: structural attack motivation; sparse mask not evaluated.
- Zhang etal., TinyNoise, NSR2021: targeted P300 attacks already established.
- Mezzina etal., Sensors2021: synchronized channel-mixing protection under a
  trusted transmission architecture; not directly comparable to signal-only detection.
- Chen and Wu, arXiv2212.00727 (2022): reviewed preprint precedent for BU/LID/MD
  and subject-held-out ERN/MI detection. Chen, Meng, Xu and Wu, JNE2024,
  DOI10.1088/1741-2552/ad8964, is an expanded journal study with three datasets
  and eight detectors according to its abstract. Its full methods are now checked separately: ERN, MI and RSVP epochs;
  supervised logistic attack detectors; LOSO and AUC-based nested tuning.
  Section4.4 uses AUC to avoid threshold selection. This differs from our
  clean-only character calibration and all-attempt reliability estimand; it
  does not establish uniqueness across all literature. See
  results/jne2024_fulltext_review_20261022.md.
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
Cloud verification covers software contracts, not independent scientific
validation. The complete validation19 controller executed all36 characters across
three model seeds and cloud aggregation, preserving completed-ledger hashes on
repeat invocation. The latest code-bound release regression passed368 portable
tests with four declared historical raw/path exclusions. Exact scopes and
failures remain in the supplement and private provenance artifacts.

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
These clean-validation observations precede the separately recorded development
attack audit. Independent-user outcomes remain pending.

## Independent results: pending

Independent OpenBMI evaluation is running; no completed independent summary
has been inspected or reported in this draft.
Do not fill tables or conclusions from expected results. Recorded-marker indexing
and schedule correspondence were inspected across all54 session1 participants.
Subject52 official train fails identity, with528 schedule/known-label mismatches
and no compatible symbol in any33 blocks under the fixed schedule. Its test
passes; subjects53/54 pass both phases. Full54-person compatibility is false.
Physical timing cause remains unresolved. No precise per-user5% assurance from36
test characters. Fixed three-seed source training completed on Kaggle at the recorded30-epoch
final checkpoints, with no independent performance outcome. The source-reference/template/calibration pipeline and fixed validation19
clean audit have completed across all3seeds. Full validation19 attack/controller/aggregation verification has completed;
independent evaluation outcomes remain pending.

## Independent methods specification: prospective, not executed results

The exact prospective settings are frozen in
configs/openbmi_independent_protocol_frozen_20261022.json, with cloud-generated
immutable allocations, targets and code-bound receipts in
splits/openbmi_independent_20261022. Settings and artifacts were committed before
independent launch. The first five of51 fixed subject-seed tasks are running
on five private Kaggle accounts. Existing historical splits and failed gates
are preserved; BNCI2015-003 remains locked.

### Participants, identity and missingness

Development subjects1-18, validation19-36 and evaluation37-54 are assigned by ID,
with model seeds20261013/14/15 and no best-seed selection. Of18 intended evaluation
participants,52 lacks a usable official training identity under the fixed loader.
Keep this person in feasibility/participant flow, with calibrated performance
unavailable. Cloud metadata verification establishes allocation eligibility for17
people only; their performance remains to be estimated. This is pre-outcome metadata eligibility, not exclusion by
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
reproduction or physical causality claim. Source-only template training has completed, with hash-bound provenance.

Detector-aware PGD combines targeted cross-entropy and a softplus detector penalty
at fixed weight1 and budget.20 with200steps/3restarts on all36 characters.
A400step/5restart sensitivity runs only on the fixed first test character of
each eligible user, for all seeds/detectors. Ordinary targeted PGD is the
weight0 comparator. The larger weights0/.1/1/10 development grid remains
development evidence and is not represented as the independent scope. Thresholds/MAD scales come only from clean calibration. Report every
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
then use equal-user2000-resample percentile bootstrap(seed20261013) for
population mean summaries. Report paired pointwise differences versus confidence
for the shared nonadaptive perturbations. Detector-specific adaptive objectives
do not support a paired comparative-efficacy claim. No selected winners or
familywise significance claims are made; exchangeable-user sampling is assumed. Cost includes observed rejection and
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

Independent benchmark outcomes, expanded closest-paper fulltext matrix,
explicitly labelled baseline adaptations,
uncertainty/cluster-aware analysis, clean-error/rejection cost, defense-aware
stress test, source-audited results tables/figures, complete methods/supplement,
author/professor review, journal scope/current category-quartile verification.

### Development detector-aware optimization audit (16 October)

On one fixed validation subject19 character, at the primary raw budget0.20,
ordinary targeted PGD was rejected by MD, adjusted MD and KDE in all three model
seeds, while confidence, entropy, BU and LID accepted it. Optimizing separately
against each detector with weights0.1,1,10 reached the wrong target and passed
that attacked detector for every seed at both200steps/3restarts and400steps/
5restarts. This does not establish simultaneous evasion of all detectors or a
population success rate. Longer optimization improved some objective endpoints
without changing these binary outcomes. Finite values, actual raw budgets and
trace consistency passed cloud checks. The finding rules out a robust-defense
claim from this example; independent participant outcomes are still pending.
Full settings, timings, gradients and limitations are preserved in
results/development_attack_audit_interpretation.md and development_attack_summary_v2.
