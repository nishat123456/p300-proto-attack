# Clean rejection and adaptive manipulation in P300 spellers: character-level reliability of existing detectors

Working manuscript,22 October2026 Central. **Not submission-ready.**

**Author and corresponding author:** M M Nishat (sole author, confirmed by the
author on22 October2026). Affiliation and correspondence contact remain to be
confirmed. No professor or other coauthor is listed. Target journal remains
unselected; no acceptance claim.

## Abstract draft

P300 speller security requires character decisions that remain useful after a
system rejects suspicious signals. Ranking attack examples by anomaly does not
by itself quantify genuine character rejection, accepted decoding errors or
successful manipulations that pass a deployed threshold. We evaluate these
outcomes for seven existing detector controls and adaptations in an offline
recorded-marker OpenBMI protocol. Source models use18 participants; clean-user
normalization and threshold calibration use disjoint official training decisions;
evaluation retains all36 official test characters per available held-out user.
The prespecified study includes17 eligible evaluation participants, three fixed
model seeds, five nonadaptive perturbation budgets and detector-aware stress
tests. One further intended participant remains in the flow with an unavailable
calibrated estimand because training labels fail identity validation. All-original-
attempt rates, participant uncertainty and explicit rejection-cost assumptions
separate reliability from successful-attack-only detection. Independent execution
is incomplete, so population results and the final results/conclusion sentences
of this abstract remain pending. Development failures motivate the protocol but
do not establish a new detector, robustness or a per-user false-alarm guarantee.

## Introduction

A P300 speller maps repeated stimulus responses to a character. Manipulating the
recorded signal can therefore change the selected symbol, even when the added
perturbation is small. Targeted P300 manipulation is established prior work
[@zhang2021tinynoise]. Generative and attention-guided EEG attacks further motivate
checking structured signal manipulations rather than only random corruption
[@jung2023gpn; @zhong2025auap]. A signal detector adds a second decision: whether
to accept the character. Rejecting an attack may be useful, but rejecting genuine
characters also imposes a cost. A detector's ranking performance alone does not
specify either outcome at a threshold calibrated for a new user.

Existing EEG security studies already examine attack detection, subject transfer,
P300 defense benchmarks and detector-aware evasion. These are not new claims of
this study. The question here is narrower: under clean-only user calibration,
what proportion of genuine character decisions is rejected, what genuine errors
remain accepted, and what proportion of all targeted manipulation attempts both
changes the character and passes the detector? We examine these outcomes together
rather than interpreting a nominal false-alarm setting or attack-ranking score
as a reliability guarantee.

The protocol separates source learning, clean-user normalization, clean threshold
calibration and held-out character evaluation. It fixes model seeds, attack
budgets and the adaptive recipe before evaluation; retains unsuccessful attacks
in end-to-end denominators; averages model seeds within participants; and reports
participant-level uncertainty. Rejection and source-query compute costs have
explicit assumptions. The empirical scope is an offline recorded-marker signal-
access setting, not a live headset attack or clinical throughput experiment.

The intended contribution is evidence about character-level operating points
and their limitations across users and attack families. Whether that evidence
supports a useful publication claim depends on the complete independent results.
The detector families, bootstrap methods and adaptive attack principles are
established. We do not claim a first P300 defense, first adaptive P300 evaluation,
a new uncertainty method or a certified robust detector.

## Related work

Targeted speller attacks and signal protection have different security
assumptions. Tiny Noise demonstrated targeted P300 manipulation
[@zhang2021tinynoise]. GPN and attention-guided universal perturbations provide
learned-generator and structured-perturbation precedents
[@jung2023gpn; @zhong2025auap]. They motivated development work here, but no
unlicensed GPN implementation was copied and an AUAP-style sparse mask is not
evaluated. Mezzina etal. examine synchronized transformation-based protection
under a trusted transmission architecture [@mezzina2021cybersecure]. That
architecture is distinct from post hoc signal-only anomaly detection.

Zhang and Gu compare confidence, entropy, Bayesian uncertainty and kernel density
estimation on CompetitionIII P300 and motor-imagery signals, including
detector-aware CW against KDE [@zhanggu2024detection]. Their study establishes
prior P300 detection and adaptive-evasion evidence. Its reviewed protocol
emphasizes flash-level attack ranking and balanced classification rather than
independently clean-calibrated character rejection and all-attempt spelling
reliability. Our conventional negative-Gaussian KDE is an explicit adaptation:
the printed positive exponent conflicts with the article's low-density rationale.
It is not an exact reproduction claim.

Chen, Meng, Xu and Wu's expanded journal study evaluates ERN, motor-imagery and
RSVP epochs using supervised clean/attack logistic detectors, subject-held-out
evaluation and nested AUC tuning [@chen2024artifacts]. Its methods explicitly use
AUC to avoid selecting a detection threshold. The journal protocol was reviewed
separately from the earlier Chen and Wu preprint [@chen2022artifacts].
Clean-only character calibration and its observed operating point therefore
represent a different estimand from the reviewed journal experiment; this does
not establish uniqueness across the full literature.

Chen, Jia and Wu benchmark nine defenses across five datasets, including an
eight-participant P300 dataset, three networks and within-/cross-session
online/offline scenarios [@chen2025benchmark]. Their epoch-level balanced accuracy
and targeted attack success establish broad P300 defense and session-shift
precedents. Conventional detector/ANFIS comparisons [@costa2026comparison] and
Covariance Entropy detection with compute reporting [@khilnani2025coven] further
limit broad detector-comparison or efficiency novelty claims. The latter review
uses accepted-manuscript HTML; the final PDF has not been visually verified.

Our frozen seven-method set is nonexhaustive: it omits Covariance Entropy and
journal cosine-distance variants. Conclusions must therefore concern these
specified adaptations and settings. The source-by-source reading scope and
protocol comparison are retained in
results/literature_protocol_matrix_20261022.md. The reviewed protocols do not
evaluate the same combined clean-only character operating-point, all-attempt
manipulation and rejection-cost estimand. This bounded distinction is not a
universal priority proof. Independent results must establish the magnitude and
practical implications beyond known adaptive failure.

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

## Independent methods: prespecified protocol

The exact prospective settings are frozen in
configs/openbmi_independent_protocol_frozen_20261022.json, with cloud-generated
immutable allocations, targets and code-bound receipts in
splits/openbmi_independent_20261022. Settings and artifacts were committed before
independent launch. Of51 fixed subject-seed tasks, forty-four have completed and their compact provenance
is preserved; five further task versions are active or queued on the five
private Kaggle accounts. Task01 has a preserved terminal failure with no exposed
log or outputs and one identical-payload retry. No scientific setting changed. Existing historical splits and failed gates
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
The OpenBMI dataset/toolbox source is Lee etal.2019 [@lee2019openbmi].
Compute clean calibration scores without true-character selection: average flash
anomaly over groups supporting the decoder-predicted symbol. The higher empirical
quantile at nominal1/5/10% defines rejection strictly above the threshold, retaining
ties. At20 decisions, nominal1% and5% both select the maximum; nominal values are
not per-user guarantees. Clean-reference calibration has established statistical
precedents [@bates2023outliers]; our empirical-quantile rule does not establish
their calibration-conditional or multiple-testing guarantees under phase shift.
Dropout calibration and every reported clean/attacked
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

FGSM and projected-gradient attack ideas are established [@goodfellow2014fgsm;
@madry2017robust]. Adaptive evaluation is required methodological practice
[@tramer2020adaptive], not a contribution by itself. Stochastic gradient averaging
uses the expectation-over-transformation idea [@athalye2018eot], without a
physical-world robustness claim. Detector-aware PGD combines targeted cross-entropy and a softplus detector penalty
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

Describe per-user two-sided95% exact binomial intervals [@clopper1934binomial]
under IID assumptions,
not serial-dependence-proof assurance. Average all three seeds within each user,
then use equal-user2000-resample percentile bootstrap(seed20261013) for
population mean summaries, using established resampling ideas [@efron1979bootstrap].
Report paired pointwise differences versus confidence
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

Complete independent benchmark outcomes, a refreshed closest-paper scope check,
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

### Full development subject19 reporting check, 22 October

The completed36-character validation19 run was processed by the cloud-only
reporter across all three seeds. At budget0.20, seed-averaged accepted targeted
manipulation under ordinary PGD was100% for confidence/entropy/BU,52.78% for LID,
and0% for MD/adjustedMD/KDE. Under each detector's separate fixed aware recipe,
the reported rate was100% for every attacked detector. This is not one shared
perturbation evading all seven. No threshold, method or source seed was selected
from these observations. All nominal settings and method/budget rows are retained
in results/report_checks_v2/development_reliability_report.md.

These are one already-inspected user's development observations, not17-user
independent findings or108 independent clean characters. No population confidence
interval is reported from one user. This extends the earlier fixed-character
audit's coverage but does not establish a new adaptive-failure phenomenon or
journal readiness. The independent cohort remains to be completed and summarized.
