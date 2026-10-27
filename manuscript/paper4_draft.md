# Clean rejection and adaptive manipulation in P300 spellers: character-level reliability of existing detectors

Working manuscript, updated27 October2026 UTC. **Not submission-ready.**

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
separate reliability from successful-attack-only detection. All51 subject-seed tasks completed. A separately disclosed post-execution
numerical-envelope analysis retains13,419 original strict-budget violations;
it does not establish original strict compliance. At nominal5%, mean genuine
character rejection ranged from3.92% to8.82% across the seven detectors, while
accepted wrong characters comprised36.00% to38.24% of all clean attempts.
At budget0.20, separate detector-aware attacks achieved94.01% to100% accepted
targeted manipulation. Participant-bootstrap intervals and all declared settings
are retained. These findings indicate limitations of these detector operating
points under the tested offline conditions, rather than a new robust defense.
The high clean-error burden limits practical interpretation. Two-task paired
sensitivity does not establish population equivalence or causal rounding-only
invariance.

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
attack audit. Independent-user outcomes are reported below under the revised numerical contract.

## Independent results under the revised numerical envelope

The complete cloud analysis retains all51 original subject-seed tasks for17
available users. Participant52 remains in the18-person flow with unavailable
calibrated performance: official training data had528 schedule/known-label
mismatches and no compatible symbol in any33 blocks under the fixed schedule.
No substitution or performance-based exclusion was made. The source-only release
review verified task identities, stage hashes, clean CSV decision consistency
and the exact numerical envelope before aggregation. It passed368 portable
tests and six envelope guards. The original strict failure remains disclosed;
these results use the separate post-execution contract, not the original strict
budget claim.

### Clean operating points

At nominal5%, the cloud report gives the following equal-user means after
within-user model-seed averaging. Intervals are pointwise95% participant-bootstrap
intervals, not simultaneous intervals or individual-user guarantees.

|Detector|Rejected clean characters, mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|
|---|---|---|---|
|Confidence|6.32% [4.52,8.17]|56.05%|37.64%|
|Entropy|5.83% [3.98,7.63]|56.15%|38.02%|
|BU|4.85% [2.34,7.90]|56.92%|38.24%|
|LID|3.92% [2.23,5.94]|58.17%|37.91%|
|MD|8.82% [3.92,15.09]|55.17%|36.00%|
|Adjusted MD|8.77% [3.92,14.92]|55.17%|36.06%|
|KDE|6.32% [2.94,10.13]|56.86%|36.82%|

Nominal1% and5% yield identical observed outcomes because both select the
maximum of20 calibration decisions. At nominal10%, the observed rejection means
range from8.39% for LID to13.13% for MD. The full report retains all operating
points, accepted-correct and accepted-wrong intervals. Genuine accepted errors
remain substantial for every tested detector. This limits the usefulness of the
underlying decoder-detector system regardless of attack rejection.

### Detector-unaware and detector-aware manipulation

At the primary0.20 budget, ordinary targeted PGD was accepted and successful on
100% of original attempts for confidence and entropy,99.89% for BU,85.62% for
LID,0.05% for MD and adjusted MD, and0% for KDE. The fixed own-detector adaptive
recipes produced the following rates over all original characters, including
unsuccessful attempts. The declared longer run contributes only its fixed first
character candidate to the corresponding character union.

|Attacked detector|Accepted targeted manipulation, mean [95% CI]|
|---|---|
|Confidence|100.00% [100.00,100.00]|
|Entropy|100.00% [100.00,100.00]|
|BU|100.00% [100.00,100.00]|
|LID|94.23% [82.68,100.00]|
|MD|94.12% [82.35,100.00]|
|Adjusted MD|94.12% [82.35,100.00]|
|KDE|94.01% [82.14,100.00]|

These are different attacks optimized separately against each detector. They do
not establish simultaneous evasion or a comparative defense ranking. Bootstrap
intervals collapsed at100% reflect observed resampling outcomes, not certainty
for future users. The contrast between MD/KDE rejection of ordinary PGD and high
own-detector adaptive acceptance limits any robustness conclusion from
nonadaptive attacks alone. All147 attack/budget/detector groups and120 paired
nonadaptive descriptive comparisons are retained without selecting a winner.
The full cloud report also retains FGSM, Gaussian controls and source-template
results across all five declared budgets.

### Secondary decoder, conditional detection and ranking summaries

A separate hash-bound CPU summary of the existing aggregate passed five reporting
guards. Mean clean decoder accuracy before rejection was0.601852, with pointwise
95% participant-bootstrap interval[0.430814,0.767443]. Conditional accuracy among
accepted clean characters at nominal5% was0.597051 for confidence,0.597046 for
entropy,0.596029 for BU,0.598654 for LID,0.597628 for MD,0.596906 for adjusted MD,
and0.605231 for KDE. Full intervals and accepted-decision denominators are
retained in results/secondary_summary_v1. These are averages of within-participant
seed summaries, not ratios of population mean rates. Rejection did not provide
high accepted-character accuracy in this decoder configuration.

For ordinary PGD at budget0.20, all-attempt nonadaptive AUC was0.995930
[0.991799,0.999047] for MD/adjusted MD and0.999425[0.998517,1.000000] for KDE.
These ranking results coexist with high own-detector adaptive accepted-targeted
rates reported above. They do not quantify adaptive AUC, which was not computed.
The conditional TPR among successful ordinary PGD attacks was0.999455
[0.998366,1.000000] for MD/adjusted MD and1.000000[1.000000,1.000000] for KDE;
the full report retains every attack group and its successful-attempt denominator.

The all17participant conditional adaptive LID TPR summary is unavailable because
participant51 has an undefined seed-level conditional denominator. That identity
is retained, and any defined-only descriptive summary is explicitly conditional,
not silently presented as complete-cohort performance. This does not remove that
participant from the primary all-original-attempt accepted-targeted endpoint.
Undefined conditional outcomes are not imputed as zero. All nominal settings,
attack groups, participant/seed records and undefined-case identities remain in
the secondary JSON and Markdown report. Secondary pointwise intervals are not
familywise or serial-dependence guarantees.

### Computation and rejection cost

The cost report retains every participant/seed/nominal operating point and its
stored ideal IID retry expression. It is not a measured repeated-spelling or
backspace-correction cost. Fixed source-query timing on Tesla T4 had BU medians
0.062255to0.064205seconds across seeds, compared with confidence medians
0.006543to0.007710seconds. These are synchronized batch1 source-query timings,
not independent-user latency or clinical throughput. Recorded first-to-last
marker plus0.8second extrema remain descriptive duration proxies. No product of
mean retry and mean timing is claimed as observed throughput, and no population
retry confidence interval is inferred from the per-seed table.

All numerical values in this section are transcribed from the Kaggle-generated
reports in results/envelope_aggregate_v1/out. Aggregate SHA256 is
e11fb26676c22c548074ecbf8113f4aa427a815df2723c383effb277cf31e9ce.
Actual browser visual review of the two cloud-generated figures remains pending;
their generation and hashes alone do not establish layout quality.

## Independent methods: prespecified protocol

The exact prospective settings are frozen in
configs/openbmi_independent_protocol_frozen_20261022.json, with cloud-generated
immutable allocations, targets and code-bound receipts in
splits/openbmi_independent_20261022. Settings and artifacts were committed before
independent launch. All51 fixed subject-seed tasks have completed and their compact
provenance is preserved. The unchanged strict aggregator stopped on an actual-budget check. Separate
post-execution numerical-envelope cloud aggregation and reporting are complete;
the original strict failure is retained. Cloud auditing traced the violations
to upward float32 rounding of the declared epsilon; a stricter implementation
passed373 portable tests and a separate source-only raw projection/release review.
The initial corrective plan specified33 affected whole-task reruns and18 valid
original tasks. The author subsequently declined further strict corrective
launches and chose a separately disclosed post-execution numerical-envelope
analysis of all51 original tasks. The original strict failure and completed
corrective evidence remain preserved. This revised analysis permits norms up to
max(original epsilon, its nearest float32 representation); it must not be
described as satisfying the original strict bound. Before combined analysis, a separate CPU release review verified original
source and task hashes, full coverage, exact envelope validity and clean-score
CSV consistency. Its new receipt was privately preserved before aggregate launch.

Paired sensitivity of the two completed corrective tasks retained1958 settings,
all147 groups and72 paired characters per group. Decoded predictions, targeted
success, own-detector recipe rejections and the declared accepted-targeted
character unions were unchanged. Cross-detector diagnostic rejection changes
remain recorded. This limited comparison does not isolate rounding causally or
prove equivalence for the full cohort. No further strict reruns are planned;
the separately labeled full-cohort results are reported above. Task01 has a preserved
terminal failure with no exposed log or outputs and one identical-payload retry.
No scientific setting changed. Existing historical splits and failed gates
are preserved; BNCI2015-003 remains locked.

### Participants, identity and missingness

Development subjects1-18, validation19-36 and evaluation37-54 are assigned by ID,
with model seeds20261013/14/15 and no best-seed selection. Of18 intended evaluation
participants,52 lacks a usable official training identity under the fixed loader.
Keep this person in feasibility/participant flow, with calibrated performance
unavailable. Cloud metadata verification establishes allocation eligibility for17
people only; their available-case performance is reported above. This is pre-outcome metadata eligibility, not exclusion by
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

### Interpretation of the independent findings

The result supports a bounded reliability audit of these seven adaptations, not
an effective new defense. A detector may reject ordinary optimization while
accepting most separately optimized detector-aware attacks. This adaptive
vulnerability is established prior methodology; the contribution here is its
joint characterization with genuine character rejection, accepted decoding
mistakes and explicit cost assumptions under clean-only user calibration.

The large accepted-clean-error burden is a major limitation, not a result to
remove by conditioning on correctly decoded characters. Together with17
conditional available cases, one unavailable participant, one recorded dataset,
coarse calibration, marker-timing uncertainty and fixed attack recipes, it
limits deployment implications and generalization. No result establishes
cross-dataset, cross-session or physical attack effectiveness. The post-execution
numerical contract further limits confirmatory interpretation. Small bound
excesses and stable paired outcomes in two tasks do not establish all-cohort
counterfactual equivalence under strict projection.

## Conclusion draft

Across the17 available held-out OpenBMI participants, the tested clean-calibrated
operating points left substantial genuine decoding errors accepted and allowed
high detector-specific adaptive targeted manipulation. These findings caution
against inferring character-level reliability from nominal calibration or
nonadaptive detection alone. They concern the specified detector adaptations,
decoder, offline signal-access assumptions and separately disclosed numerical
envelope. They neither establish a new robust detector nor guarantee reliability
for future users. Final figure review, reproducibility access and professor
scientific review remain required before submission.

## Required before submission

Actual independent figure visual review, a refreshed closest-paper scope check,
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
claim from this example; the independently held-out cohort findings are reported above.
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
journal readiness. The complete independent cohort is summarized above under the revised numerical contract.
