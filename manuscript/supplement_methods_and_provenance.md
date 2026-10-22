# Supplement draft: methods, provenance and claim limits

October21,2026 Central. Professor-review draft, not submission-ready. Independent
performance summaries, full uncertainty/cost results and final figures remain
pending. Protocol/manifest freeze is complete and independent tasks are running. Software completion is not scientific efficacy.

## A. Current detector implementation

All reference fitting uses source subjects1-18 official train only. Network
features are the final flattened EEGNet encoder (48 dimensions in executed
OpenBMI run). Up to5000 references are sampled without replacement by fixed
subject/flash-class quotas and seeded ordering. For deployment scoring, average
flash anomaly among groups supporting the decoder-predicted symbol; intended
symbol is not a score input. True copy-spelling labels are used for research
identity checks/evaluation only, not for selecting the scored symbol.

|Family|Executed implementation|Reproduction limitation|
|---|---|---|
|Confidence|Binary1-max(p,1-p), higher more anomalous|Conventional control with our character aggregation|
|Entropy|Binary softplus(logit)-logit*sigmoid(logit)|Conventional control, not a new detector|
|BU|Summed class-probability variance; dropout only, BN frozen;100draw clean calibration/reported scores|Different decision unit/reference protocol; gradient EOT30 is not a reported-score substitute|
|LID|Euclidean k20, -k/sum(log(r_i/r_k)) against source references|Zero/degenerate distances rejected explicitly, no silent epsilon repair|
|MD|Nearest source-class squared feature distance, shared LedoitWolf residual precision|Shrinkage/final layer adaptation, not exact empirical-covariance/multilayer reproduction|
|Adjusted MD|Same distance after normalized-epoch step-.001*sign(distance gradient)|Gradient input adjustment implemented; no learned multilayer aggregation|
|KDE|Class-conditional conventional negative Gaussian log-density, bandwidth.5, predicted flash class|Sign-corrected adaptation of inconsistent printed equation; no author-code reproduction claim|

Source/family equations and overlap: results/reference_baseline_specification.md
(historical checkpoint), results/zhang_gu_2024_fulltext_review.md (later verified
primary fulltext). The reviewed Chen/Wu2022 preprint motivates BU/LID/MD; the expanded2024
journal version (Chen/Meng/Xu/Wu) has not been fully verified. Do not use
preprint details as proof of the journal dataset/calibration scope; Zhang/Gu2024 already
compares P300 confidence/entropy/BU/KDE and detector-aware attack. Chen/Jia/Wu2025
is a broad benchmark precedent. Neither first detector, first benchmark nor
first adaptive P300 evaluation is a contribution here. Remaining novelty review
must establish a substantive independently calibrated reliability/cost gap.

## B. Partition and calibration contracts

Development1-18, validation19-36, independent37-54; fixed3model seeds20261013,
20261014,20261015. Frozen target allocation:13 normalization and20 calibration
from33 official train characters, all36 official test characters for evaluation,
allocation seed20261013. Raw channel statistics fit normalization characters
only. Thresholds use20 independent calibration decisions and higher empirical
quantile at nominal1/5/10%, rejection strictly greater with ties retained.
Nominal1% and5% thresholds coincide at this sample size. MAD score scale has
fixed1e-6 floor, not a target-attack-dependent scale.

Independence of file/decision roles prevents reuse leakage; it does not prove
exchangeability. The train phrase NEURAL_NETWORKS_AND_DEEP_LEARNING differs from
test PATTERN_RECOGNITION_MACHINE_LEARNING. Character frequencies, recording
phase and time therefore differ. Any IID or exchangeable calibration statement
must remain an assumption/sensitivity, not a measured guarantee. Repeated flash
samples/attacker targets/seeds do not become independent users.

Subject52 train fails unchanged identity checks (528 mismatches; no compatible
symbol in33 blocks), although its test passes. Keep all18 planned evaluation
people in feasibility/flow and mark52 calibrated outcome unavailable. Available-
case17-person performance cannot establish all18-person reliability or missing-
at-random. No inferred phrase repair, replacement participant or reuse of test
characters for calibration. Old frozen splits/gates remain unchanged and
BNCI2015-003 stays locked. New independent manifests/targets and code-bound receipts are frozen under
splits/openbmi_independent_20261022; original historical files are unchanged.

## C. Recorded-marker threat and attack units

Raw1000Hz,0.5-20Hz/CAR, decimation17 gives58.8235Hz and0.8s epochs. Stored marker
order is semantic schedule order; physical cause of first-pair reversal is
unresolved. This is offline signal access under recorded markers, not verified
photodiode onset, causal hardware protection or clinical real-time assurance.

Budget ratios.05/.10/.20/.30/.50 multiply median channel raw standard deviation
from normalization-only decisions. All perturbations apply before preprocessing.
The prospective OpenBMI budget definition is disclosed separately from historical
CompIII training-scale budgets; do not pool their raw magnitudes as identical.
Float32 budget checks use the actually applied raw sum difference. Projection
moves rounding violations one representable ULP inward; every audited optimizer
iterate and final perturbation must satisfy the declared bound.

The source template is a gradient-trained idea adaptation, not exact TinyNoise
reproduction. It uses800raw samples at1000Hz (0.8s),200Adam updates/.01/eight
individual variable-length source-trial gradients per update, no raw padding.
Train at primary.20 source raw scale and rescale the same template at each
budget. No target-user template training or selected best seed.

Current development audit fixes the first subject19 test character and a shared
wrong target, all3seeds, all baseline budgets and all7 detector-aware paths.
PGD200/3 and stronger400/5 plus weights0/.1/1/10 are retained. BU gradient EOT30
uses independent100draw reporting, calibrated at100draws. Every setting, finite/
zero-gradient flag, objective trace, actual budget and time is retained. This
one-character audit does not estimate population TPR or prove global attack
optimality. Full independent attack/generalization evidence remains required.

## D. Completed evidence and exact execution

|Stage|Source/runner|Compact evidence|Bounded result|
|---|---|---|---|
|Current broad portable suite before latest projection change|Runner268144f, result0f69ed8|results/portable_suite_v14|351passed,41warnings45.98s; four historical excluded raw/path files|
|Source training|Source381ac26, runner543cebf|results/source_training_v1|3seeds,30epochs,35640flash source-only; not accuracy evidence|
|Reference/template/calibration|Sourcef689536, runnera0b6cf6|results/reference_calibration_v1|5000x48references/templates/fixed19calibration;28tests16.66s|
|Fixed clean validation19|Source766ea82, runner01b89d6|results/validation_clean_v1|29tests13.70s,36/36correct eachseed; method/seed rejection0-3/36 at nominal5%|
|Real attack/convergence audit|Source3eefe17, runner9cae903|kaggle/jobs/openbmi_attack_audit|Complete,31tests; fixed-character individually adaptive evasion, evidence1582f56|

Every dataset/model/reference/schedule artifact is hash-bound in compact JSON.
Training losses and final checkpoint completion are not accuracy or security
metrics. Prior failed packaging/test runs remain preserved. Portable exclusions:
test_comp3.py, test_letter_head.py, test_bank_report.py and
 test_kaggle_kill_test_run.py, due documented historical raw/path limitations.
The latest applied-projection changes require their new cloud test evidence;
351 earlier passes are not silently relabeled as post-change verification.

## E. Reproduction and data handling

Exact runnable cloud launch folders are kaggle/jobs/openbmi_source_train,
openbmi_reference_calibration, openbmi_validation_clean and openbmi_attack_audit.
Read each payload-manifest and pinned revision before use. Completed own private
source/refcal artifacts are attached through Kaggle kernel_sources, never copied
through a local model/raw-data download. All tests, EEG preprocessing, fitting,
metrics and statistical analysis run on Kaggle. Local activity is source/document
editing, packaging, compact evidence reading/copying and git only.

Compact JSON/TXT/Markdown/logs are preserved in private GitHub. Per-character CSV,
raw EEG, models and NPZ features remain in owned private Kaggle outputs under the
current retrieval scope. Credentials are never printed or committed. Reproduction
must preserve full output provenance, all methods/seeds/negative findings, and
unchanged historical frozen artifacts. Do not submit, email or pay fees without
explicit user authorization.

## F. Completion audit: required evidence still missing

|Requirement|Current evidence|Remaining work|
|---|---|---|
|Specific literature gap|Closest primary studies reviewed; broad/adaptive precedents disclosed|Finish remaining protocol comparisons; demonstrate substantive independent contribution|
|Valid frozen independent protocol|Candidate settings and pre-outcome eligibility amendment|Final integration/source review and new immutable cloud manifests before sealed outcomes|
|Baselines/generalization|Source fitting and one validation user's clean results|Independent users, all planned methods/budgets/seeds, cross-session claims only if actually tested|
|Adaptive evasion|Fixed-character audit complete, each detector individually evaded|Independent stress tests and population uncertainty; preserve negative finding|
|Clean spelling cost/uncertainty|Accounting code and one-user IID descriptions|User-cluster uncertainty, genuine-error behavior, measured cost; no fabricated throughput|
|Manuscript/supplement/reproducibility|Draft methods/provenance and verified development failures|Real independent results/figures, complete sources/limitations and professor review|

No arbitrary detection percentage or passing software-test count establishes
Q1 readiness. If evidence cannot support the intended claim, preserve the gap
and required next work instead of declaring success.

## G. Dedicated execution integration update, 18 October

The validation-only dedicated runner integration completed across all3 seeds:
manifest-bound calibration,36 clean decisions, first-character approved-recipe
attack and a second invocation with unchanged checkpoint bytes. Current portable
scope passed363 tests/41warnings43.88s. Exact launcher06781cf/source dd7c11f;
results/execution_integration_v1_report.md and compact tree preserve evidence.
This replaces pending software integration status for those exercised paths,
not independent evaluation or a forced mid-setting crash test. No final sealed
release receipt exists. Source-query timing and complete aggregate/orchestration
remain pending; journal readiness is not established.


## Execution release, 21 October 2026 Central

Protocol settings0535c5e, PREREG append8f931ae, metadata release9ce8906, first
five-task launcher dc8a039. Protocol SHA256:
a5699d95b0d35343da207dec8d657f380d1fd49a7664424682cd6e7c910b9916.
All54 original metadata records were rechecked on Kaggle.53 metadata allocations
and17 evaluation receipts were generated;52 retained unavailable in participant
flow. Receipts bind exact protocol/manifest bytes and every source/script Python
file hash. Independent runners reject changed code; launcher wrappers remain
outside that scientific code inventory. All raw/model/reference/CSV artifacts
stay in private Kaggle outputs. Compact metadata and reports are retained in
private GitHub. Public reproducibility release/access arrangements remain pending.

Full fixed validation19 controller completed all36 characters/all3 seeds and
cloud aggregation; repeat completed-ledger SHA remained unchanged. This is not
proof of forced mid-setting crash recovery. Source/query timing completed on
Kaggle, with exact-query identity removed only from timing LID/KDE references
to avoid self-distance; deployment references are unchanged. Timing is not
clinical latency. Cloud regressions v18/v19 passed367/368 tests; same four
historical raw/path exclusions. These scopes establish implementation behavior,
not independent statistical efficacy or journal readiness.

Population means use equal participant weights after seed averaging and
pointwise percentile95% subject bootstrap (2000 resamples, seed20261013).
With one development participant the population interval is unavailable.
Intervals assume exchangeable participants and are conditional on17 available
calibrated users. Paired differences use shared nonadaptive perturbations;
separately attacked detector objectives are not comparative efficacy tests.
