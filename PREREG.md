# Pre-registration: Paper 4

**Committed:** 2026-09-25, before any decoder or attack was trained. Changes after this date go in the Amendments section with a date and reason; the original text is not edited.

## Hypotheses
- **H1 (attack):** at matched targeted success and matched Linf budget, prototype-guided GPN perturbations are harder to detect and more physiologically plausible than (a) target-only GPN and (b) Tiny Noise templates, including with cross-subject prototype banks.
- **H2 (defense):** the genuine-vs-induced ERP detector detects Tiny Noise, PGD, and target-only GPN attacks with TPR >= 0.80 at 5% FPR, and we report (not assume) its TPR against prototype-guided attacks.

## Primary outcomes
- **H1:** AUC of the clean-vs-attacked CNN detector at 80% targeted success, cross-subject bank, per subject, averaged over subjects. Direction: lower is stealthier.
- **H2:** TPR at 5% FPR of the genuine-vs-induced detector, per attack family.

## Secondary outcomes
L2 / RMS / SNR at matched success; area under success-vs-eps curve; surrogate-to-victim transfer success; ERP waveform correlation, P300 latency and amplitude, topography correlation, PSD distance; false-alarm rate of the defense on clean users.

## Fixed settings
- **Budget grid (Linf):** eps = {0.05, 0.10, 0.20, 0.30, 0.50} x the subject's median per-channel standard deviation of raw training EEG.
- **Matched-success levels:** 50%, 65%, 80%. The 80% level is primary. A method that never reaches a level is reported at its maximum success, not extrapolated.
- **Attack targets:** 10 pre-drawn symbols per test trial, never the true symbol (`splits/*_attack_targets.json`, seed 20260925 + subject code). Same targets for every method.
- **Splits:** Comp III official 85 train chars -> 70 train / 15 val (seed 0); official 100 test chars -> test (`splits/comp3_*.json`). Test data is not used until final evaluation.
- **Surrogate:** EEGNet. **Victims:** DeepConvNet, ShallowConvNet, xDAWN+LDA. Victims are never used for selection or tuning.
- **Defense training pool:** PGD, target-only GPN, random noise. **Held out from defense training:** Tiny Noise, all prototype-guided variants.
- **Seeds:** decoders 3 seeds; attacks 1 seed with bootstrap CIs over trials.

## Statistics
Per-subject results; bootstrap 95% CIs (2000 resamples over trials); paired Wilcoxon signed-rank across subjects for (proto-GPN nearest vs target-only GPN) and (proto-GPN nearest vs Tiny Noise); linear mixed-effects with subject and dataset as random effects; Holm correction across the secondary outcomes.

## Decision gate (after the Comp III A <-> B kill test)
- H1 primary outcome favors proto-GPN (nearest) over both target-only GPN and Tiny Noise, and the wrong-letter control does worse than nearest: **full attack + defense paper**.
- Otherwise: **defense-led paper**; prototype results reported as a negative result and adaptive stress test.
- If the defense also fails on its training attack families (TPR at 5% FPR < 0.5): stop and reassess.

## Amendments

### 2026-09-29: post-hoc authenticity-defense rescue

The original kill test was unblinded before this amendment. H1 failed and the
original H2 implementation produced held-out clean false-alarm rates near 50
percent despite selecting its threshold for 5 percent on the training fold.
The following analysis is explicitly exploratory and does not replace the
pre-registered H1 or H2 result.

Before running the rescue, we fixed an interpretable authenticity detector with
temporal-shape, P300-window, spatial-dispersion, spectral, and repeat-consistency
features. Evaluation uses five folds defined by test trial index modulo five.
For outer fold k, fold (k+1) modulo five contains genuine calibration samples
only, the remaining three folds fit an L2 logistic regression with C=0.1 and
balanced classes, and fold k is evaluated. Thresholds are fixed from the clean
calibration scores at nominal false-alarm rates 1, 5, and 10 percent. We report
both calibrated held-out TPR and observed held-out clean FAR for every attack.

The bounded rescue is viable only if both subjects have observed FAR no greater
than 10 percent and median calibrated TPR on the registered training families
no lower than 70 percent at the nominal 10 percent operating point. A stronger
Q1-candidate signal requires observed FAR no greater than 5 percent and median
TPR no lower than 80 percent on held-out attack families at the nominal 5
percent point. Failure of the viability gate stops defense expansion.

### 2026-09-29: external defense-validation protocol

This amendment was written after all Comp III results were unblinded but before
any attack or defense outcome was computed on BNCI 2014-008 or BNCI 2015-003.
Comp III is development data from this point forward. BNCI 2014-008 is the first
external validation dataset; within each of its eight subjects, the first 70
percent of character trials in recording order are training, the next 15
percent are validation, and the final 15 percent are test. BNCI 2015-003 is the
locked confirmation dataset: its official train recording is available for
source-subject training and target-user clean calibration, and its official
test recording is used only after the detector and thresholds are frozen.

The defense is fit on source subjects, then adapted to a held-out target subject
using clean calibration data only. Attack labels, attack scores, and victim
outputs from the target subject are unavailable during adaptation. Thresholds
are fixed by the target user's clean calibration scores at nominal FAR values
of 1, 5, and 10 percent. Primary reporting uses observed held-out FAR and TPR,
with subject and character as the units of inference.

The Q2 gate requires observed FAR no greater than 10 percent for every external
dataset, median held-out-family TPR of at least 70 percent, and improvement over
the registered distance and spectral baselines. The Q1-candidate gate requires
all three datasets, observed FAR no greater than 5 percent for every dataset,
median held-out-family TPR of at least 80 percent with no subject below 70
percent, plus a defense-aware adaptive attack and clean spelling-cost analysis.
If the frozen detector fails the Q2 gate on BNCI 2014-008, BNCI 2015-003 test
outcomes will not be used for tuning.

### 2026-10-04: defense-only scope and budget-identity correction

The user selected a defense-led manuscript, with a one-week submission target.
This is a target for preparation, not a guarantee of acceptance or publication.
Existing attacks remain evaluation stress tests; no attack novelty is claimed.
The original failed H1 result and prior calibration failures remain recorded.
Before selecting a new method or research question, compare the closest existing
defense studies and document the specific candidate gap.

The exploratory cross-subject evaluator keyed feature sets by an attack label
without epsilon. Multiple budget files overwrote one another, and the retained
metrics were reused across budget rows. Source fitting also reused those sets.
Historical budget-specific cross-subject claims are withdrawn pending rerun.
The correction uses unique manifest filenames, rejects duplicate filenames,
and has a regression test that verifies distinct source and evaluation budgets.
All tests and corrected evaluation run on Kaggle. Previously inspected data
remain development evidence. BNCI2015-003 remains locked. Existing scientific
gates are unchanged, and no positive result or journal quartile is guaranteed.

### 2026-10-04: independent-normalization clean-only repair audit

The prior BNCI2014-008 clean outcomes are already known. This diagnostic is
exploratory, not fresh external confirmation or a replacement primary outcome.
Keep all frozen splits, source model settings and seed unchanged. Fit target
preprocessing and feature normalization on the 24 training characters; select
the empirical 5% threshold using only five validation characters; inspect six
test characters. Do not tune on this repeat's outcomes. Report normalization
sample count and observed FAR per subject, without claiming a precise 5% FAR
guarantee or attack TPR. The original calibration-fit behavior is preserved as
the CLI default. All tests and processing run on Kaggle. BNCI2015-003 stays
locked. This isolates a fitting-stage issue; it is not claimed as a novel method.

### 2026-10-04: fixed feature-baseline development diagnostic

After the independent-normalization BNCI audit failed, compare three fixed
conventional scorers on inspected Comp III feature exports: logistic C=0.1,
random forest with 200 trees/minimum leaf 10/sqrt features/balanced classes,
and clean-only Ledoit-Wolf Mahalanobis distance. Seed 20261004. No target attack
is used for fitting or hyperparameter choice. Target normalization uses the
three genuine folds excluding evaluation and threshold-calibration folds.
Thresholds retain the existing empirical quantile at nominal 1/5/10 percent.
All attack/budget identities remain separate; report all rows and AUC plus
observed FAR/TPR, including held-out Tiny Noise and prototype variants.
No winner is claimed as a confirmed method from these unblinded development
results. This diagnoses nonlinear separability and conventional anomaly
baselines; it is not a reproduction of Chen/Wu's network-feature detector.
Their study already evaluates EEG detector approaches, so this comparison is
not claimed as novel: https://arxiv.org/abs/2212.00727
All tests/computation run on Kaggle; BNCI2015-003 stays locked.

### 2026-10-04: baseline diagnostic seed deviation disclosure

After outputs were available, inspection found the inherited Kaggle runner
explicitly passed seed 20260929 rather than the amendment/default 20261004.
The completed v2 comparison is therefore a disclosed protocol deviation and
exploratory diagnostic, not an exact amended-seed run. Preserve its outputs
and report their actual configuration; do not select a winning method or
claim confirmation from these results. Existing gates remain unchanged.

### 2026-10-04: score and information-loss audit

Instrument the fixed comparison with per-fold calibration/clean/induced score
quantiles, extrema, threshold values, exceedance counts and ties. Keep the actual
completed v2 seed 20260929 and every model/fold/budget setting unchanged to
isolate instrumentation. No threshold is changed or chosen from attack scores.
Test the existing feature extractor's invariance to electrode permutation and
window order using synthetic inputs on Kaggle. This tests lost identity/order,
not whether a new representation improves detection. No replacement method is
selected. Prior EEG detector work already exists (Chen/Wu, JNE 2024); report
this as development diagnosis, not novelty. BNCI2015-003 remains locked.

### 2026-10-04: bounded channel/time representation diagnostic

Prior-work review confirms EEG adversarial detectors and spatial/temporal EEG
processing already exist (Chen/Wu JNE 2024, Tiny Noise NSR 2021, cybersecure
fake-P300 defense 2021). No first-spatial-defense or new-method claim is made.
Export existing summary features plus 12 chronological averaged-ERP bins per
channel, preserving artifact channel indices. This is an ablation of lost
channel/time identity, not individual-flash or repetition-sequence recovery.
Keep all channels; no label-based electrode/time selection. Original summary
features remain the paired control. Fixed logistic C=0.1, source training pool,
independent target normalization/calibration/evaluation folds, all budget/attack
identities and seed 20260929 are retained for subsequent comparison. Export
derived features only; raw waveforms remain on their existing Kaggle accounts.
No raw EEG is downloaded. Cross-montage transfer is not tested by this ablation.
This is exploratory inspected Comp III development; no target-output tuning
or confirmation claim. Tests/processing run on Kaggle. BNCI2015-003 stays locked.

### 2026-10-05: OpenBMI metadata and sample-size feasibility audit

Fixed subject1, sessions1 and2, both official ERP phases. This subset is selected
by index before outcomes, not by decoding or detection performance. Kaggle-only
audit retrieves public source files in cloud working storage, records hashes,
schema, channel metadata, event alignment and documented versus inferred
60-flash block counts. Do not treat inferred blocks or Target/NonTarget labels
as verified symbol membership or independent decision identifiers. No attack,
detector fitting or performance evaluation occurs. Raw EEG is never downloaded
locally or copied into output artifacts. Report IID zero-alarm confidence bounds
and exchangeable rank resolution as design assumptions, not EEG guarantees.
Only focused audit assertions execute; this is not a full-repository test run.
Existing frozen splits/gates unchanged and BNCI2015-003 remains locked. Source
metadata: official MOABB Lee2019_ERP documentation and public OpenBMI deposit.

### 2026-10-05: OpenBMI ordering and decision-identity follow-up

Keep fixed subject1 and both sessions. Kaggle-only metadata diagnostics inspect
onset duplicates, class-run storage order, stable chronological gap summaries,
label vocabulary, strict indexing bounds and target counts in inferred60-flash
blocks. Sorting is diagnostic, not a validated loader transformation. No symbol
mapping is invented from Target/NonTarget labels; no attack/defense evaluation
runs. Raw EEG remains cloud-only, compact summaries only are retrieved. Existing
splits/gates unchanged and BNCI2015-003 stays locked. Original OpenBMI toolbox
and GigaScience documentation are references, not code copied into our runner.

### 2026-10-12: original OpenBMI schedule consistency audit

Fixed subject1 both sessions, cloud-only. Inspect original random_cell_order.mat
structure/coverage; compare known copy-spelling target labels with original and
chronological event-order schedule interpretations, without fitting decoders.
Count mismatches and within-block timing anomalies. Label agreement is necessary
but not sufficient to establish physical flash identity or indexing origin.
No memberships guessed, performance runs or frozen-file changes. Raw stays cloud;
compact summaries only. BNCI2015-003 locked. Exact runner fingerprint recorded.

### 2026-10-12: timestamp reversal and epoch alignment audit

Fixed subject1 both sessions. On Kaggle record original reversal positions and
sample differences. Compare provided smt epochs to x[t+ival+shift] for fixed
shifts-2,-1,0,1,2 and predeclared event indices0,1,59,60,last plus first3
reversal starts. Original prep_segmentation uses MATLAB one-based indexing.
Report errors/correlations without automatically choosing alignment or assuming
provided epochs are unfiltered. No model tuning/performance evaluation. Existing
gates/splits unchanged; BNCI2015-003 locked. Raw EEG never retrieved locally.

### 2026-10-12: constant-baseline alignment diagnostic

Prior raw/smt slices mismatch. Keep same fixed offsets/events and compute
per-channel temporal-mean-centered relative errors to isolate a constant offset.
Kaggle-only, no model fitting or chosen alignment. A remaining discrepancy
requires provenance/preprocessing review, not blind offset choice. Raw remains
cloud; compact results only. Existing gates/splits and BNCI2015-003 lock retained.

### 2026-10-12: bounded OpenBMI loader candidate validation

Preserve stored event order, original five-sequence schedule and documented
MATLAB index conversion. Candidate constructs raw character trials with known
copy-spelling targets and rejects count/label/schedule/boundary mismatches.
Validate synthetic ordering/index/negative/duplicate/label/end-boundary tests
on Kaggle, then predeclared subjects2/3 session1, both phases. Do not choose
subjects by performance. Trial identity includes session. No frozen splits
created/changed, no decoder/defense fitting. Timestamp reversal physical cause
remains unresolved; candidate cannot be labeled deployed/all-subject validated.
All tests/processing Kaggle-only; raw EEG/code tree excluded from retrieval.

### 2026-10-12: approved loader contract and sample-design cloud validation

User approved exact private source/scripts/tests payload to private Kaggle job
mustaqimnishat/p3a-openbmi-contract-design. Run portable suite with documented
raw/path exclusions. Compute one-sided95% exact bounds0/1/2 alarms and chance of
upper5% certification at IID rates1/3/5%, fixed n6/20/33/36/59/69/100/200/500.
Design sensitivity only, no EEG processing/model performance or frozen allocation.
All tests/calculations on Kaggle; compact outputs only. Existing gates/splits and
BNCI2015-003 lock unchanged.

### 2026-10-14: development-only real-data network execution smoke

Fixed subject1 session1 official training phase only. First10 characters fit
EEGNet for2 epochs, next3 reference activations, next1 query. Provided baseline-
adjusted epochs resampled1000->60Hz are an execution diagnostic, not final raw-
injection preprocessing. Scaling fits training characters only. Smoke8 dropout
draws/k20/epsilon.001 are fixed functionality settings, not benchmark choices.
Exercise BU/LID/shared-shrinkage MD/confidence-adjusted MD and decoder-predicted
character aggregation. Output finite/shape checks and provenance only, no
accuracy/FAR/TPR/AUC, no evaluation subjects or model/checkpoint output. Portable
tests run first on Kaggle. Raw stays cloud, compact JSON/log retrieval only;
old gates/splits unchanged, BNCI2015-003 locked.

### 2026-10-14: session1 all-subject structural coverage

After raw integration passes on development data, audit all54 documented
OpenBMI subjects, session1 train/test metadata, schedule-label consistency,
loader boundaries/IDs and fixed sampled epoch alignment. No model fitting,
performance metric or subject selection by outcomes. Hardened loader checks
remain mandatory; stop/report incompatibility rather than excluding subjects
silently. Compact metadata only retrieved, raw files removed after each subject.
This is not a performance freeze or confirmation run; old splits/gates and
BNCI2015-003 lock unchanged.

### 2026-10-15: outcome-independent metadata eligibility and reporting

All54 OpenBMI session1 subjects were structurally inspected before independent
model outcomes. Subject52 official train fails fixed-schedule label identity
(528 mismatches; no schedule-compatible symbol in any33 blocks); its test passes.
Retain all planned18 evaluation participants in feasibility/flow reporting,
explicitly mark52 calibrated-character estimand unavailable, never assign an
invented performance score or silently substitute a subject. Primary calibrated
analysis eligibility requires both phases pass unchanged identity/boundary checks.
If manifest verification agrees,17 assessable and1 unavailable must be disclosed;
available-case results do not establish all-person performance or random missingness.
Do not repurpose36 test characters for13norm/20cal to force a passing allocation.
No performance benchmark is unlocked by this amendment: full attack/statistical/
cost protocol and new immutable cloud manifests remain prerequisites. Source1-18
training proceeds under previously recorded settings. Old frozen splits/gates
unchanged, BNCI2015-003 locked. See results/metadata_eligibility_protocol_decision.md.

### 2026-10-18: prospective resource-bounded adaptive scope

After the full cloud workload estimate and concrete scope options were presented,
the user authorized the agent to choose what supports the Q1/Q2 submission goal.
Adopt optionB before any sealed participant performance: retain all17 eligible
evaluation users (18 in flow with52 unavailable), all36 test characters, all3
model seeds, all7 detectors, all5 budgets for clean/nonadaptive comparisons.
Detector-aware evaluation uses primary0.20, fixed weight1,200steps/3restarts;
400steps/5restarts sensitivity uses the fixed first official test character per
eligible user, every model seed and detector. Ordinary PGD is the weight0
comparator, not duplicated per detector. Preserve every setting and negative
result in the earlier development grid. This is one explicit adaptive recipe,
not the former full weight/budget grid or a worst-case robustness guarantee.
Resource rationale: results/resource_scope_decision_20261018.md and verified
cloud workload outputs. No threshold, user or defense selected from sealed
outcomes. New workload accounting, executable guards/tests and immutable final
protocol/manifests remain required before performance unlock. Historical frozen
splits/gates and BNCI2015-003 lock are unchanged.

## Independent reliability settings freeze, 21 October 2026 Central (22 October UTC)

Prospective settings are frozen in
`configs/openbmi_independent_protocol_frozen_20261022.json` (commit0535c5e),
with the completed gate review in `results/independent_release_review_20261022.md`.
This appends to, and does not modify, earlier frozen allocations or gates.
No sealed37-54 performance has been inspected. Subject52 remains unavailable,
not replaced; all18 remain in participant flow and17 are the conditional
calibrated estimand. BNCI2015-003 remains locked.

The existing optionB methods/budgets/targets/seed/allocation/adaptive recipe
are unchanged. This records explicit decoder/preprocessing/training settings,
equal-subject pointwise percentile bootstrap intervals after averaging model
seeds, and exact code-bound release requirements. Single-user development
results have no population uncertainty interval. Cloud v18 verified the
reporting implementation; new code-hash release-guard regression remains
required before cloud receipt generation. Each final manifest is generated
from original cloud metadata evidence against this protocol's exact bytes.
Execution is prohibited until newly generated manifests and code-bound receipts
are inspected and committed. A settings file alone cannot unlock performance.
Changing execution code requires a new prospective provenance release; preserve
previous receipts and completed outcomes, never silently rebind a changed run.

Novelty claims remain conditional: expanded JNE2024 methods have not been
verified and preprint content must not be attributed to the journal version.
The study measures character-level clean rejection, accepted errors and
all-attempt manipulation with cost under a recorded-marker offline threat;
it does not claim a novel detector, first adaptive P300 evaluation or certified
robustness. Independent findings, including negatives, determine manuscript
claims rather than an outcome-selected headline or method winner.


### Amendment: strict numerical budget correction, 26 October 2026

After all51 original independent jobs completed, the unchanged aggregate rejected
actual perturbation norms exceeding their declared Python epsilon. Cloud audit
of1836 shards/49929 settings found13419 violations and no nonfinite rows. The
complete cloud summary established every violation equals upward float32
rounding of its declared epsilon, with maximum relative excess
4.065375469261503e-08. This is a numerical contract defect, not permission to
increase epsilon or add a tolerance. Original outputs and failure history remain
immutable evidence and are not silently accepted as satisfying the strict bound.

The prospective correction rounds an upward-rounded representable float32 bound
inward and compares actual representable raw differences in float64/Python
precision against the original epsilon. The record-time guard also avoids
implicit float32 conversion of a Python scalar. No attack steps, restarts,
weights, targets, detector thresholds, model seed, normalization or statistical
estimand changes. Fixed synthetic/cloud contract tests precede corrected EEG
execution and new code-bound receipts are required. Old receipts are never
rewritten or rebound.

The corrective rerun scope is every declared setting and character for each
subject-seed task containing any strict numerical violation in the complete
audit. It is selected exclusively by numerical validity, before combined
performance summaries are accepted, and never by success, detector outcome or
favorable seed. Numerically valid original tasks are preserved and may be reused
under an explicit canonical lineage map. Corrected tasks replace whole tasks in
that canonical analysis, without mixing rows within a task. Preserve all original
and corrected versions separately. A new prospective release must bind corrected
source inventory, exact affected-task list and original audit/report hashes;
remaining original settings and protocol semantics stay fixed. No corrected
performance run is authorized merely by this amendment or a synthetic pass.
Complete cloud provenance review and strict all-row budget checks remain required
before combined reporting. Disclose this post-execution numerical correction and
both original failure and corrective execution in the manuscript/supplement.

### Amendment numerical sensitivity analysis option B 26 October 2026

The author now requests disclosure of floating-point tolerance and validation of
its effect. Before inspecting paired comparison outcomes, a separate descriptive
analysis fixes tasks10 and15, the two complete corrective tasks preserved at this
checkpoint. Selection uses availability, not performance. Compare every character,
setting and detector against its original version, with exact identity, target,
protocol, manifest and model/input hashes checked. Report prediction, targeted
success, nominal5 rejection and accepted-targeted decision changes, as well as
score differences. No task, setting or detector may be dropped because of changes.

The numerical diagnostic envelope is exactly max(epsilon, float32(epsilon)), not
a fitted absolute or relative slack. Report original strict violations separately;
corrective results must still satisfy the original strict epsilon. This comparison
is a post-execution sensitivity analysis and does not edit the primary aggregator,
old receipts, original bounds or the existing whole-task correction plan. No
harmlessness or cohort equivalence claim follows from the small excess or a
limited paired sample. Full-run differences also include possible stochastic or
hardware variation; this design does not isolate rounding causally. Any proposal
to reuse all original tasks under a revised numerical contract requires a separate
explicitly disclosed analysis release and appropriate validation, not silent
acceptance as originally preregistered strict results. Preserve all versions.

### Sensitivity reporting clarification 26 October 2026

The v1 paired report preserves all detector-score diagnostics for every attack
recipe. Its pooled detector-aware groups include recipes aimed at other detectors
and therefore are not the own-detector primary endpoint. Preserve that report.
Before claiming decision-level stability, add own-detector recipe summaries and
the original declared per-character short/long accepted-targeted union. Retain
all147 groups and72 paired characters per group for tasks10+15. This additive
reporting clarification does not select favorable outcomes, change attacks or
bounds, or establish population equivalence or a causal rounding effect.

### Author decision to discontinue further strict reruns 26 October 2026

The author explicitly declines further strict corrective reruns and chooses the
disclosed numerical-envelope analysis route. Do not launch additional strict
corrective tasks. Preserve all original and completed corrective outputs and any
already-running task provenance. The previous33task plan remains historical,
not an instruction to continue launching. This decision does not establish that
original outputs satisfy the old strict contract. Any tolerance-based full-cohort
analysis must be a separate code-bound post-execution amendment, retain the
original strict violations, use the exact epsilon float32 conversion envelope,
and disclose the limited two-task paired sensitivity evidence and its limits.
No performance-selected seeds, tasks or detector outcomes may be omitted.

### Revised numerical envelope analysis contract 26 October 2026

Following the author's decision against further strict reruns, the proposed full
cohort analysis retains all51original whole subject-seed tasks under a separately
labeled post-execution numerical contract. It accepts an applied raw norm only
when finite, nonnegative and no greater than max(original Python epsilon,
nearest float32 representation of that epsilon). This is not compliance with the
original strict bound and does not retroactively change original receipts.
Report all13419original strict violations and preserve original failure evidence.
No absolute/relative slack is fitted to results. Attack settings, nominal
thresholds, allocations, targets, seeds, denominators, participant averaging and
uncertainty procedures stay unchanged. The two-task paired sensitivity is limited
support, not proof of cohort equivalence or causal rounding-only invariance.

A separate Kaggle CPU review must verify original source inventory against the
old receipts, original portable regressions and new envelope guards, all51task
compact JSON/stage hashes and clean CSV threshold/decision/summary consistency.
It must create a new analysis receipt binding this contract, complete input
hashes and the separate reporter implementation. Inspect and privately preserve
that receipt before executing combined performance aggregation. Original strict
aggregator bytes remain unchanged. Scientific evaluation outputs and corrected
runs stay separate, with no mixture of favorable rows or best-seed selection.
If any hash, coverage or envelope check fails, preserve the failure and resolve
it without silently weakening that check. All scientific computation stays on
Kaggle; raw data/models and CSV must not be downloaded locally.
