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
