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
