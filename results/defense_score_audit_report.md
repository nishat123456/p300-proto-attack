# Defense score and information-loss audit

Completed Kaggle `mustaqimnishat/p3a-defense-score-audit` v1. Date: 4 October
2026 in America/Chicago (completion inspected at 21:50 local time).
**297 portable tests passed, 41 warnings**; all computation ran on Kaggle.
Instrumentation preserved baseline-v2 model/fold/budget settings and its actual
seed 20260929. No attack-conditioned threshold adjustment or model selection
was made. This remains inspected Comp III development evidence.

## Why informative AUC can coexist with zero PGD detection

Forest B-to-A PGD at budget 0.2 retains AUC 0.86845 but TPR 0% at nominal 5%
FAR and observed clean FAR 3%. The inspected diagnostics (`combo_007.npz`)
show clean calibration cutoffs above every attacked score in every fold:

| Fold | Clean-calibrated threshold | Maximum attacked score | Attacks at/above threshold |
| --- | ---: | ---: | ---: |
| 0 | 0.981363 | 0.943995 | 0 / 200 |
| 1 | 0.986046 | 0.948612 | 0 / 200 |
| 2 | 0.958730 | 0.943995 | 0 / 200 |
| 3 | 0.971406 | 0.943995 | 0 / 200 |
| 4 | 0.966778 | 0.943995 | 0 / 200 |

Many attacked scores repeat near 0.943995. AUC reflects ranking against the
whole clean distribution; successful ranking against most clean examples does
not imply exceeding its upper tail. The observations establish the immediate
reason for this miss. They do not uniquely identify the underlying causal
mechanism (forest partition behavior, representation shift, normalization, or
their interaction). This is not explained by equality ties at the threshold:
no attacked score equals it in these folds.

Forest B-to-A Tiny Noise at the same budget retains TPR 9%. The inspected
`combo_072.npz` diagnostics contain 27/8/19/7/29 attacked exceedances out of
200 per fold, at the same clean-calibrated cutoffs; equality counts are zero.
Thus most Tiny Noise attacks also lie below the required cutoff. These counts
are direct cloud outputs, not locally computed metrics.

Twenty genuine calibration characters force the current `higher` quantile at
nominal 1% and 5% to the same observed maximum. The quantile convention is an
explicit existing choice, not automatically a coding error. Lowering the
cutoff after inspecting target attacks would not be a valid repair or a
guarantee of maintained FAR.

## Prototype result remains weak

At budget 0.2 and nominal 5%, forest causal cross-subject nearest-prototype TPR
is 19.6% A-to-B and 32.5% B-to-A, with observed FAR 4% and 3%. B-to-A AUC is
0.8946, again illustrating informative ranking without useful upper-tail
sensitivity. The prior full budget/method comparison remains the performance
record; this instrumentation supplies a diagnosis, not a new winning method.

## What the information-loss test establishes

The Kaggle synthetic regression test passes when electrodes are permuted and
stored windows reordered: the current 36 engineered features remain equal
within tolerance. The extractor uses channel-aggregated moments, dispersion,
peak-latency summaries, averaged spectra, and symmetric window-consistency
statistics. It does not retain electrode identity or ordered window structure.
Its input ERP is already an average, so this compact export does not recover
individual flash responses. Stored random windows are not automatically the
stimulus-aligned repetition sequence.

This demonstrates a real representation limitation. It does **not** establish
that electrode-specific, flash-specific, or ordered features separate these
attacks, nor that the full waveforms are separable under low-FAR calibration.
Those claims require a separately frozen, prior-work-grounded evaluation with
waveforms retained on Kaggle and compatible electrode/timing semantics.

## Decision and safeguards

Do not fix the miss by moving thresholds using known target attack outcomes.
Do not describe forest as a validated defense. The next defensible step is a
prior-work comparison focused on representations and their calibration under
shift, followed by one bounded representation diagnostic if a useful gap is
identified. Existing EEG detector work already exists; neither score auditing
nor preserving electrode identity is itself a novelty claim.

BNCI2015-003 remains locked; splits/gates are unchanged. No replacement method
was selected or launched by this audit. No new external attack TPR or population
FAR guarantee is established. A Q1/Q2-ready positive defense claim remains
unsupported.

## Provenance

Source instrumentation/test amendment `cc16868`; exact submitted runner
`a67016d`; seed 20260929 explicitly freezes the actual previous run. Runtime
config reports immutable base bundle `97ef266ab027dcaef9718a6d075e6d2abf4c3867`;
the overlay record supplies the executed correction commit. Raw CSV/config,
per-fold diagnostic JSON, test scope, overlay provenance and cloud log are
retained unchanged under `results/defense_score_audit/`. Only compact outputs
were downloaded, never raw EEG or the working code tree. Local operations
were file inspection, artifact copying and Git/Notion reporting, not analysis.
