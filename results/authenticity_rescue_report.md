# P300 authenticity-defense rescue

**Run date:** 2026-09-29  
**Status:** Post-hoc exploratory rescue, frozen in the 2026-09-29 PREREG amendment before execution  
**Kaggle jobs:** `mustaqimnishat1/p3a-auth-rescue-a`, `mustaqimnishat2/p3a-auth-rescue-b`

## Verdict

**Mechanical viability gate: PASS.** At the nominal 10% operating point, the
observed clean FAR was 9% for subject A and 10% for subject B. The median TPR on
the registered training-family rows was 100% for both subjects, above the 70%
gate.

**Q1-candidate gate: FAIL.** At the nominal 5% operating point, clean FAR was
controlled at 5% for A and 3% for B, but median TPR on held-out attack-family
rows was 81.0% for A and only 22.6% for B. The required threshold was at least
80% for both subjects.

**Decision: continue only as a bounded defense study; do not begin the broad
Q1 multi-dataset expansion yet.** The calibration failure of the original
defense is fixed, but attack generalization is strongly subject dependent. The
next experiment must address cross-subject calibration and demonstrate that B
can reach the fixed-FAR target. If that focused repair fails, stop the defense
paper.

## Frozen-gate summary

| Subject | Nominal FAR | Observed FAR | Training-family median TPR | Held-out-family median TPR | Held-out median AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 5% | 5% | 100.0% | 81.0% | 0.986 |
| B | 5% | 3% | 100.0% | 22.6% | 0.852 |
| A | 10% | 9% | 100.0% | 94.9% | 0.986 |
| B | 10% | 10% | 100.0% | 49.3% | 0.852 |

Training families are PGD, target-only GPN, and random noise. Held-out families
are Tiny Noise and all prototype-guided variants. Medians are across the
configuration and epsilon rows in `results/authenticity_rescue.csv`.

## Held-out behavior at nominal 5% FAR

| Subject | Attack family | Median calibrated TPR | Median AUC |
| --- | --- | ---: | ---: |
| A | Tiny Noise | 99.9% | 1.000 |
| A | Proto medoid | 88.3% | 0.991 |
| A | Proto nearest | 72.5% | 0.975 |
| A | Proto random | 84.7% | 0.988 |
| A | Proto wrong | 88.4% | 0.992 |
| B | Tiny Noise | 100.0% | 1.000 |
| B | Proto medoid | 23.7% | 0.863 |
| B | Proto nearest | 17.0% | 0.823 |
| B | Proto random | 17.0% | 0.790 |
| B | Proto wrong | 18.9% | 0.830 |

The detector reliably catches Tiny Noise and the registered gradient-based
attacks. Its failure is specific to prototype-guided transfer on subject B,
where AUC remains moderately informative but the clean-calibrated threshold has
poor sensitivity.

## Interpretation and limits

- The original 49-57% clean FAR is reduced to 3-5% at the nominal 5% point, so
  the nested clean-calibration design solves the immediate calibration defect.
- Subject A satisfies the stronger held-out-family signal. Subject B does not,
  which prevents a Q1-ready claim and makes a two-subject average misleading.
- Random noise produced too few successful induced trials for stable
  evaluation: it contributed one evaluable fold and 0% TPR. The mechanical
  training-family median still passes because PGD and both target-only variants
  are detected at 100%, but random noise must not be described as validated.
- Each calibration fold contains 20 genuine trials. The 1% and 5% nominal
  thresholds therefore collapse to the same empirical upper-tail threshold;
  this dataset cannot resolve a 1% operating point reliably.
- This is a post-hoc result on two subjects from one dataset. It does not yet
  establish cross-dataset generalization, adaptive-attack robustness, or a
  publishable clinical false-alarm rate.

## Next bounded experiment

Use only training and validation data to fit a subject-normalized or
cross-subject calibration layer, then rerun the same untouched test folds. The
go criterion remains at least 80% median held-out-family TPR with no more than
5% observed FAR for each subject. No new attack development or multi-dataset
expansion should start until subject B meets that criterion.
