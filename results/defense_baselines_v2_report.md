# Fixed feature defense comparison: exploratory Comp III

Kaggle `mustaqimnishat/p3a-defense-baselines` v2 completed 4 October 2026.
Remote portable tests: **295 passed, 41 warnings**. Exclusions are documented
in `defense_baselines_v2/cloud_test_scope.txt`. All tests and data analysis ran
on Kaggle. Tables below are direct readings of the cloud CSV, not local analysis.

Three fixed methods share feature identities, source training families and
independent target normalization/calibration/evaluation folds. Logistic uses
C=0.1. Forest uses 200 trees, minimum leaf 10, balanced classes. Mahalanobis fits
Ledoit-Wolf covariance to genuine source features only. This is a conventional
feature diagnostic, not a new method or reproduction of a published network
feature detector. No target attacks train or tune a model.

## Budget 0.2, nominal FAR 5%: TPR

| Detector / direction | PGD | Target causal | Target signal | Tiny Noise | Proto medoid causal | Proto nearest causal | Observed clean FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic A to B | 99.9% | 44.1% | 91.8% | 68.7% | 12.0% | 12.4% | 5% |
| Logistic B to A | 100% | 85.4% | 100% | 100% | 34.7% | 39.2% | 6% |
| Forest A to B | 100% | 49.8% | 96.0% | 84.3% | 19.5% | 19.6% | 4% |
| Forest B to A | 0% | 89.0% | 95.8% | 9.0% | 26.8% | 32.5% | 3% |
| Mahalanobis A to B | 96.3% | 2.9% | 14.9% | 24.0% | 2.6% | 2.6% | 2% |
| Mahalanobis B to A | 100% | 20.8% | 80.2% | 62.9% | 6.9% | 7.4% | 6% |

Prototype rows use cross-subject banks. Signal-specific variants, same-subject
banks, random/wrong-letter controls, noise and all nominal FARs are preserved
in the full CSV. Random noise has too few evaluable folds for reliable claims.

## Full nearest-prototype causal cross-subject budget sweep

TPR at nominal 5% FAR, with identical method-specific clean FAR for these rows.

| Detector / direction | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 | Clean FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic A to B | 4.0% | 4.1% | 12.4% | 19.2% | 28.5% | 5% |
| Logistic B to A | 6.9% | 20.2% | 39.2% | 43.1% | 45.7% | 6% |
| Forest A to B | 8.3% | 7.5% | 19.6% | 27.0% | 38.4% | 4% |
| Forest B to A | 2.7% | 8.1% | 32.5% | 48.4% | 75.8% | 3% |
| Mahalanobis A to B | 0% | 3.7% | 2.6% | 2.7% | 3.2% | 2% |
| Mahalanobis B to A | 4.0% | 6.6% | 7.4% | 10.7% | 27.6% | 6% |

The forest's 75.8% at budget 0.5 in one direction is an easier, larger-budget
case; it does not establish the gate across families, budgets or subjects.

## AUC versus actual threshold behavior

At budget 0.2, nearest-prototype causal AUC is 0.759/0.848 for logistic,
0.701/0.895 for forest, and 0.505/0.688 for Mahalanobis (A-to-B/B-to-A).
An informative AUC does not ensure useful TPR at the clean-calibrated threshold.
In particular, forest B-to-A PGD has AUC 0.868 but TPR 0% at nominal 5%;
Tiny Noise has AUC 0.868 but TPR 9%. This failure must not be hidden by calling
forest the best detector. Score distributions/thresholds need separate audit
before selecting a replacement model. No hyperparameters were changed afterward.

## Scientific conclusion

There is no consistently adequate defense in this comparison. A nonlinear
classifier can improve some rows, but is not a reliable remedy and does not
establish that the features contain enough information for low-FAR detection.
Clean-only feature distance is particularly weak against causal prototype
attacks. Independent fitting stages improve the design, not necessarily the
outcomes. Avoid broad expansion of this architecture as a validated defense.

This is development evidence on two inspected subjects, not external validation,
novelty proof or a Q1/Q2-ready result. Per-fold calibration has only 20 genuine
characters, so nominal 1% and 5% empirical thresholds can coincide; observed
FAR is not a population guarantee. Existing gates remain unchanged.
BNCI2015-003 remains locked. Prior-work comparison must precede any specific
replacement defense direction. Further diagnosis should distinguish feature
information from score/threshold transport before additional training.

## Provenance and deviation

- Source/model/test amendment: `ca47e4d`; exact repaired runner: `c5cf52c`.
- v1 failed before scientific computation due to embedded-source packaging;
  v2 repaired this using base64 JSON. No v1 results exist.
- Runtime config reports immutable base bundle `97ef266ab027dcaef9718a6d075e6d2abf4c3867`;
  overlays are recorded separately in `overlay_provenance.json` and the runner.
- **Seed deviation:** the amendment and script default specified 20261004,
  but the inherited cloud runner explicitly passed **20260929**. These results
  use 20260929 and must not be described as fully matching the amended seed.
  The deviation is disclosed rather than silently altering historical records.
- CSV/config/fold details/test scope/overlay record/log are retained unchanged.
  No raw EEG or working code tree was downloaded.
