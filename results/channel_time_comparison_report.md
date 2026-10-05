# Paired channel/time representation diagnostic

Kaggle mustaqimnishat/p3a-channel-time-compare v1 completed, inspected
4 October 2026 America/Chicago (5 October UTC). **299 portable tests passed,
41 warnings in 49.59s**. All tests and scientific computation ran on Kaggle.
Only compact CSV/JSON/test-scope/log outputs were retrieved for this report.

## Fixed comparison

Logistic C=0.1, seed20260929, source attack training pool unchanged;
three target genuine normalization folds, separate threshold and evaluation
folds. Compare original36 summary features versus804 features (36 plus12
chronological averaged-ERP bins per64 channel indices). Every attack/budget
is retained in the CSVs, with fold diagnostics in JSON. Previously inspected
Comp III is exploratory development; no method was selected as confirmed.

Exact runner e4b1a23; overlays ad121be; runtime base bundle97ef266.
The runtime config's base Git hash does not describe the overlaid source.
Remote test exclusions are documented in cloud_test_scope.txt.

## Budget0.2, nominal5% false alarms

TPR percentages below are the reported fold means, conditional on successful
induced examples, not all attempted attacks. Observed clean FAR is5% A-to-B
and6% B-to-A for36 features;3% in both directions for804 features (five-fold
rows). Sparse noise rows have fewer folds and their own reported FAR.

| Attack |36 A-to-B |804 A-to-B |36 B-to-A |804 B-to-A |
|---|---:|---:|---:|---:|
|PGD|99.9|65.6|100.0|90.7|
|Target-only causal|44.1|58.3|85.4|26.0|
|Target-only signal|91.8|93.5|100.0|100.0|
|Tiny Noise|68.7|3.9|100.0|50.5|
|Proto-medoid causal cross|12.0|30.1|34.7|6.0|
|Proto-medoid signal cross|15.5|36.8|40.7|24.4|
|Proto-nearest causal cross|12.4|30.6|39.2|3.6|
|Proto-nearest causal same|9.2|33.7|40.6|27.7|
|Proto-nearest signal cross|11.6|32.2|38.0|21.2|
|Proto-nearest signal same|22.3|26.9|13.2|17.6|
|Proto-random causal cross|11.5|29.5|41.8|2.8|
|Proto-random signal cross|14.1|17.0|26.6|12.1|
|Proto-wrong causal cross|11.9|24.9|46.3|15.5|
|Proto-wrong signal cross|14.4|33.7|35.5|40.0|
|Random noise|0.0|0.0|0.0|0.0|

Nearest-causal cross-subject AUC at0.2 is0.7593 to0.8494 A-to-B,
but0.8476 to0.6216 B-to-A. Tiny Noise AUC falls0.9485 to0.5386 A-to-B
and0.99995 to0.92455 B-to-A. No universal improvement in ranking or
clean-calibrated detection was established.

## All budgets: nearest causal cross-subject TPR at nominal5%

|Budget|36 A-to-B|804 A-to-B|36 B-to-A|804 B-to-A|
|---|---:|---:|---:|---:|
|0.05|4.0|12.5|6.9|3.5|
|0.10|4.1|15.4|20.2|3.2|
|0.20|12.4|30.6|39.2|3.6|
|0.30|19.2|35.3|43.1|4.3|
|0.50|28.5|46.8|45.7|7.5|

## Decision and limitations

The bounded representation repair did not establish robust held-out detection.
It improves some A-to-B configurations but worsens B-to-A prototype transfer
and Tiny Noise. No broad expansion or Q1/Q2-ready defense claim is supported.
This diagnostic does not mechanically replace the frozen multi-dataset gates;
those remain unmet or unevaluated. It also does not prove all richer features
or other defenses fail. Lower observed FAR is not a precise population5%
guarantee, and this comparison is at the same nominal, not equal observed FAR.

Channel indices/time bins preserve additional information but do not recover
individual flashes or demonstrate cross-montage compatibility. Twenty clean
calibration samples per fold cannot precisely resolve very small FAR. No
threshold was moved using target attacks and no hyperparameter search was run.

BNCI2015-003 remains locked; splits/gates are unchanged. Next scientific
choice should be discussed with the professor: a narrow reliability/negative
result study versus a distinct defense requiring literature review and new
validation. Do not launch another representation search merely to chase a
positive result on already inspected target data.
