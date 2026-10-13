# Loader contracts and sample-design sensitivity

Kaggle p3a-openbmi-contract-design v1 completed12 October2026.
**314 portable tests passed,41 warnings in49.07s**. Exact runnerd13500e,
overlay5d21063. Explicit raw-data/path exclusions retained in cloud_test_scope;
not literally every repository test. All calculations/tests ran on Kaggle.
Compact JSON/log only retrieved; no EEG processing/model performance fitting.

Input contracts now reject invalid finite/integer schedules/events, nonpositive
sampling rate, nonbinary labels, channel mismatches and invalid phrases. This
verifies software contracts, not trigger provenance or all-subject compatibility.

## IID evaluation design, one-sided95% exact bound

|Independent decisions|Upper FAR with0 alarms|With1 alarm|With2 alarms|Maximum alarms for upper bound<=5%|
|---|---:|---:|---:|---:|
|36|7.98%|12.51%|16.47%|None|
|59|4.95%|7.79%|10.29%|0|
|100|2.95%|4.66%|6.16%|1|
|200|1.49%|2.35%|3.11%|4|
|500|0.60%|0.95%|1.25%|16|

At59 decisions, probability of satisfying the upper5% bound is55.27% if true
FAR1%,16.58% if3%,4.85% if5%. At200 it is94.83%,28.10%,2.64%; at500,
99.998%,66.56%,3.43%. Integer acceptance thresholds explain nonmonotonic
probability near small count changes; do not cherry-pick sample sizes from this.
A true rate exactly5% seldom yields a95% upper bound<=5%, as expected for a
conservative certification test. This is not a journal requirement.

These are IID independent Bernoulli clean-decision calculations, not empirical
EEG rates, calibration guarantees, participant-level simultaneous bounds or
power calculations for detection comparisons. Session dependence and
nonstationarity may violate assumptions. Rank resolution assumes exchangeability
and does not certify our empirical percentile as a conformal procedure.

OpenBMI36 test characters per session cannot support that per-user certification
even with zero alarms. Pooling participants changes the estimand and cannot be
called individual-user assurance. Combining sessions must explicitly account for
shift and maintain separate calibration/evaluation. Thousands of flash epochs
are not thousands of independent spelling decisions.

## Decision

Software contract validation passes; strong per-user low-FAR assurance remains
unsupported by the current sample budget. Before performance experiments choose
a scientifically honest estimand: exploratory per-user observed FAR with
intervals and a separately defined population/session summary, or obtain more
independent per-user decisions. Do not relax frozen gates to manufacture a pass.
Trigger/reversal provenance still unresolved. Existing gates/splits unchanged;
BNCI2015-003 locked. No new benchmark allocation or training selected here.
