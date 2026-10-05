# OpenBMI metadata and sample-size feasibility audit

Kaggle mustaqimnishat/p3a-openbmi-audit v1 completed on5 October2026.
Source09499cf, exact runner734f493; saved config includes runner SHA256.
Five focused arithmetic/schema assertions passed on Kaggle. This was not a
full repository suite, loader validation or defense performance evaluation.
Only compact JSON/Markdown/log outputs were retrieved; raw EEG stayed cloud.

## Observed metadata: fixed subject1, both sessions

Both session files loaded:62 named channels,1000Hz. In each session, train
has1980 aligned flash events (330 Target,1650 NonTarget) and test2160
(360 Target,1800 NonTarget). These equal33/36 blocks of60, matching documented
character counts, but independent decision IDs and symbol membership remain
unverified. Fields include x,t,y_dec,y_logic,y_class,class and epoched smt;
no explicit flashed-symbol/group field was identified by this schema audit.

**All four event-time vectors failed strict-increasing order.** This is an
observed compatibility blocker, not proof the recordings are corrupt. Possible
class sorting, duplicates or storage ordering require a further cloud metadata
inspection. Do not blindly sort t without applying the same permutation to
labels/epochs and verifying original character/stimulus identity. Counts and
Target/NonTarget labels alone cannot reconstruct arbitrary flashed groups.

The within-recording check passed under a permissive0<=t<=sample-count check;
index origin and strict upper indexing bounds were not established. Nasion
reference is documented in the source description, not independently measured.
No performance outcomes or all54-subject compatibility were audited.

## Cloud-only design arithmetic

For zero alarms among independent Bernoulli clean decisions, one-sided95%
upper FAR bounds from the cloud output are:

|Independent evaluation n|Upper bound|
|---|---:|
|6|39.30%|
|20|13.91%|
|33|8.68%|
|36|7.98%|
|59|4.95%|
|100|2.95%|
|299|1.00%|

At least59 independent held-out clean decisions with zero alarms are required
for this idealized one-sided95% upper bound to be at most5%. This is a best-case
zero-alarm calculation, not a power analysis, calibration sample requirement,
multiple-user simultaneous guarantee or EEG guarantee. One observed alarm needs
a different interval. Serial dependence/nonstationarity may invalidate IID
assumptions. Rank resolution1/(n+1) assumes exchangeability and does not mean
an empirical percentile is a validated conformal method.

Only36 official test characters per session do not meet the59 zero-alarm
best-case per-user evaluation requirement. Combining sessions requires an
explicit estimand and independent calibration allocation; it cannot silently
remove session shift. Many subjects improve group evidence but cannot certify
every individual. Flash epochs, repetitions and attack targets are not new
independent clean characters.

## Decision

File access and basic counts are feasible. Character-level loader compatibility
is **not established**. Next requires a cloud-only ordering/group metadata audit
and original recording/stimulus documentation. Do not launch targeted attacks
or a detector benchmark until decision identity and stimulus memberships can
be reconstructed without guessing. If the public release lacks them, use a
dataset that supplies them or explicitly narrow to an epoch-level study with
a different claim, after literature review and protocol approval.

Current results do not rescue the defense or demonstrate Q1/Q2 readiness.
Frozen gates/splits remain unchanged and BNCI2015-003 remains locked. The
completed feasibility monitor can be retired; no performance run is authorized
by this audit report.
