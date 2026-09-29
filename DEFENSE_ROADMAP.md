# Defense-led publication roadmap

**Frozen direction:** 2026-09-29, after the Comp III kill test and exploratory
authenticity rescue.

## Central claim

A P300 defense should verify that decision evidence is consistent with a
genuine, stimulus-evoked response. A small clean calibration set from a new user
should control the false-alarm rate while a source-trained detector identifies
unseen manipulations.

## Dataset roles

| Dataset | Role | Outcome access |
| --- | --- | --- |
| BCI Competition III Dataset II | Development only | Already unblinded |
| BNCI 2014-008, 8 ALS participants | External validation | Freeze method before attack outcomes |
| BNCI 2015-003, 10 participants | Locked confirmation | No tuning after test outcomes |

BNCI 2014-008 uses the first 70% of character trials for training, the next 15%
for validation, and the final 15% for testing, in recording order. BNCI
2015-003 retains its official train and test recordings: source-subject training
and target-user clean calibration use only the train recording, while the test
recording remains locked.

## Method

1. Fit the authenticity classifier on source subjects and registered attack
   families.
2. Normalize scores or features with target-user clean calibration data only.
3. Select a conformal or empirical quantile threshold from clean calibration.
4. Evaluate fixed thresholds on unseen subjects and held-out attack families.
5. Report the rejection burden and spelling accuracy on clean users.
6. Stress test the final frozen detector with a defense-aware adaptive attack.

## Gates

### Q2-ready

- At least two datasets.
- Observed FAR no greater than 10% for every dataset.
- Median held-out attack TPR at least 70%.
- Improvement over distance, spectral, and prior adversarial-detector baselines.

### Q1-candidate

- All three datasets and at least 20 subjects.
- Observed FAR no greater than 5% for every dataset.
- Median held-out attack TPR at least 80%, with no subject below 70%.
- Adaptive-attack evaluation, clean spelling cost, subject-clustered confidence
  intervals, and complete feature ablations.

## Stop rules

- If subject-normalized calibration cannot raise Comp III subject B to 70% TPR
  at 5% FAR, do not expand the current detector architecture.
- If the frozen detector fails the Q2 gate on BNCI 2014-008, do not inspect or
  tune on BNCI 2015-003 test outcomes.
- Treat all Comp III repair results as exploratory because its test outcomes are
  already known.

## Execution tracker

- [x] Comp III attack kill test and mechanical defense-led verdict.
- [x] Nested clean-calibration rescue and fixed-FAR report.
- [ ] Verified BNCI 2014-008 loader and immutable split manifests.
- [ ] Verified BNCI 2015-003 loader preserving official train/test recordings.
- [ ] Subject-normalized calibration repair on Comp III development data.
- [ ] Freeze detector v2 and evaluation protocol.
- [ ] BNCI 2014-008 external validation and Q2 gate.
- [ ] BNCI 2015-003 locked confirmation.
- [ ] Prior-defense baselines, adaptive attack, statistics, and manuscript.
