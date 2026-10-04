# Cross-subject authenticity defense: exploratory Comp III result

**Superseded on 2026-10-04:** budget-specific TPR claims in this historical report
are invalid because attack labels omitted epsilon and overwrote feature sets.
See `cross_subject_auth_v6_report.md` and corrected v6 artifacts. This file is
preserved as an audit trail and must not be cited as verified budget performance.

## Status

This is an exploratory development result on Competition III subjects A and B. It is not external validation and does not establish Q1 readiness. The frozen BNCI evaluation remains necessary.

The evaluation trained the feature-level authenticity model on one subject and tested it on the other. Thresholds were chosen from a separate genuine calibration fold. Five folds were used when enough induced examples existed. All processing and 287 portable repository tests ran on Kaggle; the exact exclusions are recorded in `results/cross_subject_auth/cloud_test_scope.txt`.

## Key result at nominal FAR 1% and epsilon multiplier 0.2

| Attack family | Direction | Held-out TPR | Observed FAR |
|---|---:|---:|---:|
| PGD | A to B | 1.000 | 0.040 |
| PGD | B to A | 1.000 | 0.070 |
| Target-only, causal, cross-subject bank | A to B | 1.000 | 0.040 |
| Target-only, causal, cross-subject bank | B to A | 1.000 | 0.070 |
| Prototype medoid, causal, cross-subject bank | A to B | 0.398 | 0.040 |
| Prototype medoid, causal, cross-subject bank | B to A | 0.990 | 0.070 |
| Prototype nearest, causal, cross-subject bank | A to B | 0.342 | 0.040 |
| Prototype nearest, causal, cross-subject bank | B to A | 0.965 | 0.070 |
| Prototype medoid, signal, cross-subject bank | A to B | 0.158 | 0.040 |
| Prototype medoid, signal, cross-subject bank | B to A | 0.498 | 0.070 |

## Interpretation

The defense transfers well for conspicuous attacks such as PGD and target-only injection. It is much less reliable for prototype attacks, with a strong direction asymmetry: causal prototype detection is only 34–40% from A to B but 96–99% from B to A. Signal-only prototype variants are weaker still.

The nominal 1% FAR was not achieved on the held-out subject. Observed FAR was 4% for A-to-B and 7% for B-to-A. Each held-out genuine fold contains only 20 examples, so FAR is quantized in 5 percentage-point increments, and cross-subject calibration shifted further in one direction. These results therefore support pursuing subject-normalized calibration, but they do not pass a strict fixed-FAR deployment claim.

## Decision

Continue the defense-led paper only through the frozen BNCI external evaluation. The present result is promising for strong attacks and identifies calibration plus prototype-transfer as the central technical problem. A publishable claim must be limited to attack-family-dependent transfer unless the BNCI study shows stable fixed-FAR performance across subjects and protocols.

## Provenance

- Kaggle kernel: `mustaqimnishat/p3a-cross-auth`, version 5
- Seed: `20260929`
- Feature bundle commit: `97ef266ab027dcaef9718a6d075e6d2abf4c3867`
- Cloud job fix commit: `1f3e9cb`
- Raw table: `results/cross_subject_auth/cross_subject.csv`
- Fold details: `results/cross_subject_auth/folds.json`
- Configuration: `results/cross_subject_auth/config.json`
