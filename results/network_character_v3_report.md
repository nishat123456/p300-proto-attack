# Network-to-character adapter regression checkpoint

Kaggle contract-designv3 completed; inspected14 October2026.
319 portable tests passed,41warnings56.07s. Explicit raw/path exclusions;
not literally all tests. Source/runner694232c, compact outputs retained.
Dropout-only inference preserves BatchNorm/modes/RNG and is reproducible;
character selection uses decoder evidence rather than anomaly score or true label.
These are synthetic software checks, not detector performance or exact paper
reproduction. Recorded-marker timing limitation and independent protocol gaps
remain. No immutable allocation or large performance job yet.

Candidate partition helper now rejects phase identity overlap, role leakage,
incorrect33/36 counts and out-of-range subjects. Proposed13norm/20cal/36eval
and subject1-18/19-36/37-54 roles remain proposals. No frozen files generated;
cloud regression checks precede actual metadata verification and protocol freeze.
