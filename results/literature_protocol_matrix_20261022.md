# Primary protocol comparison for Paper4

22 October2026. This compares reviewed protocols, not journal readiness or an
objective novelty score. Own source-specific reviews retain exact reading scope.

|Primary study|Verified protocol|Implication for our claim|
|---|---|---|
|Zhang/Gu2024 JPCS|CompIII P300/MI, confidence/entropy/BU/KDE, detector-aware CW, flash-level AUC/BCA/RCA|P300 attack detection and detector-aware failure already exist; include KDE and no first/adaptive claim|
|Chen/Meng/Xu/Wu2024 JNE|ERN/MI/RSVP epochs, supervised balanced attack/clean logistic detector, LOSO/nested AUC; AUC avoids threshold selection|Subject-transfer detection exists; our clean-only character operating point differs from the reviewed protocol|
|Chen/Jia/Wu2025 NeuralNetworks|Five datasets including8-person P300, within-/cross-session online/offline, aligned robust classifiers, epoch BCA/targeted ASR|P300 defense benchmarking and session shift already exist; neither is our novelty|
|Conceicao daCosta etal ICEIS2026|MI EEGNet, supervised conventional/ANFIS detection,80/20 split, uncertainty features|Conventional detector comparisons are precedents; broad benchmarking alone is insufficient|
|Khilnani/Kirar/Gautam2025 SPL accepted manuscript|MI/ERN epochs, FGSM/pulse contamination, logistic AUROC, entropy computation time; HTML manuscript only|EEG entropy detection and compute-cost reporting exist; CovEn omitted from frozen set, which is nonexhaustive|

## Specific, conditional distinction

The prespecified estimand is clean character rejection and accepted wrong
characters on independent P300 users, alongside all-original-attempt targeted
manipulation before/after separately attacking each detector, with independent
normalization/calibration roles, participant uncertainty and explicit rejection
cost assumptions. The reviewed protocols do not evaluate this same combined
estimand. This is evidence of a protocol distinction, not proof of universal
priority, a new detector, a new uncertainty method or a result.

Useful scientific insight must still emerge from the complete independent
study. Repeating known adaptive evasion, adding participants or counting tests
is not enough. Coarse20-calibration/36-test character samples, phase/phrase shift,
offline recorded markers,17 conditional available cases and fixed adaptive recipe
limit conclusions. No per-user5% assurance, worst-case robustness or clinical
throughput is claimed. Do not change frozen settings or select a headline from
favorable target outcomes. Retain negative and uninformative findings.

## Sources and remaining uncertainty

See results/zhang_gu_2024_fulltext_review.md,
results/jne2024_fulltext_review_20261022.md,
results/nn2025_fulltext_review_20261022.md,
results/iceis2026_primary_review.md and
results/coven2025_author_manuscript_review_20261022.md.
ExpandedJNE andNN full-text gaps are resolved; CovEn review is accepted HTML text,
not visually verified final PDF. The search is not an exhaustive systematic
review, and related work must be refreshed before submission. The user's external
novelty scores remain subjective and are not recalculated from these checks.
