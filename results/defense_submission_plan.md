# Defense manuscript preparation: 4-11 October 2026

User direction: defense only; existing attacks are stress tests. Submission-ready
work within one week is a conditional target, not promised journal publication.

1. Compare the closest defense papers before committing to a new question.
2. Correct and remotely test the cross-subject budget-identity bug; supersede
   the historical budget-specific TPR claims after the Kaggle rerun.
3. Audit cross-dataset feature compatibility and calibration sample requirements.
   Separate normalization fitting from threshold calibration. Do not claim that
   six test characters establish a precise 5% subject-level false-alarm rate.
4. Freeze a bounded defense comparison before further outcomes. Use registered
   baselines, unseen attack families, subject-level uncertainty, and observed FAR.
5. Add adaptive evasion and clean spelling cost if the study passes feasibility.
6. Draft around verified findings. Stop or narrow claims if gates fail.

No BNCI2015-003 confirmation outcomes are unlocked by this plan. No existing
split is modified, and all tests, EEG processing, and experiments run on Kaggle.

## Closest verified precedent

- P300 fake-wave defense with secret synchronized transformations (2021):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8709057/
- Chen and Wu, Adversarial Artifact Detection in EEG-Based BCIs (preprint 2022;
  Journal of Neural Engineering 21:056043, 2024):
  https://arxiv.org/abs/2212.00727

These establish that P300 defense and EEG adversarial detection already exist.
Independent calibration under shift with character-level unseen/adaptive attacks
is a candidate evaluation gap, not established novelty. A 2026 detector model
comparison also needs full-text review before finalizing the comparison matrix:
https://www.scitepress.org/publishedPapers/2026/148196/pdf/index.html

## Current correction status

Correction v6 completed on Kaggle; 288 portable tests passed, including budget
identity coverage. Corrected causal prototype detection at epsilon 0.2 is only
12-13% A to B and 28-32% B to A at nominal 5% FAR. Current architecture must not
be broadly expanded as a validated defense. See cross_subject_auth_v6_report.md
for verified rows, provenance qualifications and calibration feasibility.
