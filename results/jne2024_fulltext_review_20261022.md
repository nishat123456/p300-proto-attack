# Expanded JNE2024 primary full-text check

22 October2026 Central. Author publication list links the publisher-formatted
16-page PDF: https://drive.google.com/file/d/1i-GevRmiBZXiTAMY3FqM9F7sERAgk9-u/view
Author index: https://sites.google.com/site/drwuhust/publications/completepubs
DOI10.1088/1741-2552/ad8964. PDF title page confirms Chen/Meng/Xu/Wu; the index
also lists Bian, so use publisher title-page authors. PDF SHA256
1e123ebbe87d9acbc5893db02b225a6c9626020b14f10016120a01d95f9df574.
Original retained temporarily, not copied into repository.

Sections4.1-4.5, Figure4 and Table1 were extracted and pages6-8 visually checked.
Datasets are ERN16 participants, MI9, and RSVP64 available/10 sampled for LOSO,
with five repeated processes. The decision unit is an EEG epoch. Section4.3
fits logistic attack detectors with balanced clean/adversarial training examples.
White-box evaluation selects correctly classified clean epochs with successful
attacks; black-box testing includes correctly classified clean epochs regardless
of attack success. Section4.4 explicitly uses AUC to avoid threshold selection;
section4.5 tunes LID and MD parameters using nested training-fold AUC.

Thus clean-only per-user character thresholds, all-original-character denominators
and spelling rejection-cost accounting are distinct from this reviewed protocol.
This supports a specific protocol difference, not a first-detector claim or
proof that no other paper duplicates the study. Our fixed LID/shared-shrinkage
MD/character aggregation remain adaptations, not exact reproduction. The2022
preprint must remain separately cited. Independent scientific insight is pending.
