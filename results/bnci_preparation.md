# BNCI preparation verified on Kaggle

Kernel: mustaqimnishat/p3a-prepare-bnci, version 6, completed 2026-09-29.
Remote loader/split tests: 8 passed in 0.88 seconds. No local tests or data processing were run for this verification.

- BNCI2014-008: all 8 subjects loaded, 35 characters each, 120 flashes per character. Chronological splits are 24 training, 5 validation, 6 test characters.
- BNCI2015-003: all 10 subjects loaded, 5 official training and 5 test characters each. Subjects s1/s2 use 540 singleton flashes per character; s3-s10 use 180 column/row flashes per character.
- 36 new split and pre-drawn attack-target manifests were retrieved. Their embedded hash fields match the remote audit. Existing Comp III manifests were not changed. Raw EEG remains off the repository.

Provenance caveat: the audit's commit 3887236 identifies the attached base code bundle, not the complete executed code. Kernel v6 replaced the BNCI2015 loader and loader tests using exact embedded source snapshots from fd5a439. Reproduce with that commit's kaggle/jobs/prepare_bnci/run.py. The original audit is preserved verbatim.

This establishes dataset loading and split preparation only. No external attack or defense results exist from this job, and it does not establish the Q1 gate.
