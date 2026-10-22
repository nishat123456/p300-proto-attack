# Independent task packaging

Package only the next fixed indices, for example:

    python kaggle/dispatch/stage_openbmi_tasks.py --task-index 5 6 7 8 9

This copies source text and compact frozen release files; it performs no tests,
EEG processing, fitting, metric computation or experiment. Generated folder
creation refuses existing task folders to avoid accidentally replacing launch
provenance. Review/commit the payload before pushing to the assigned account.
Do not rerun completed or healthy tasks. Do not change src/scripts while using
existing receipts: cloud guards compare every Python file to the frozen inventory.

Use configs/openbmi_five_account_dispatch_candidate.json only for its fixed
ascending subject/seed round-robin routing, unchanged from before outcomes.
Scientific authorization comes from the committed final release, not its older
candidate status label. Each task is one subject/seed/all36 characters. First
five tasks00..04 were launched already. Subsequent tasks may start as their
assigned account becomes free; no need to wait for every account in a wave.

Credentials stay in local existing profiles and are never embedded in payloads.
Retrieve only compact JSON/TXT/MD/log. Kernel output can span multiple API pages;
do not mistake a first-page result for complete artifacts. Keep per-character
CSV, EEG, models and reference NPZ on Kaggle for final cloud aggregation. Further
private artifact sharing across accounts needs the applicable authorization;
the current approved source/refcal viewer access is already verified.
