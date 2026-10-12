# Original OpenBMI schedule consistency

Kaggle v6 completed12 October2026, corrected runner5dcde3e. v5 comparison
was skipped due incorrect exact-five-entry validator; historical outputs retained.
First five entries of the ten-entry rc_order artifact match original decoder
Nsequence5. Selected schedules pass12x6 symbol-range and twice-per-symbol
coverage checks. Five focused remote assertions passed, not full loader tests.

Subject1 both sessions: **zero label mismatches in stored event order** for all
33 training and36 test characters. Chronological sorting produces8 mismatches
in each training phase and20 in each test phase. Each original60-event block
contains one nonincreasing timestamp step. Thus blind chronological sorting
breaks the documented schedule-label correspondence. Preserve stored order;
resolve the within-block timestamp reversal rather than silently sorting.

This establishes consistency of known copy-spelling labels with the original
schedule for this subset, not independent physical flash identity, index origin,
all-subject schedule reuse, or validated raw-signal epoch alignment. Do not use
agreement with true labels as proof of deployment without oracle labels. Next
bounded audit should identify the reversal position and verify MATLAB indexing
and epoch alignment against source metadata before building the loader.

No attack/detector performance runs, local scientific analysis or raw downloads.
Compact evidence preserved; gates/splits unchanged, BNCI2015-003 locked.
