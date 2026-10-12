# OpenBMI timestamp and epoch comparison

Kaggle alignmentv1 completed12 October2026, runner28e281a. Five focused
schema/arithmetic assertions, not full loader tests. No local analysis/raw
retrieval. Reversals are at positions0 then60 etc, the first pair in each block.
They are not a general global-order issue. Physical cause remains unresolved.

Fixed raw versus provided-epoch comparisons do not match: at shift-1 relative
L2 errors1.2967/1.7212/4.2803/1.4517 for session1 train/test and session2
train/test. Mean correlations0.2724/0.3750/0.0610/0.4718. The MATLAB indexing
convention is documented but these comparisons do not empirically establish
alignment. No offset chosen from results. Provided epochs may be preprocessed;
this is a possibility, not a verified explanation.

Next fixed cloud diagnostic removes per-channel epoch means before comparing
same slices/offsets/events, isolating constant baseline offsets. This is not
preprocessing selected to improve detector performance. Gates/splits unchanged;
BNCI2015-003 locked. No validated loader or performance claim.
