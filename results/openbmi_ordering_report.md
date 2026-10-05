# OpenBMI event ordering and identity follow-up

Kaggle p3a-openbmi-order v1 completed5 October2026. Exact runner25b2303,
source base9177168; config contains SHA256. Five focused arithmetic/schema
assertions passed remotely, not a full suite or loader validation. All scientific
processing on Kaggle; only compact summaries/log retrieved, no raw EEG.

## Observed subject1, both sessions

- All four train/test phase vectors have zero duplicate onsets.
- Nonincreasing steps:33 in each train phase,36 in each test phase.
- Labels alternate extensively (658 train/720 test transitions), so a simple
  globally class-sorted explanation is not supported.
- Stable chronological diagnostic sorting yields33/36 inferred blocks of60,
  each with10 target flashes. This supports consistency with five sequences;
  it does not independently establish decision boundaries or flashed groups.
- Sorted median gap217 samples at1000Hz; minimum gaps16/39/34/42 samples for
  session1 train/test, session2 train/test. These short gaps merit inspection
  before assuming the sorted order is the actual flash presentation sequence.
- Both zero-based and one-based bounds checks pass. This does not distinguish
  index origin. Label fields encode target/nontarget only.

No sorting transformation has been approved as a production loader. Preserve
aligned labels/epochs and original event identity in any eventual reconstruction.

## Newly located original decoding artifact

[Original ERP_performance.m](https://github.com/PatternRecognition/OpenBMI/blob/master/GigaScience/function_ERP/ERP_performance.m)
loads random_cell_order.mat, derives flashes per character from the sequence
count and schedule rows, assigns flash evidence to symbol groups using that
schedule, and compares decoded characters with the documented test sentence.
The public source tree contains
GigaScience/function_ERP/random_cell_order.mat. This is a concrete stimulus
membership source to investigate, not proof it matches every released recording.
The original GigaScience Anaylsis_ERP.m specifies five sequences and36 symbols.
No MATLAB source was copied into this project.

Next bounded compatibility step is a Kaggle-only schedule-validation audit:
inspect the public schedule, compare dimensions/coverage and alignment to both
session recordings, and verify known copy-spelling labels without decoder fitting.
Resolve anomalously short sorted gaps and index origin before constructing trials.
Do not reconstruct memberships from Target/NonTarget labels or assume schedule
reuse across subjects without evidence. Character compatibility remains unverified.

No attack/detector performance run occurred. Existing gates/splits unchanged;
BNCI2015-003 remains locked. Sample-size limitations from the prior audit remain.
Completed ordering monitor retired; a performance benchmark is not launched.
