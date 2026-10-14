# Reference detector family implementation checkpoint

13 October2026. Source: Chen/Wu arXiv2212.00727v1, equations9/10/11-13,
https://arxiv.org/pdf/2212.00727. This is a source-reviewed family adaptation,
not a certified exact JNE2024 reproduction. Only formulas were independently
implemented; no external repository code copied.

Implemented BU as summed class-probability variance over dropout draws; LID
as-k/sum(log(r_i/r_k)) over Euclidean source-reference distances; network-feature
Mahalanobis as minimum squared distance to source class means with shared
Ledoit-Wolf residual covariance. Shrinkage is an explicit adaptation. The
paper's input-gradient confidence perturbation and layer combination are NOT
implemented yet; do not label plain score an exact MDmax detector reproduction.
Zero-distance LID requires explicit self-exclusion/duplicate policy, not silently
adding epsilon. Dropout sampling wrapper and network-layer/character aggregation
are pending; no performance computed or hyperparameters frozen here.

Three synthetic tests verify probability uncertainty/disagreement, LID equation
and self-distance rejection, and distance/source-label relabeling behavior. Tests
execute on Kaggle portable suite, never locally. Additional protocol/negative
coverage required before performance integration.

Chen/Jia/Wu2025 author README identifies NT/AT/TRADES/SCR/SAP/IT/RSE/SEAT/
SEAT-DA and online/offline EA; it is a robust-classification benchmark, not
proof our calibrated-rejection question is new. Zhang/Gu2024 publisher PDF
retrieval remains inaccessible; methods details unknown. Keep fulltext gaps
visible. No generic first-defense/benchmark claim.
