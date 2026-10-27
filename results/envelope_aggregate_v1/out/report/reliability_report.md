# Character-level reliability report

Independent available-case17 users;18 planned,52 unavailable

All rates retain original character denominators. Model seeds are averaged within each user before equal-user means. Intervals are pointwise subject-bootstrap descriptions under exchangeability, not per-user guarantees. Adaptive rows are separate attacks against each detector, not simultaneous evasion or comparative efficacy.

## Clean decisions at nominal5%

|Method|Observed rejection mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|
|---|---|---|---|
|confidence|6.32% [4.52, 8.17]|56.05% [39.98, 71.19]|37.64% [22.06, 53.49]|
|entropy|5.83% [3.98, 7.63]|56.15% [40.03, 71.51]|38.02% [22.28, 54.19]|
|bu|4.85% [2.34, 7.90]|56.92% [40.85, 73.21]|38.24% [22.54, 53.98]|
|lid|3.92% [2.23, 5.94]|58.17% [41.23, 74.56]|37.91% [22.22, 53.38]|
|md|8.82% [3.92, 15.09]|55.17% [38.88, 71.68]|36.00% [20.92, 51.09]|
|md_adjusted|8.77% [3.92, 14.92]|55.17% [38.88, 71.68]|36.06% [21.02, 51.09]|
|kde|6.32% [2.94, 10.13]|56.86% [40.52, 72.82]|36.82% [21.62, 52.29]|

## All declared clean operating points

|Nominal FAR|Detector|Observed rejection mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|
|---|---|---|---|---|
|1%|confidence|6.32% [4.52, 8.17]|56.05% [39.98, 71.19]|37.64% [22.06, 53.49]|
|1%|entropy|5.83% [3.98, 7.63]|56.15% [40.03, 71.51]|38.02% [22.28, 54.19]|
|1%|bu|4.85% [2.34, 7.90]|56.92% [40.85, 73.21]|38.24% [22.54, 53.98]|
|1%|lid|3.92% [2.23, 5.94]|58.17% [41.23, 74.56]|37.91% [22.22, 53.38]|
|1%|md|8.82% [3.92, 15.09]|55.17% [38.88, 71.68]|36.00% [20.92, 51.09]|
|1%|md_adjusted|8.77% [3.92, 14.92]|55.17% [38.88, 71.68]|36.06% [21.02, 51.09]|
|1%|kde|6.32% [2.94, 10.13]|56.86% [40.52, 72.82]|36.82% [21.62, 52.29]|
|5%|confidence|6.32% [4.52, 8.17]|56.05% [39.98, 71.19]|37.64% [22.06, 53.49]|
|5%|entropy|5.83% [3.98, 7.63]|56.15% [40.03, 71.51]|38.02% [22.28, 54.19]|
|5%|bu|4.85% [2.34, 7.90]|56.92% [40.85, 73.21]|38.24% [22.54, 53.98]|
|5%|lid|3.92% [2.23, 5.94]|58.17% [41.23, 74.56]|37.91% [22.22, 53.38]|
|5%|md|8.82% [3.92, 15.09]|55.17% [38.88, 71.68]|36.00% [20.92, 51.09]|
|5%|md_adjusted|8.77% [3.92, 14.92]|55.17% [38.88, 71.68]|36.06% [21.02, 51.09]|
|5%|kde|6.32% [2.94, 10.13]|56.86% [40.52, 72.82]|36.82% [21.62, 52.29]|
|10%|confidence|10.84% [8.28, 13.67]|53.38% [38.02, 68.14]|35.78% [20.92, 50.98]|
|10%|entropy|11.33% [8.22, 14.60]|52.67% [37.53, 67.43]|36.00% [21.02, 51.20]|
|10%|bu|9.86% [6.37, 13.51]|54.19% [38.73, 69.66]|35.95% [21.35, 50.00]|
|10%|lid|8.39% [6.26, 10.62]|55.94% [39.49, 71.79]|35.68% [21.29, 50.11]|
|10%|md|13.13% [7.35, 20.54]|52.67% [36.27, 69.17]|34.20% [19.66, 48.47]|
|10%|md_adjusted|13.07% [7.30, 20.48]|52.72% [36.44, 69.23]|34.20% [19.66, 48.47]|
|10%|kde|10.89% [6.15, 16.23]|54.14% [38.24, 69.93]|34.97% [20.58, 49.07]|

## All methods and declared budgets

|Attack|Budget|Detector|Defended targeted / all attempts, mean [95% CI]|
|---|---|---|---|
|FGSM|0.05|bu|99.51% [98.75, 100.00]|
|FGSM|0.05|confidence|100.00% [100.00, 100.00]|
|FGSM|0.05|entropy|100.00% [100.00, 100.00]|
|FGSM|0.05|kde|0.16% [0.00, 0.49]|
|FGSM|0.05|lid|64.11% [44.99, 82.41]|
|FGSM|0.05|md|0.16% [0.00, 0.49]|
|FGSM|0.05|md_adjusted|0.16% [0.00, 0.49]|
|FGSM|0.10|bu|98.15% [95.81, 99.89]|
|FGSM|0.10|confidence|100.00% [100.00, 100.00]|
|FGSM|0.10|entropy|100.00% [100.00, 100.00]|
|FGSM|0.10|kde|0.00% [0.00, 0.00]|
|FGSM|0.10|lid|8.06% [0.93, 19.50]|
|FGSM|0.10|md|0.00% [0.00, 0.00]|
|FGSM|0.10|md_adjusted|0.00% [0.00, 0.00]|
|FGSM|0.20|bu|94.99% [89.71, 98.75]|
|FGSM|0.20|confidence|100.00% [100.00, 100.00]|
|FGSM|0.20|entropy|100.00% [100.00, 100.00]|
|FGSM|0.20|kde|0.00% [0.00, 0.00]|
|FGSM|0.20|lid|0.00% [0.00, 0.00]|
|FGSM|0.20|md|0.00% [0.00, 0.00]|
|FGSM|0.20|md_adjusted|0.00% [0.00, 0.00]|
|FGSM|0.30|bu|93.14% [86.76, 97.66]|
|FGSM|0.30|confidence|100.00% [100.00, 100.00]|
|FGSM|0.30|entropy|100.00% [100.00, 100.00]|
|FGSM|0.30|kde|0.00% [0.00, 0.00]|
|FGSM|0.30|lid|0.00% [0.00, 0.00]|
|FGSM|0.30|md|0.00% [0.00, 0.00]|
|FGSM|0.30|md_adjusted|0.00% [0.00, 0.00]|
|FGSM|0.50|bu|90.74% [83.39, 96.19]|
|FGSM|0.50|confidence|100.00% [100.00, 100.00]|
|FGSM|0.50|entropy|100.00% [100.00, 100.00]|
|FGSM|0.50|kde|0.00% [0.00, 0.00]|
|FGSM|0.50|lid|0.00% [0.00, 0.00]|
|FGSM|0.50|md|0.00% [0.00, 0.00]|
|FGSM|0.50|md_adjusted|0.00% [0.00, 0.00]|
|Gaussian_control|0.05|bu|0.87% [0.22, 1.69]|
|Gaussian_control|0.05|confidence|0.98% [0.33, 1.80]|
|Gaussian_control|0.05|entropy|0.98% [0.33, 1.80]|
|Gaussian_control|0.05|kde|0.87% [0.22, 1.69]|
|Gaussian_control|0.05|lid|0.93% [0.27, 1.74]|
|Gaussian_control|0.05|md|0.87% [0.22, 1.69]|
|Gaussian_control|0.05|md_adjusted|0.87% [0.22, 1.69]|
|Gaussian_control|0.10|bu|0.82% [0.22, 1.53]|
|Gaussian_control|0.10|confidence|0.93% [0.33, 1.63]|
|Gaussian_control|0.10|entropy|0.93% [0.33, 1.63]|
|Gaussian_control|0.10|kde|0.82% [0.22, 1.53]|
|Gaussian_control|0.10|lid|0.87% [0.27, 1.58]|
|Gaussian_control|0.10|md|0.82% [0.22, 1.53]|
|Gaussian_control|0.10|md_adjusted|0.82% [0.22, 1.53]|
|Gaussian_control|0.20|bu|0.98% [0.33, 1.69]|
|Gaussian_control|0.20|confidence|1.09% [0.44, 1.80]|
|Gaussian_control|0.20|entropy|1.03% [0.44, 1.69]|
|Gaussian_control|0.20|kde|0.98% [0.33, 1.69]|
|Gaussian_control|0.20|lid|1.03% [0.38, 1.74]|
|Gaussian_control|0.20|md|0.93% [0.33, 1.58]|
|Gaussian_control|0.20|md_adjusted|0.93% [0.33, 1.58]|
|Gaussian_control|0.30|bu|0.93% [0.33, 1.58]|
|Gaussian_control|0.30|confidence|0.98% [0.38, 1.63]|
|Gaussian_control|0.30|entropy|0.98% [0.38, 1.63]|
|Gaussian_control|0.30|kde|0.98% [0.33, 1.69]|
|Gaussian_control|0.30|lid|0.98% [0.38, 1.63]|
|Gaussian_control|0.30|md|0.93% [0.33, 1.58]|
|Gaussian_control|0.30|md_adjusted|0.93% [0.33, 1.58]|
|Gaussian_control|0.50|bu|0.93% [0.38, 1.53]|
|Gaussian_control|0.50|confidence|0.98% [0.44, 1.58]|
|Gaussian_control|0.50|entropy|0.98% [0.44, 1.58]|
|Gaussian_control|0.50|kde|0.93% [0.33, 1.63]|
|Gaussian_control|0.50|lid|0.93% [0.38, 1.53]|
|Gaussian_control|0.50|md|0.87% [0.33, 1.53]|
|Gaussian_control|0.50|md_adjusted|0.87% [0.33, 1.53]|
|PGD|0.05|bu|100.00% [100.00, 100.00]|
|PGD|0.05|confidence|100.00% [100.00, 100.00]|
|PGD|0.05|entropy|100.00% [100.00, 100.00]|
|PGD|0.05|kde|1.47% [0.00, 4.03]|
|PGD|0.05|lid|97.44% [92.76, 99.95]|
|PGD|0.05|md|8.66% [1.20, 18.46]|
|PGD|0.05|md_adjusted|8.61% [1.20, 18.30]|
|PGD|0.10|bu|100.00% [100.00, 100.00]|
|PGD|0.10|confidence|100.00% [100.00, 100.00]|
|PGD|0.10|entropy|100.00% [100.00, 100.00]|
|PGD|0.10|kde|0.05% [0.00, 0.16]|
|PGD|0.10|lid|91.99% [78.70, 99.68]|
|PGD|0.10|md|3.43% [0.54, 7.90]|
|PGD|0.10|md_adjusted|3.43% [0.54, 7.90]|
|PGD|0.20|bu|99.89% [99.67, 100.00]|
|PGD|0.20|confidence|100.00% [100.00, 100.00]|
|PGD|0.20|entropy|100.00% [100.00, 100.00]|
|PGD|0.20|kde|0.00% [0.00, 0.00]|
|PGD|0.20|lid|85.62% [71.78, 94.88]|
|PGD|0.20|md|0.05% [0.00, 0.16]|
|PGD|0.20|md_adjusted|0.05% [0.00, 0.16]|
|PGD|0.30|bu|98.69% [96.13, 100.00]|
|PGD|0.30|confidence|100.00% [100.00, 100.00]|
|PGD|0.30|entropy|100.00% [100.00, 100.00]|
|PGD|0.30|kde|0.00% [0.00, 0.00]|
|PGD|0.30|lid|61.27% [43.18, 78.49]|
|PGD|0.30|md|0.00% [0.00, 0.00]|
|PGD|0.30|md_adjusted|0.00% [0.00, 0.00]|
|PGD|0.50|bu|96.79% [90.36, 100.00]|
|PGD|0.50|confidence|100.00% [100.00, 100.00]|
|PGD|0.50|entropy|100.00% [100.00, 100.00]|
|PGD|0.50|kde|0.00% [0.00, 0.00]|
|PGD|0.50|lid|18.19% [6.53, 34.21]|
|PGD|0.50|md|0.00% [0.00, 0.00]|
|PGD|0.50|md_adjusted|0.00% [0.00, 0.00]|
|detector_aware|0.20|bu|100.00% [100.00, 100.00]|
|detector_aware|0.20|confidence|100.00% [100.00, 100.00]|
|detector_aware|0.20|entropy|100.00% [100.00, 100.00]|
|detector_aware|0.20|kde|94.01% [82.14, 100.00]|
|detector_aware|0.20|lid|94.23% [82.68, 100.00]|
|detector_aware|0.20|md|94.12% [82.35, 100.00]|
|detector_aware|0.20|md_adjusted|94.12% [82.35, 100.00]|
|source_template|0.05|bu|99.62% [98.86, 100.00]|
|source_template|0.05|confidence|99.84% [99.51, 100.00]|
|source_template|0.05|entropy|99.95% [99.84, 100.00]|
|source_template|0.05|kde|14.11% [4.25, 26.47]|
|source_template|0.05|lid|90.14% [75.98, 98.42]|
|source_template|0.05|md|12.69% [4.30, 24.67]|
|source_template|0.05|md_adjusted|12.75% [4.30, 24.73]|
|source_template|0.10|bu|100.00% [100.00, 100.00]|
|source_template|0.10|confidence|100.00% [100.00, 100.00]|
|source_template|0.10|entropy|100.00% [100.00, 100.00]|
|source_template|0.10|kde|0.33% [0.00, 0.98]|
|source_template|0.10|lid|58.82% [40.95, 76.25]|
|source_template|0.10|md|0.16% [0.00, 0.49]|
|source_template|0.10|md_adjusted|0.16% [0.00, 0.49]|
|source_template|0.20|bu|100.00% [100.00, 100.00]|
|source_template|0.20|confidence|100.00% [100.00, 100.00]|
|source_template|0.20|entropy|100.00% [100.00, 100.00]|
|source_template|0.20|kde|0.00% [0.00, 0.00]|
|source_template|0.20|lid|5.77% [0.00, 16.23]|
|source_template|0.20|md|0.00% [0.00, 0.00]|
|source_template|0.20|md_adjusted|0.00% [0.00, 0.00]|
|source_template|0.30|bu|100.00% [100.00, 100.00]|
|source_template|0.30|confidence|100.00% [100.00, 100.00]|
|source_template|0.30|entropy|100.00% [100.00, 100.00]|
|source_template|0.30|kde|0.00% [0.00, 0.00]|
|source_template|0.30|lid|0.00% [0.00, 0.00]|
|source_template|0.30|md|0.00% [0.00, 0.00]|
|source_template|0.30|md_adjusted|0.00% [0.00, 0.00]|
|source_template|0.50|bu|100.00% [100.00, 100.00]|
|source_template|0.50|confidence|100.00% [100.00, 100.00]|
|source_template|0.50|entropy|100.00% [100.00, 100.00]|
|source_template|0.50|kde|0.00% [0.00, 0.00]|
|source_template|0.50|lid|0.00% [0.00, 0.00]|
|source_template|0.50|md|0.00% [0.00, 0.00]|
|source_template|0.50|md_adjusted|0.00% [0.00, 0.00]|

## Figure interpretation

![Observed clean rejection](clean_rejection.png)

![Accepted manipulation](defended_manipulation.png)

Color scales are fixed0-100%. Tables carry pointwise population intervals; colors do not establish significance. Compare detector-specific adaptive rows only as stress tests. The fixed longer run covers the first character per user/seed, not every character. No winner, robust-defense or journal-readiness verdict is generated.

## Limitations

Twenty calibration and36 test characters give coarse user estimates. Serial dependence, phrase/phase shift,17 conditional available cases, recorded-marker timing and fixed adaptive recipe limit inference. Rejection is not clinical throughput. BNCI2015-003 remains locked.
