# Character-level reliability report

DEVELOPMENT ONLY: fixed already-inspected subject19

All rates retain original character denominators. Model seeds are averaged within each user before equal-user means. Intervals are pointwise subject-bootstrap descriptions under exchangeability, not per-user guarantees. Adaptive rows are separate attacks against each detector, not simultaneous evasion or comparative efficacy.

## Clean decisions at nominal5%

|Method|Observed rejection mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|
|---|---|---|---|
|confidence|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|entropy|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|bu|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|lid|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|md|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|md_adjusted|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|kde|1.85% (population CI unavailable)|98.15% (population CI unavailable)|0.00% (population CI unavailable)|

## All declared clean operating points

|Nominal FAR|Detector|Observed rejection mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|
|---|---|---|---|---|
|1%|confidence|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|entropy|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|bu|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|lid|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|md|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|md_adjusted|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|1%|kde|1.85% (population CI unavailable)|98.15% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|confidence|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|entropy|0.00% (population CI unavailable)|100.00% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|bu|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|lid|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|md|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|md_adjusted|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|
|5%|kde|1.85% (population CI unavailable)|98.15% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|confidence|1.85% (population CI unavailable)|98.15% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|entropy|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|bu|2.78% (population CI unavailable)|97.22% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|lid|5.56% (population CI unavailable)|94.44% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|md|7.41% (population CI unavailable)|92.59% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|md_adjusted|6.48% (population CI unavailable)|93.52% (population CI unavailable)|0.00% (population CI unavailable)|
|10%|kde|4.63% (population CI unavailable)|95.37% (population CI unavailable)|0.00% (population CI unavailable)|

## All methods and declared budgets

|Attack|Budget|Detector|Defended targeted / all attempts, mean [95% CI]|
|---|---|---|---|
|FGSM|0.05|bu|75.93% (population CI unavailable)|
|FGSM|0.05|confidence|99.07% (population CI unavailable)|
|FGSM|0.05|entropy|100.00% (population CI unavailable)|
|FGSM|0.05|kde|0.00% (population CI unavailable)|
|FGSM|0.05|lid|21.30% (population CI unavailable)|
|FGSM|0.05|md|0.00% (population CI unavailable)|
|FGSM|0.05|md_adjusted|0.00% (population CI unavailable)|
|FGSM|0.10|bu|50.93% (population CI unavailable)|
|FGSM|0.10|confidence|100.00% (population CI unavailable)|
|FGSM|0.10|entropy|100.00% (population CI unavailable)|
|FGSM|0.10|kde|0.00% (population CI unavailable)|
|FGSM|0.10|lid|0.00% (population CI unavailable)|
|FGSM|0.10|md|0.00% (population CI unavailable)|
|FGSM|0.10|md_adjusted|0.00% (population CI unavailable)|
|FGSM|0.20|bu|21.30% (population CI unavailable)|
|FGSM|0.20|confidence|99.07% (population CI unavailable)|
|FGSM|0.20|entropy|99.07% (population CI unavailable)|
|FGSM|0.20|kde|0.00% (population CI unavailable)|
|FGSM|0.20|lid|0.00% (population CI unavailable)|
|FGSM|0.20|md|0.00% (population CI unavailable)|
|FGSM|0.20|md_adjusted|0.00% (population CI unavailable)|
|FGSM|0.30|bu|12.96% (population CI unavailable)|
|FGSM|0.30|confidence|98.15% (population CI unavailable)|
|FGSM|0.30|entropy|100.00% (population CI unavailable)|
|FGSM|0.30|kde|0.00% (population CI unavailable)|
|FGSM|0.30|lid|0.00% (population CI unavailable)|
|FGSM|0.30|md|0.00% (population CI unavailable)|
|FGSM|0.30|md_adjusted|0.00% (population CI unavailable)|
|FGSM|0.50|bu|11.11% (population CI unavailable)|
|FGSM|0.50|confidence|100.00% (population CI unavailable)|
|FGSM|0.50|entropy|100.00% (population CI unavailable)|
|FGSM|0.50|kde|0.00% (population CI unavailable)|
|FGSM|0.50|lid|0.00% (population CI unavailable)|
|FGSM|0.50|md|0.00% (population CI unavailable)|
|FGSM|0.50|md_adjusted|0.00% (population CI unavailable)|
|Gaussian_control|0.05|bu|0.00% (population CI unavailable)|
|Gaussian_control|0.05|confidence|0.00% (population CI unavailable)|
|Gaussian_control|0.05|entropy|0.00% (population CI unavailable)|
|Gaussian_control|0.05|kde|0.00% (population CI unavailable)|
|Gaussian_control|0.05|lid|0.00% (population CI unavailable)|
|Gaussian_control|0.05|md|0.00% (population CI unavailable)|
|Gaussian_control|0.05|md_adjusted|0.00% (population CI unavailable)|
|Gaussian_control|0.10|bu|0.00% (population CI unavailable)|
|Gaussian_control|0.10|confidence|0.00% (population CI unavailable)|
|Gaussian_control|0.10|entropy|0.00% (population CI unavailable)|
|Gaussian_control|0.10|kde|0.00% (population CI unavailable)|
|Gaussian_control|0.10|lid|0.00% (population CI unavailable)|
|Gaussian_control|0.10|md|0.00% (population CI unavailable)|
|Gaussian_control|0.10|md_adjusted|0.00% (population CI unavailable)|
|Gaussian_control|0.20|bu|0.00% (population CI unavailable)|
|Gaussian_control|0.20|confidence|0.00% (population CI unavailable)|
|Gaussian_control|0.20|entropy|0.00% (population CI unavailable)|
|Gaussian_control|0.20|kde|0.00% (population CI unavailable)|
|Gaussian_control|0.20|lid|0.00% (population CI unavailable)|
|Gaussian_control|0.20|md|0.00% (population CI unavailable)|
|Gaussian_control|0.20|md_adjusted|0.00% (population CI unavailable)|
|Gaussian_control|0.30|bu|0.00% (population CI unavailable)|
|Gaussian_control|0.30|confidence|0.00% (population CI unavailable)|
|Gaussian_control|0.30|entropy|0.00% (population CI unavailable)|
|Gaussian_control|0.30|kde|0.00% (population CI unavailable)|
|Gaussian_control|0.30|lid|0.00% (population CI unavailable)|
|Gaussian_control|0.30|md|0.00% (population CI unavailable)|
|Gaussian_control|0.30|md_adjusted|0.00% (population CI unavailable)|
|Gaussian_control|0.50|bu|0.00% (population CI unavailable)|
|Gaussian_control|0.50|confidence|0.00% (population CI unavailable)|
|Gaussian_control|0.50|entropy|0.00% (population CI unavailable)|
|Gaussian_control|0.50|kde|0.00% (population CI unavailable)|
|Gaussian_control|0.50|lid|0.00% (population CI unavailable)|
|Gaussian_control|0.50|md|0.00% (population CI unavailable)|
|Gaussian_control|0.50|md_adjusted|0.00% (population CI unavailable)|
|PGD|0.05|bu|100.00% (population CI unavailable)|
|PGD|0.05|confidence|100.00% (population CI unavailable)|
|PGD|0.05|entropy|100.00% (population CI unavailable)|
|PGD|0.05|kde|0.00% (population CI unavailable)|
|PGD|0.05|lid|99.07% (population CI unavailable)|
|PGD|0.05|md|0.00% (population CI unavailable)|
|PGD|0.05|md_adjusted|0.00% (population CI unavailable)|
|PGD|0.10|bu|100.00% (population CI unavailable)|
|PGD|0.10|confidence|100.00% (population CI unavailable)|
|PGD|0.10|entropy|100.00% (population CI unavailable)|
|PGD|0.10|kde|0.00% (population CI unavailable)|
|PGD|0.10|lid|97.22% (population CI unavailable)|
|PGD|0.10|md|0.00% (population CI unavailable)|
|PGD|0.10|md_adjusted|0.00% (population CI unavailable)|
|PGD|0.20|bu|100.00% (population CI unavailable)|
|PGD|0.20|confidence|100.00% (population CI unavailable)|
|PGD|0.20|entropy|100.00% (population CI unavailable)|
|PGD|0.20|kde|0.00% (population CI unavailable)|
|PGD|0.20|lid|52.78% (population CI unavailable)|
|PGD|0.20|md|0.00% (population CI unavailable)|
|PGD|0.20|md_adjusted|0.00% (population CI unavailable)|
|PGD|0.30|bu|99.07% (population CI unavailable)|
|PGD|0.30|confidence|100.00% (population CI unavailable)|
|PGD|0.30|entropy|100.00% (population CI unavailable)|
|PGD|0.30|kde|0.00% (population CI unavailable)|
|PGD|0.30|lid|25.93% (population CI unavailable)|
|PGD|0.30|md|0.00% (population CI unavailable)|
|PGD|0.30|md_adjusted|0.00% (population CI unavailable)|
|PGD|0.50|bu|99.07% (population CI unavailable)|
|PGD|0.50|confidence|100.00% (population CI unavailable)|
|PGD|0.50|entropy|100.00% (population CI unavailable)|
|PGD|0.50|kde|0.00% (population CI unavailable)|
|PGD|0.50|lid|0.00% (population CI unavailable)|
|PGD|0.50|md|0.00% (population CI unavailable)|
|PGD|0.50|md_adjusted|0.00% (population CI unavailable)|
|detector_aware|0.20|bu|100.00% (population CI unavailable)|
|detector_aware|0.20|confidence|100.00% (population CI unavailable)|
|detector_aware|0.20|entropy|100.00% (population CI unavailable)|
|detector_aware|0.20|kde|100.00% (population CI unavailable)|
|detector_aware|0.20|lid|100.00% (population CI unavailable)|
|detector_aware|0.20|md|100.00% (population CI unavailable)|
|detector_aware|0.20|md_adjusted|100.00% (population CI unavailable)|
|source_template|0.05|bu|88.89% (population CI unavailable)|
|source_template|0.05|confidence|100.00% (population CI unavailable)|
|source_template|0.05|entropy|100.00% (population CI unavailable)|
|source_template|0.05|kde|0.00% (population CI unavailable)|
|source_template|0.05|lid|96.30% (population CI unavailable)|
|source_template|0.05|md|0.93% (population CI unavailable)|
|source_template|0.05|md_adjusted|0.93% (population CI unavailable)|
|source_template|0.10|bu|100.00% (population CI unavailable)|
|source_template|0.10|confidence|100.00% (population CI unavailable)|
|source_template|0.10|entropy|100.00% (population CI unavailable)|
|source_template|0.10|kde|0.00% (population CI unavailable)|
|source_template|0.10|lid|22.22% (population CI unavailable)|
|source_template|0.10|md|0.00% (population CI unavailable)|
|source_template|0.10|md_adjusted|0.00% (population CI unavailable)|
|source_template|0.20|bu|100.00% (population CI unavailable)|
|source_template|0.20|confidence|100.00% (population CI unavailable)|
|source_template|0.20|entropy|100.00% (population CI unavailable)|
|source_template|0.20|kde|0.00% (population CI unavailable)|
|source_template|0.20|lid|0.00% (population CI unavailable)|
|source_template|0.20|md|0.00% (population CI unavailable)|
|source_template|0.20|md_adjusted|0.00% (population CI unavailable)|
|source_template|0.30|bu|100.00% (population CI unavailable)|
|source_template|0.30|confidence|100.00% (population CI unavailable)|
|source_template|0.30|entropy|100.00% (population CI unavailable)|
|source_template|0.30|kde|0.00% (population CI unavailable)|
|source_template|0.30|lid|0.00% (population CI unavailable)|
|source_template|0.30|md|0.00% (population CI unavailable)|
|source_template|0.30|md_adjusted|0.00% (population CI unavailable)|
|source_template|0.50|bu|100.00% (population CI unavailable)|
|source_template|0.50|confidence|100.00% (population CI unavailable)|
|source_template|0.50|entropy|100.00% (population CI unavailable)|
|source_template|0.50|kde|0.00% (population CI unavailable)|
|source_template|0.50|lid|0.00% (population CI unavailable)|
|source_template|0.50|md|0.00% (population CI unavailable)|
|source_template|0.50|md_adjusted|0.00% (population CI unavailable)|

## Figure interpretation

![Observed clean rejection](clean_rejection.png)

![Accepted manipulation](defended_manipulation.png)

Color scales are fixed0-100%. Tables carry pointwise population intervals; colors do not establish significance. Compare detector-specific adaptive rows only as stress tests. The fixed longer run covers the first character per user/seed, not every character. No winner, robust-defense or journal-readiness verdict is generated.

## Limitations

Twenty calibration and36 test characters give coarse user estimates. Serial dependence, phrase/phase shift,17 conditional available cases, recorded-marker timing and fixed adaptive recipe limit inference. Rejection is not clinical throughput. BNCI2015-003 remains locked.
