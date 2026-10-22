# Recorded computation and rejection cost

DEVELOPMENT participant19 only, no population cost inference

## Fixed source-query computation

|Seed|Hardware|Method|Median seconds|P95 seconds|
|---|---|---|---|---|
|20261013|Tesla T4|confidence|0.006543|0.012585|
|20261013|Tesla T4|entropy|0.006010|0.006445|
|20261013|Tesla T4|bu|0.064050|0.067586|
|20261013|Tesla T4|lid|0.006569|0.006805|
|20261013|Tesla T4|md|0.006513|0.006851|
|20261013|Tesla T4|md_adjusted|0.009902|0.010321|
|20261013|Tesla T4|kde|0.007894|0.008173|
|20261014|Tesla T4|confidence|0.007710|0.012589|
|20261014|Tesla T4|entropy|0.005960|0.006071|
|20261014|Tesla T4|bu|0.064205|0.066400|
|20261014|Tesla T4|lid|0.006724|0.007315|
|20261014|Tesla T4|md|0.006635|0.007691|
|20261014|Tesla T4|md_adjusted|0.009938|0.010211|
|20261014|Tesla T4|kde|0.007948|0.008230|
|20261015|Tesla T4|confidence|0.007247|0.009028|
|20261015|Tesla T4|entropy|0.005897|0.006230|
|20261015|Tesla T4|bu|0.062255|0.065998|
|20261015|Tesla T4|lid|0.006568|0.006746|
|20261015|Tesla T4|md|0.006528|0.006669|
|20261015|Tesla T4|md_adjusted|0.009787|0.010010|
|20261015|Tesla T4|kde|0.007998|0.008271|

Timing is the synchronized batch1 source query with five warmups/30 repeats. It is not measured independent-user latency or clinical throughput. Timing-only reference identity exclusion is preserved in the input provenance; deployment references are unchanged.

## All declared clean operating points

|Participant|Seed|Nominal|Method|Rejected /36|Stored ideal IID extra attempts per accepted decision|
|---|---|---|---|---|---|
|19|20261013|0.01|confidence|0/36|0.000000|
|19|20261013|0.05|confidence|0/36|0.000000|
|19|20261013|0.1|confidence|0/36|0.000000|
|19|20261013|0.01|entropy|0/36|0.000000|
|19|20261013|0.05|entropy|0/36|0.000000|
|19|20261013|0.1|entropy|0/36|0.000000|
|19|20261013|0.01|bu|0/36|0.000000|
|19|20261013|0.05|bu|0/36|0.000000|
|19|20261013|0.1|bu|0/36|0.000000|
|19|20261013|0.01|lid|1/36|0.028571|
|19|20261013|0.05|lid|1/36|0.028571|
|19|20261013|0.1|lid|3/36|0.090909|
|19|20261013|0.01|md|0/36|0.000000|
|19|20261013|0.05|md|0/36|0.000000|
|19|20261013|0.1|md|0/36|0.000000|
|19|20261013|0.01|md_adjusted|0/36|0.000000|
|19|20261013|0.05|md_adjusted|0/36|0.000000|
|19|20261013|0.1|md_adjusted|0/36|0.000000|
|19|20261013|0.01|kde|1/36|0.028571|
|19|20261013|0.05|kde|1/36|0.028571|
|19|20261013|0.1|kde|3/36|0.090909|
|19|20261014|0.01|confidence|0/36|0.000000|
|19|20261014|0.05|confidence|0/36|0.000000|
|19|20261014|0.1|confidence|1/36|0.028571|
|19|20261014|0.01|entropy|0/36|0.000000|
|19|20261014|0.05|entropy|0/36|0.000000|
|19|20261014|0.1|entropy|2/36|0.058824|
|19|20261014|0.01|bu|2/36|0.058824|
|19|20261014|0.05|bu|2/36|0.058824|
|19|20261014|0.1|bu|2/36|0.058824|
|19|20261014|0.01|lid|0/36|0.000000|
|19|20261014|0.05|lid|0/36|0.000000|
|19|20261014|0.1|lid|1/36|0.028571|
|19|20261014|0.01|md|2/36|0.058824|
|19|20261014|0.05|md|2/36|0.058824|
|19|20261014|0.1|md|3/36|0.090909|
|19|20261014|0.01|md_adjusted|2/36|0.058824|
|19|20261014|0.05|md_adjusted|2/36|0.058824|
|19|20261014|0.1|md_adjusted|2/36|0.058824|
|19|20261014|0.01|kde|1/36|0.028571|
|19|20261014|0.05|kde|1/36|0.028571|
|19|20261014|0.1|kde|2/36|0.058824|
|19|20261015|0.01|confidence|0/36|0.000000|
|19|20261015|0.05|confidence|0/36|0.000000|
|19|20261015|0.1|confidence|1/36|0.028571|
|19|20261015|0.01|entropy|0/36|0.000000|
|19|20261015|0.05|entropy|0/36|0.000000|
|19|20261015|0.1|entropy|1/36|0.028571|
|19|20261015|0.01|bu|1/36|0.028571|
|19|20261015|0.05|bu|1/36|0.028571|
|19|20261015|0.1|bu|1/36|0.028571|
|19|20261015|0.01|lid|2/36|0.058824|
|19|20261015|0.05|lid|2/36|0.058824|
|19|20261015|0.1|lid|2/36|0.058824|
|19|20261015|0.01|md|3/36|0.090909|
|19|20261015|0.05|md|3/36|0.090909|
|19|20261015|0.1|md|5/36|0.161290|
|19|20261015|0.01|md_adjusted|3/36|0.090909|
|19|20261015|0.05|md_adjusted|3/36|0.090909|
|19|20261015|0.1|md_adjusted|5/36|0.161290|
|19|20261015|0.01|kde|0/36|0.000000|
|19|20261015|0.05|kde|0/36|0.000000|
|19|20261015|0.1|kde|0/36|0.000000|

## Recorded marker proxy

|Participant|Seed|First-to-last marker plus0.8s, minimum seconds|Maximum seconds|
|---|---|---|---|
|19|20261013|13.313000|13.324000|
|19|20261014|13.313000|13.324000|
|19|20261015|13.313000|13.324000|

All36 marker proxies per participant/seed remain in the bound clean JSON. These extrema are descriptive, not uncertainty intervals. Retry overhead is a stored IID sensitivity expression, not measured repeated spelling, correction/backspace cost or a serial-dependence guarantee. No product of average timing and average retry is presented as observed latency. Participant uncertainty for primary reliability rates is reported separately; this per-seed cost table does not infer a population retry effect. No method or seed is selected.
