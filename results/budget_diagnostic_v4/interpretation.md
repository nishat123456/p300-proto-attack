# Strict budget diagnostic, 26 October 2026

Version4 completed, covering1836 character shards and49929 settings. The report records13419 strict applied-bound violations and zero nonfinite rows. All original experimental outputs and strict checks remain unchanged. The first reported violations equal the float32-rounded declared epsilon, which exceeds the recorded Python floating-point bound. Whether every violation has that explanation and its largest magnitude is being summarized on Kaggle, not computed locally.

Private p3a-budget-summary version1 uses the completed diagnostic cloud output and tests a stricter candidate projection: round an upward-rounded float32 bound inward and verify the actual representable raw change in float64 against the original epsilon. It checks every offending epsilon and fixed synthetic scale/offset cases, without inspecting performance. This is not an epsilon/tolerance increase or applied experimental change. Any actual scientific correction requires a prospective release and preservation/accounting of original outcomes. Rerun scope is not selected from performance.

Original combined results remain withheld. Small numerical excess, if confirmed, does not itself establish harmlessness or validate the original strict budget claim.
