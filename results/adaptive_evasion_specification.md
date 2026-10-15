# Detector-aware stress-test implementation checkpoint

14 October2026. Independently implemented projected raw-signal optimization of
targeted character cross-entropy plus lambda softplus((detector_score-threshold)/
clean_score_scale). Caller supplies differentiable full decoder and detector
paths; raw budget projection each step, random restarts and best-objective
iterate retention. No victim-training or target-outcome model selection.
This is an evaluation stress test, not a novel attack claim.

Candidate fixed evaluation grid: lambda0/0.1/1/10,200steps,3restarts; also400steps
and5restarts at primary budget0.2 to check optimization saturation. Retain all
attack attempts and report strongest successful nonrejected manipulation across
attack settings separately from per-setting results. This is adversary selection,
not selecting a winning defense on target data. All perturbed trial targets and
budgets must be frozen before evaluation; primary/secondary roles pending final
protocol. Existing five raw-unit budgets stay unchanged.

Clean score scale comes only from calibration MAD with a documented positive
floor, not target attack dispersion. Threshold fixed from clean calibration;
no threshold retuning during attack. For stochastic dropout, EOT30 draws per
gradient update and independent100draw verification are candidate settings to
freeze; low-gradient failures require convergence and gradient checks, not a
robustness claim. Score computations must remain differentiable through raw
preprocessing, epoch extraction and character aggregation. Current NumPy LID/MD
scoring requires torch equivalent integration before using this engine; do not
attack a detached surrogate and call it a white-box detector-aware evaluation.

Synthetic regression cases cover zero budget/projection/reproducibility,
targeted behavior, changed evasion tradeoff, and invalid calibration scale.
Tests UNRUN queued for Kaggle after healthy coverageaudit. No outcomes produced.
