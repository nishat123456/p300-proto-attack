# Cloud integration failures, October 15, 2026

Reference-tests v2 passed all 18 synthetic tests in 10.38 seconds, then the real raw adaptive smoke failed its combined finite/budget/gradient assertion. The failing boolean was not emitted. A diagnostic-only rerun now writes the booleans before retaining the same strict assertion. No passing real adaptive result is claimed.

Contract-design v8 stopped at subject 52 with `schedule and true-character labels disagree`. Logs show metadata inspection through subject 51, not complete 54-subject compatibility. Preserve the partial audit and do not exclude subject 52, infer a new schedule, or unlock evaluation performance. Next inspect subject 52 event/label identity on Kaggle before any protocol freeze.

No independent performance evidence or submission readiness follows from these jobs. Raw EEG and models were not retrieved. Existing gates, splits and BNCI2015-003 lock remain unchanged.

## Diagnostic update

V4 identifies only `within_budget=false`: finite perturbation, finite nonzero raw gradient, epsilon0.3678136825561524. Float32 subtraction after projection can round beyond the declared bound when the raw baseline is large. The returned delta is now explicitly clamped to the same budget, with a large-baseline regression test; verification remains pending Kaggle. This does not loosen epsilon or assert security. Subject52 follow-up records both phases' mismatch event indices, target counts and original loader rejection without declaring compatibility or fitting a model.
