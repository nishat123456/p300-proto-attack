# Current portable suite status, October15,2026

V11 stopped with10 collection errors, not failing numerical assertions: project test helper imports were shadowed by an installed third-party `tests` package, and the packaging omitted kaggle/jobs/train_decoders/run.py used by a runner test. Preserve failure evidence. Add a project tests package initializer and the missing runner to the private cloud payload; rerun the same current portable suite with exactly the same four historical raw/path exclusions. No additional test is excluded, assertion weakened or scientific gate changed. Current broad suite remains unverified until the corrected job completes. The preceding28 integration tests remain their narrower evidence, not a substitute.

V12 corrected namespace shadowing and runner omission, then exposed3 legacy flat `defense_helpers` imports that were incompatible with package imports. Normalize these3 imports to tests.defense_helpers, with no assertion changes or new exclusions. V13 remains pending; broad suite is not claimed passed.

V13 collected and ran:340 passed,11 failed,41 warnings45.50s. All11 failures were missing uploaded job metadata/runner or scripts/kaggle.sh fixtures. Add the two declared metadata files, kill-test runner and shell script to the same private payload, then rerun unchanged assertions and exclusions. This is not a340-only success claim or reason to drop the11 tests.

V14 COMPLETE:351 passed,41 warnings45.98s on Kaggle. Four historical raw/path
file exclusions remain explicit in cloud_test_scope.json; this is the full
current portable scope, not literally every repository test. No numerical
assertions were relaxed and no additional tests were excluded. Failuresv11-13
remain preserved. This verifies software behavior, not detector performance.
