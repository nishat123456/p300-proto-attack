# Release review packaging failure

V1 passed six numerical-envelope guards but stopped during original portable-suite collection because the review source payload omitted kaggle/jobs/train_decoders/run.py. No cohort transfer or combined performance analysis occurred. Preserve failure and restore the already established portable-package file set at the original source revision. Do not add exclusions or relax assertions.
