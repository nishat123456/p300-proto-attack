# Bounded OpenBMI loader candidate validation

Kaggle p3a-openbmi-loader v1 completed12 October2026, exact runner2aeed3a,
source basec335257 and runtime fingerprint retained. **Five pytest tests passed
in0.13s** on Kaggle: stored-order/index behavior and rejection of negative,
duplicate, label-mismatched and insufficient-end-boundary inputs. Five inherited
arithmetic/schema assertions also ran. Not a full repository test suite.

Fixed subjects2/3, session1, both phases: each candidate loader constructs33
training and36 test raw character trials, preserving stored order and session
identity. Every character's schedule agrees with target/nontarget labels;
original-order mismatch0 in each phase. Chronological-order mismatch8 train/20
test per subject, so blind sorting remains inappropriate.

Sampled mean-centered epoch relative errors at documented MATLAB offset-1:
subject2 train1.7365e-16,test2.4903e-16; subject3 train2.6418e-16,test3.2215e-16.
These support raw/epoch alignment for fixed inspected events, not all epochs.
Boundary checks were applied to every constructed character. Trial IDs include
subject/session/split/index; compact records preserve all IDs.

Combined with subject1 both-session audits, evidence now covers three subjects,
but only subject1 session2. It is not validation of all54 subjects or a deployed
speller. Known copy-spelling phrases and shared original schedule are research
labels, not independently measured stimulus logs. First-pair timestamp reversal
cause remains unresolved; stored-order schedule consistency does not identify
its physical cause. Native flash_codes are candidate sequence-position codes,
not verified hardware trigger identities. No production loader or chronology
claim should hide these limitations.

Next: review acquisition/stimulus trigger provenance and decide the bounded
benchmark estimand, train/calibration/evaluation partition and independent-user
sample feasibility before any large decoder/defense experiments. Run additional
predeclared metadata coverage if required, never choose users by attack results.
Do not claim Q1/Q2 readiness from loader completion.

All scientific work ran on Kaggle; only compact JSON/Markdown/log retrieved,
no EEG/code tree downloaded. Frozen gates/splits unchanged and BNCI2015-003
locked. No defense training or performance benchmark started.
