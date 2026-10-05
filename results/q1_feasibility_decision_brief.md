# Paper 4: literature gap and data feasibility decision brief

Prepared 5 October 2026 for professor review. Defense-only scope. This is a
source-grounded planning assessment, not a new experiment or journal promise.
All tests, EEG processing and scientific analysis remain Kaggle-only.

## Recommendation

**Do not expand the current detector as a validated new defense. Proceed with
one bounded feasibility stage for a character-level defense-reliability study.**
Its publishable novelty is not established yet. If full-text comparison or data
compatibility fails, stop or narrow to a reproducibility/negative-result report.
A Q1 submission needs a substantial finding and credible independent evidence;
a particular detection percentage is not a journal acceptance criterion.

Candidate question, not yet a frozen protocol: how well do source-trained P300
attack detectors retain useful character-level detection after independent
clean-user calibration, across sessions, unseen attack families and users, and
what clean spelling cost does rejection impose?

The contribution would be the demonstrated operating-point failure mechanisms
and a reproducible evaluation, rather than inventing logistic regression,
spatial features, calibration, or adversarial EEG detection. Do not claim first.

## Closest literature and overlap

|Primary source|What it establishes|Implication for our study|
|---|---|---|
|[Chen and Wu, Adversarial Artifact Detection](https://arxiv.org/pdf/2212.00727), JNE2024 DOI10.1088/1741-2552/ad8964; reviewed preprint|BU, LID and network-feature Mahalanobis detection; ERN/MI results, leave-one-subject-out, nested tuning, ROC-AUC; discusses attack-dependent differences|EEG detector comparison and subject transfer already exist. Our engineered-feature Mahalanobis is not an exact reproduction. Character-level independently calibrated operating points and spelling cost are candidate distinctions, not proven unique. A P300 illustration is present but the reported experimental tables use ERN/MI.|
|[Mezzina et al., Sensors2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709057/)|P300 noise/fake-signal attacks and synchronized channel-mixing authentication; five subjects, simulated transmission attacks, secret seed assumption|P300 authenticity and channel-based defense already exist. Its trusted headset/channel architecture differs from post-injection signal-only detection; its99.996% simulated rejection cannot be compared directly with our TPR. It also discusses counteraction and latency, so practical cost is not entirely unexplored.|
|[Zhang et al., Tiny Noise, NSR2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8288388/)|Targeted P300/SSVEP speller manipulation and transfer|Character-level attack success and physiological signal structure already have precedent. Use its threat model/baseline carefully; attacks are stress tests, not our claimed contribution.|
|[Conceicao da Costa et al., ICEIS2026](https://www.scitepress.org/publishedPapers/2026/148196/pdf/index.html), DOI10.5220/0014819600004018|Publisher-indexed full-text excerpts describe RF/SVM/KNN detection with an80/20 split, FGSM/DeepFool examples, ANFIS and ROC-AUC|Another conventional detector comparison already exists. Direct PDF retrieval remains intermittently inaccessible; dataset/subject separation and all calibration details are not verified. Do not encode missing evidence as an absent experiment.|
|[On Adaptive Attacks to Adversarial Example Defenses](https://arxiv.org/abs/2002.08347)|Defense-specific adaptive evaluation methodology|Non-adaptive success does not justify robust security claims; a future detector must be attacked jointly with the speller, including preprocessing/threshold logic.|

Searches also surfaced an August2026 EEG-defense survey
[publication page](https://www.sciencedirect.com/science/article/pii/S0920548926000292)
and the title “Adversarial sample detection for EEG-based brain-computer
interfaces” in a secondary listing. Full source identification/access for the
latter remains unresolved. These are explicit review gaps, not evidence of
novelty. A systematic exhaustive review has not been completed.

## What our verified evidence supports

- Original registered prototype stealth advantage failed in the CompIII pilot.
- Corrected feature baselines show weak, direction-dependent held-out prototype
  detection. Historical budget-overwrite results remain superseded.
- The fixed804-dimensional repair improves nearest causal detection12.4% to
  30.6% A-to-B at budget0.2, but worsens39.2% to3.6% B-to-A; Tiny Noise detection
  also worsens. This does not support a general improvement.
- The score audit shows that high AUC can coexist with poor clean-calibrated
  upper-tail detection. This mathematical distinction is not itself novel;
  a broader empirical study must establish its magnitude and consequences.
- The independent BNCI2014-008 clean audit failed its frozen all-subject gate.
  It measured no external attack TPR. CompIII and inspected BNCI2014-008 remain
  development evidence, not untouched confirmation.

Evidence: results/channel_time_comparison_report.md,
results/defense_score_audit_report.md, results/defense_baselines_v2,
results/bnci_independent_cal and appended PREREG history. Exact804 runner
 e4b1a23, source overlay ad121be, result commit3594569. Remote comparison passed
299 portable tests. No new measurements were generated for this assessment.

## Data feasibility: count decisions, not flash epochs

|Dataset|Independent character evidence available|Feasible role and limitation|
|---|---|---|
|CompIII A/B|Frozen70/15/100 per subject; official test already inspected|Development and implementation checks only; two people cannot support broad subject-generalization claims. Repeated attack targets are not new clean characters.|
|BNCI2014-008|Eight subjects; frozen24/5/6 characters|Development compatibility stress test. Five calibration and six test characters cannot establish precise5% per-user FAR. Do not change frozen splits to manufacture precision.|
|BNCI2015-003|Ten subjects; official5/5 characters in our verified audit|Remains locked. Too few per-user decisions for a precise low-FAR claim, even if used eventually; do not unlock now.|
|[Lee2019/OpenBMI ERP, official MOABB documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_ERP.html)|54 participants, two sessions;33 training and36 test characters per documented session;1980/2160 are flash epochs|Best currently identified feasibility candidate for subject/session-level reliability. More participants improve group evidence; they do not give thousands of independent per-user characters. Character order/group membership/raw metadata must be validated on Kaggle before selecting it.|
|[Brain Invaders bi2014a official deposit](https://doi.org/10.5281/zenodo.3266223)|Different game selection protocol; character-equivalent counts not audited here|Possible second external protocol only after reconstructing independent selections and stimulus membership. Target/non-target epochs alone are insufficient for chosen-character attack evaluation. Do not promise a subject count from unverified loader subsets.|

A statistical design must separate normalization, threshold calibration and
clean evaluation. It must specify whether FAR is per person, average across
people, or a session-level property. Pooling participants does not validate a
per-person guarantee, and flash/target repetitions are not independent sample
multipliers.

For an idealized independent Bernoulli clean-evaluation sample with zero alarms,
the one-sided95% exact upper bound is 1 - 0.05^(1/n). This formula illustrates
why a zero count in six characters proves little; it is not an EEG guarantee.
Likewise an exchangeable rank-based calibration rule has tail resolution tied
to1/(n+1). Dependence, nonstationarity and source/target shift require explicit
assessment, not an unsupported distribution-free claim. Sample-size/uncertainty
calculations and sensitivity analysis must be executed on Kaggle, not locally.
No precision budget or statistical power has been computed in this brief.

## Concrete next work and decision boundaries

1. Complete the closest full-text matrix, including the2026 comparison and
   unresolved2024 title. Record threat model, task/decision unit, partitions,
   attack-family holdout, calibration data, operating-point metrics, adaptive
   evaluation and clean cost. Unknown entries remain unknown. Stop the novelty
   claim if an equivalent study already exists without a substantive extension.
2. Prepare a Kaggle-only metadata/loader audit for OpenBMI, using a predeclared
   small subject subset without choosing subjects by detection outcome. Verify
   independent character counts, session IDs, stimulus memberships, target
   labels, channel names/references and file access. Output compact audits only;
   no raw downloads and no confirmation-result inspection. This is not yet an
   attack or defense performance run.
3. Run a cloud-only sample-size and uncertainty design for a clearly specified
   group/session estimand. If public data cannot support per-user5% assurance,
   narrow that claim explicitly or seek additional data. Do not reuse correlated
   flash epochs as clean spelling decisions.
4. Only if novelty and compatibility survive, propose an append-only registered
   benchmark: network-feature BU/LID/Mahalanobis reference implementations,
   existing engineered controls, training-only tuning, independent calibration,
   held-out subjects/sessions/families, multiple seeds and subject-level
   uncertainty. Current baseline implementations are not sufficient substitutes
   for literature reproduction. No method winner selected from target results.
5. Evaluate end-to-end defended targeted success among all attack attempts as
   well as conditional detector TPR, clean character accuracy/rejection,
   genuine-error false alarms, and spelling-delay cost under an explicit retry
   policy. Use predicted-character evidence for deployment claims; oracle true
   labels are permitted only for labeled research analysis and must be named.
6. Freeze a defense-aware stress test with attack success plus detector evasion,
   budgets injected before preprocessing, adequate optimization checks and no
   victim-assisted method selection. Negative outcomes remain reportable.

No new scientific method or confirmatory protocol is frozen by this brief.
Existing splits/gates remain unchanged and BNCI2015-003 remains locked.

## Pursue-or-stop conclusion

**Pursue the feasibility audit, not another outcome-driven detector repair.**
There is a plausible narrower reliability contribution, but neither exhaustive
novelty nor adequate external evidence is established. If only the current
small datasets are usable, a broad low-FAR defense claim is unsupported; prepare
a transparent pilot/negative-result paper for professor review instead. If a
larger subject/session benchmark is feasible, seek approval of its frozen scope
before large experiments. A manuscript outline by11 October is possible; a
Q1-ready completed validation or publication by then is not currently supported.
