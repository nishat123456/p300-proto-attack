# Paper 4: independent novelty audit request

Prepared 21 October 2026. Temporary review brief for the author to paste into Perplexity Research. This is unpublished work in progress, not a completed paper or a claim of journal readiness. No access to our private repository is required to evaluate the proposal below.

## Instructions to the research auditor

Act as a skeptical EEG/BCI security reviewer. Search primary literature published or publicly posted on or before 21 October 2026, including preprints and conference papers. Determine whether this proposal has already been done substantially the same way. Check full methods and supplements where possible, not just titles or abstracts. Verify DOI, authors, year, dataset and actual decision unit. Mark inaccessible details UNKNOWN rather than assuming they are absent. Separate ordinary domain-adversarial transfer learning from malicious adversarial-example security.

Give a justified novelty score out of 100 using the rubric at the end. The score is a subjective comparison, not an acceptance probability. Do not reward software tests, private infrastructure, more model seeds, routine dataset substitution or merely combining standard metrics as novel research. Do not invent reassuring gaps or citations. If the closest paper defeats our proposed contribution, say so clearly.

## 1. Working research question

For offline P300 speller character decisions, how do existing adversarial-input detectors behave when their operating thresholds are calibrated on clean data independent of normalization, evaluated on previously unseen users, and challenged by detector-aware targeted attacks? What clean rejection and accepted-error costs accompany the apparent security benefit?

Candidate title: **Character-level reliability of adversarial detection in P300 spellers: independent calibration, user variation, adaptive evasion and rejection cost.** This title is provisional. We are not proposing a new detector or attack algorithm.

The candidate contribution is a controlled empirical reliability study. It must show an informative, reproducible finding beyond the already known fact that adaptive attacks can defeat detectors. Whether that contribution is sufficiently distinct is the central question for this audit.

## 2. Motivation and abandoned hypothesis

The initial supplied motivations were Jung et al.'s Generative Perturbation Network (IEEE JBHI, 2023) and Zhong et al.'s Attention-guided Universal Adversarial Perturbations (Expert Systems with Applications, 2025). We proposed conditioning P300 attacks on genuine response prototypes. The Competition III development pilot did not establish the proposed stealth advantage; wrong-letter prototype guidance behaved similarly. We moved to defense reliability rather than claiming a successful new attack.

Tiny Noise P300 speller attacks and existing EEG detectors are precedents, not our inventions. An AUAP reproduction was not completed. Existing detector families below are independently implemented adaptations, not certified reproductions of every author implementation.

## 3. Current prospective protocol

- Dataset: OpenBMI ERP/P300, session 1, 54 documented participants. Competition III and inspected BNCI2014-008 results are development evidence, not untouched confirmation.
- Fixed roles: subjects 1–18 source development/training; 19–36 validation; 37–54 planned evaluation. Subject 52's official training metadata fails unchanged schedule/label identity checks. Retain all 18 in evaluation flow, with 17 eligible calibrated estimates and one explicitly unavailable. No invented labels or replacement participant. This is conditional available-case performance, not all-person reliability.
- Per eligible target user: 33 official training characters partitioned into 13 normalization and 20 clean threshold-calibration characters; 36 official test characters for outcomes. Roles are disjoint. Same identities across model seeds. Phase/phrase shift means exchangeability is not established merely by splitting data.
- Decoder: fixed source-only EEGNet, three predeclared seeds, final epoch of fixed training schedule. Target-user normalization uses only the 13 normalization characters. No independent-outcome selection of architecture, threshold, winner or best seed.
- Seven detector scores: confidence, predictive entropy, dropout uncertainty (BU), local intrinsic dimensionality (LID), network-feature Mahalanobis distance (MD), confidence-adjusted MD and class-conditional Gaussian KDE. Character anomaly aggregates flashes supporting the decoder-predicted symbol, without using the true intended symbol as detector input.
- Adaptations: shared Ledoit–Wolf MD covariance; final encoder features rather than a claimed exact learned multilayer MD reproduction; explicit conventional negative Gaussian KDE exponent. These are not new-method claims.
- Calibration: clean-only higher empirical quantiles, strict greater-than rejection retaining ties; nominal 1%, 5% and 10%, primary 5%. With only 20 calibration characters, the 1% and 5% thresholds coincide. There is NO per-user 5% FAR guarantee. BU uses 100 draws for calibration and all reported scores; 30 draws only for adaptive gradient estimation.
- Threat model: additive perturbation of recorded raw EEG before preprocessing, known recorded flash markers and full white-box model/detector access for the adaptive test. This does not demonstrate a live headset attack or physical stimulus-timing control. Stored event-order anomalies remain disclosed.
- Nonadaptive baselines: targeted FGSM, targeted PGD (200 steps, 3 restarts), Gaussian control and a source-trained targeted template inspired by prior speller attacks. Five raw budget ratios: 0.05/0.1/0.2/0.3/0.5, primary 0.2. Budget scale is fixed from target normalization characters only. Actual representable raw perturbations are bounded.
- Resource-bounded adaptive recipe: each detector individually attacked at primary 0.2, fixed penalty weight 1, 200 steps/3 restarts. Longer 400-step/5-restart sensitivity on the fixed first test character per eligible user, all seeds/detectors. Ordinary PGD is the zero-penalty comparator. Original development weight/budget grid remains disclosed. This is not exhaustive worst-case robustness or one perturbation jointly evading all detectors.
- Outcomes: per-user observed clean FAR; defended targeted success over ALL original attempts; conditional detection among successful manipulations with its denominator; accepted-correct and accepted-wrong rates; clean accuracy; detector-unaware AUC and budget curves.
- Uncertainty: average seeds within each user, equal participant weighting and paired participant bootstrap, with descriptive pointwise comparisons. Exact binomial intervals are IID descriptions, not guarantees for serial EEG decisions. No false independence from counting three seeds as three participants.
- Cost: clean rejection/accepted-error accounting, recorded first-to-last marker duration plus epoch proxy, and synchronized batch-one GPU inference timing on a fixed source query. Ideal IID retry formulas are sensitivity illustrations, not measured clinical typing throughput.
- BNCI2015-003 remains locked. No final independent performance release has been issued. Final immutable protocol/manifests and remaining execution verification precede sealed outcomes.

## 4. What is actually completed versus pending

Completed development evidence: source models/references/templates; seven scoring paths; clean calibration; three-seed execution of calibration, 36 clean decisions and first-character attacks; completed-checkpoint resume checks. The latest portable cloud scope passed 366 tests. Software checks do not demonstrate scientific novelty or efficacy.

On ONE fixed validation19 character at budget 0.2, ordinary PGD was rejected by MD, adjusted MD and KDE, while confidence/entropy/BU/LID accepted it. Directly optimizing against each detector reached the wrong target and passed that attacked detector across all three seeds and nonzero tested weights, at both tested run lengths. This is not population TPR, a joint-evasion result or a new discovery that adaptive attacks exist.

Source-query timing completed after a timing-only explicit leave-query-out reference fix: the original source query could be its own LID reference. The zero-distance guard and independent detector reference sets were not loosened.

Pending: full 36-character subject-controller/aggregation validation, final freeze, the independent 17-user study, complete figures/uncertainty/cost synthesis and final manuscript. No independent result, Q1/Q2 readiness or acceptance is claimed.

## 5. Closest known primary precedents to audit

| Work and link | Known overlap | Specific remaining comparison question |
|---|---|---|
| Zhang and Gu, **Adversarial sample detection for EEG-based brain-computer interfaces**, JPCS 2024. DOI [10.1088/1742-6596/2761/1/012037](https://doi.org/10.1088/1742-6596/2761/1/012037) | Prior project full-text review found CompIII P300 and MI detectors including confidence/entropy/BU/KDE, plus detector-aware CW against KDE. Thus P300 detection and adaptive P300 detector failure are already established. | Does the full paper already implement character-level independent clean calibration, user-level FAR uncertainty and rejection cost substantially matching our proposed study? Locate exact sections. Publisher retrieval failed during today's refresh, so the earlier reviewed PDF must be independently checked. |
| Chen and Wu, **Adversarial Artifact Detection in EEG-Based Brain-Computer Interfaces**, arXiv 2022/JNE 2024. [arXiv:2212.00727](https://arxiv.org/abs/2212.00727); DOI [10.1088/1741-2552/ad8964](https://doi.org/10.1088/1741-2552/ad8964) | Multiple EEG detector families, two datasets, three CNNs, white/black-box detection; prior project review identifies subject-transfer and reference-statistic precedents. | Which clean operating-point, participant independence and adaptive rejection-cost analyses are actually reported? |
| Chen, Jia and Wu, **Data alignment based adversarial defense benchmark for EEG-based BCIs**, Neural Networks 2025. DOI [10.1016/j.neunet.2025.107516](https://doi.org/10.1016/j.neunet.2025.107516); [author repository](https://github.com/xqchen914/EEGAdversarialBenchmark) | Nine defenses, five EEG datasets, three neural networks and four scenarios. A broad multi-dataset EEG defense benchmark is not new. | Does its full paper/supplement include our calibrated character-rejection estimand? Author README is verified; do not substitute it for unreviewed full protocol details. |
| Conceição da Costa et al., **Model Comparison for Adversarial Attack Detection in EEG-Based Brain-Computer Interfaces**, ICEIS 2026. DOI [10.5220/0014819600004018](https://doi.org/10.5220/0014819600004018); [publisher](https://www.scitepress.org/PublishedPapers/2026/148196/) | Prior project review found GrazA MI/EEGNet, several attacks, conventional and uncertainty-aware detector comparison. | Is there an equivalent independently calibrated P300 character/user-cost evaluation? Abstract silence is insufficient evidence of absence. |
| Zhang et al., **Tiny noise, big mistakes**, NSR 2021. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8288388/) | Targeted speller perturbation templates and P300 attacks are established. | Which attack/repetition/transfer or character-level protocol details overlap our template stress test? |
| Mezzina et al., **A Cybersecure P300-Based BCI**, Sensors 2021. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709057/) | P300-specific integrity protection already exists, under a different trusted transformation architecture. | Is our observable-score rejection question materially different from its security assumptions and user-cost assessment? |

Also check broader selective-classification/calibration, finite-sample false-alarm and adaptive-defense evaluation literature. A new application of established calibration arithmetic alone may be incremental. The recent publisher-listed **Density-consistency framework for attack-agnostic robust inference** ([publisher lead](https://www.sciencedirect.com/science/article/pii/S0957417426024188)) is an adjacent medical-image rejection/adaptive-evaluation lead, NOT yet a verified EEG/P300 equivalent. Verify its identity and methods before using it as evidence.

## 6. Required output from Perplexity

1. Closest 5–10 primary papers, with verified DOI/URL, publication date and exact methods sections supporting overlap.
2. A matrix: P300 versus another task; flash versus character decision; user-held-out evaluation; normalization/calibration/evaluation independence; clean FAR with uncertainty; successful-attack versus all-attempt denominators; detector-aware evasion; rejection/typing/computation cost; protocol/artifact availability.
3. Classify every proposed contribution as already done, incremental combination, plausibly distinct, or unknown. Distinguish same-to-same duplication from only sharing a detector or dataset.
4. Give two separate novelty scores: the work actually completed and the proposed completed independent study. Use a range and confidence level. Explain which missing primary source could change your score. Never turn either score into a journal acceptance probability.
5. Explicitly answer: if our independent study merely repeats known adaptive detector failure on more users, is there enough contribution? What minimally necessary, genuinely distinct result would change that judgment? Do not prescribe a new method without checking its precedents.
6. Recommend proceed, narrow/reframe, or stop. Provide the strongest argument AGAINST proceeding and a concrete claim sentence we can defend without saying “first.”

### Scoring rubric (100 total; subjective, not validated)

- Research-question/estimand distinction from closest prior work: 25.
- Nontrivial protocol or methodological contribution beyond routine good practice: 20.
- Generalizable empirical insight beyond known detector failure: 25.
- Evidence breadth and credible independent validation: 20.
- Reusable research artifact value beyond ordinary code release: 10.

Assign zero where unsupported. Show component scores, avoid double-counting and separately rate methodological soundness and completion status. Do not inflate novelty because the study is expensive or the authors want a Q1/Q2 venue.
