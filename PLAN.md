# Paper 4: Prototype-Guided Attacks on P300 Spellers, and How to Detect Them

**Working title:** Physiologically Grounded Attacks on P300 Spellers: Cross-Subject Prototype Guidance, Detectability, and a Genuine-vs-Induced ERP Defense

**Target venues:** IEEE TNSRE or IEEE JBHI (primary), J. Neural Engineering (stretch), Computers in Biology and Medicine (fallback). Verify SJR/JCR quartile before submission.

**Status (2026-09-25):** fresh rebuild from scratch. The earlier P1-P4 work is not reused; its design is replaced by this plan.

## ACTIVE: 10-day sprint (2026-09-25 to 2026-10-04), Kaggle T4

The 5-month plan below is the full version. The sprint delivers a complete first draft with real results on a reduced scope. Code: `p300-proto-attack/` in this folder.

| Day | Date | Work | GPU (T4) |
|---|---|---|---|
| D1 | Sep 25 | P0 + P1: repo, Comp III loader, tests, frozen splits, PREREG.md (**done**); verify BNCI codes (**done**: 2014-008 OK, 2015-003 OK, 2014-009 dropped) | 0 |
| D2 | Sep 26 | P2: differentiable preprocessing, EEGNet surrogate, DeepConvNet/Shallow/xDAWN-LDA victims, letter head; Kaggle kernel pipeline via CLI; train A and B | ~1 h |
| D3 | Sep 27 | P3 bank + retrieval; attack code: GPN port, proto-GPN (both forms), PGD, Tiny Noise, noise | ~0.5 h (smoke) |
| D4 | Sep 28 | P4 kill test A <-> B, eps sweep, clean-vs-attacked detector | ~6 h |
| D5 | Sep 29 | GATE; genuine-vs-induced defense; BNCI 2014-008 + 2015-003 loaders + decoders | ~3 h |
| D6-D7 | Sep 30 - Oct 1 | Full leave-one-subject-out runs, 3 datasets, all baselines and victims | ~12-15 h |
| D8 | Oct 2 | Defense eval, simple adaptive attacker, jitter/repetition robustness, stats, figures | ~3 h |
| D9-D10 | Oct 3 - 4 | Full paper draft + supplement | 0 |

**GPU budget:** ~25-30 T4 hours total. Kaggle gives 30 h/week (12 h max per session), and the sprint spans two quota weeks, so it fits with margin for reruns. Use both T4s (2x T4 accelerator) to run two jobs in parallel.

**Cut from the sprint (future work or revision):** Lee2019 and Brain Invaders (dataset size and loader time), expert rating (so no IRB needed), AUAP sparse mask, multi-seed attacks (decoders keep 3 seeds; attacks use bootstrap CIs). Sprint scope: 3 datasets, 20 subjects (Comp III 2, BNCI 2014-008 8, BNCI 2015-003 10).

**Dataset check (2026-09-25):** BNCI 2014-008 has per-flash Row1-6/Col1-6 codes, 35 chars/subject: usable. BNCI 2015-003 has per-flash codes 1-36 plus a target flag: usable through the flash-group head (exact code meaning to confirm in the loader). BNCI 2014-009 flashes groups Gr1-12 with no symbol-membership map in the file and ~6 chars per file: dropped from the sprint.

**Changelog**
- v3 (2026-09-25): 10-day sprint schedule on Kaggle T4; scope reduced as above.
- v2 (2026-09-25): added defense as a full contribution; perturbation injected into raw signal before a differentiable preprocessing pipeline; letter head generalized from row/column to arbitrary flash groups (unlocks Lee2019 and Brain Invaders); matched-success fallback metrics; optional blinded expert rating; timeline revised to about 5 months.
- v1 (2026-09-25): initial rebuild plan.

---

## 1. Research question and contributions

Prior work already shows P300 spellers can be forced to output an attacker-chosen character (Zhang et al., "Tiny noise, big mistakes", NSR 2021, same BCI Comp III dataset, >90% success). GPN (Jung et al., JBHI 2023) already generates target-conditioned EEG perturbations. So "we can attack a P300 speller" is not a contribution.

**Contributions**
1. **Attack:** a prototype-guided generator that uses genuine target-letter ERPs (from other people) to build targeted perturbations, in both a signal-specific and a causal (precomputable) form.
2. **Defense:** a genuine-vs-induced ERP detector that flags attack-induced P300 responses, evaluated against all attacks including our own.
3. **Evidence:** a multi-dataset, subject-held-out comparison of targeted speller attacks and their detectability, with the perturbation injected into the raw signal before the speller's preprocessing.

**Hypotheses (pre-registered)**
- **H1 (attack):** at matched targeted success and budget, prototype-guided perturbations are harder to detect and more physiologically plausible than (a) target-only GPN and (b) Tiny Noise templates, including when prototypes come from other subjects.
- **H2 (defense):** the genuine-vs-induced detector detects Tiny Noise, PGD, and target-only GPN attacks at high TPR at 5% FPR, with a measurable (reported, not assumed) drop against prototype-guided attacks.

**Primary outcome (H1):** detector AUC (attacked vs clean) at matched targeted success rate (80%), with the fallback in Section 6 if 80% is not reached.
**Primary outcome (H2):** TPR at 5% FPR of the genuine-vs-induced detector, per attack.
**Secondary outcomes:** L2 norm at matched success; surrogate-to-victim transfer; ERP plausibility score; cross-subject success; detector cost (false alarms on clean users).

The paper is publishable under either H1 outcome: if prototypes win, "a stealthier attack exists and here is what still catches it"; if they lose, "current attacks, including physiologically guided ones, are detectable by our defense." See Section 10.

---

## 2. Threat model (state explicitly in the paper)

| Assumption | Setting |
|---|---|
| Model access | White-box to a **surrogate** only. Victim is never queried during training or tuning. |
| Victim preprocessing | Unknown to attacker in the transfer setting; attacker uses a standard pipeline (Section 5.8). |
| Injection point | **Raw recorded signal, before any filtering, re-referencing, or downsampling** (e.g. compromised amplifier/software, or EM pickup as in Tiny Noise). The attack must survive the speller's own preprocessing. |
| Training EEG | Attacker has labeled EEG from **other people** (cross-subject bank). Same-subject bank is reported as an upper bound only. |
| Flash timing | Attacker knows stimulus onsets (same assumption as Tiny Noise). Timing-jitter robustness is tested. |
| Causality | **Signal-specific** (generator sees the source trial; not causal; upper bound) and **causal/universal** (generator sees only target letter + prototype + flash schedule; precomputable; the fair comparison to Tiny Noise). |
| Defender | Has clean calibration data from the user and genuine target ERPs; does not know which attack is used (non-adaptive) or knows the attack family (adaptive). |

---

## 3. Datasets

The method uses a **flash-group representation** (Section 5.1): for every flash, the set of symbols that lit up. Row/column spellers are the special case where each group is one row or one column. Any dataset qualifies if per-flash group membership (or row/column codes) and the spelled symbol are recoverable. Target/NonTarget labels alone are **not** enough.

| Dataset | Subjects | Channels / Fs | Paradigm | Role | Check before use |
|---|---|---|---|---|---|
| BCI Comp III Dataset II | 2 (A, B) | 64 / 240 Hz | 6x6 row/column | Development, kill test (A to B) | Use official 85 train + 100 test chars (test labels are public). |
| BNCI 2014-008 (ALS) | 8 | 8 / 256 Hz | 6x6 row/column | Clinical relevance, cross-subject | Row/col codes in original .mat; MOABB may expose only Target/NonTarget |
| BNCI 2014-009 | 10 | 16 / 256 Hz | 6x6 row/column | Cross-subject | Same check |
| Lee2019 ERP (OpenBMI) | 54 | 62 / 1000 Hz | 6x6, random-set flashing | **Scale: main cross-subject result** | Confirm per-flash symbol sets are in the release |
| Brain Invaders (bi2014a / bi2015a) | ~40+ | 16-32 / 512 Hz | 36 targets in 12 flash groups | Optional generality | Confirm group membership and target identity are recoverable |
| BNCI 2015-003 | 10 | 8 / 256 Hz | 6x6 | Optional | Same check |

Target: **at least 3 datasets and 60+ subjects** in the final paper. Cross-subject prototype banks are built **within a dataset** (same montage and layout). No mixing across montages.

**Comp III layout:** `ABCDEF / GHIJKL / MNOPQR / STUVWX / YZ1234 / 56789_`. Codes 1-6 columns, 7-12 rows. 15 reps x 12 flashes = 180 flashes per char. SOA 175 ms (42 samples at 240 Hz).

---

## 4. Repository layout

```
p300-proto-attack/
  pyproject.toml            # pinned deps
  configs/                  # one YAML per experiment, all hyperparams live here
    data/  preproc/  model/  attack/  defense/  eval/  experiment/
  src/p3a/
    data/
      loaders/              # comp3.py, bnci.py, lee2019.py, braininvaders.py -> unified Trial
      trial.py              # Trial: raw eeg[C,T], onsets, flash_groups[F, n_symbols] (bool),
                            #        target_symbol, subject, dataset, layout, fs
      epochs.py             # flash epoching, target-conditioned epoching
      splits.py             # frozen splits, written once to splits/*.json
      layout.py             # symbol sets per layout; row/col -> flash-group conversion
    preproc/
      pipeline.py           # differentiable: bandpass (FIR conv), re-reference, decimate, scale
    models/
      eegnet.py  deepconvnet.py  shallowconvnet.py
      xdawn_lda.py          # classical victim (Tiny Noise's pipeline family)
      letter_head.py        # flash scores + flash_groups -> symbol logits
      speller.py            # raw EEG -> preproc -> flash decoder -> letter head (end to end)
    prototypes/
      bank.py               # HDF5 bank, strict split + subject filtering
      features.py           # target-conditioned ERP features
      retrieval.py          # random / medoid / nearest / wrong-letter
    attacks/
      gpn.py                # original GPN (ported from AIRLABkhu repo)
      proto_gpn.py          # prototype-conditioned GPN (signal-specific + causal)
      pgd_letter.py         # non-generative targeted PGD baseline
      tiny_noise.py         # Tiny Noise template reproduction
      noise.py              # amplitude-matched random noise
      sparse_mask.py        # optional AUAP-style channel/time mask
    defenses/
      genuine_vs_induced.py # main defense (Section 5.9)
      baselines.py          # clean-vs-attacked CNN, band-power anomaly, ERP-Mahalanobis
    eval/
      success.py            # targeted success, per subject
      budget.py             # Linf, L2, RMS, SNR; eps sweep; matched-success + AUC fallbacks
      plausibility.py       # waveform corr, P300 latency/amplitude, topography corr, PSD distance
      rating_export.py      # blinded ERP panels for expert rating
      stats.py              # bootstrap CIs, Wilcoxon, mixed-effects, Holm
    utils/  seed.py  checkpoint.py  state.py  logging.py
  scripts/                  # 01_prepare_data.py ... 10_make_figures.py
  tests/                    # pytest, CPU, < 2 min
  splits/  prototype_banks/  checkpoints/  results/   # large artifacts gitignored
  PLAN.md  PREREG.md  README.md
```

**Stack:** Python 3.11, PyTorch 2.x, MNE, braindecode (EEGNet/Deep4 references), pyriemann (xDAWN), MOABB (downloads only; loaders are ours), h5py, YAML configs, pytest, statsmodels, pandas, matplotlib.

**Compute:** one GPU (Lightning/Kaggle/lab) is enough. Data prep and tests run on CPU. Lee2019 at 1000 Hz needs disk (~tens of GB raw); cache the decimated version.

---

## 5. Core components

### 5.1 Differentiable letter head (flash-group form)
Each trial has flash scores `s[f]` from the per-flash decoder and a boolean membership matrix `G[f, k]` (flash f lit symbol k).
```
evidence[k] = sum_f G[f,k] * s[f] / sum_f G[f,k]      # mean score of flashes containing k
L_target    = CrossEntropy(evidence / tau, target_symbol)
```
For row/column spellers this equals the usual row score + column score up to scale. Unit tests: (1) on clean Comp III data, argmax matches the standard row/col decision; (2) the row/col and flash-group code paths give identical rankings.

### 5.2 Target-conditioned features
For a source trial and attacker target k*: epoch the source at flashes with `G[f, k*] = 1`, average. This is the query feature. Never reuse the source's own target labels. For row/col datasets this also splits into row-part and column-part features.

### 5.3 Prototype bank
One entry per genuine character trial: raw `eeg`, `onsets`, `flash_groups`, `target_symbol`, `subject_id`, `dataset`, `layout`, `split`, `preproc_version`. Features computed after the standard preprocessing (Section 5.8) for 0-800 ms. Standardization stats fitted on bank members only.
Modes: `same_subject` (upper bound), `cross_subject` (attacked subject fully excluded).

### 5.4 Retrieval
`random`, `medoid`, `nearest` (standardized Euclidean; also correlation distance), and **`wrong_letter`** (nearest prototype of a different letter; causal control).

### 5.5 Prototype-guided GPN
- Signal-specific: `delta = G(x_raw, onehot(target), proto_feat)`
- Causal: `delta = G(onehot(target), proto_feat, flash_schedule)`
- `x_adv_raw = x_raw + clip(delta, -eps, eps)`; the loss is computed on `speller(x_adv_raw)` through the differentiable preprocessing.
- `L = L_target + lambda_p * D(F_target(P(x_adv_raw)), F(proto)) + lambda_s * ||delta||^2/(C*T)`, where P is the preprocessing.
- Surrogate frozen. Victims never touched.

**Redundancy check (mandatory):** `L_target` already pushes a P300 onto target flashes. Guidance must beat `lambda_p = 0` (target-only GPN) and the `wrong_letter` control.

### 5.6 Baselines (all in the main table)
1. Tiny Noise (their algorithm, re-fit against our surrogate through our preprocessing; plus their original xDAWN+BLDA setting)
2. Targeted PGD with the letter loss (non-generative)
3. Target-only GPN (`lambda_p = 0`)
4. Proto-GPN: random / medoid / nearest / wrong-letter
5. Amplitude-matched random noise
6. Optional: AUAP-style sparse channel/time mask on the best method

### 5.7 Victims
Surrogate: EEGNet. Victims: DeepConvNet, ShallowConvNet, xDAWN+LDA, each trained separately with different seeds. Report clean letter accuracy for every model and subject before any attack.

### 5.8 Preprocessing inside the speller (differentiable)
Standard P300 pipeline implemented as fixed torch ops: bandpass 0.5-20 Hz (FIR as conv1d, zero-phase via forward-backward or linear-phase with delay compensation), common average reference (optional per dataset), decimation to ~60-64 Hz with anti-alias, per-channel scaling fitted on train. Attacks optimize through it; the perturbation lives at raw sampling rate.
**Transfer test:** victim uses a *different* pipeline (e.g. 0.1-30 Hz, no CAR, different decimation) to show robustness to unknown preprocessing.
Unit test: numerical agreement with the MNE equivalent to within tolerance.

### 5.9 Defense: genuine-vs-induced ERP detector
- **Input:** per-trial target-conditioned ERP for the decoded letter (the response the speller's decision relied on), plus its spatial pattern and the residual (trial minus user's clean template).
- **Model:** small CNN or Riemannian (xDAWN covariance + tangent space + logistic regression); pick on validation.
- **Training data:** genuine target ERPs vs ERPs induced by a **training pool of attacks** (PGD, target-only GPN, noise) at several eps. Tiny Noise and proto-GPN held out for testing, so detection of unseen attack families is measured.
- **Calibration:** per-user threshold from clean calibration data at 5% FPR.
- **Evaluations:** non-adaptive (attacker unaware); adaptive (attacker adds a detector-evasion term to its loss; report the resulting success-vs-detection trade-off curve). Report the false-alarm cost on clean users and the effect on clean spelling speed if flagged trials are re-spelled.

---

## 6. Evaluation protocol

**Matched-success comparison:** sweep eps on a fixed grid for every method. Interpolate norm and detector metrics at 80% targeted success.
**Fallbacks (always reported, so the comparison never breaks):**
- If a method does not reach 80% within the grid: report the highest success it reaches, and exclude it from the 80% row with a footnote rather than extrapolating.
- **Area under the success-vs-eps curve** (normalized).
- **Detector AUC at success levels 50 / 65 / 80%**, shown as curves.

**Targeted success:** fraction of (source trial, target symbol) pairs where the speller outputs the target. Targets: a fixed pre-registered random subset of 10 non-source symbols per trial (same subset for every method). No cherry-picking.

**Stealth (attack side):** clean-vs-attacked CNN (trained on held-out subjects and a held-out eps); band-power anomaly; ERP-Mahalanobis; the genuine-vs-induced detector from 5.9. AUC and TPR at 5% FPR.

**Plausibility:** waveform correlation with genuine target ERP, P300 peak latency and amplitude vs genuine distribution, topography correlation at peak, PSD distance.

**Optional blinded expert rating:** 2-3 raters with EEG experience see pairs of averaged target ERPs (genuine vs attack-induced, randomized, method hidden) and choose which is genuine plus a 1-5 realism score. ~100 pairs per rater. Report per-method identification accuracy (50% = indistinguishable) and inter-rater agreement (Fleiss' kappa). Confirm whether IRB review is needed with the university before collecting.

**Robustness:** flash-timing jitter (0, 10, 20, 50 ms); repetitions (5, 10, 15); victim preprocessing mismatch (5.8).

**Statistics:** all results per subject; bootstrap 95% CIs; paired Wilcoxon (method vs target-only GPN; method vs Tiny Noise); linear mixed-effects with subject and dataset as random effects; Holm correction; at least 3 seeds per model.

---

## 7. Phases, deliverables, gates

| Phase | Work | Deliverable | Est. |
|---|---|---|---|
| **P0** Setup | Repo, env, CI tests, configs, seed/checkpoint/state utils. Write and date-commit PREREG.md (hypotheses, primary outcomes, eps grid, success levels, target subsets). | `PREREG.md` | 3 days |
| **P1** Data (Comp III) | Loader to unified Trial with flash groups; onset detection; official train/test; train/val inside the 85; tests (180 flashes, 30 targets, row/col == flash-group). | `splits/comp3_*.json`, passing tests | 4 days |
| **P2** Speller | Differentiable preprocessing (with MNE agreement test); EEGNet surrogate; DeepConvNet/Shallow/xDAWN-LDA victims; letter head; clean accuracy A and B. | Clean accuracy table | 1.5 weeks |
| **P3** Bank + retrieval | Bank (same/cross-subject), features, 4 retrieval modes, per-symbol prototype counts. | Bank HDF5, sanity plots | 4 days |
| **P4** Kill test | Target-only GPN, proto-GPN (nearest/random/wrong-letter), Tiny Noise, PGD, noise; eps sweep; clean-vs-attacked CNN; first version of the genuine-vs-induced detector; plausibility. Subject A to B. | Kill-test report | 3 weeks |
| **GATE 1** | Section 10. | Go / pivot | |
| **P5** Datasets | Lee2019, BNCI 2014-008/009 (Brain Invaders optional) loaders with flash groups; decoders per subject. | Clean accuracy per subject | 2.5 weeks |
| **P6** Full attack experiments | Leave-one-subject-out banks, all baselines, all victims, both GPN variants, 3 seeds, preprocessing-mismatch transfer. | Results parquet | 3 weeks |
| **P7** Defense | Final genuine-vs-induced detector; unseen-attack evaluation; adaptive attacker trade-off; false-alarm cost; baseline detectors. | Defense tables | 2 weeks |
| **P8** Stealth + robustness + rating | Plausibility metrics, jitter, repetitions, optional expert rating. | Tables + rating results | 1.5 weeks |
| **P9** Analysis | Stats, retrieval-distance vs cost, ablations (lambda_p, window, distance metric, sparse mask). | Figures | 1.5 weeks |
| **P10** Writing | Paper, supplementary, public code, reproducibility checklist. | Submission | 3 weeks |

Rough total: **about 5 months** of focused work for one person; less if split. Kill-test answer about mid-November 2026 if started now.

Cut from scope: word/sentence extension, language-model-assisted spellers, physical EM injection. Mention as future work.

---

## 8. Reproducibility rules

- Splits written once to `splits/*.json` and never regenerated; tests assert their hashes.
- Every run saves config, git commit, seed, data version, bank version, preprocessing version.
- Checkpoints hold model, optimizer, epoch, config, seed, best val metric. `results/state/current_phase.json` tracks progress; resume from latest valid checkpoint.
- Test data untouched until P6. Validation only for early stopping and hyperparameters.
- Victims never used for selection or tuning. Held-out attack families never used to train the defense.
- Large artifacts (raw EEG, banks, checkpoints) outside git; code, configs, splits, summary CSVs in git.

---

## 9. Paper outline

1. Introduction: communication BCIs and ALS users; integrity risk; what Tiny Noise and GPN established; open questions (stealth, physiological realism, detection).
2. Related work: EEG adversarial attacks, speller attacks, UAPs, generative attacks, feature-guided attacks in vision, EEG adversarial detection and defense.
3. Threat model (Section 2), including raw-signal injection.
4. Method: flash-group speller model, differentiable preprocessing, target-conditioned features, prototype bank and retrieval, proto-GPN (both variants), genuine-vs-induced defense.
5. Experimental setup: datasets, decoders, baselines, matched-success protocol and fallbacks, detectors, statistics.
6. Results: clean accuracy; success vs eps; matched-success norm; transfer incl. preprocessing mismatch; stealth; plausibility and expert rating; cross-subject; defense (unseen attacks, adaptive attacker, false-alarm cost); robustness; ablations.
7. Discussion: implications for speller safety, recommended defenses, limitations (timing knowledge, injection access, dataset scope).
8. Conclusion.

---

## 10. Decision gate after P4

| Result on Subject A to B | Decision |
|---|---|
| Proto-GPN beats target-only GPN **and** Tiny Noise on the H1 primary outcome, without losing the others; wrong-letter control does worse | **Go**: full attack + defense paper as planned |
| Beats them same-subject only | Go; rely on P5-P6 cross-subject results before claiming H1 |
| No advantage over target-only GPN, or wrong-letter works as well | **Reframe as defense-led paper**: genuine-vs-induced detector as the main contribution, prototype guidance reported as a negative result and as an adaptive-attack stress test. Continue P5-P10. |
| Defense detector also fails on known attacks (TPR at 5% FPR near chance) | Stop and reassess with the team before scaling up |

---

## 11. Open items before P0

- Confirm who owns the paper, author order, and who runs compute.
- Confirm GPU environment (Lightning / Kaggle / lab machine).
- Confirm flash-group or row/column code availability: BNCI 2014-008, 2014-009, Lee2019, Brain Invaders.
- Check with the university whether the optional expert rating needs IRB review.
- Read AUAP (paper 6) fully; decide whether the sparse mask is a baseline, an ablation, or dropped.

## References
- Zhang, Wu et al. Tiny noise, big mistakes. National Science Review 8(4), 2021. Code: github.com/ZhangXiao96/Speller-Attacks
- Jung et al. Generative Perturbation Network for Universal Adversarial Attacks on BCIs. IEEE JBHI 27(11), 2023. Code: github.com/AIRLABkhu/Generative-Perturbation-Networks
- Zhong et al. Attention-guided universal adversarial perturbations for EEG-based BCIs. Expert Systems with Applications 268, 2025.
- Liu et al. Universal adversarial perturbations for CNN classifiers in EEG-based BCIs. arXiv:1912.01171.
- Lee et al. EEG dataset and OpenBMI toolbox for three BCI paradigms. GigaScience, 2019.
- Sabour et al. Adversarial manipulation of deep representations. ICLR 2016.
