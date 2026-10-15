# Zhang/Gu2024 primary full-text review

14 October2026. Publisher PDF retrieved successfully after earlier failures;
DOI10.1088/1742-6596/2761/1/012037 title/authors independently confirmed via
Crossref registration metadata. Local source reading only, no EEG analysis.
https://iopscience.iop.org/article/10.1088/1742-6596/2761/1/012037/pdf

Sections2.1/2.3/2.4/3.1 andTable1 reviewed. CompIII P3002subjects, flash-level
binary samples, band0.1-20Hz/zscore/0-1s epochs; MI CompIV2a9subjects. Detectors
confidence,predictive entropy,Bayesian uncertainty and final-feature class KDE.
FGSM/PGD/CW, plus detector-awareCW against KDE and transfer attack. EEGNet as
defended model; P300 substituteCNN, MI substituteShallowCNN. MC50/bandwidth0.5,
PGD/FGSMepsilon0.1 andCWconfidence5. Detection includes normal samples and
misclassification-inducing attacks. Outputs emphasize detectorAUC/BCA/RCA, not
character-level independently clean-calibrated FAR or spelling rejection cost.
Do not infer unreported split/calibration details are absent from implementation.

Table1 P300 KDEAUC0.944FGSM/0.942PGD/0.984CW under detector-unaware threat,
falling0.397 for detector-awareCW. Thus firstadaptiveP300 detector evaluation is
not a valid claim. Their KDE baseline must be included in our comparison,
not omitted in favor of easier detectors. No direct score comparison until same
threat/decisionunit/partition/budget protocol.

PrintedEq7 was visually checked: positive squared-distance exponent conflicts
with text saying lower density far from source samples. Our Gaussian KDE uses
negative exponent with denominatorbandwidth**2 and stablelogmeanexp, and returns
negative logdensity as anomaly. This is explicitly a sign-corrected conventional
adaptation, not exact reproduction; no claimed author-code validation. Full
original PDF is not copied into repository; citation and review retained.

Candidate distinct focus remains character-level independent calibration,
user/session shift, uncertainty and rejection cost under recorded-marker offline
threat model. Broad detector comparison and adaptiveP300 security are precedents,
not contributions. Novelty still requires comparison to remaining full studies.
