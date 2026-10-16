# Fixed-character development attack audit, 16 October 2026

Cloud-generated source: development_attack_summary_v2/out/attack_summary.md and JSON. Exact summary runner:2592cd7. Optimization source3eefe17, launcher9cae903. All three seeds each76 settings, fixed subject19 first test character, target13 versus intended15. This is one character repeated across model seeds and settings, not independent participants or a TPR estimate.

## Observed results

At all five budgets, FGSM, source template and PGD reach the fixed wrong target in all three models; Gaussian controls do not. At primary0.20, ordinary targeted PGD is accepted by confidence, entropy, BU and LID, while MD, adjusted MD and KDE reject it in each model.

For every detector individually, its detector-aware attack with each nonzero weight0.1/1/10 both reaches the target and passes that detector at primary0.20 in each seed. This holds at both200steps/3restarts and400steps/5restarts. It does not show that one perturbation evades all detectors simultaneously. All seven scores are preserved for every perturbation in JSON.

The clean-probe detector gradients are finite and nonzero for every method in all seeds. All recorded rows are finite and respect actual representable raw budgets; traced best objectives are nonincreasing. These checks do not establish gradient quality everywhere or physical feasibility.

Longer optimization reduces several LID/MD/adjusted-MD/KDE objective endpoints, but does not change the displayed target-plus-own-detector acceptance on this character. Confidence/entropy/BU endpoints largely plateau. This is evidence against claiming convergence from binary outcomes alone, not proof of global convergence. The longer BU nonzero-weight paths take about233seconds per setting versus about70seconds at the shorter setting; these are optimization timings, not detector inference latency or clinical spelling cost.

## Implication and next gate

The audited detectors cannot be described as robust under this white-box recorded-marker threat model. Apparent rejection of ordinary PGD by the distance/density baselines disappears under direct detector-aware optimization on this development character. Preserve this negative result; do not change thresholds or pick a favorable seed/method.

The remaining study may evaluate the reliability gap between nonadaptive detection and adaptive evasion, with independent clean false alarms and spelling cost. That is a candidate evaluation scope already under development, not a newly validated novel defense. Full independent protocol review and new immutable cloud manifests remain prerequisites before sealed subjects37-54. Do not unlock the sealed benchmark merely because software checks passed. BNCI2015-003 remains locked. Population reliability, cross-user generalization and journal readiness remain unestablished.
