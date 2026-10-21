# P300 adaptive attack and detector evaluation

Development research code for prototype-guided P300 attacks, calibration, and detector-aware evaluation. This is not a validated clinical defense or a completed independent-user study.

## Start here

- [Pre-registration and dated amendments](PREREG.md)
- [Clean validation report](results/validation_clean_v1_report.md)
- [Adaptive development finding](results/development_attack_audit_interpretation.md)
- [Verified execution integration](results/execution_integration_v1_report.md)
- [Methods and provenance supplement](manuscript/supplement_methods_and_provenance.md)

The reviewed development snapshot is `b8e9fe413085886aa9fb019ae4fa5e44f650ccba` (18 October 2026). Later documentation does not change the experimental protocol or unlock independent outcomes.

## Main recorded findings

The initial prototype-guidance hypothesis did not establish the registered stealth advantage. Later fixed calibration on OpenBMI development participant 19 showed clean rejection counts varying across methods and model seeds. On one fixed development character, directly optimizing individual detectors could evade the detector being attacked.

This is not simultaneous evasion of every detector by one perturbation, population-level detection performance, cross-user generalization, or proof of global worst-case optimization. Independent evaluation and statistical aggregation remain gated. BNCI2015-003 confirmation outcomes remain locked.

## Environment and reproduction

Python >=3.10; package dependencies are declared in `pyproject.toml`. A typical isolated setup is:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

These commands install the package; they do not reproduce a scientific result. The exact cloud runners, input identities, seeds, configuration and declared test exclusions must be taken from the linked reports and corresponding `kaggle/jobs` packages. The recorded portable suite had 363 passes, 41 warnings and four declared historical raw-data/path exclusions; no fresh test run is asserted by this README.

Raw EEG, checkpoints, authentication files and private participant records are not distributed in this repository. Obtain datasets from their original sources under their applicable terms and reproduce in a suitable authorized environment. Do not interpret a directory named `data/raw` as permission to redistribute data.

## Attribution and status

Research work is directed and owned by M M Nishat as confirmed in the project review, with AI-assisted implementation/documentation. This statement does not assign coauthor roles or claim sole invention of established methods. See code, protocol and source citations for specific adaptations. No first-ever attack, journal acceptance or verified defense efficacy is claimed. No new reuse license is granted by making the repository viewable.
