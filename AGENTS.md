# Project rules for coding agents

Project: prototype-guided adversarial attacks on P300 spellers, plus a genuine-vs-induced ERP defense. Read `PLAN.md` (sprint section at top; copy of ../PLAN.md) and `PREREG.md` before coding.

## Environment
- Python venv: `.venv/bin/python` (run everything with it). Tests: `.venv/bin/python -m pytest -q`.
- Package code lives in `src/p3a/` (installed editable). Scripts in `scripts/`. Tests in `tests/`.
- Local machine: Apple M2, 8 GB RAM, CPU/MPS only, ~8 GB free disk. Heavy training runs on Kaggle T4 (see Prompt 5 in PROMPTS.md).
- Local data: `data/raw/comp3/Subject_A_Train.mat` and `true_labels_{a,b}.txt`. Other .mat files are on Kaggle only.

## Existing API (do not change signatures)
- `p3a.data.layout`: `Layout`, `COMP3` (36 symbols, 6x6), `rc_code_to_group(layout, code) -> bool[36]`, `symbol_rc_codes(layout, symbol) -> (row_code, col_code)`.
- `p3a.data.trial.Trial`: `eeg [C,T] float32 raw`, `fs`, `onsets [F]`, `flash_groups bool[F,K]`, `flash_codes [F]`, `target_symbol`, `layout`, `subject_id`, `dataset`, `trial_index`, `source_split`, `trial_id`, `target_flash_mask(symbol=None)`.
- `p3a.data.loaders.comp3.load_subject(root, subject, split) -> list[Trial]` (180 flashes per trial, SOA 42 samples at 240 Hz).
- `p3a.data.splits`: `load(path)`, `write_once(path, split)`. Split files in `splits/` are FROZEN.

## Hard rules
1. Never edit or regenerate anything in `splits/`, and never edit `PREREG.md` except by appending under "Amendments".
2. Never use test trials for training, early stopping, or hyperparameter choice. Validation split only.
3. Victim models (DeepConvNet, ShallowConvNet, xDAWN+LDA) are never used inside any attack or for any selection.
4. Every script takes a `--seed` and writes its full config + git commit hash next to its outputs.
5. Every new module gets pytest tests that run on CPU in seconds (use tiny synthetic data when real data is not needed).
6. Do not copy code from the AIRLABkhu GPN repo (it has no license). Implement from the description given in the prompt.
7. Keep the code simple and readable. No new frameworks (no Lightning, no Hydra). Plain PyTorch, numpy, scipy, scikit-learn, pandas.
8. No em dashes in any written text or comments.
9. When a task is done: run the full test suite, show the output, and list the files you created or changed.
