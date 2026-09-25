# Prompts for the coding agent (Antigravity + DeepSeek)

Paste one prompt at a time, in order. After each one, run the tests and send the result to Claude for review before moving on.
Open the folder `p300-proto-attack/` as the workspace. Every prompt assumes the agent has read `AGENTS.md`.

---

## Prompt 1: Differentiable preprocessing

```
Read AGENTS.md, then ../PLAN.md (sprint section) and PREREG.md.

Create src/p3a/preproc/pipeline.py with a torch nn.Module `Preprocessor`:

- __init__(n_channels, fs, band=(0.5, 20.0), target_fs=60.0, car=True)
  - FIR bandpass taps from scipy.signal.firwin(numtaps, band, pass_zero=False, fs=fs),
    numtaps = 2*int(fs) + 1 (odd, linear phase). Store as a registered buffer.
  - decimation factor q = int(round(fs / target_fs)); fs_out = fs / q. Expose both as attributes.
  - buffers `ch_mean` and `ch_std`, shape [C], initialized to 0 and 1.
- forward(x): x is raw EEG [B, C, T] float32. Steps, all differentiable:
  1. depthwise conv1d with the FIR taps (groups=C), reflect padding of numtaps//2 on both sides, so output length == T and there is no phase delay.
  2. common average reference if car=True (subtract mean over channels).
  3. decimate by taking every q-th sample (x[..., ::q]). The 20 Hz lowpass is the anti-alias filter.
  4. standardize: (x - ch_mean[:, None]) / ch_std[:, None].
  Returns [B, C, ceil(T/q)].
- fit_scaling(list_of_raw_arrays): runs steps 1-3 without grad on the given raw trials (each [C, T] numpy), sets ch_mean/ch_std per channel over all samples of all trials.
- to_config() / from_config(): so the preprocessor can be saved in checkpoints (config + state_dict).

Tests in tests/test_preproc.py (synthetic data, CPU):
- output length is ceil(T/q) and dtype float32.
- a 10 Hz sine passes with amplitude ratio > 0.9; a 45 Hz sine and a 0.05 Hz sine are attenuated to < 0.1 (check before decimation by calling the filter step directly; expose it as a method `bandpass(x)`).
- filter output matches scipy.signal.convolve(x, taps, mode="same") on the interior samples (exclude numtaps//2 at each edge) to within 1e-4.
- zero phase: the peak of a filtered 5 Hz sine aligns with the input peak within 1 sample.
- gradients flow: x.requires_grad_(); forward(x).sum().backward(); x.grad is not None and finite.
- after fit_scaling on random data, the standardized output has per-channel mean ~0 and std ~1 (tolerance 0.05).
```

---

## Prompt 2: Epoching and the letter head

```
Read AGENTS.md. Create two modules.

1) src/p3a/data/epochs.py
- `downsample_onsets(onsets, q)` -> onsets // q (int64).
- `epoch_window(fs_out, tmax=0.8)` -> L = int(round(tmax * fs_out)).
- `gather_epochs(x, onsets_ds, L)`: x [B, C, T'] torch, onsets_ds [B, F] long -> epochs [B, F, C, L].
  Implement with index arithmetic (torch.gather or unfold + index_select), differentiable w.r.t. x, no Python loop over flashes.
  Raise if any onset + L > T'.
- `batch_trials(trials, device)`: stack a list of Trial objects into tensors:
  eeg [B, C, T] float32, onsets [B, F] long, groups [B, F, K] bool, targets [B] long (symbol index).
  All trials in a batch have equal C, T, F (true for Comp III).

2) src/p3a/models/letter_head.py
- `symbol_evidence(flash_scores, groups)`: flash_scores [B, F], groups [B, F, K] bool
  -> evidence [B, K] = sum_f groups[b,f,k] * s[b,f] / sum_f groups[b,f,k].
- `symbol_logits(flash_scores, groups, tau=1.0)` -> evidence / tau.
- `first_reps(onsets, groups, flash_scores, n_reps, flashes_per_rep)`: keep only the first n_reps * flashes_per_rep flashes (for accuracy at fewer repetitions).

Tests (tests/test_epochs.py, tests/test_letter_head.py):
- gather_epochs returns the right slices for a hand-built tensor, and gradients reach x.
- on real Comp III Subject A train data (skip if file missing): with flash scores = 1 for flashes where
  trial.target_flash_mask() is true and 0 otherwise, argmax of symbol_logits equals the true target for all 85 trials.
- row/column equivalence: for random flash scores on a real trial, argmax of symbol_evidence equals the symbol
  at (argmax over row codes of mean score per row code, argmax over column codes of mean score per column code).
```

---

## Prompt 3: Decoders

```
Read AGENTS.md. Create flash-level P300 decoders. Each takes epochs [N, C, L] (preprocessed, 60 Hz, L = 48 for Comp III)
and returns one logit per epoch [N] (P300 present).

src/p3a/models/eegnet.py: EEGNet (Lawhern et al. 2018). F1=8, D=2, F2=16, temporal kernel 32, depthwise spatial conv over C,
  avg-pool 4, separable conv kernel 16, avg-pool 4, dropout 0.25, ELU, batchnorm, flatten, linear to 1. Use nn.LazyLinear
  or compute the flatten size from (C, L) in __init__.
src/p3a/models/deepconvnet.py: DeepConvNet (Schirrmeister et al. 2017) adapted to short epochs: blocks of
  (temporal conv k=5 -> [block 1 only: spatial conv over C] -> BN -> ELU -> maxpool 2 -> dropout 0.5) with 25, 50, 100 filters,
  then linear to 1.
src/p3a/models/shallowconvnet.py: ShallowConvNet: temporal conv k=13 (40 filters), spatial conv over C (40), BN, square,
  avg-pool k=15 stride=5 (pad if needed so it works for L=48), log(clamp(min=1e-6)), dropout 0.5, linear to 1.
src/p3a/models/xdawn_lda.py: class XdawnLDA with fit(epochs_np [N,C,L], y [N]) and decision_function(epochs_np) -> [N].
  xDAWN: target-class average ERP P [C, L]; signal covariance S from all epochs concatenated in time; spatial filters =
  top 4 generalized eigenvectors of (P P^T, S) via scipy.linalg.eigh. Features = filtered epochs [N, 4, L] flattened.
  Classifier = sklearn LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"). Save/load with joblib.
src/p3a/models/registry.py: build_model(name, n_channels, L) for "eegnet", "deepconvnet", "shallowconvnet".

Tests: each torch model maps a random [7, 64, 48] batch to shape [7], backward works; also works for C=8 and C=16 with L=51.
XdawnLDA separates a synthetic dataset (injected ERP in targets) with AUC > 0.9.
```

---

## Prompt 4: Speller wrapper, decoder training script, Kaggle-safe data paths

```
Read AGENTS.md.

A) Update src/p3a/data/loaders/comp3.py WITHOUT changing load_subject's behavior for local paths:
   - add `find_file(root, name)`: return root/name if it exists, else the first match of root.rglob(name)
     (on Kaggle the files sit in nested folders like "Dataset p300/Subject_A_Train.mat/Subject_A_Train.mat").
   - add optional `labels_dir` parameter to load_subject (default: same as root) for true_labels_*.txt.
   Existing tests must still pass.

B) src/p3a/models/speller.py: nn.Module `Speller(preprocessor, decoder, L)`.
   forward(eeg [B,C,T], onsets [B,F], groups [B,F,K]) -> dict(flash_scores [B,F], logits [B,K]).
   Pipeline: preprocessor -> gather_epochs(downsampled onsets) -> decoder on [B*F, C, L] -> reshape -> symbol_logits.
   Also `epochs(eeg, onsets)` returning preprocessed epochs [B,F,C,L] (used by prototype features and detectors).
   save(path) / load(path) storing: preprocessor config + state, decoder name + state, L, fs, training config, seed.

C) scripts/train_decoder.py
   Args: --data-root, --labels-dir, --subject (A|B), --model (eegnet|deepconvnet|shallowconvnet|xdawn_lda), --seed,
         --out-dir, --epochs 100, --batch 256, --lr 1e-3, --patience 15, --device auto.
   Steps:
   1. load train and test trials; split official train into train/val using splits/comp3_{subject}.json (trial_id).
   2. build Preprocessor(fs=240), fit_scaling on TRAIN trials only.
   3. precompute preprocessed epochs (no grad) for train/val/test; labels = target_flash_mask per flash.
   4. torch models: BCEWithLogits with pos_weight = n_neg/n_pos, Adam, early stopping on VAL flash AUC, keep best state.
      xdawn_lda: fit on train epochs directly.
   5. metrics.json: val_auc, test_flash_auc, test_letter_acc at 15 reps and at 5 reps (first 5*12 flashes), n_params,
      train_time_s, seed, git commit, full args.
   6. save best checkpoint as <out-dir>/<subject>_<model>_s<seed>.pt (or .joblib for xdawn_lda) via Speller.save.
   Set all seeds (python, numpy, torch, cudnn deterministic where possible).

D) scripts/train_all_decoders.py: loops subjects [A, B] x models [eegnet, deepconvnet, shallowconvnet, xdawn_lda]
   x seeds [0, 1, 2], calls the training function in-process, writes results/decoders_summary.csv.

Local smoke test (must pass on my Mac, CPU, Subject A only): add a --max-epochs-smoke flag or --smoke that uses 10 train
trials, 3 val, 0 test, 2 epochs; run it for eegnet and xdawn_lda and show the output. Do NOT run the full training locally.
```

---

## Prompt 5: Kaggle pipeline and decoder training job

```
Read AGENTS.md. Kaggle CLI is installed and logged in as user "mustaqimnishat".
Data dataset (public, attach, do not upload): manthandixit726262/bci-competition-iii-dataset-ii

Create:
1. scripts/make_code_bundle.py: copies src/, scripts/, splits/, configs/, tests/, pyproject.toml, AGENTS.md, PREREG.md and
   data/raw/comp3/true_labels_*.txt into kaggle/code_bundle/ (clean it first). Writes kaggle/code_bundle/dataset-metadata.json:
   {"title": "p300-proto-attack-code", "id": "mustaqimnishat/p300-proto-attack-code", "licenses": [{"name": "CC0-1.0"}]}.
   Also writes GIT_COMMIT.txt with the current commit hash. Must never copy .venv, .mat files, checkpoints, results.
2. kaggle/jobs/train_decoders/kernel-metadata.json:
   id "mustaqimnishat/p3a-train-decoders", title "p3a-train-decoders", code_file "run.py", language "python",
   kernel_type "script", is_private true, enable_gpu true, enable_internet true,
   dataset_sources ["mustaqimnishat/p300-proto-attack-code", "manthandixit726262/bci-competition-iii-dataset-ii"].
3. kaggle/jobs/train_decoders/run.py:
   - find the code bundle under /kaggle/input (rglob for pyproject.toml), copy it to /kaggle/working/code, pip install -e it.
   - print nvidia-smi and torch.cuda.is_available().
   - run scripts/train_all_decoders.py with --data-root /kaggle/input --labels-dir <bundle>/data/raw/comp3
     --out-dir /kaggle/working/out/decoders --device cuda.
   - stream logs to stdout; on any exception print the traceback and still write whatever finished.
4. scripts/kaggle.sh with subcommands:
   bundle   -> python scripts/make_code_bundle.py && kaggle datasets create -p kaggle/code_bundle --dir-mode zip
               (if it already exists: kaggle datasets version -p kaggle/code_bundle -m "$MSG" --dir-mode zip)
   push JOB -> kaggle kernels push -p kaggle/jobs/JOB --accelerator NvidiaTeslaT4
   status JOB, output JOB (download to kaggle/out/JOB/, which is gitignored).

Then run: bash scripts/kaggle.sh bundle, wait until the dataset is ready (kaggle datasets status), then
bash scripts/kaggle.sh push train_decoders, and show me the status output. If the accelerator name is rejected,
print the error and stop; do not guess other names.
```

---

## Prompt 6: Prototype bank, features, retrieval

```
Read AGENTS.md. All features are computed with a trained Speller's preprocessing (Speller.epochs).

src/p3a/prototypes/features.py
- `target_conditioned_erp(epochs [F,C,L], groups [F,K], symbol_idx)` -> mean over flashes containing the symbol [C, L].
  Torch and numpy versions (torch one differentiable; used in the attack loss).
- `FeatureScaler`: fit(list of [C,L]) -> per-(c,t) mean and std; transform; save/load.

src/p3a/prototypes/bank.py
- `PrototypeBank`: entries with fields feature [C,L] (target ERP of the trial's TRUE symbol), symbol_idx, subject_id,
  dataset, trial_id, split. Methods: build(trials, speller, subject_ids, allowed_splits), save/load (HDF5 via h5py),
  symbols_available(), entries_for(symbol_idx), counts_per_symbol().
- Modes: same_subject (attacked subject, official-train "train" split trials only, per splits/comp3_X.json) and
  cross_subject (all trials, train and test, of every OTHER subject in the same dataset). A bank must refuse any entry
  whose subject equals the attacked subject in cross_subject mode (assert, with a test).

src/p3a/prototypes/retrieval.py
- `retrieve(bank, scaler, query_feature, target_idx, mode, rng)` with mode in
  random (uniform among entries of target_idx), medoid (entry with min sum of standardized Euclidean distance to the
  other entries of that symbol; precompute), nearest (min standardized Euclidean distance to the query),
  wrong_letter (nearest entry among symbols != target_idx).
  Returns (entry index, distance). If the target symbol has no entries: return None and let the caller skip or fall back
  (log it). Query = target_conditioned_erp of the SOURCE trial at the flashes of the ATTACKER target (not its true symbol).

scripts/bank_report.py: builds both banks for Comp III A and B (needs trained EEGNet checkpoints), prints per-symbol counts
and a histogram of nearest distances, writes results/bank_report.csv.

Tests: synthetic bank with 3 symbols: medoid/nearest/wrong_letter return the expected entries; cross_subject excludes the
attacked subject; query uses the attacker target flashes (build a trial where true and target symbols differ and check
which flashes are averaged).
```

---

## Prompt 7: Generators (GPN-style, prototype-conditioned)

```
Read AGENTS.md. Do NOT copy code from the AIRLABkhu repo. Implement from this description of GPN (Jung et al., JBHI 2023):
a ResNet-style encoder-decoder over the EEG treated as a 1-channel image [1, C, T']: reflect-pad 3 + conv 7x7 (64 filters) +
BN + ReLU; conv 3x3 stride 2 (128) + BN + ReLU; 2 residual blocks (conv3x3-BN-ReLU-conv3x3-BN + skip, ReLU); transposed
conv 3x3 stride 2 (64) + BN + ReLU; reflect-pad 3 + conv 7x7 to 1 channel + tanh. The perturbation is then rescaled so
its L-infinity norm equals epsilon: delta = out / max|out| * eps (per sample).

Create src/p3a/attacks/generators.py:

1) `Conditioner`: builds a conditioning vector from
   - target one-hot [K] -> linear 64,
   - prototype feature [C, L] (standardized) -> small conv encoder (conv over time, global avg pool) -> 64
     (a zero tensor when no prototype is used),
   - eps multiplier (scalar) -> linear 16.
   Concatenate -> MLP -> vector z [128].

2) `SignalGPN` (signal-specific; sees the source trial):
   - input: raw eeg [B, C, T] at 240 Hz. Downsample by average pooling with factor 4 in time to [B, C, T/4]
     (pad T to a multiple of 4). Add a second input channel: the target-flash map = 1 at downsampled onsets of flashes
     that contain the attacker target, 0 elsewhere, broadcast over C. Image input [B, 2, C, T/4].
   - the GPN body above, with FiLM conditioning from z (per-channel gamma, beta) after each residual block.
   - output [B, C, T/4] -> linear interpolation back to T -> L-inf rescale to eps (eps given per sample, raw units).
3) `CausalGPN` (precomputable; never sees the source EEG):
   - from z, an MLP + 1D transposed convs produce a template w [C, 192] (800 ms at 240 Hz).
   - delta[b] = sum over flashes f of trial b that contain the attacker target of w placed at onset f (overlaps add),
     cropped to T, then L-inf rescale to eps.
4) `use_prototype` flag on both: when False the prototype input is zeros (this is "target-only GPN").

Tests: shapes for B=2, C=64, T=7794; ||delta||_inf == eps (tolerance 1e-5) for each sample; gradients reach generator
params; CausalGPN output is identical for two different source EEGs with the same schedule/target/prototype.
```

---

## Prompt 8: Attack training script

```
Read AGENTS.md. Create scripts/train_attack.py and src/p3a/attacks/train.py.

Args: --data-root --labels-dir --subject --surrogate <EEGNet checkpoint> --method
      {target_only, proto_random, proto_medoid, proto_nearest, proto_wrong} --generator {signal, causal}
      --bank {same_subject, cross_subject} --lambda-p 1.0 --lambda-s 0.1 --epochs 60 --batch 8 --lr 1e-4 --seed --out-dir.

- eps scale s = median over channels of the per-channel std of the subject's raw TRAIN EEG. eps multipliers grid
  {0.05, 0.10, 0.20, 0.30, 0.50} (from PREREG.md). Each training sample draws a multiplier uniformly from the grid;
  eps_raw = multiplier * s. The multiplier is also a generator input.
- training pairs: TRAIN trials, each epoch a fresh random attacker target != true symbol. Validation: VAL trials with a
  fixed seeded target per trial.
- prototype per sample from retrieval (mode from --method), using the bank and FeatureScaler; skip samples whose target
  has no prototype and count them.
- loss = CE(surrogate symbol logits of x + delta, target)
       + lambda_p * mean((scaler(target_conditioned_erp(surrogate.epochs(x+delta), target)) - scaler(prototype))^2)   [proto methods only]
       + lambda_s * mean(delta^2) / eps_raw^2
  Surrogate frozen (eval mode, requires_grad False). Victims never loaded.
- keep the checkpoint with best VAL targeted success averaged over the eps grid; log per-epoch train loss parts and
  val success per eps to a CSV.
- save generator checkpoint + config + eps scale s + git commit.

Local smoke test on Subject A with --epochs 1 on 4 trials (surrogate from a smoke EEGNet checkpoint).
```

---

## Prompt 9: Baselines

```
Read AGENTS.md. Create src/p3a/attacks/baselines.py. All baselines produce delta in RAW units with ||delta||_inf <= eps_raw
and are evaluated through the same Speller.

1) targeted_pgd(speller, eeg, onsets, groups, target, eps_raw, steps=40, step=eps/10, random_start=True):
   per-sample L-inf PGD on CE(symbol logits, target). Batched.
2) TinyNoiseTemplate (re-implementation of the idea of Zhang et al., NSR 2021, via gradient optimization):
   one universal template w [C, 192] per subject and eps, placed at every flash that contains the attacker target
   (same placement as CausalGPN), trained on TRAIN trials with random targets to minimize CE(symbol logits, target),
   clipped to eps after each step (Adam, 30 epochs). No prototype, no conditioning. save/load.
3) random_noise(shape, eps_raw, rng): Gaussian noise over the whole trial rescaled to L-inf = eps_raw.

scripts/train_tiny_noise.py trains the template for each eps in the grid for one subject.
Tests: L-inf bound holds; PGD raises targeted success on a toy speller built from a tiny random decoder.
```

---

## Prompt 10: Evaluation runner

```
Read AGENTS.md. Create scripts/evaluate_attacks.py and src/p3a/eval/{budget.py,success.py,store.py}.

For one attacked subject: load the surrogate EEGNet, the victims (DeepConvNet, ShallowConvNet, XdawnLDA; seed 0), test trials,
and the pre-drawn targets from splits/comp3_{S}_attack_targets.json (10 per test trial). For every method
(noise, pgd, tiny_noise, target_only x {signal,causal}, proto_{random,medoid,nearest,wrong} x {signal,causal}, and
proto_nearest with same_subject bank x {signal,causal}) and every eps multiplier in the grid:
- compute delta, x_adv = x + delta.
- success: surrogate and each victim symbol argmax == target (15 reps and 5 reps).
- budget: linf, l2, rms of delta; snr_db = 10*log10(sum x^2 / sum delta^2).
- retrieval distance (proto methods).
- store (float16, one npz per method+eps): decision ERP = target_conditioned_erp(surrogate.epochs(x_adv), target) [C,L];
  the same for the clean trial (store once per trial and target); 4 random 1-second windows of the preprocessed x_adv
  (same window positions for clean, stored once).
One row per (method, generator, bank, eps, trial, target) into results/<subject>/attacks.parquet.
Resumable: skip (method, eps) combinations whose npz already exists. Batch everything on GPU.
Tests on a toy speller: row count, npz shapes, budget math.
```

---

## Prompt 11: Detectors and the defense

```
Read AGENTS.md. Create src/p3a/defenses/{window_cnn.py, genuine_vs_induced.py, simple.py} and scripts/run_detectors.py.
Inputs come from the npz files written by evaluate_attacks.py. All splits of the data for detectors are by TEST TRIAL ID:
fold 1 = even trial index, fold 2 = odd (2-fold, report the mean).

1) Clean-vs-attacked window CNN (one detector per method and eps; strongest defender): input 1-second preprocessed windows
   [C, fs_out]; small CNN (temporal conv, spatial conv, pool, linear). Metric: AUC and TPR at 5% FPR on the held-out fold.
2) Genuine-vs-induced ERP detector (the defense): input decision ERP [C, L].
   genuine = clean test trials' ERP at their TRUE symbol; induced = attacked ERP at the attacker target where the surrogate
   attack succeeded. Train on the pool {pgd, target_only (both generators), noise} only; evaluate on every method.
   Models: logistic regression on standardized flattened ERP and a small CNN; choose by validation inside the training fold.
   Threshold: 95th percentile of genuine scores in the training fold (5% FPR). Report TPR per method and eps, and the
   false alarm rate on held-out genuine trials.
3) Simple detectors: band-power anomaly (z-score of 20-30 Hz and 0-2 Hz power vs clean) and ERP-Mahalanobis distance
   (Ledoit-Wolf covariance of clean decision ERPs, PCA to 20 dims). AUC per method and eps.
4) Plausibility metrics per attacked trial vs the mean genuine ERP of the same target symbol (from clean trials of the
   subject): waveform correlation (Pz, Cz), P300 peak latency and amplitude in 250-500 ms at Pz, topography correlation at
   the genuine peak latency, PSD distance (Welch, 1-20 Hz).
Write results/<subject>/detectors.csv and plausibility.csv.
```

---

## Prompt 12: Kill-test Kaggle job and report

```
Read AGENTS.md. Create kaggle/jobs/kill_test/ (same pattern as train_decoders) whose run.py, for subjects A and B:
1. loads the decoder checkpoints from the train_decoders kernel output (add "mustaqimnishat/p3a-train-decoders" to
   kernel_sources in kernel-metadata.json),
2. builds banks (cross_subject: A attacked with B's prototypes and vice versa; same_subject for the upper bound),
3. trains all attack generators (Prompt 8 methods x {signal, causal}) and the Tiny Noise templates,
4. runs evaluate_attacks.py and run_detectors.py,
5. writes everything to /kaggle/working/out/kill_test/.
Checkpoint after every trained model so a restarted run resumes. Must fit in 12 hours on one T4; print elapsed time per
stage.

Then create scripts/kill_test_report.py (runs locally on the downloaded output) producing results/kill_test_report.md with:
clean decoder accuracy; targeted success vs eps per method (surrogate and victims); norms and detector AUC interpolated at
50/65/80% success (report max success if a level is not reached, never extrapolate); genuine-vs-induced TPR per method;
plausibility table; per-symbol prototype counts; and the PREREG decision-gate verdict computed mechanically.
Plots saved as PNG next to the report.
```

---

## Prompt 13: BNCI loaders

```
Read AGENTS.md. Add loaders producing the same Trial objects.

src/p3a/data/loaders/bnci2014_008.py: files A01.mat..A08.mat from
https://bnci-horizon-2020.eu/database/data-sets/008-2014/ (download if missing into a cache dir).
Structure (scipy.io.loadmat squeeze_me=True, struct_as_record=False): data.X [samples, 8], data.y (0 none, 1 NonTarget,
2 Target), data.y_stim (0 none, 1-12 with classes_stim Row1..Row6, Col1..Col6), data.trial (start sample of each of 35
characters), fs 256, channels Fz Cz Pz Oz P3 P4 PO7 PO8. Split the continuous signal into character trials at data.trial
boundaries; flash onsets = rising edges of y_stim != 0; map codes to 6x6 row/column groups (NOTE: here 1-6 are ROWS and
7-12 are COLUMNS, the opposite of Comp III; write a test for it); the target symbol = the cell at the intersection of the
row and column flagged Target. Symbols are cell indices 0-35 (use a Layout with generic symbols).

src/p3a/data/loaders/bnci2015_003.py: files s1.mat..s10.mat from https://bnci-horizon-2020.eu/database/data-sets/003-2015/
(s{n}.train and s{n}.test are [11, samples]: row 0 time or counter, rows 1-8 EEG, row 9 stimulus code 0-36, row 10 target
flag). FIRST inspect the files and the dataset description on the BNCI page to confirm what codes 1-36 mean and the sampling
rate; write your findings as a comment at the top of the loader, and stop to ask if it is unclear. Map each code to a flash
group over the 36 symbols accordingly.

For both: frozen train/val/test splits via splits.write_once (per subject: first 70% of characters train, next 15% val,
last 15% test, in recording order), pre-drawn attack targets exactly like scripts/01_make_splits.py, and tests on counts.
Append an Amendment to PREREG.md describing these splits.
```

---

## Prompt 14: Full experiments on all datasets

```
Read AGENTS.md. Generalize train_all_decoders.py, train_attack.py, evaluate_attacks.py and run_detectors.py to take
--dataset {comp3, bnci2014_008, bnci2015_003} and leave-one-subject-out cross-subject banks within each dataset.
Create Kaggle jobs full_bnci008 and full_bnci003 (same pattern, resumable, one subject per stage, checkpoint after each).
Each job must fit in 12 h on one T4; if it cannot, split subjects across two jobs (a and b) and say so.
```

---

## Prompt 15: Statistics and figures

```
Read AGENTS.md and PREREG.md. Create scripts/analyze.py producing results/final/ with:
- per-subject tables for every outcome; bootstrap 95% CIs (2000 resamples over trials);
- paired Wilcoxon across subjects: proto_nearest vs target_only, proto_nearest vs tiny_noise (primary outcome first);
- linear mixed-effects (statsmodels mixedlm) outcome ~ method with random intercepts for subject (and dataset);
- Holm correction across secondary outcomes;
- Spearman correlation between retrieval distance and required eps / success;
- figures: success vs eps per method; detector AUC vs success level; genuine-vs-induced TPR per method;
  ERP overlays (genuine vs induced) at Pz; topographies at peak.
Everything computed only from results files, no retraining.
```
