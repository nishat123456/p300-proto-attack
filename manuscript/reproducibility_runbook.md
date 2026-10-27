# Reproducibility and completion runbook

Working document, updated27 October2026. Execution and revised aggregation are complete; the reviewable package is incomplete. This is an execution record and prospective completion checklist, not a successful reproduction or journal-readiness claim.

## Fixed study identity

Use `configs/openbmi_independent_protocol_frozen_20261022.json` and the byte-identical protocol in `splits/openbmi_independent_20261022/`. Protocol SHA256:

`a5699d95b0d35343da207dec8d657f380d1fd49a7664424682cd6e7c910b9916`

The receipts bind every Python source under `src/` and `scripts/`. Keep this scientific inventory unchanged during the released evaluation. A repair affecting scientific execution requires a prospective release and explicit accounting for existing outputs, not rebinding old receipts. Reporting code outside this inventory must preserve its own exact hash.

Source participants1-18, validation19-36, evaluation37-54. Participant52 remains in the18-person flow but is unavailable for the calibrated estimand because official training labels fail identity validation. Do not relabel, replace or infer missing-at-random. Report17 available participants conditionally. BNCI2015-003 stays locked. Historical frozen files and gates are not overwritten.

## Cloud execution and recovery

All scientific processing, tests, score/statistical aggregation and figure generation run on Kaggle. Local work is limited to source/document preparation and compact evidence transfer. Use the staged `kaggle/jobs/openbmi_independent_task00` through `task50` payloads and fixed routing in `configs/openbmi_five_account_dispatch_candidate.json`. The routing file alone does not authorize performance; the frozen receipts do.

Each task is one participant/model-seed pair. Required seeds are20261013/14/15. Preserve version, owner, task, start/completion receipts, subject execution ledger and all36 character shards. A completed task has27 settings per ordinary character and34 on the fixed first character, including the prespecified longer adaptive runs. Model/reference/template arrays and clean score CSV remain on Kaggle.

Never restart a live job solely because output is quiet. After a terminal failure, preserve the exact status and available log/partial artifacts before deciding on recovery. An unchanged retry is distinguished from a changed-code release. Task01v1 currently has terminal ERROR with null failureMessage and no exposed output/log; cause is unknown. Its single exact-payload v2 retry is recorded separately, not erased by a later success.

Completed-ledger no-op resume was validated on participant19. Forced interruption during a setting was not demonstrated. Do not claim stronger recovery testing.

## Evidence collection

Retrieve only compact JSON/TXT/MD/log. Use official output pagination, since the CLI's first100-file page is insufficient for some jobs. Never download raw EEG, model/PT files, reference NPZ, code trees or CSV. Do not locally recompute scientific summaries from retrieved JSON. Preserve all outcomes, not just successful attacks or favorable detectors.

Before aggregation, retain one authoritative completed version per fixed task plus any failure history. Remove no original evidence. Avoid mounting both an original result and its copied duplicate under the aggregation root, because the aggregator deliberately rejects duplicate identities. Any needed new private output-sharing permissions must be explicitly authorized at the applicable access boundary. Existing source/refcal access does not prove new independent outputs are mounted.

## Original strict aggregation, preserved historical failure

The following original command failed its strict actual-budget assertion and is retained for provenance, not as the accepted revised-analysis command. Do not rerun completed jobs. Mount or stage all51 completed subject-seed output trees inside a dedicated cloud root. Clean score CSV must accompany the matching clean JSON on cloud for AUC. Do not substitute local JSON-only copies. Use the frozen scientific source inventory and matching protocol:

```bash
python scripts/aggregate_openbmi_reliability.py \
  --root /kaggle/working/independent_results \
  --protocol splits/openbmi_independent_20261022/protocol.json \
  --out-dir /kaggle/working/out/aggregate \
  --git-commit EXACT_EXECUTION_REVISION \
  --seed 20261013
```

`EXACT_EXECUTION_REVISION` must be the actual packaged revision, with the frozen source hashes checked. Do not use `--validation-integration` for independent results. The aggregator must find all17 participants, three seeds, all36 original characters, all21 clean nominal/method summaries per task, and all147 attack/budget/detector groups. Missing or duplicate shards are failures, not grounds for denominator reduction.

Interpret seed-averaged participant outcomes as equal-participant summaries. Report pointwise subject-bootstrap intervals, paired descriptive comparisons and the fixed adaptive recipe's limits. Serial character dependence and phase/phrase shift prevent per-user nominal-FAR guarantees. Do not interpret three seeds as three independent participants. Keep conditional successful-attack TPR secondary to original-attempt denominators.

## Report and figure generation, on Kaggle only

```bash
python reporting/build_reliability_report.py \
  --aggregate /kaggle/working/out/aggregate/reliability_aggregate.json \
  --protocol splits/openbmi_independent_20261022/protocol.json \
  --out-dir /kaggle/working/out/report \
  --git-commit EXACT_REPORT_REVISION
```

Do not pass `--development-only` to a final independent report. That mode is restricted to the already-inspected participant19 integration evidence. It cannot provide a population result. Keep all nominal1/5/10% clean tables and all declared attack budgets. Fixed color scales and descriptive intervals do not select a winner or establish significance.

CPU report checks exercised real participant19 aggregate data and five rejection/acceptance guards. The reporter has now processed the complete17-person revised numerical-envelope aggregate; exact compact evidence is in results/envelope_aggregate_v1. Development PNG files were generated on Kaggle. The first file/script previews were blank, but actual notebook previewv2 subsequently rendered both exact images; browser visual review is recorded in results/report_preview_v2. Final independent figure layout and captions require actual visual review, not generation alone.

## Cost and publication completion gates

Preserve the source-query timing evidence with hardware, synchronization, five warmups,30 repeats and median/p95. Report detector overhead separately from marker-derived trial duration and idealized IID retry cost. None establishes clinical throughput, backspace behavior or live physical trigger causality.

Before professor review, reconcile every manuscript number against cloud aggregate provenance; include participant flow, unavailable52, all methods/budgets, adaptive convergence scope, uncertainty, clean rejection and accepted errors, source timing and spelling-cost limitations. Finish results and captions with negative findings. Check bibliography metadata, authors/order/affiliations and the target venue's current requirements. Public code/artifact access and dataset redistribution terms remain to be arranged; private preservation is not public reproducibility. Obtain professor scientific review. No submission, email or payment is authorized by this runbook.

## Cost presentation command

On Kaggle, use the clean output root (including all51 JSON identities), the
cloud source-cost output root, and the same frozen protocol:

```bash
python reporting/build_cost_report.py \
  --clean-root /kaggle/working/independent_results \
  --timing-root /kaggle/input/source_cost_outputs \
  --protocol splits/openbmi_independent_20261022/protocol.json \
  --out-dir /kaggle/working/out/cost_report \
  --git-commit EXACT_REPORT_REVISION
```

The timing-root path is a mount placeholder, to be replaced by the actual
verified cloud attachment path. Do not use `--development-only` for independent
reporting. Cloud cost-report checksv1 passed nine checks and generated the real
participant19 descriptive cost table, sourceb2cb475/launchereea4b24. It validates
complete seed/nominal/marker coverage, model/reference/protocol binding and stored
retry consistency. It has now processed all17participant/3seed independent outputs under the revised numerical-envelope release. It reports stored per-seed costs without a population retry
interval; primary reliability intervals remain in the separate frozen aggregate.

## Revised numerical envelope analysis release 26 October2026

The original strict aggregate failure and immutable execution receipts remain
historical evidence. The author declined further strict corrective launches.
The separate post-execution contract accepts only finite nonnegative norms up to
max(original epsilon,nearest float32 epsilon). It is not original strict-budget
compliance and must be labeled in results, supplement and output provenance.

CPU release reviewv2 completed368original portable tests with41warnings in43.38s
and6new envelope guards, with the same four historical excluded tests. It verified
all51original task ledgers,3927compactJSON and51cleanCSV cloud-to-cloud, including
cleanCSVthreshold/decision/summary consistency. All49929settingrows passed the
exact envelope;13419originalstrictviolations remain disclosed. Fullperformance
was not computed during review. Preserve the failedv1 packaging evidence.

Inspected new receipt:
analysis_releases/openbmi_float32_envelope_20261026/analysis_release_receipt.json
SHA25646674a236cdf444e946defd2c323e49dc0688273315a5bcd4001418edbfe40ed.
The private preservation commit is26e8163. Original src/scripts revision0959347
is restored in the reviewed cloud package; new analysis reporters have separate
hashes. No old receipt was rebound. Exact launchbinding is
kaggle/jobs/openbmi_envelope_aggregate/launch-binding.json, launchera7357a1.

Private CPU job mustaqimnishat/p3a-envelope-aggregate v1 mounts the completed
primary review's code/archive. It verifies the exact inspected receipt and every
recorded source/JSON/CSV hash before running the separately labeled envelope
aggregate, existing reliability reporter and existing cost reporter. No raw EEG,
model or CSV is downloaded locally, and no secondary-account restriction is
bypassed. The already-preserved primary archive supplies the complete cohort.
Download only compactoutJSON/TXT/MD/log using paginated officialAPI; keep figures
on Kaggle for actual browser visual review. Do not download source trees orPNG.

The aggregate completed and compact evidence is privately preserved b2f20ec in
results/envelope_aggregate_v1. Completion and report provenance bind aggregate
SHA256 e11fb26676c22c548074ecbf8113f4aa427a815df2723c383effb277cf31e9ce.
The51tasks/17users,147population groups,84clean metric summaries and120paired
descriptive comparisons are present. Manuscript results/discussion/conclusion
were drafted8d98142 from cloud reports. Actual independent figure visualQA,
final source consistency review, reviewer access and professor review remain
pending. No additional GPU run is authorized. Do not poll or restart this
completed aggregate.

## Reviewer artifact access, not yet activated

The exact revised runner is kaggle/jobs/openbmi_envelope_aggregate/run.py, bound
by launch-binding.json to source26e8163 and the preserved review receipt. It
invokes reporting/aggregate_openbmi_numerical_envelope.py in the reviewed cloud
package, not the strict historical aggregator command above. All scientific
reproduction remains cloud-only. Review receipt and compact result hashes allow
identity checks but do not alone provide a reader access to private artifacts.

Before a journal submission, arrange authorized reviewer access to the exact
source package, protocol and original receipts, new analysis receipt, complete
original derived outputs and clean scores, original strict failure, both paired
sensitivity reports, source timing and final cloud figures. Dataset acquisition
must follow original provider terms; no raw data or model redistribution is
assumed. Do not publish private artifacts, signed links or credentials under this
checklist. Public release and submission remain unauthorized. The browser-only
figure check currently awaits primary-account passkey verification already
requested from the author. No access-restriction bypass is permitted.
