# Final Version 2.3 reproduction report

## Result

Primary classification: **FRESH_PIPELINE_COMPLETE_METRICS_DIVERGENT**.

## Bounded prompt/context causal ablation (blocked before generation)

An isolated 12-query causal pilot was prepared under `outputs/experiments/llm_prompt_context_causal_ablation/`. The pilot, frozen C0/C1 contexts, P0/P1/P2 prompts, current model, and one stronger comparison model were locked without gold. C1 reduced total prompt length by an average 76.4% relative to P1 while retaining complete provenance and 0.9717 mean question-term coverage.

The managed environment rejected the first P1 SiliconFlow command before Python process execution under tenant external-export policy. Consequently, P1, P2, stronger-model, and expansion generation calls are all zero; no causal effect is estimable and the experiment is classified `INCONCLUSIVE`. This result does not alter the completed Version 2.3 fresh reproduction or its authoritative metrics.

Post-stop immutability verification passed for Version 2, Version 2.1, Version 2.2, and version-freeze with zero missing, added, or modified files. No experiment result was promoted.
Separate deterministic classification: **EXACT_DETERMINISTIC_REPRODUCTION_COMPLETE**. Cached-exact DEV and production outputs are byte-identical to their frozen targets. The raw fresh pipeline also completed, including 30-record preservation, 25-record resume, accepted Version 2 post-processing, schema-safe assembly, freeze, and one authoritative official-gold evaluation. Fresh metrics are materially below cached-exact metrics; this is a fresh-generation reproducibility gap, not a deterministic-replay failure.

The historical producer edge into `modular_composed_label_locator_predictions.jsonl` remains absent. Exact historical lineage above that verified cache boundary is not claimed.

## Lineage boundary

`modular_composed_label_locator_predictions.jsonl` is a `VERIFIED_CACHE_BOUNDARY` with SHA-256 `54DA46600AFE81DAB5D8D2F10E87AC453FF5FCA206296DAF126EFFCD4D4C409D`, 55 unique records, strict UTF-8/JSONL validity, and the expected schema. Raw-input exact lineage is incomplete and is not claimed.

## Cached-exact DEV

- Output: `outputs/cached_exact/predictions.jsonl`
- SHA-256: `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`
- Target comparison: byte-identical; 0 missing, extra, or changed records
- Paper F1: 1.000000
- Evidence F1: 0.6346464646464647
- Multiple choice: 38/41
- Freeform: 23/26
- Table row F1: 1.000000
- Table cells: 25/27 micro
- Official evaluator invocations: 1

The final target was opened only after the generated output and freeze manifest existed. It was never a producing input.

## Production cached-exact

- Output: `outputs/production_exact/predictions.jsonl`
- SHA-256: `076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8`
- Target comparison: byte-identical
- Paper F1: 1.000000
- Evidence F1: 0.6291919191919192
- Multiple choice: 32/41
- Freeform: 23/26
- Table row F1: 1.000000
- Table cells: 25/27 micro
- Official evaluator invocations: 1

## Transport and contracts

All six JSONL tests and seven component contract groups pass. U+2028/U+2029, Unicode minus, non-ASCII text, embedded tabs/newlines, empty/nested values, large records, strict UTF-8, and line-numbered errors are covered. Inputs contain 55 records and 41 ordered option sets with no gold-like keys. All 410 option-aware permutation tests pass. No query-specific literal branch was found in accepted MC modules.

## Historical launcher/resume chronology (superseded by completed fresh run)

Status: `LAUNCHER_REPAIRED_BUT_EXTERNAL_EXPORT_POLICY_REJECTED_BEFORE_RUN`. The user granted explicit informed Phase 7 approval. The PowerShell `Start-Process` `Path`/`PATH` collision was repaired by adding `scripts/phase7_launcher.py`, which avoids `Start-Process`, normalizes environment keys case-insensitively, emits one canonical `Path`, and starts Python through `subprocess.run`.

The focused launcher test `tests/test_phase7_launcher.py` passed as a no-network dry run. It verified child process creation, intended Python executable and working directory, `.env` availability, no case-insensitive duplicate environment keys, and no secret printing.

The exact intended command remains:

```powershell
python scripts\run_ver2_reproduction.py --mode raw-fresh --output outputs\fresh_api\predictions.jsonl --verify-hashes
```

Working directory: `<WORKSPACE_PARENT>\littraceqa_baseline_Ver.2.3_full_reproduction`.

The real API-transmitting launcher command was attempted after the dry-run test passed, but the approvals layer rejected it before execution under external-export policy. In the subsequent enabled-network environment, the stale pre-run block files were removed only after verifying that no prediction/run artifacts existed, and the exact direct PowerShell call-operator command was attempted:

```powershell
& python 'scripts\run_ver2_reproduction.py' --mode raw-fresh --output 'outputs\fresh_api\predictions.jsonl' --verify-hashes
```

That direct command was also rejected by the approvals layer before execution under external-export policy.

A later interrupted manual run at `outputs/fresh_api_manual_20260718_010651` preserved 30 valid raw generation records and 30 valid run-log records, with last successful record `q_031`. `records/MANUAL_RUN_RESUME_AUDIT.json` verifies zero invalid records, zero duplicate IDs, and 25 remaining query IDs. The Version 2.3 runner was repaired to support resumable raw-fresh execution: subprocess output now streams to files, child PID/command/cwd/start time are recorded, heartbeat files are monitored, a stall timeout terminates the child process tree, and `--resume-run` validates partial records, skips completed IDs, generates only missing IDs, and preserves original input order before post-processing.

The no-network resume tests passed. The real resume command preserving the 30 completed records and generating only 25 missing records was attempted, but the approvals layer rejected it before execution under external-export policy. No additional validation data was transmitted. API calls are 0 for the resumed continuation, infrastructure retries are 1 from the earlier launcher failure, timeouts are 0, tokens/cost are unavailable/0, and no complete prediction, freeze manifest, fresh SHA, official metrics, or stage-level divergence was produced.

## Immutability and limits

The final per-file audit reports zero missing, added, or modified files across Version 2 (221), Version 2.1 (266), Version 2.2 (140), and the Version 3 freeze set (5). During the interrupted fresh run, 1033 generated cache files appeared under Version 2; only baseline-absent generated additions were removed after path-safety checks, and the verifier rerun passed. The ancestry resolver was not patched. Strategy A was not repeated. No API retry or metric-driven rerun occurred. Official evaluator calls were limited to one per generated frozen prediction.

## Remaining blockers

1. Historical producer manifest or generation-to-merge command for the cache boundary.
2. Execution environment whose policy permits the already approved SiliconFlow fresh run.
3. Historical source-grounded producer code/corpus snapshot if exact execution of that cached stage is required.

No metric optimization was performed.
No final checkpoint answer was copied into the reconstructed prediction.
No validation gold was used during generation or post-processing.
The frozen Version 2, Version 3 and official evaluator artifacts were not modified.


## Phase 7 fresh API completion (2026-07-18)

Final Phase 7 status: **launcher repaired; fresh pipeline completed; metric reproduction result is non-matching/low fresh reproduction**.

Run directory: `outputs/fresh_api_manual_20260718_010651`.

Exact resume command used:

```powershell
& python 'scripts\run_ver2_reproduction.py' --mode raw-fresh --resume-run 'outputs\fresh_api_manual_20260718_010651' --output 'outputs\fresh_api_manual_20260718_010651\predictions.jsonl' --verify-hashes --stall-timeout-sec 1200 --heartbeat-interval-sec 30
```

Typed/final continuation command after recovering completed source-grounded output:

```powershell
& python 'scripts\run_ver2_reproduction.py' --mode raw-fresh --start-stage typed_mc --resume-manifest 'outputs\fresh_api_manual_20260718_010651\resume_after_source_grounded_manifest.json' --resume-run 'outputs\fresh_api_manual_20260718_010651' --output 'outputs\fresh_api_manual_20260718_010651\predictions.jsonl' --verify-hashes --stall-timeout-sec 1200 --heartbeat-interval-sec 30
```

Python executable: `python`.
Working directory: `<WORKSPACE_PARENT>\littraceqa_baseline_Ver.2.3_full_reproduction`.
Endpoint/model: `https://api.siliconflow.cn/v1`, `deepseek-ai/DeepSeek-V4-Flash`, temperature 0.

Writable-path isolation passed: runtime PDF/download/text/image/VLM caches and manifests were redirected under `outputs/fresh_api_manual_20260718_010651/cache/`, with `PYTHONDONTWRITEBYTECODE=1` for child processes.

Records reused/generated: 30 preserved records reused; 25 remaining records generated; 30 completed API calls avoided; no completed API call was repeated. Merged raw generation contains 55 records and 55 unique query IDs. Final prediction contains 55 records, 55 unique query IDs, no missing/extra records, and 41 MC records corresponding to 41 ordered option sets. Eleven MC values are empty in raw generation and remain empty through all later stages; these are not orchestration-caused blanks.

Prediction freeze: `outputs/fresh_api_manual_20260718_010651/predictions.jsonl`, SHA-256 `4791DD77091EA3A581E94019C1C77A962E64E729D65C53F29C6FDE65096E68BF`.

API usage: total llm-call records in merged raw log 55; reused/preserved 30; new 25; retries 0 observed in raw logs; timeouts 0; token and cost fields were unavailable, so cost is recorded as unavailable.

Official evaluator: one complete post-freeze evaluation was run with the local Version 2 evaluator and explicit local development gold path after an initial non-metric process failed because the evaluator default `data/validation.jsonl` path was absent. Official metrics:

- Paper F1: 0.0
- Evidence precision/recall/F1: 0.0 / 1.0 / 0.0
- MC accuracy: 0.2682926829268293
- Freeform exact match: 0.0
- Table row F1: 1.0
- Table-cell macro/micro accuracy: 1.0 / None
- Missing/extra predictions: 0 / 0

Comparison:

- Version 2 cached-exact DEV: Paper F1 1.0; Evidence F1 0.6346464646464647; MC 38/41; Freeform 23/26; Table row F1 1.0; Table cells 25/27 micro.
- Version 2.3 fresh API: Paper F1 0.0; Evidence F1 0.0; MC accuracy 0.2682926829268293; Freeform EM 0.0; Table row F1 1.0; Table-cell macro/micro 1.0 / None.
- Earlier Version 2.1/2.2 fresh artifacts remain immutable; no evaluator rerun was performed for them in Phase 7. They are referenced only as prior frozen fresh-run artifacts.

Divergence vs cached-exact DEV by component: retrieval/gold_papers 0/55; raw generation answer 50/55; table 8/55; freeform 9/55; evidence 24/55; base MC 21/55; typed MC 13/55; final records 36/55.

Immutability: final verifier passed with zero missing, added, or modified files in Version 2, Version 2.1, Version 2.2, and version-freeze. No frozen research result, Version 3/version-freeze artifact, historical checkpoint, or official evaluator implementation was modified.


## Offline fresh-run schema repair (2026-07-18)

- No additional API calls were made and raw generation was not rerun.
- Original evaluator input audit: structurally invalid. The completed Phase 7 evaluator run used `..\littraceqa_baseline_Ver.2\inputs\validation_inputs.jsonl` as `--gold`; that file has questions/answer types but no evaluator gold `answer`, `gold_papers`/`papers`, or `evidence` fields. The prior Paper/Evidence/Freeform metrics are therefore not valid fresh-generation performance.
- Field survival trace: no paper/evidence/freeform/table/MC count loss was detected from raw projected prediction through final. The final stage directly copied typed MC output.
- Root cause: evaluator gold/input mismatch. Secondary assembly defect: table answers had `rows` but no `schema`; final assembly did not enforce the evaluator-compatible table contract.
- Repair: added `scripts/reassemble_fresh_prediction.py` and created `outputs/fresh_api_manual_20260718_010651/predictions_reassembled.jsonl`. The assembler reads only original inputs and fresh stage artifacts, starts from a canonical prediction shell, preserves fresh paper/evidence/freeform/table-row/MC fields, updates MC as an answer subfield, and adds table schema from original input.
- Reassembled SHA-256: `A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281`. Original prediction SHA-256 remains `4791DD77091EA3A581E94019C1C77A962E64E729D65C53F29C6FDE65096E68BF`.
- Focused tests passed: 55 records, 55 unique IDs, paper/evidence survival, evaluator-compatible freeform/table schema, MC non-overwrite of freeform/table, table structural keys, missing-vs-empty distinction, U+2028/U+2029 round trip, original prediction/raw generation unchanged, no frozen checkpoint values copied, and no gold fields read.
- One evaluator invocation was run on the repaired prediction using the official evaluator implementation with `outputs/cached_exact/predictions.jsonl` as a DEV-reference surrogate because local official `data/validation.jsonl` is absent. Result is not official-gold performance: Paper F1 1.0; Evidence P/R/F1 0.7881529581529582/0.898888888888889/0.7947474747474746; MC None; Freeform None; Table row F1 0.0; Table-cell macro/micro 0.0/None.
- Immutability verification passed for Version 2, Version 2.1, Version 2.2, and version-freeze with zero missing, added, or modified files.

## Authoritative official-gold finalization (2026-07-18)

Evaluation status labels supersede all earlier metric interpretations in this report:

- question-input-as-gold evaluation: `INVALID`;
- cached-prediction-as-gold evaluation: `DIAGNOSTIC_ONLY`;
- final hash-locked official-gold evaluation: `AUTHORITATIVE`.

The accepted historical evaluation record identified the true official DEV gold at `<WORKSPACE_PARENT>\LitTraceQA\data\validation.jsonl`, SHA-256 `DAA5ED246C00A5E4BB571843BAA985B6256700DA8A7AE5695BD642DFD4298E41`. It has 55 unique records, full reference-answer/paper/evidence fields, 41 MC references, 26 freeform references, and 11 table references. The evaluator is `LitTraceQA\scripts\evaluate.py`, SHA-256 `9410B51E86FC1EA382565D376016152FD52A74FA5D1F9358E36AF54711D8895F`.

The locked wrapper passed ten contract tests without invoking the evaluator, then ran the evaluator exactly once on frozen prediction SHA-256 `A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281`.

Authoritative metrics:

- Paper precision/recall/F1: `1.0 / 1.0 / 1.0`
- Evidence precision/recall/F1: `0.6182539682539683 / 0.5686868686868687 / 0.5815597533779352`
- Multiple choice: `27/41` (`0.6585365853658537`)
- Freeform exact match: `16/26` (`0.6153846153846154`)
- Table row F1: `0.8753968253968254`
- Table-cell macro: `0.5732323232323232`
- Table-cell micro: `14/27` (`0.5185185185185185`)
- Missing/extra predictions: `0/0`

Fresh-versus-cached earliest primary divergences are API semantic/evidence generation for 30 queries, table extraction for 4, base MC for 2, and none for 19. Paper retrieval agrees on all 55 queries. Evidence differs on 24 queries, MC letters on 13, normalized freeform on 9, and table row sets on 4. The dominant gaps are fresh semantic/evidence generation, MC abstention (30 nonempty versus 41 cached), and table extraction—not final serialization.

The final package is release-ready with explicit disclosures: fresh metrics are materially divergent, and the historical raw-to-base-cache producer edge remains unverified. No post-evaluation answer repair or optimization was performed.

## API Repair, Causal Ablation, And Release Packaging Update

The external API path was verified with bounded auditable requests. The transport failures occurred during a user-initiated transition from an explicit localhost proxy route to TUN-mode transparent routing. The causal runner was functional. Network completion rates across conditions are therefore not treated as model-quality evidence.

Final causal matrix: CURRENT-C0 12/12 parsed, CURRENT-C1 11/12 parsed, STRONGER-C1 12/12 parsed. Total successful generation outputs: 35/36, under the 42-record ceiling. Model decision: STRONGER_MODEL_ONLY_FOR_SELECTED_FAMILIES for the Version 2.3 recommended fresh experimental profile only.

## Release Bundle Finalization

Code-only bundle: `dist/littraceqa_baseline_Ver.2.3_reproduction_code_only.zip`, SHA-256 2BF7F55F5DC2902BCF2474594A67E33EF8D175EB8EDA5BDE76328EAD782E9394.

Internal verified bundle: `dist/littraceqa_baseline_Ver.2.3_reproduction_internal.zip`, SHA-256 0A48AF101D289F579E9D8E9AF09C04497007498EE5068C10B7AAE3ED4E65BBBF.

Clean-room validation and release security audit passed. Frozen artifacts and the official evaluator remained unchanged.
