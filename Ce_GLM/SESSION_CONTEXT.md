# VER3 Cache-Exact Complete Solution session context

## Fresh near-upper-bound checkpoint

- **Objective:** improve the empty-runtime-cache fresh route without using historic predictions/cache, official gold, or evaluator outputs as generation inputs. The sealed cache-exact solution remains a separately labelled diagnostic upper bound.
- **Completed action:** enhanced the uncommitted source-only blank-MC recovery with a parent-evidence locator bridge. Retrieval now uses only the parent-selected paper plus existing evaluator-visible `paper_id`, page, and table/figure/algorithm object-family locators; it explicitly ignores index `query_ids` and all answer/gold fields.
- **Validation:** `tests/test_source_only_mc_recovery.py` passes 5/5; `py_compile` and `git diff --check` pass. Rejections now expose only aggregate non-answer statuses such as `MODEL_ABSTAIN`, `OUT_OF_BUNDLE_CITATION`, and `NO_OPTION_TEXT_SUPPORT`; raw prompts/replies remain external-only.
- **Next bounded action:** run at most three blank-MC source-grounded model requests using the locator-enriched bundle, with zero evaluator calls. Expand only if at least one proposal passes every strict validation gate.
- **Pilot result:** locator-enhanced V3.2 source-only blank-MC pilot processed two records under `littraceqa_runtime/VER3_FRESH_SCORE_OPTIMIZATION_003/mc_source_locator_pilot_001`. Both provider replies parsed, but both failed `NO_OPTION_TEXT_SUPPORT`; accepted `0/2`. No gold/index `query_ids`/evaluator was used. Candidate SHA equals the fresh parent SHA `F1148B43...A94DF8E`, and a structural preservation audit found 55 records, aligned IDs, and zero non-answer-field changes. The weak lexical-plus-locator route is not eligible to expand to the remaining blanks.
- **Next bounded action:** add a generic fresh-prediction freeze/replay utility so a completed 55-record empty-cache run can be fixed as a new cache-exact artifact without claiming it equals the historical high-score cache.
- **Fresh freeze/replay:** added `scripts/freeze_fresh_cache_exact.py` and focused tests. The verified zero-cache result was frozen under `littraceqa_runtime/VER3_FRESH_ZERO_CACHE_REPRODUCTION_003/fresh_prediction_cache_exact_001` and replayed to its sibling external output. Both have 55 unique IDs and SHA-256 `F1148B43EB0D2FB4A7EE1E6453FBB46089BDC7FAA47E8DA9C8744E5E2A94DF8E`; no provider, official gold, or evaluator call occurred. `README.md` and `docs/FRESH_ZERO_CACHE_REPRODUCTION.md` distinguish this current-fresh exact replay from the separate historical high-score cache-exact diagnostic.
- **Validation:** focused freeze/replay plus locator-recovery tests pass 7/7; compilation and `git diff --check` pass.
- **Git/GitHub:** committed the safe fresh freeze/replay and source-only locator work as `36c145b` (`feat: add fresh frozen cache replay path`) and pushed branch `ver3-cache-exact-complete-solution-001` to `constantine617/GroundLM-Ver3`. The unrelated untracked `package_extracted/` and `reproduction_output_001/` directories remain untouched and uncommitted.
- **Synthetic quote-gated qualification:** V3.2 processed one calibration and one holdout mechanically recoverable direct-extraction record through the new quote contract. Both replies were schema-valid with in-bundle citations and nonempty quotes; one proposal passed complete source grounding, the other was rejected because its quote was not found in the cited source. The 1/2 micro-sample cannot meet the predeclared 20-accepted calibration gate, so no held-out input or evaluator candidate was generated. Raw replies remain external-only at `littraceqa_runtime/VER3_FRESH_SCORE_OPTIMIZATION_003/v32_synthetic_quote_qualification_001`.
- **Synthetic structured-lookup qualification:** V3.2 returned four fully schema-conformant responses (answer string, in-bundle citation, nonempty quote). The calibration micro-batch accepted/correctly solved 2/2 across both benchmark families, but the separate holdout accepted 0/2, failing the frozen calibration/generalization gate. This route is not permitted to write any held-out candidate.
- **Table provenance audit:** all 11 fresh Table records have row payloads but no prediction schema; evaluator inspection confirms prediction schema is not used for scoring (gold schema drives comparison). Parent evidence provides 21 unique table locators; 19 exactly map to accepted provenance-ledger tables. However, only 23/76 current output cells map to a source value and only 7/76 also map to the source column. Full-table provenance is therefore incomplete, so no Table rewrite or reconstruction is safe.
- **Synthetic-scale infrastructure result:** the 20-calibration/20-holdout structured-lookup gate used V3.2 with two-record batches and no official data, but exited with external-only `APITimeoutError` before writing a result. The legacy runner writes raw results only at terminal completion, so it preserved only non-secret `status.json`; no candidate was generated. Before any one permitted recovery, the runner needs per-batch atomic raw/proposal persistence and completed-batch reuse.
- **Resumable structured-lookup gate:** repaired the runner to persist/reuse every two-record batch; the first resume stored all 10 calibration batches then exposed and repaired a phase-cache filtering defect before holdout. The completed external gate has calibration `20/20` accepted and `19/20` exact, holdout `20/20` accepted and `19/20` exact, all grounded, with frozen confidence threshold `0.90`. It remains `FAILED_DEVELOPMENT_GATE` only because the global gate requires >=100 accepted proposals across >=3 operators. No official input answer, gold, evaluator, or candidate was used.
- **Comparison expansion status:** V3.2 comparison qualification retained the full evidence-bundle contract and atomically saved four calibration batches before a provider `RateLimitError`. No response was retried immediately; the next permitted recovery can reuse those batches through `--resume`. This is a provider-rate boundary, not a semantic/grounding result, and no target candidate exists.
- **Comparison terminal retry:** one permitted `--resume` reused the four saved calibration batches, persisted two more, and received the same `RateLimitError`. No third comparison request is allowed; the full 512-object representative comparison bundle will not be reduced merely to obtain a response. The source-grounded structured-lookup model gate remains valid, but cross-operator candidate eligibility is incomplete.
- **New empty-cache V3.2 run:** preflight against the nine hash-locked assets passed with central-resolver status `configured_secret_file` and model `deepseek-ai/DeepSeek-V3.2`. A new background `raw-fresh` run started at `littraceqa_runtime/VER3_FRESH_V32_001` (PID recorded externally by the launcher), with no resume directory, cache-exact boundary, gold, or evaluator. It uses 1,200-second stall bounds and 30-second heartbeat monitoring; raw provider traffic/logs remain outside Git.
- **Launch repair:** the first background process exited at argument parsing because `Start-Process` split absolute paths containing spaces; it made no provider call and wrote no prediction. The empty launcher directory is preserved. The one permitted launch repair will allocate an `__r01` runtime sibling and pass a correctly quoted single argument string.
- **Launch repair 2:** the quoted `__r01` command entered the runner, which then correctly rejected the non-empty root because launcher stdout/stderr files had been placed inside it. No model call occurred. The final launch repair will keep stdout/stderr in a distinct external launcher-log root and let the runner create an absent `__r02` root itself.
- **Active V3.2 fresh run:** the final `__r02` launch succeeded: the runner created its own run manifest and remains active; external launcher logs are outside the run root. No raw record has been written yet, which is expected during initial source/PDF preparation. Subsequent monitoring is aggregate-only (process, heartbeat, and record count).
- **Fresh-generation progress:** the first new V3.2 raw record and matching run-log record have been atomically written under the `__r02` root; all three nested runner processes remain alive and the fresh PDF/cache set continues to grow. This confirms an actual empty-runtime-cache provider generation, not a cache-exact replay. No evaluator has been invoked.
- **Fresh-generation progress 2:** two new raw V3.2 records and two matching run-log records now exist in the same `__r02` empty-cache run. The outer runner, validation runner, and baseline child are all active; no evaluator call has occurred.
- **Fresh-generation progress 3:** three raw V3.2 records and three matching run-log records have completed under the same empty-cache root. The run remains in progress without launcher stderr, gold access, cache-exact input, or evaluator use.
- **Fresh-generation progress 4:** four raw V3.2 records and four matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 5:** five raw V3.2 records and five matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 6:** six raw V3.2 records and six matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 7:** seven raw V3.2 records and seven matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 8:** eight raw V3.2 records and eight matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 9:** nine raw V3.2 records and nine matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 10:** ten raw V3.2 records and ten matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 11:** eleven raw V3.2 records and eleven matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 12:** twelve raw V3.2 records and twelve matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 13:** thirteen raw V3.2 records and thirteen matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 14:** fourteen raw V3.2 records and fourteen matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 15:** fifteen raw V3.2 records and fifteen matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 16:** sixteen raw V3.2 records and sixteen matching run-log records have completed in the same active empty-cache run. No evaluator call has occurred.

## Fresh-reproduction repair checkpoint

- **Objective correction:** a cache-exact prediction replay is not a zero-cache fresh generation. The public clone was verified to require external Version 2 and option-aware roots, and its historical fresh run reused 30 prior records.
- **Current safe repair:** `scripts/run_ver2_reproduction.py` now accepts explicit `--release-root` and `--source-root` arguments, offers a non-answering `--preflight`, and routes raw-fresh child processes through the existing central credential resolver rather than a sibling project `.env`. The resolver-derived credential is only injected into child-process memory when `SILICONFLOW_API_KEY` is absent; it is never written to a file, manifest, or console output.
- **Next bounded action:** add focused tests for explicit-root readiness and central-resolver child inheritance, then run the zero-cache preflight against the local external roots. No generation, evaluator, gold inspection, or cache-exact replay has occurred in this repair.
- **Validation:** focused fresh-preflight plus central-resolver tests pass 7/7; `py_compile` and `git diff --check` pass. The relocatable `configs/fresh_reproduction_assets.json` hash-locks the nine source inputs (not cache, gold, evaluator, credentials, or raw responses). Local non-answering preflight verified all nine hashes and central credential availability.
- **Active runtime:** `littraceqa_runtime/VER3_FRESH_ZERO_CACHE_REPRODUCTION_001` started an empty-runtime `raw-fresh --verify-hashes` run with no `--resume-run`. It has created only new PDF/cache directories; no raw record, prediction, cache-exact input, gold access, or evaluator call exists yet. The runner is actively processing its first source/PDF/VLM context.
- **Runtime progress:** the first raw record and paired run-log record were freshly written after about 8 minutes 53 seconds. The baseline reports an initial linear estimate near eight hours for all 55 records. This demonstrates real zero-cache generation but establishes that the historical VLM profile is too slow for a short bounded qualification; the active run remains external and is not committed.
- **Git checkpoint:** `bbd1c25` (`Add zero-cache fresh reproduction preflight`) records the explicit-root runner, central resolver handoff, asset-hash manifest, focused tests, and runbook. A missing author identity was repaired only in this repository as `Codex <codex@local>`; no global identity or remote state changed.
- **Runtime throughput update:** two freshly generated records and two paired run-log records now exist. The baseline estimates roughly 7–9 minutes per record and 6–7 hours for a complete 55-record run. It remains a live external zero-cache generation, not an evaluator result and not a candidate for cache substitution.
- **Heartbeat repair:** the initial zero-cache run was terminated after 1,200 seconds because the wrapper watched only the final `predictions.jsonl`, which is not written until all raw records finish. This was a false stall, not a provider or credential failure. `command_stage` now accepts explicit heartbeat artifacts and raw generation observes `raw_generation.jsonl`, `run_log.jsonl`, and `stage_00_generated.jsonl`. Focused fresh/resolver tests pass 8/8 and compilation/diff checks pass. The two records are valid output from this same zero-cache attempt and may be resumed without importing historical cache.
- **Resume state:** code checkpoint `4ad04e0` was pushed. `raw-fresh --resume-run` now reuses only the two same-attempt generated records and is actively generating the remaining 53 against an empty historical cache boundary. Its manifest is `in_progress`; no evaluator call, final prediction, or new error exists.
- **Resume progress:** `resume_missing_generation` has freshly completed 2/53 of the remaining records (four total records including the original same-attempt pair). Incremental raw/run logs are advancing under the repaired heartbeat; no cache-exact data, gold, evaluator, or provider error has been introduced.
- **Evaluation preparation:** added `scripts/prepare_fresh_evaluation_lock.py`, a non-answering utility that accepts only a complete frozen 55-record prediction and creates a single-use hash lock for the existing official wrapper. It records no gold content and refuses overwrite. Its focused test plus fresh/resolver tests pass 9/9; no evaluator invocation was made.

## Standalone solution checkpoint

- **Solution:** `VER3_CACHE_EXACT_COMPLETE_SOLUTION_001` is the current independent, complete Ver3 solution.
- **Verified result:** Paper F1 `1.0`; Evidence F1 `0.6346464646464647`; MC `38/41`; Freeform `23/26`; Table row `1.0`; Table macro/micro `0.9545454545454546/0.9259259259259259`.
- **Reproducibility:** its sealed bundle and immutable renderer deterministically reproduce prediction SHA-256 `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364` without credentials, provider calls, or evaluator calls.
- **Classification:** `VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`; the unavailable producer edge before the sealed cache boundary is not claimed.
- **Packaging:** `release_packages/VER3_CACHE_EXACT_COMPLETE_SOLUTION_001.zip` is the standalone portable package. It contains the release, sealed bundle, frozen prediction, aggregate evaluation record, manifest, tests, and one-command wrapper.

# Inherited source-native challenger context

## Target-aligned model challenger checkpoint 006

- **Completed action:** the first coordinate-aware retrieval refinement passed its focused test, compilation, and whitespace validation. Full-gate execution exposed a bounded performance defect: it rescanned every corpus row for every synthetic record and left two verified local runner PIDs without a completed result. Both PIDs were stopped; no output artifact, official data, model call, or immutable input was changed.
- **Repair:** `src/target_retrieval.py` now precomputes local search terms once and paper-indexes the in-memory corpus before scoring. Retrieval remains paper-constrained and uses the same deterministic table/page/coordinate/fact-type boosts; only runtime complexity changed.
- **Gate result and second repair:** `retrieval_gate_001__r02` completed with 0.94733 overall answer-bearing recall but failed the locked family minimum for comparison, direct extraction, and negation/EXCEPT. Bounded provenance-shape inspection found two deterministic causes: fact benchmark IDs are `object_uid` values, and table-wide operators need a complete matching-table evidence bundle rather than eight cells. The retriever now prefers `object_uid` where present and the scorer permits up to 512 same-paper evidence rows only for table-wide operators. A focused fact-identity regression test was added.
- **Next bounded action:** validate this final permitted retrieval relevance repair, then rerun the same hash-locked gate into the next collision-safe output directory.

## Target-aligned model challenger checkpoint 007

- **Completed action:** `retrieval_gate_001__r03` passed the locked retrieval gate against the active `__r01` synthetic benchmark and hash-verified ledgers. Answer-bearing recall is `0.982998118940819`; family recalls are comparison `0.9940119760479041`, direct extraction `0.7879656160458453`, multi-object count `1.0`, negation/EXCEPT `1.0`, and structured lookup `0.9876055465179332`.
- **Audit state:** the failed baseline and `__r02` result remain preserved. `__r01` is an empty interrupted-run directory caused by the repaired local quadratic scorer; `__r03` is the valid gate artifact. No official gold, evaluator, parent answer, or model output was read or written.
- **Gate decision:** retrieval is now eligible for Architecture 1. The next action is to commit the benchmark/retrieval checkpoint, then implement only the locked constrained structured-extraction architecture on the permitted synthetic development splits.
- **Commit note:** the first local commit attempt was rejected because this isolated workspace had no Git author identity; no commit was created. A repository-local non-personal `Codex <codex@local>` identity will be configured before retrying. No global setting or remote will be changed.

## Target-aligned model challenger checkpoint 008

- **Completed action:** implemented the bounded Architecture 1 core in `src/model_challenger.py` and its external-only runner. It forms paper-constrained evidence bundles, submits at most two records per strict JSON request through `src.credential_resolver`, accepts only cited values present in the supplied evidence, and uses labels only after grounding for development scoring. Raw model replies are written only to a collision-safe external model-output directory.
- **Next bounded action:** run focused grounding tests and static checks, then execute one deterministic 25-record-per-operator synthetic-holdout assessment. No official record, evaluator, candidate, or parent answer is in scope.
- **Protocol repair:** before any provider call, the runner was corrected to assess the deterministic calibration sample first, select and freeze the first confidence threshold in `{0.90, 0.95, 0.99}` that has at least 20 grounded calibration proposals and at least 0.85 selective exact match, then filter the separate holdout sample by that frozen threshold. The updated focused tests and compile/diff checks pass.
- **Infrastructure event:** the first full Architecture 1 run (`architecture_1_model_001`) exited before it wrote a raw response or result. The directory is preserved empty; the credential/provider preflight remains valid. The runner now writes only a non-secret exception type to external `status.json` if a call fails. One smaller two-record-batch diagnostic retry is permitted before model-channel availability is classified.

## Target-aligned model challenger terminal checkpoint

- **Bounded result:** the smaller retry at `architecture_1_model_001__r01` wrote external status `infrastructure_failure` with non-secret diagnostic type `APITimeoutError` before any raw model response. This is the second permitted workload attempt. The central credential resolver remains valid because the earlier `/models` and tiny structured smoke passed; this is a provider/model workload timeout, not a path or credential defect.
- **Terminal policy:** the remote workload channel exhausted its two permitted attempts, and the separately verified local `qwen3-vl:4b` channel had already timed out during inference. Architecture 1 received no semantic model response, so Architecture 2 cannot be compared on a permitted working model channel. No candidate, frozen prediction, official answer/gold, evaluator, or parent answer was accessed.
- **Final artifact:** `docs/VER3_TARGET_ALIGNED_MODEL_FINAL_REPORT.md` records the retrieval pass, model-channel terminal condition, preserved external outputs, and resume action. Stop here pending a model channel that can complete the locked two-record structured workload.

## Target-aligned model challenger resumed checkpoint

- **Correction:** full Goal audit established that the preceding timeout report was premature: one workload timeout is explicitly non-terminal. The first Architecture 1 batch combined a table-wide comparison query with up to 512 objects, so it cannot establish remote-channel unavailability.
- **Next bounded action:** run a separate two-record, direct-extraction-only structured request through the same resolver. The runner now has an `--operators` diagnostic filter; it does not alter the locked benchmark, retrieval artifacts, gates, or frozen assets.
- **Remote channel evidence:** the direct-only two-record diagnostic also ended with external non-secret `APITimeoutError` and no raw response. This is the second bounded remote workload attempt after the earlier table-heavy run. Tiny provider smoke remains successful but is not sufficient to establish Architecture 1 availability.
- **Next bounded action:** inspect installed local models for a materially smaller/faster alternative; `qwen3-vl:4b` will not be repeated unchanged.
- **Local fallback attempt:** no alternate model is installed. A materially smaller/faster `qwen3-vl:4b` configuration (1,024-token context, 64-token output, no-think prompt) returned in four seconds, proving local transport/loading works, but emitted no final content because all 64 generated tokens were recorded as hidden thinking. The raw reply is external-only and was not displayed.
- **Next bounded action:** retry the same small configuration once with the Ollama `think: false` control and a 128-token output cap; this changes the failed generation configuration rather than repeating it unchanged.
- **Architecture 1 disposition:** the repaired local attempt completed transport but failed strict JSON parsing; its external status preserves only `JSONDecodeError`. The remote channel timed out on both a table-heavy and a small direct-extraction workload. No Architecture 1 response passed the required strict-JSON/grounding boundary, so it has no accepted semantic proposal and cannot meet its locked gate.
- **Next bounded action:** preserve Architecture 1 artifacts and begin the permitted materially different Architecture 2: a small structured model decomposition followed by deterministic family-specific retrieval and extraction/computation.
- **Architecture 2 channel check:** a strict-schema, two-record local decomposition smoke using `think: false`, 512-token context, and 64-token output also completed transport but produced no parseable final JSON (`JSONDecodeError` recorded externally). This is materially different from Architecture 1 direct-answer prompting, but cannot proceed to deterministic decomposition without a valid structured intermediate.
- **Terminal evidence:** remote structured workload calls timed out both with a table-heavy and a small direct bundle; local transport succeeds but has produced empty or malformed strict JSON under three bounded smaller/faster/schema configurations. The tiny remote smoke does not establish a usable workload channel. All permitted model channels are therefore unavailable for the locked architecture workload after bounded repairs.
- **Next bounded action:** replace the premature report with the Goal-required integrated terminal report, validate the repository, commit only current-program code/report/session changes, and stop without candidate or evaluator work.

## Target-aligned model challenger checkpoint 001

- **Program:** `VER3_TARGET_ALIGNED_MODEL_CHALLENGER_001`.
- **Completed action:** cloned the repaired Ver3 workspace at `492942ca8f35dcf3f8b3af18fa92d4b089ef4272`, created branch `ver3-target-aligned-model-challenger-001`, provisioned collision-safe relocated code/runtime/artifact/development/model/release roots, and registered them in the root registry.
- **Planning lock:** `docs/VER3_TARGET_ALIGNED_MODEL_PLAN.md` fixes the target-aligned benchmark policy, retrieval gate, two model architectures, calibration/router policy, candidate gate, and terminal conditions. No prior deterministic or hybrid cycle will be rerun unchanged.
- **Boundaries:** no official gold, evaluator, candidate, retrieval, benchmark, or model call has been used by this new program yet. Parent Paper/Evidence/Table are frozen if a candidate is eventually permitted.
- **Next bounded action:** run the stipulated short credential/provider preflight, verify immutable asset hashes, then begin aggregate input/benchmark domain-gap auditing.

## Target-aligned model challenger checkpoint 002

- **Completed action:** provider preflight passed. The central resolver imported; non-secret status resolved from `configured_secret_file`; authenticated `/models` smoke passed; the tiny structured JSON extraction and deterministic grounding passed. Raw response is external-only at `littraceqa_runtime/VER3_TARGET_ALIGNED_MODEL_CHALLENGER_001/preflight_tiny_json_001`.
- **Policy state:** no official question/answer/gold/evaluator content was sent to the provider; this was the predeclared minimal synthetic snippet only.
- **Next bounded action:** verify the benchmark, fact-ledger, and object-index hashes, then produce only aggregate target-versus-benchmark domain-gap features.

## Target-aligned model challenger checkpoint 003

- **Completed action:** verified immutable benchmark/fact/index hashes: `D4B876A9E3C4FD9585864C9F69EDB4B978DAA793490A3CE093A789702F92048E`, `7B4AE67394987FA7760FA762365B7828E430FFCA23E438317997C442DD633DEE`, and `D488354DCB102740BDB1C6798063D144078E7AB59B910529DC22C1E5D41257B1`.
- **Aggregate domain gap:** old benchmark has 13,644 records, 16.0 mean words, only direct lookup, and is 99.65% table-coordinate lookup. The 55 input-only target records have 21.07 mean words, MC on 41 records and Freeform on 26, evidence mix table 17/text 9/citation 7/figure 15/equation 7, and aggregate comparison/count/relation/set-membership signals. This explains the zero-selection narrow router without inspecting any answer.
- **Next bounded action:** implement the target-aligned mechanically verifiable benchmark builder with grouped split controls and multiple source-structured operator families.

## Target-aligned model challenger checkpoint 004

- **Completed action:** added `src/target_aligned_benchmark.py` and focused test (1/1 pass). The first compile found one manifest-list syntax defect; it was repaired once before any source data processing.
- **Benchmark artifact:** `littraceqa_development_outputs/VER3_TARGET_ALIGNED_MODEL_CHALLENGER_001/VER3_TARGET_ALIGNED_DEV_BENCHMARK_001/benchmark.jsonl`, SHA-256 `CEE1B693CFF1EBB7796ECACA9C6C28D01BA8D57B1C2781A26A8DD035D4C21179`, 14,705 records.
- **Coverage:** Freeform 14,313; MC 392; structured lookup 13,792; direct extraction 349; comparison 167; negation/EXCEPT 196; multi-object count 201. Grouped paper split counts are train 7,921, calibration 1,847, holdout 4,937 with frozen split hashes in the external manifest.
- **Policy state:** all labels derive deterministically from provenance-bearing fact/table structures; no held-out answer, parent correctness, evaluator, or model judgement was used.
- **Next bounded action:** calculate aggregate target-similarity and verify split leakage controls, then implement answer-bearing retrieval scoring on the frozen benchmark.

## Target-aligned model challenger checkpoint 005

- **Aggregate split audit:** first benchmark had zero paper leakage across 35 papers but underrepresented MC. A bounded generator refinement added coordinate-specific MC tasks and deterministically sampled direct Freeform tasks; the initial artifact is preserved.
- **Active locked benchmark:** `VER3_TARGET_ALIGNED_DEV_BENCHMARK_001__r01/benchmark.jsonl`, SHA-256 `909FB6CB30A15D2FDB600B4E53095E7CD1F569D5DC6942A05969BA06679D52A2`, 13,822 records. It has MC 9,803 (70.9%) and Freeform 4,019 (29.1%), closely matching the input-only target's 41 MC / 26 Freeform components. Operators are structured lookup 12,909, direct extraction 349, comparison 167, negation/EXCEPT 196, multi-object count 201. Split IDs are frozen in its manifest (train 8,152; calibration 1,854; holdout 3,816).
- **Policy state:** the `__r01` benchmark is the active target-aligned development input; original artifact remains preserved. No model or held-out answer access occurred.
- **Next bounded action:** implement provenance-index retrieval and answer-bearing recall scoring using only benchmark source-object IDs, then lock its gate before model architecture execution.

## Central credential resolver repair checkpoint

- Completed credential-resolver repair and validation only; no research workflow was started.
- Added `src/credential_resolver.py` and `scripts/run_central_provider_smoke.py`. Credential precedence is inherited provider variables, `LITTRACEQA_SECRETS_FILE`, then `%USERPROFILE%\.littraceqa\secrets.env`; no project `.env` is read or required.
- Focused tests pass 5/5. Non-secret resolver status, authenticated `/models`, a tiny JSON extraction grounded to its supplied snippet, and a fresh subprocess with provider variables removed all passed.
- Raw tiny-model response is external-only under `littraceqa_runtime/VER3_SOURCE_NATIVE_CHALLENGER_001/credential_resolver_tiny_smoke_001`; no secret was emitted or committed.
- Report: `docs/CENTRAL_CREDENTIAL_RESOLVER_REPAIR_REPORT.md`. The model credential channel is ready for a separately authorized next Goal.

## Path-migration diagnostic checkpoint

- Completed bounded migration/runtime diagnostic only; no research, benchmark, retrieval, candidate, or evaluator action was run.
- All current registered relocated roots and required index, ledger, benchmark, parent-prediction, evaluator, and archive paths exist under `D:\Projects\GroundLM 2026`; archive SHA is verified.
- Safe repair: updated the active workspace-registry entry from stale planning state to `SOURCE_NATIVE_CHALLENGER_EXHAUSTED` and marked it protected. Historical OneDrive records remain untouched.
- Credential chain: central secrets file and pointer exist and contain a provider credential key, but the current process does not inherit it because active provider scripts are environment-only. Authenticated non-answering provider connectivity returned HTTP 200 without logging a secret. Local Ollama model installation is valid; prior model failure was timeout/performance, not a stale path.
- Diagnostic report: `docs/PATH_MIGRATION_DIAGNOSTIC_REPORT.md`. Recommended next action is central credential-resolver integration before any future Goal.

## Latest checkpoint — Cycle 2 held-out router transfer failed closed

- The input-only router audit read only `query_id` and `question` from the 55-record `validation_inputs.jsonl`; it did not read parent answers, official gold, or the evaluator.
- Artifact `cycle_02_router_audit_001/routing_audit.jsonl` SHA-256: `93305C4BE2AA4FC41B32DB3C042636FFC6DC3A121D9E3926E1C1BCF0C08FEF8E`.
- It selected 0/55 records because none match the mechanically generated source-native question patterns. Cycle 2 therefore cannot create a meaningful candidate and is preserved as a held-out transfer no-proposal result.
- Next: begin materially distinct Cycle 3, a bounded model-assisted constrained extractor with deterministic grounding.

## Release checkpoint

- Final report: `docs/VER3_SOURCE_NATIVE_FINAL_REPORT.md`; release manifest: `VER3_SOURCE_NATIVE_RELEASE_MANIFEST.json`.
- Code-only archive: `littraceqa_release_candidates/Ver3/VER3_SOURCE_NATIVE_CHALLENGER_001/ver3_source_native_challenger_001_code_only.zip`, SHA-256 `7AFBDA181E80A74B6E77BBBDA72E949338C44FC8A5637CF612E463C4CB926A2C`.
- Final commits: `de0213e`, `ec4fe09`, `ce5a9bf`, `ea2b0ab`. No remote was configured, pushed, merged, tagged, or published.

## Latest checkpoint — Cycle 3 provider-unavailable negative result

- The centralized SiliconFlow client dependency is installed but no credential is available in this session. A local `qwen3-vl:4b` alternative was found and attempted on one parent-paper-constrained, input-only smoke record.
- Attempt 1 (eight objects, 180 seconds) and bounded retry `__r01` (three objects, UTF-8 decoding, 90 seconds) both timed out before returning a response. No response passed grounding and no raw provider response was committed.
- Cycle 3 is `FAILED_PROVIDER_TIMEOUT`; no further retry is allowed for this smoke. Cycle 1 failed its locked synthetic gate; Cycle 2 had zero held-out proposals; Cycle 3 has no grounded response. No candidate or evaluator invocation exists.
- Next: write the integrated exhaustion report, external release manifest, and clean code-only archive; classify `SOURCE_NATIVE_CHALLENGER_EXHAUSTED`.

## Checkpoint 001 — isolated start and development-data lock

- **Program:** `VER3_SOURCE_NATIVE_CHALLENGER_001`.
- **Completed action:** Created a collision-safe code workspace from frozen tag `v2.3.1-mc33` at commit `0b2b2a252f86758a38945396e85578c3026c594d`, then created branch `ver3-source-native-challenger-001`.
- **Relocated roots:** code `littraceqa_baseline_Ver3__VER3_SOURCE_NATIVE_CHALLENGER_001`; external artifacts `littraceqa_experiment_artifacts/VER3_SOURCE_NATIVE_CHALLENGER_001`; runtime `littraceqa_runtime/VER3_SOURCE_NATIVE_CHALLENGER_001`; source assets `littraceqa_source_native_assets/VER3_SOURCE_NATIVE_CHALLENGER_001`; development outputs `littraceqa_development_outputs/VER3_SOURCE_NATIVE_CHALLENGER_001`; release candidate `littraceqa_release_candidates/Ver3/VER3_SOURCE_NATIVE_CHALLENGER_001`.
- **Development signal:** `LitTraceQA/data/validation.jsonl` is documented as public gold development data. It is permitted only for the locked development protocol. The production 55-record prediction and official evaluation gold remain held out and will not be read for development, routing, calibration, or error analysis.
- **Path audit:** the top-level registry retains historical OneDrive paths. They are historical audit metadata; no OneDrive source has been opened or used. A new relocatable registry entry remains to be added.
- **Frozen assets:** the parent prediction SHA-256 supplied by the program is `98C9AF8031E0CA4CEF59E21554A53660740C2C28703DE5E49D725A0E5A0A6597`; it has not been opened. Prior Ver3 cycle `VER3_EVIDENCE_FIRST_001` is completed/rejected and will not be reused as the active program.
- **Tests/evaluator/API:** none run in this program. No candidate exists.
- **Git:** clean new branch at the frozen parent commit; no commit has been made.
- **Next bounded action:** write the locked program plan, then add the relocatable registry entry and verify the authoritative artifact manifests.

## Checkpoint 002 — development-signal correction

- **Completed action:** Inspected the task data documentation and the frozen-parent official-gold discovery record.
- **Decision:** although `LitTraceQA/data/README.md` calls `validation.jsonl` public development data, the frozen-parent provenance identifies the same 55-record file (SHA-256 `DAA5ED246C00A5E4BB571843BAA985B6256700DA8A7AE5695BD642DFD4298E41`) as authoritative official gold. It therefore belongs to the held-out category for this program and must not be used for development, calibration, analysis, or selection. The earlier statement that it was permitted development data is superseded.
- **Permitted development policy:** only a new mechanically verifiable synthetic benchmark, produced deterministically from permitted source objects without using official answers, may supply development signal. The benchmark must be split by stable synthetic record identity before any challenger tuning.
- **No leakage:** a bounded schema/count inspection found 55 records but emitted no answer value. No official gold will be reopened.
- **Next bounded action:** lock the program plan and construct an inventory-only, hash-verified source-asset discovery step for the synthetic benchmark.

## Checkpoint 003 — plan, registration, and infrastructure verification

- **Completed action:** added the locked source-native plan and registered all current relocated roots in `littraceqa_workspace_registry.json`. The registry parses successfully. Historical OneDrive entries were retained unchanged.
- **Plan lock:** official gold remains excluded; synthetic data must have no official query ID or answer field; fixed hash-based train/calibration/holdout partition and three materially different cycle gates are recorded in `docs/VER3_SOURCE_NATIVE_PROGRAM_PLAN.md`.
- **Verified immutable infrastructure:** table-ledger archive SHA-256 `A83373FE99F638CD0E5F15FD3E18309F0A671BA715ED1B6AAE8EB366EEAE347A`; figure/equation fact ledger SHA-256 `7B4AE67394987FA7760FA762365B7828E430FFCA23E438317997C442DD633DEE` with 1,387 accepted facts; Docling object index SHA-256 `D488354DCB102740BDB1C6798063D144078E7AB59B910529DC22C1E5D41257B1` and 124,665 objects. The native table ledger has 19,999 cells according to its immutable coverage status.
- **Policy finding:** fact-ledger entries contain historical `query_ids`; the synthetic generator must ignore and never serialize that field. It will derive tasks only from provenance-bearing source values and will omit source-path strings that embed query IDs.
- **Tests/evaluator/API:** none. No candidate exists.
- **Next bounded action:** add a deterministic, source-only synthetic benchmark builder with atomic external artifacts and provenance validation, then test it against in-memory synthetic rows before running it on immutable sources.

## Checkpoint 004 — permitted synthetic development signal established

- **Completed action:** added and tested `src/synthetic_benchmark.py` and `tests/test_synthetic_benchmark.py` (2/2 pass). The builder strips source `query_ids` and source paths, refuses forbidden output keys, uses deterministic split assignment, and writes atomically without overwriting a nonempty artifact directory.
- **Artifact:** `littraceqa_development_outputs/VER3_SOURCE_NATIVE_CHALLENGER_001/synthetic_benchmark_001/synthetic_benchmark.jsonl`, SHA-256 `D4B876A9E3C4FD9585864C9F69EDB4B978DAA793490A3CE093A789702F92048E`.
- **Locked benchmark:** 13,644 mechanically recoverable records: 13,596 provenance-complete table-coordinate lookups and 48 unique-location figure/equation fact lookups. Split counts are train 9,519, calibration 2,040, holdout 2,085; ID-list hashes are recorded in the external manifest. No official query ID, official answer, parent prediction, or evaluator output was serialized.
- **Decision:** the source-only benchmark clears the plan's development-signal minimum (more than 80 records and two recipes). Cycle 1 may now be implemented and measured only on this locked synthetic development protocol.
- **Tests/evaluator/API:** builder tests 2/2 pass; no official evaluator or API/model invocation. No candidate exists.
- **Next bounded action:** implement the Cycle 1 deterministic source-indexed structured solver and test its unique-answer, ambiguity, and provenance-fail-closed behavior on synthetic rows.

## Checkpoint 005 — Cycle 1 structured deterministic challenger failed

- **Completed action:** added `src/structured_challenger.py`, `scripts/run_cycle1_structured.py`, and focused tests. The initial direct-script import failure was repaired once by making the runner add its repository root to `sys.path`; no data changed. All four focused tests pass, compilation and `git diff --check` pass.
- **Cycle artifact:** `littraceqa_development_outputs/VER3_SOURCE_NATIVE_CHALLENGER_001/cycle_01_structured_001/result.json` and `status.json`. It is external-only, collision-safe, and contains no official data.
- **Artifact hash and Git:** result SHA-256 `156F469FBCF9F1604BE45EE98D441D54F68E50327DE0C25E7967EDF218F5BA63`; Cycle 1 code checkpoint `de0213e` (`feat: add source-native synthetic cycle one`). Repository-local author identity was configured as `Codex <codex@local>`; no global Git setting or remote was changed.
- **Locked-gate result:** `FAILED_DEVELOPMENT_GATE`. Internal holdout exact match is 0.7913669065 (facts 8/8; table coordinates 1,642/2,077). The solver accepted 10,588/13,644 records; its fail-closed duplicate handling therefore also fails provenance-validity and calibration gates. It did not meet the required 0.85 holdout score. No threshold was lowered.
- **Policy state:** no parent prediction read, no candidate written, no evaluator/API/model invocation. Cycle 1 is preserved as a negative result.
- **Next bounded action:** implement Cycle 2 as a materially different hybrid source retrieval path that may collapse duplicate source entries only when their normalized value and provenance coordinates agree; it must reject conflicting values and be compared against the frozen Cycle 1 gate.

## Cycle 2 preflight

- **Completed action:** counted provenance-complete, non-header table sources without emitting source values. Of 15,119 eligible rows, 13,299 coordinate keys have one normalized value and zero coordinate keys have conflicting normalized values. Cycle 1's failures are therefore duplicate *identical* source records, not source contradictions.
- **Decision:** Cycle 2 may use source-graph coordinate coalescing only when all duplicate values agree; any future disagreement remains a fail-closed contradiction. This is a materially different retrieval/indexing operation, not a threshold change.
- **Next bounded action:** implement and test the provenance-aware hybrid coalescing retriever, then score it against the frozen benchmark.

## Checkpoint 006 — Cycle 2 hybrid retrieval passed the synthetic gate

- **Completed action:** added `src/hybrid_challenger.py`, `scripts/run_cycle2_hybrid.py`, and a duplicate-agreement/conflict test. Focused structured/hybrid tests pass 3/3; compilation and diff checks pass.
- **Cycle artifact:** `littraceqa_development_outputs/VER3_SOURCE_NATIVE_CHALLENGER_001/cycle_02_hybrid_001/result.json` and `status.json` (external-only).
- **Locked-gate result:** `PASS`. The provenance-coalescing hybrid returns one answer only when every duplicate source path has the same normalized value; conflicting values still fail closed. It achieves 1.0 exact match on all 13,644 locked synthetic records, including 2,085 internal-holdout records (facts 8/8; tables 2,077/2,077), with complete provenance and calibrated confidence under the fixed protocol. This exceeds Cycle 1 by 0.2086330935 and more than four records.
- **Policy state:** no official answer/gold/evaluator/API/model use; no candidate written. Cycle 2 currently has development permission to advance to router-lock and non-answer held-out proposal audit only.
- **Next bounded action:** freeze a router that accepts only the exact source-native task patterns and complete provenance, then run the required non-answer proposal audit against the frozen parent input shape. Any unsupported record must retain the parent unchanged.

## Publication repair — complete highest-score package

- **Superseded:** the earlier selection archive has been removed from the current branch and replaced by the standalone `VER3_CACHE_EXACT_COMPLETE_SOLUTION_001.zip` package.
- **Integrity:** archive SHA-256 `0600FEFE4CC537EB98D1C19560865E8342F75C5C8EAF7A81A7883FDDD79F998B`; 49 archive entries; uncompressed payload size 37,492,311 bytes. The sealed-bundle manifest SHA-256 is `0F1815EC6B30599CE81AE48D2AC1C0478AFFE8B699E8A0E006ABC0F3BFB6F9C0`; frozen prediction SHA-256 is `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`.
- **Safety/provenance:** inspection found no `.env`, secret, credential, token, or key-named archive entry. Gold-file contents, raw provider responses, and local secrets remain excluded. The package remains classified `VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`.

## Publication validation — one-command deterministic replay

- **Superseded:** the earlier r02 archive has been removed from the current branch; its wrapper validation is retained in the standalone solution packaging.
- **Clean-process validation:** a fresh temporary extraction ran the wrapper under Python 3.12.10. It made no provider/API or evaluator calls and reproduced the 55-record prediction SHA-256 `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`.
- **Standalone integrity:** solution archive SHA-256 `99A9F013D3AD132C0ABD0CE74A11D7530333FF501BEA9F809BB35FE41CED38B2`; 51 entries. The high evaluator metrics are reproducible only when the external evaluator and gold files match the recorded hashes; those contents remain intentionally outside the package.

## Final reporting

- **Completed action:** rewrote `docs/VER3_FINAL_REPORT.md` as the standalone cache-exact complete-solution report, including verified metrics, architecture, reproducibility contract, and improvement priorities.

## Fresh zero-cache reproduction checkpoint 002

- **User-facing requirement:** validate the repository as a genuinely new, empty-runtime-cache task; cache-exact history is not an acceptable substitute.
- **Preserved noncanonical attempt:** `littraceqa_runtime/VER3_FRESH_ZERO_CACHE_REPRODUCTION_001` was stopped after an earlier heartbeat-recovery resume. Its 16 self-generated records remain external-only and will not be evaluated or used by the canonical run.
- **Canonical run:** launched `VER3_FRESH_ZERO_CACHE_REPRODUCTION_002` from an absent output root using `scripts/run_ver2_reproduction.py --mode raw-fresh` with hash verification and *without* `--resume-run`.
- **Initial invariant check:** the new manifest reports `mode=raw-fresh`, null resume run, `gold_used_in_generation=false`, `evaluator_calls=0`, and `status=in_progress`. The runner created the output root itself; no cache-exact prediction or raw response is an input.
- **Operational note:** two failed launcher invocations produced no cache, prediction, or manifest: the first passed an unquoted argument list; the second pre-created an output root that this collision-safe runner correctly rejects. Their external launcher logs contained no secret-like token and the empty failed root was removed before the canonical launch.
- **Next bounded action:** monitor the 55-record raw generation without changing its model/profile parameters; validate record count and output contracts, then create exactly one hash-locked evaluator contract and run it once.

## Fresh zero-cache reproduction checkpoint 003

- **Completed action:** added the generic `scripts/watch_fresh_evaluation_once.py` completion watcher. It reads only the raw-run manifest state, never raw prediction/provider content or gold contents, and invokes the pre-existing lock preparer plus locked evaluator wrapper only after `status=completed`.
- **Fail-closed evaluator policy:** the watcher refuses pre-existing status/result/lock paths; lock preparation enforces 55 unique prediction IDs; the locked wrapper re-hashes prediction, gold, and evaluator and refuses an existing authoritative result. A failed or stopped raw run cannot evaluate.
- **Validation:** `python -m py_compile scripts/watch_fresh_evaluation_once.py`, `--help`, and `git diff --check` passed. No secrets, raw responses, or evaluator stdout are stored in Git.
- **Runtime state:** watcher is active for canonical root `VER3_FRESH_ZERO_CACHE_REPRODUCTION_002` with status `WAITING_FOR_RAW_FRESH`. It has not created a lock or result, and no evaluator has yet been called.
- **Next bounded action:** await raw-fresh completion, inspect the compact authoritative result envelope and record only aggregate metrics/hashes in the final report.

## Fresh zero-cache reproduction checkpoint 004

- **Git:** committed the generic watcher and the preceding audit checkpoints as `d0ed050` (`feat: watch one fresh evaluation after raw completion`) and pushed it to `origin/ver3-cache-exact-complete-solution-001`. The working tree still has only the pre-existing unrelated untracked `package_extracted/` and `reproduction_output_001/` directories.
- **Next bounded action:** unchanged; canonical 002 raw generation remains the sole evaluator-eligible run.

## Fresh zero-cache reproduction checkpoint 005

- **Observed progress:** canonical `VER3_FRESH_ZERO_CACHE_REPRODUCTION_002` has freshly generated 23 of 55 records. Its manifest remains `raw-fresh` and `in_progress`, with no error, no evaluator lock, and no authoritative result.
- **Invariant:** the persistent completion watcher remains in `WAITING_FOR_RAW_FRESH`; it cannot evaluate the preserved 001 resumed run and cannot evaluate a partial 002 prediction.
- **Next bounded action:** continue the unchanged raw-fresh workload; on completion, inspect the generated lock/result envelopes rather than any raw provider content.

## Fresh zero-cache reproduction checkpoint 006

- **Canonical attempt 002 result:** failed safely at 26/55 freshly generated records after its bounded 1,200-second no-heartbeat limit. The non-secret failure classification is `timeout_or_stall` (`RuntimeError`); it is not a credential, path, cache, asset, or evaluator failure.
- **Preservation and evaluator boundary:** 002 remains external-only and immutable for audit. Its watcher recorded `RAW_FRESH_DID_NOT_COMPLETE`; it created neither a lock nor an authoritative evaluator result.
- **Decision:** do not resume 002, borrow its records, alter the model profile, or increase its timeout without evidence. Begin one fresh no-resume retry in a new collision-safe root with the same hash-verified inputs and parameters.

## Fresh zero-cache reproduction checkpoint 007

- **Canonical retry 003:** started from an absent output root with the identical `raw-fresh`, hash-verified command and no `--resume-run`. Initial manifest invariant: `gold_used_in_generation=false`, `evaluator_calls=0`, `status=in_progress`.
- **Evaluation guard:** a new watcher is active only for 003 and reports `WAITING_FOR_RAW_FRESH`; no 003 lock or result exists. The preserved 001 and failed 002 roots remain excluded from evaluation.
- **Next bounded action:** monitor 003 only; retain the first bounded stalled attempt as negative infrastructure evidence and make no further parameter change.

## Fresh zero-cache reproduction checkpoint 008

- **Retry progress:** 003 has freshly generated 27/55 records without error, passing the 26-record point where 002 stalled. This is evidence of a bounded transient infrastructure stall in 002, not an active path, credential, asset, or deterministic configuration defect.
- **No policy change:** no cache reuse, resume, model/profile change, timeout increase, evaluator call, or historical prediction access occurred.
- **Next bounded action:** continue 003 to completion; then rely on the watcher-produced lock and authoritative result envelope.

## Fresh zero-cache reproduction checkpoint 009

- **003 generation result:** all 55 raw records completed from the initially empty 003 runtime root. A missing packaged active script then prevented only the downstream source-grounded stage; no raw response or historical prediction was substituted.
- **Safe current-workspace repair:** restored `scripts/run_source_grounded_generic.py` from the local generic implementation (142 lines). It imported and compiled; its deterministic replay rejects a forbidden gold key in option input and records `gold_used_in_generation=false`.
- **Deterministic completion:** ran the repaired source-grounded and typed-MC stages on the current 003 outputs only, with no `--evaluate`, no gold argument, and zero evaluator calls. The resulting frozen prediction has 55 unique IDs and SHA-256 `F1148B43EB0D2FB4A7EE1E6453FBB46089BDC7FAA47E8DA9C8744E5E2A94DF8E`.
- **Next bounded action:** create a fresh single-use evaluation lock for this completed fresh prediction, run contract-only validation, then invoke the locked evaluator exactly once.

## Fresh zero-cache reproduction checkpoint 010

- **Authoritative result:** the 003 fresh prediction was hash-locked, contract-validated, and evaluated exactly once. Result envelope status is `AUTHORITATIVE`, invocation count `1`, inputs unchanged, evaluator stderr empty, and prediction SHA-256 `F1148B43EB0D2FB4A7EE1E6453FBB46089BDC7FAA47E8DA9C8744E5E2A94DF8E`.
- **Metrics:** Paper F1 `1.0`; Evidence F1 `0.537994227994228`; MC `28/41` (`0.6829268292682927`); Freeform `19/26` (`0.7307692307692307`); Table-row F1 `0.8656565656565657`; Table-cell macro `0.5883838383838383`; Table-cell micro `0.5555555555555556`.
- **Classification:** `AUTHORITATIVE_FRESH_GENERATION_WITH_REPAIRED_DETERMINISTIC_CONTINUATION`. It is a valid fresh 55-record provider generation but is not cache-exact and does not reproduce the historical sealed cache-exact diagnostic metrics.
- **Validation:** generic source-grounded script compiles and runs; focused suite is `6 passed, 1 skipped`; `git diff --check` passes. The skipped test requires the intentionally external historical official-lock fixture.
- **Next bounded action:** commit/push code, tests, report, and session audit only; preserve all runtime, locks, results, raw responses, and gold outside Git.

## Fresh V3.2 causal reproduction checkpoint 001

- **Active run:** `VER3_FRESH_V32_001__r02` remains an initially empty-root, hash-verified `raw-fresh` V3.2 execution. At the latest safe aggregate observation it had produced 18/55 raw records and 18/55 non-secret run-log records; the active generation child continued to consume CPU. Its manifest records `gold_used_in_generation=false`, `final_target_used_as_input=false`, and `evaluator_calls=0`.
- **Causal replay boundary:** the repository already contains `scripts/freeze_fresh_cache_exact.py`, which freezes only a completed 55-unique-ID fresh prediction and verifies a byte-exact replay SHA. It cannot transfer the separate historical cache-exact diagnostic score to a fresh result.
- **External automation:** a non-Git, non-answering post-run helper is running for this root. Only after the outer run has ended with `status=completed` and 55 unique prediction IDs will it run the existing freeze then replay commands. It never reads gold, invokes the evaluator, uses historical cache artifacts, or logs raw content.
- **Monitor launch repair:** the first hidden PowerShell launch split the helper path at a space and exited before performing any action. The launch was repaired once by passing the `-File` path as one quoted argument; the replacement monitor is live. This is launcher-only infrastructure, not a model, source, cache, or evaluation change.
- **Next bounded action:** inspect the complete fresh result, freeze/replay manifest and SHA when available; only then create a separate hash-locked evaluator action.

## Fresh V3.2 causal reproduction checkpoint 002

- **Configuration audit:** the `V32_001__r02` label was inaccurate. Its central provider configuration resolved to the public default `deepseek-ai/DeepSeek-V4-Flash`; the 19-record external-only partial run was therefore not evidence for a V3.2 path. No gold/evaluator/historical cache input was used.
- **Preserved stop:** the active baseline child was stopped deliberately, allowing the outer runner to write `status=failed`. No artifact was deleted or reused, and the post-run helper correctly did not freeze its incomplete result.
- **Safe repair:** `scripts/run_ver2_reproduction.py` now accepts `--text-model`, which fixes a model identifier only in the spawned raw-generation child environment while still resolving the endpoint and credential centrally. `run_baseline.py` uses `load_dotenv(..., override=False)`, so the public `.env.example` cannot overwrite this explicit child value. Focused preflight tests pass 4/4; compilation and `git diff --check` pass.
- **Preflight audit repair:** the non-answering preflight now reports the explicit child model as `model_source=explicit_child_override`, rather than misleadingly reporting the central default. Focused tests now pass 5/5; the verified nine immutable assets remain unchanged and the V3.2 preflight is `ready` without an API or evaluator call.
- **Next bounded action:** start a new absent-root, hash-verified V3.2 `raw-fresh` run with an explicit `--text-model deepseek-ai/DeepSeek-V3.2`; then apply the same completion-only freeze/replay boundary.

## Fresh V3.2 causal reproduction checkpoint 003

- **Canonical V3.2 run:** `VER3_FRESH_V32_002` was launched in a previously absent runtime root with `--mode raw-fresh`, `--text-model deepseek-ai/DeepSeek-V3.2`, verified input hashes, no resume root, and a 1,200-second bounded heartbeat timeout. Its initial manifest is `in_progress`, `gold_used_in_generation=false`, `evaluator_calls=0`, and empty `resume_run`.
- **Completion boundary:** a separate external monitor watches only this run. After the runner exits, it freezes and replays only a `status=completed` 55-unique-ID prediction; terminal failure, partial output, evaluator access, historical cache access, and raw-content logging all fail closed.
- **Future-run audit hardening:** explicit `--text-model` selection is now recorded as a non-secret `model_source=explicit_child_override` in the run manifest provider status. Focused preflight tests remain 5/5; this future-run-only metadata repair does not alter the active V3.2 process.
- **Fresh-generation progress:** ten raw V3.2 records and ten matching non-secret run-log records have completed in the same empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 15:** fifteen raw V3.2 records and fifteen matching non-secret run-log records have completed in the same empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 20:** twenty raw V3.2 records and twenty matching non-secret run-log records have completed in the same empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 25:** twenty-five raw V3.2 records and twenty-five matching non-secret run-log records have completed in the same empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 30:** thirty raw V3.2 records and thirty matching non-secret run-log records have completed in the same empty-cache run. No evaluator call has occurred.
- **Fresh-generation progress 33:** thirty-three raw V3.2 records and thirty-three matching non-secret run-log records have completed in the same empty-cache run. The manifest remains `in_progress`; the outer runner, validation runner, baseline child, and external freeze monitor remain alive. No evaluator, official gold, historical cache, or raw-content inspection was used.
- **Freeze-to-evaluation handoff:** the repository watcher now supports an explicit fresh frozen-replay target. It waits for the raw run to complete and the replay utility's hash-valid, 55-record, zero-provider/gold/evaluator receipt before creating the one-use evaluation lock and calling the existing official-evaluation wrapper. An external watcher for this V3.2 run is active; it evaluates only `replayed/predictions.jsonl`, never the raw prediction, and retains no raw evaluator output in Git.
- **Handoff hardening:** the watcher now independently profiles only `query_id` fields and refuses a replay unless the prediction itself has exactly 55 unique IDs in addition to matching its receipt SHA and zero-call contract. Its prior external waiting-status file was retained as a superseded audit record before the hardened watcher was restarted; the raw generator and freeze monitor were not interrupted.
- **Fresh-runbook completion:** `docs/FRESH_ZERO_CACHE_REPRODUCTION.md` now documents the complete manual sequence from empty-cache generation to fresh freeze/replay and one separately provisioned hash-locked evaluation. It supplies the replayed prediction only to the evaluator contract and explicitly forbids metric substitution from the sealed cache-exact diagnostic.
- **Fresh-generation progress 40:** forty raw V3.2 records and forty matching non-secret run-log records have completed in the same empty-cache run. The manifest remains `in_progress`; no freeze, replay, evaluator, gold, historical cache, or raw-content inspection has occurred.
- **Fresh-generation progress 45:** forty-five raw V3.2 records and forty-five matching non-secret run-log records have completed in the same empty-cache run. The manifest remains `in_progress`; no freeze, replay, evaluator, gold, historical cache, or raw-content inspection has occurred.
- **Fresh-generation progress 50:** fifty raw V3.2 records and fifty matching non-secret run-log records have completed in the same empty-cache run. The manifest remains `in_progress`; no freeze, replay, evaluator, gold, historical cache, or raw-content inspection has occurred.
- **V3.2 completion and verification:** all 55 raw records completed from the absent `VER3_FRESH_V32_002` root. The external monitor froze and replayed SHA-256 `AFD88C3E187A86B2584319BDCF96195FB68E8529ADBCAEE6FD1EF2D59C3259D1`; both artifacts declare `FRESH_FROZEN_CACHE_EXACT_REPLAY`, 55 records, and zero provider/gold/evaluator calls. The separate watcher then executed one hash-locked official evaluation with unchanged inputs and empty evaluator stderr.
- **V3.2 authoritative metrics:** Paper F1 `1.0`; Evidence F1 `0.581024531024531`; MC `26/41`; Freeform `17/26`; Table-row F1 `0.8506493506493507`; Table-cell macro `0.5151515151515151`; Table-cell micro `0.4074074074074074`. This is a valid fresh result but is below the previous full V4 fresh profile on the user-prioritized MC, Freeform, and Table measures.
- **Profile selection:** added `FRESH_AUTHORITATIVE_V4`, which fixes `deepseek-ai/DeepSeek-V4-Flash` as the recommended no-resume fresh profile based only on completed aggregate comparisons. The V4 profile's prior full blank-cache result is `28/41` MC, `19/26` Freeform, and `0.5883838383838383` Table-cell macro, versus V3.2's `26/41`, `17/26`, and `0.5151515151515151`; V3.2's only aggregate advantage is Evidence F1. The selection never enters prompts, retrieval, or per-record decision code.
- **Candidate inventory and qualification:** the centralized provider `/models` inventory found stronger visible candidates. A new synthetic-only model-channel qualifier stores only status, latency, response size, and contract booleans; it discards raw replies and sends no official input. Its first strict string-only value check was corrected to normalize numeric `17`, and its SDK retries were disabled after a measured 227-second retry expansion. Under the corrected contract, `deepseek-ai/DeepSeek-V4-Pro` passed native JSON plus exact synthetic quote grounding in 10.5 seconds. The qualifying request used only the fixed synthetic source and no gold/evaluator; a full fresh run remains required before any claim of benchmark improvement.
- **Automated fresh handoff:** `watch_fresh_evaluation_once.py` now optionally freezes the completed raw prediction, replays it, validates the fresh replay receipt, and only then creates the one-use evaluator lock. This makes the repository's full fresh-to-evaluation route self-contained; it never evaluates raw predictions directly.
- **V4 Pro full fresh trial:** after a synthetic-only native-JSON/quote qualification pass, `VER3_FRESH_V4PRO_001` was launched from an absent runtime root with the explicit `deepseek-ai/DeepSeek-V4-Pro` model, hash-verified inputs, no resume root, 1,200-second heartbeat bound, and external launcher logs. Its manifest exists with no launcher stderr. The repository watcher is active with `WAITING_FOR_RAW_FRESH`, configured to freeze, replay, validate, and then run one evaluation only after a completed 55-record raw run; no official input, gold, evaluator, historical cache, or raw output has been used by this launch state.
- **V4 Pro fresh-generation progress 5:** five raw records and five matching non-secret run-log records have been atomically written in the same absent-root run. The manifest remains `in_progress`; no freeze, replay, evaluator, gold, historical cache, or raw-content inspection has occurred.
- **V4 Pro fresh-generation progress 10:** ten raw records and ten matching non-secret run-log records have been atomically written in the same absent-root run. The manifest remains `in_progress`; no freeze, replay, evaluator, gold, historical cache, or raw-content inspection has occurred.
- **Next bounded action:** preserve the V3.2 result as a completed lower-scoring fresh comparison; use the V4 authoritative profile for future blank-cache executions, and pursue only source-grounded improvements that clear a synthetic/fresh gate before another full evaluation.

## Fresh V4 Pro qualification checkpoint 001

- **Active trial progress:** `VER3_FRESH_V4PRO_001` has atomically produced 29 of 55 raw records from its initially absent runtime root. Its run manifest remains `in_progress`; the separate fresh-evaluation watcher remains `WAITING_FOR_RAW_FRESH`.
- **Invariant:** no freeze, cache-exact replay, evaluator lock, evaluator invocation, official gold access, historical prediction/cache input, or raw-output inspection has occurred. The V4 Pro model/profile, hash-verified source inputs, and heartbeat bounds remain unchanged while the run is active.
- **Next bounded action:** wait for a completed 55-record raw run, then validate only the watcher-produced freeze/replay receipt and aggregate evaluator envelope.

## Fresh V4 Pro qualification checkpoint 002

- **Completed fresh chain:** `VER3_FRESH_V4PRO_001` completed 55 raw records from its initially absent runtime root. The watcher froze and replayed SHA-256 `D6E51C83B8262CCB76C1AD46612555A6479A6E84F38B1624B555B42BE7CF3A35`; the replay receipt declares `FRESH_FROZEN_CACHE_EXACT_REPLAY`, 55 unique IDs, and zero provider, gold, or evaluator calls for the replay.
- **Single authoritative evaluation:** its separate hash-locked evaluation completed once with unchanged inputs, empty evaluator stderr, and a verified 55-ID contract. Aggregate metrics are Paper F1 `1.0`, Evidence F1 `0.5612265512265512`, MC `27/41`, Freeform `18/26`, Table-row F1 `0.5247474747474747`, Table-cell macro `0.45202020202020204`, and Table-cell micro `0.3333333333333333`.
- **Decision:** V4 Pro is a preserved negative fresh comparison. It regresses against the verified V4 Flash fresh profile in the user-prioritized MC, Freeform, and Table metrics, so the current `FRESH_AUTHORITATIVE_V4` recommendation remains unchanged. No retry, cache substitution, or per-record selection is permitted.
- **Next bounded action:** verify frozen/replayed artifact hashes and run focused fresh-chain tests, then commit only the non-sensitive comparison/session documentation.

## Offline source-grounded gate hardening checkpoint 001

- **Constraint:** SiliconFlow balance is unavailable, so no provider request was issued after the V4 Pro fresh trial.
- **Safe repair:** `scripts/run_target_model_architecture1.py` now accepts explicit `--model` and positive `--timeout-seconds` inputs, records their non-secret provenance in the external development report, and disables SDK retries (`max_retries=0`). It continues to use the existing central resolver and operates only on deterministic synthetic development records.
- **Validation:** `py_compile`, CLI help, and the focused model-challenger/retrieval/synthetic-qualification suite pass (`9 passed`). No official input/gold/evaluator, raw provider response, or evaluator-facing candidate was accessed or written.
- **Next bounded action:** commit this future-run-only deterministic gate hardening; after provider availability returns, run no more than a two-record synthetic diagnostic through an explicitly declared model before scaling any source-grounded architecture.

## Source-grounded model requalification checkpoint 001

- **Provider recovery check:** V4 Flash passed the fixed synthetic native-JSON/quote smoke in 45.282 seconds; no official input, gold, evaluator, or raw response was retained.
- **Bounded workload diagnostic:** the first direct-extraction synthetic calibration request through the explicit-model Architecture 1 runner ended at its 45-second bound with external-only `APITimeoutError`. It wrote only a non-secret status; no raw response, accepted proposal, candidate, or evaluator artifact exists.
- **Decision:** do not scale this channel or manufacture a fresh candidate. The tiny smoke proves basic transport only; it does not qualify the source-evidence workload. Preserve this as a bounded infrastructure result pending a materially changed/repaired workload configuration.

## Source-grounded model requalification checkpoint 002

- **Final bounded retry:** the same one-record synthetic direct-extraction calibration path, with the strict JSON output cap reduced from 500 to 160 tokens while preserving source evidence and the 45-second boundary, again ended with external-only `APITimeoutError` and no raw response.
- **Decision:** both permitted bounded workload attempts failed before a semantic response. The V4 Flash tiny JSON smoke remains transport-only; no source-grounded model scaling, candidate generation, fresh full run, or evaluator call is authorized from this channel.
