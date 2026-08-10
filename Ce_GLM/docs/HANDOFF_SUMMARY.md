# Version 2.3 reproduction handoff

## Status

Work is paused at the Phase 1 checkpoint-lineage quality gate. The workspace is:

`<WORKSPACE_PARENT>\littraceqa_baseline_Ver.2.3_full_reproduction`

Phase 0 passed. Phase 1 established the stable downstream Version 2 lineage but could not prove one upstream producing edge. Phases 2–7 were not started. Mode A, production reproduction, Mode B, integration, API transmission, and official evaluation were not started after the block.

## Primary target

- Target: `littraceqa_baseline_Ver.2/checkpoints/CANONICAL_DEV_BEST_TYPED_MC_predictions.jsonl`
- Verified SHA-256: `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`
- Expected metrics: Paper F1 1.000000; Evidence F1 0.6346464646464647; MC 38/41; Freeform 23/26; Table row F1 1.000000; Table cells 25/27 micro.
- The target was hashed as required by preflight. Its contents were not opened for a pre-reproduction field diff.

## Verified lineage

Hash-verified downstream nodes:

1. `parent_19of26_25of27_predictions.jsonl` — `188D081433DA039171AFDF56EF6248F79520019EEA8A7DAD5B1B4FD336FD8344`.
2. `modular_difference_completeness_23of26_25of27` — `C3B51991E37EC1DC2E778E3FC41C2EA41D6D96DF4793C24A19F447D966915B0C`.
3. `DEV_BEST_OPTION_AWARE` — `9338B351E7DBC677F58C7BC847D8754E251D86E26746BDB9911677A34C741839`.
4. `CANONICAL_DEV_BEST_EVIDENCE_SAFE` — `A9A99BB552363E030F5876664AE8BB4024381A72C66A0782D993603A3D0D7B89`.
5. `CANONICAL_DEV_BEST_SOURCE_GROUNDED_MC` — `D791909E0C342BA2393D617ABD007DE95D8127036545FE9B3D44AEC59A94DCAA`.
6. DEV typed target, hash only — `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`.
7. Production option-aware parent — `944513E5084522C3233842B9AE15DCD7BB2CEE1E44DC683DC827C8968FEDBBDE`.
8. Production typed target, hash only — `076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8`.

Pre-target field-change aggregation verified that completeness-to-DEV-option-aware changed only 15 MC letters, DEV-option-aware-to-evidence-safe changed evidence on two records, and evidence-safe-to-source-grounded changed only four MC letters. The final target was not diffed before a Mode A freeze.

Metadata and archived-artifact existence also resolved the accepted upstream table/evidence chain through pairwise row pruning, evidence subset/page-context replays, schema-aligned column repair, Docling page alignment, equation-location evidence, section-bounded count, and the four completeness renderers. These nodes are listed precisely in `records/HANDOFF_STATUS.json` and `records/PHASE_1_CHECKPOINT_ANCESTRY.json`.

## Exact blocker

The unresolved artifact is:

`littraceqa_baseline_uq_experiments/outputs/modular_composed_label_locator_predictions.jsonl`

It exists (28,181 bytes) and is consumed by later archived table/RAG commands. Known consumers include `audit_predicted_table_row_existence.py`, `replay_unsupported_table_rows.py`, `run_full_docling_object_rag_pipeline.sh`, and `prepare_complete_docling_manifest.py`.

No verified manifest or command collected in this session produces it from raw validation input, retrieval/base outputs, or archived API responses. `outputs/all_target_experiment/run_manifest.json` names it as `base_prediction_path`, confirming that manifest consumes it rather than creates it. Inferring a producer from filenames would violate the Phase 1 instructions.

Phase 1 therefore stopped at its explicit quality gate: every final field could not be traced through a verified raw-input/retrieval/base-generation edge. The task required stopping before implementation in that condition.

## Bounded producer-provenance result (2026-07-17)

Strategy A was executed and then stopped. The exact artifact path, its SHA-256 `54DA46600AFE81DAB5D8D2F10E87AC453FF5FCA206296DAF126EFFCD4D4C409D`, `output_prediction`, and explicit command fields were checked across exactly 10 compact candidates:

1. `outputs/experiments/VERSION3_RESEARCH_SESSION_SUMMARY.json`
2. `outputs/all_target_experiment/run_manifest.json`
3. `outputs/all_target_experiment/full_docling_run_manifest.json`
4. `outputs/all_target_experiment/docling_manifest_summary.json`
5. `outputs/all_target_experiment/docling_remaining35_run_summary.json`
6. `outputs/final_release_audit/FINAL_RELEASE_MANIFEST.json`
7. `outputs/version_freeze/CHECKPOINT_MANIFEST.json`
8. `outputs/canonical_dev_leader_build/provenance_summary.json`
9. `outputs/external_parser_manifest_summary.json`
10. `outputs/complete_paper_selection_v1_checkpoint_audit/summary.json`

No verified producer edge was found. Three manifests mention the artifact only as an existing input: `run_manifest.json` uses `base_prediction_path`, `full_docling_run_manifest.json` uses `base_predictions`, and `docling_manifest_summary.json` uses `predictions`. None of the candidates contains the artifact SHA-256, an `output_prediction` record, or an explicit producing command.

Phase 1 therefore still cannot pass. The next bounded action is for the user to supply either (a) the archived producer manifest identifying this exact output hash and its hash-linked upstream inputs, or (b) the exact generation-to-merge command together with the archived input/output hashes. Phase 2 and all execution modes remain unstarted.

No immutable artifact was modified. No prediction content was inspected, no recursive repository scan was performed, and the ancestry resolver was not patched.

## Material already inspected

Small release/configuration files:

- Version 2 `README.md`, `VERSION.md`, `requirements.txt`, `environment.yml`, `configs/release.json`.
- Version 2 `docs/REPRODUCTION_GUIDE.md`, `docs/ACTIVE_MODULES.md`, `docs/CHECKPOINT_LINEAGE.md`.

Compact lineage material:

- DEV/production option-aware metadata and evidence-safe metadata.
- Verified completeness replay manifest.
- DEV/production typed replay manifests.
- Evidence replay summaries.
- Final-release audit manifest and checkpoint-lineage report.
- Relevant checkpoint metadata and selected accepted experiment-registry rows.
- `artifacts/CHECKSUMS.sha256` and the all-target experiment run manifest.
- Bounded README/CHANGELOG command excerpts for row pruning and pairwise adjudication.

Implementation inspection was bounded to seven AST-probed producer entry points and exact sections needed from the completeness, option-aware, source-grounded, typed, and fresh entry paths. Large JSONL files were processed locally for counts, hashes, schemas, or aggregate differences only; no full prediction, log, metadata corpus, or object index was printed.

## Files created

The session created the Version 2.3 directory structure plus:

- Phase 0 preflight, source-directory hashes, source manifest, and four immutable per-file baselines.
- Phase 1 lineage probe and checkpoint ancestry records.
- `scripts/phase0_preflight.py`, `scripts/phase1_lineage_probe.py`, and `scripts/phase1_checkpoint_ancestry.py`.
- This handoff, `records/HANDOFF_STATUS.json`, and the updated session context.

No algorithm, checkpoint, prediction, evaluator, or earlier Version directory was changed. All writes were confined to the new Version 2.3 workspace.

## Commands already executed

The principal executable commands were:

```powershell
& python `
  'littraceqa_baseline_Ver.2.3_full_reproduction\scripts\phase0_preflight.py'

& python `
  'littraceqa_baseline_Ver.2.3_full_reproduction\scripts\phase1_lineage_probe.py'

& python `
  'littraceqa_baseline_Ver.2.3_full_reproduction\scripts\phase1_checkpoint_ancestry.py'
```

The ancestry command was run initially and after each of two repairs. Other commands were read-only `Get-Content`, `Get-Item`, shallow `Get-ChildItem`, and targeted `rg` queries over exact names/hashes. No replay, generation, evaluator, API, or integrated pipeline command was executed.

## Repair-cycle accounting

Two ancestry-resolver repair cycles were consumed:

1. Added metadata aliases and accepted experiment-registry edges.
2. Merged duplicate metadata/registry nodes and unioned parent edges.

The two-cycle limit for that resolver is exhausted. A continuation should not patch it again unless the user explicitly resets the limit.

## Bounded continuation strategies

Do not execute these in this conversation. A new conversation may choose one:

1. Search only manifests, indexes, or run summaries for an output hash or explicit producer record naming `modular_composed_label_locator_predictions.jsonl`. Do not inspect prediction contents.
2. Inspect only the bounded provenance/schema of the archived `selective_uq_rag_visual` raw and run logs to determine whether a verified hash link from those artifacts to the missing base prediction exists.
3. If neither provides producer proof, request the missing run manifest or generation-to-merge command from the user and leave Phase 2 blocked.

## NEW CONVERSATION STARTING CONTEXT

```text
Resume Version 2.3 full-code reproduction from:
<WORKSPACE_PARENT>\littraceqa_baseline_Ver.2.3_full_reproduction

Read first:
1. records/SESSION_CONTEXT.md
2. records/HANDOFF_STATUS.json
3. docs/HANDOFF_SUMMARY.md
4. records/PHASE_1_LINEAGE_PROBE.json
5. records/PHASE_1_CHECKPOINT_ANCESTRY.json

Phase 0 passed. Phase 1 stopped at the mandatory quality gate because no verified producer edge reconstructs:
littraceqa_baseline_uq_experiments/outputs/modular_composed_label_locator_predictions.jsonl
from raw input, retrieval/base outputs, or archived API responses.

The artifact exists and later stages consume it. Do not infer lineage from filenames. Two ancestry-resolver repair cycles are already consumed; do not patch that resolver again. Mode A, Mode B, integration, API generation, and official evaluation have not started. Inspect only the missing producer provenance first, using one bounded strategy from the handoff.
```
