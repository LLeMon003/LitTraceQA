# VER3 Target-Aligned Model Challenger: integrated terminal report

## Classification

`TARGET_ALIGNED_MODEL_CHALLENGER_EXHAUSTED`: all permitted model channels were unavailable for the required structured-evidence workload after bounded repairs. No candidate was created, frozen, evaluated, or replayed.

## Workspace, inputs, and policy

- Workspace: `D:\Projects\GroundLM 2026\littraceqa_baseline_Ver3__VER3_TARGET_ALIGNED_MODEL_CHALLENGER_001`, created from repaired-resolver commit `492942ca8f35dcf3f8b3af18fa92d4b089ef4272` on branch `ver3-target-aligned-model-challenger-001`.
- Relocated roots were registered below `D:\Projects\GroundLM 2026`; no active OneDrive path was accessed. No path repair was required during this Goal.
- Frozen parent: tag `v2.3.1-mc33`, commit `0b2b2a252f86758a38945396e85578c3026c594d`, prediction SHA-256 `98C9AF8031E0CA4CEF59E21554A53660740C2C28703DE5E49D725A0E5A0A6597`.
- Immutable source hashes: mechanical benchmark `D4B876A9E3C4FD9585864C9F69EDB4B978DAA793490A3CE093A789702F92048E`; fact ledger `7B4AE67394987FA7760FA762365B7828E430FFCA23E438317997C442DD633DEE`; Docling index `D488354DCB102740BDB1C6798063D144078E7AB59B910529DC22C1E5D41257B1`; table archive `A83373FE99F638CD0E5F15FD3E18309F0A671BA715ED1B6AAE8EB366EEAE347A`.
- Official gold, official evaluator, parent answers, and query-specific held-out analysis were never used. Input-only questions were used only for aggregate target-shape audit.

## Development benchmark and retrieval

The old benchmark was 99.65% table-coordinate/direct lookup, whereas aggregate held-out inputs include MC/Freeform, text/citation/figure/equation evidence and compositional signals. This explains the old zero-selection router without inspecting answers.

The locked `VER3_TARGET_ALIGNED_DEV_BENCHMARK_001__r01` has 13,822 deterministic source-derived records: 9,803 MC and 4,019 Freeform; 12,909 structured lookup, 349 direct extraction, 167 comparison, 196 negation/EXCEPT, and 201 multi-object count. Its split counts are train 8,152, calibration 1,854, and holdout 3,816; IDs and split hashes are frozen in its external manifest.

The valid `retrieval_gate_001__r03` result passed: answer-bearing recall `0.982998118940819`; comparison `0.9940119760479041`; direct extraction `0.7879656160458453`; multi-object count `1.0`; negation/EXCEPT `1.0`; structured lookup `0.9876055465179332`. Failed/empty earlier gate directories remain preserved. Retrieval is internal-only and did not modify evaluator-facing Evidence.

## Provider and model channels

The central resolver imported, resolved a non-secret status from the configured user secret file, passed authenticated `/models`, and passed a tiny grounded structured extraction. No credential value, header, or raw secret was logged or committed.

Architecture 1 implemented paper-constrained bundles, strict two-record JSON batches, deterministic cited-value grounding, and calibration-before-holdout threshold selection. Its remote table-heavy and small direct workload requests both produced external-only `APITimeoutError` status without a raw model answer. Its local smaller/faster fallback (1,024-token context, 64 then 128 output, `think: false`) loaded and returned but produced empty or malformed final JSON, so no response passed strict parsing or grounding.

Architecture 2 changed the design to model-assisted family decomposition followed by deterministic extraction/computation. Its strict-schema local decomposition smoke likewise returned no parseable final JSON. Without a valid structured intermediate, Architecture 2 cannot be scored or routed. Raw provider/local replies remain only in collision-safe external model-output directories.

## Gates, router, and evaluation

No architecture obtained an accepted proposal; therefore neither architecture can meet the 100-proposal, two-family, three-operator, >=0.85 selective-exact-match development gate. No calibrated threshold or router was frozen, no held-out proposal run occurred, and the candidate gate was never eligible.

Accordingly: Paper, evaluator-facing Evidence, Table answers, and every unrelated parent field remain unchanged; no prediction, candidate SHA, official metric, evaluator invocation, clean-clone replay, release archive, tag, push, merge, or publication exists for this Goal.

## Commits and next direction

Current-program commits are `efdad45` (target-aligned benchmark/retrieval) and `95be165` (Architecture 1 diagnostic infrastructure and earlier checkpoint). This terminal report corrects the earlier premature timeout interpretation. A pre-existing local `origin` points to the source-native workspace; it was not fetched, pushed, or changed.

Before resuming, provide a text model endpoint that completes a two-record structured evidence request and returns schema-valid JSON within a bounded timeout. Resume with a fresh collision-safe Architecture 2 run, retain this benchmark/retrieval lock, and do not create or evaluate a candidate until an architecture clears its frozen synthetic development gate.
