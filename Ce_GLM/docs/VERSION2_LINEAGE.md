# Version 2 lineage and cache boundary

## Decision

`littraceqa_baseline_uq_experiments/outputs/modular_composed_label_locator_predictions.jsonl` is classified as a **VERIFIED_CACHE_BOUNDARY**.

Its SHA-256 is `54DA46600AFE81DAB5D8D2F10E87AC453FF5FCA206296DAF126EFFCD4D4C409D`. It is strict UTF-8 and valid one-object-per-line JSONL with 55 records, 55 unique query IDs, no duplicates, and one stable top-level schema: `answer`, `evidence`, `gold_papers`, and `query_id`.

## Final bounded producer investigation

The archived `selective_uq_rag_visual` raw, run, and decision logs were streamed locally. They are valid JSONL and cover 55 unique query IDs. None contains the cache artifact path, its hash, an `output_prediction` field, or a command that produces it. Strategy A had already reached the same result across ten compact manifests.

The historical producer edge is therefore absent, not verified. Raw-input exact lineage remains incomplete. This limitation prevents a Level 2 exact raw-lineage claim unless the missing producer record is recovered.

## Downstream policy

Cached-exact reproduction may start from this boundary and must execute every accepted downstream stage. It may not start from the final DEV target, any downstream checkpoint, official gold, or evaluator output. Every stage must record its input/output hashes and execution classification.

The verified downstream order is:

`VERIFIED_CACHE_BOUNDARY` → table repair ancestry → evidence ancestry → freeform completeness chain → option-aware MC → evidence-safe cleanup → source-grounded MC → typed MC → final prediction.

The detailed machine-readable decision and graph are in `records/CACHE_BOUNDARY_DECISION.json` and `records/VERSION2_LINEAGE_GRAPH.json`.
