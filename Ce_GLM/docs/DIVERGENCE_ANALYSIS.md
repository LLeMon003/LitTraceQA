# Divergence analysis

## Historical producer edge

The producer of `modular_composed_label_locator_predictions.jsonl` remains unavailable. Strategy A found only consumer manifests. Strategy B found no artifact path/hash or producing command in the archived `selective_uq_rag_visual` raw, run, or decision logs. The artifact itself passes all integrity checks and is classified `VERIFIED_CACHE_BOUNDARY`.

## Completeness hash transcription

The initial handoff contained a 65-character transcription with `...C2E6A...`. The accepted renderer chain generated the valid archived 64-hex SHA `C3B51991E37EC1DC2E778E3FC41C2EA41D6D96DF4793C24A19F447D966915B0C`. Version 2.3 records and constants were corrected; no immutable source was edited.

## Earliest deterministic divergence

The first Mode A option-aware execution used the later full-Docling object index and differed on exactly one field:

- Query: q_041
- Field: `answer.multiple_choice.gold`
- Archived confidence: `0.778923`, ineligible/abstain
- Full-Docling confidence: `0.837077`, eligible

All other 54 records and all non-MC fields were identical. A q_041-only diagnostic proved that both the base index SHA `222C63...9DE6` and v2 index reproduce the archived score, while the full-Docling index does not. The accepted base index was selected, after which Mode A reproduced the final target byte-for-byte.

## Source-grounded wrapper note

The current source-grounded module has a modification timestamp after its archived replay output. A generic wrapper is functional for fresh parents but does not reconstruct the historical source-grounded SHA from current source corpora. Cached-exact mode therefore records the archived source-grounded checkpoint as an explicit `CACHE_HIT` and does not claim that stage was freshly executed.

## Fresh API branch

The fresh run ultimately completed through resumable execution: 30 verified raw records were preserved and 25 missing records were generated, avoiding 30 repeated calls. The final evaluator-compatible prediction is frozen at SHA-256 `A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281`.

The authoritative gold evaluation yields Paper F1 1.0; Evidence F1 0.581560; MC 27/41; Freeform 16/26; Table row F1 0.875397; and Table cells 14/27 micro. These are materially below cached-exact DEV except for paper retrieval.

Paper sets agree on 55/55 records. Evidence differs on 24, MC letters on 13, normalized freeform on 9, and table row sets on 4. Earliest primary divergences are fresh API semantic/evidence generation for 30 queries, table extraction for 4, base MC for 2, and none for 19. The dominant reproducibility bottleneck is fresh semantic/evidence generation plus downstream MC abstention, not retrieval or final serialization.

The question-input-as-gold evaluation is `INVALID`; the cached-prediction-as-gold evaluation is `DIAGNOSTIC_ONLY`; the hash-locked official-gold result is `AUTHORITATIVE`. Full aggregates and query-level taxonomy are in `records/FRESH_VS_CACHED_DIVERGENCE.json` and `records/FRESH_FAILURE_TAXONOMY.jsonl`.
