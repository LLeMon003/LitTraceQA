# Fresh versus cached-exact divergence

This comparison is diagnostic and post-freeze. Neither prediction was modified. Official gold was used only to classify correctness after both predictions were frozen.

## Retrieval and evidence

Paper selection is exactly reproduced: 55/55 fresh records have the same selected-paper sets as cached-exact, and both obtain Paper P/R/F1 1.0.

Evidence sets are identical on 31/55 queries and differ on 24. Mean evaluator-key Jaccard is 0.73593. Against gold, fresh evidence has 53 TP, 82 FP, and 75 FN; cached-exact has 63 TP, 45 FP, and 65 FN. Four differing queries are page-level mismatches with the same paper/source family; no pure object-ID-only mismatch was found. The fresh result overproduces equation/algorithm and text/figure evidence relative to the cached checkpoint while yielding lower precision and recall.

Bounded audit examples: `q_021`, `q_022`, `q_023`, `q_028`, `q_029`.

## Multiple choice

There are 41 MC queries. Nonempty letters progress from 23 after option-aware base MC, to 28 after source-grounded MC, to 30 after typed MC; cached-exact has 41. Correct answers progress 19, 25, and 27, versus cached-exact 38. Fresh and cached letters differ on 13 queries; 11 fresh answers remain blank.

Source-grounded decisions: 33 incomplete preserves, 1 supported preserve, 1 no-unique-option preserve, and 6 replacements. Typed decisions: 34 operator abstentions, 4 unsupported abstentions, 1 incomplete abstention, and 2 changes. The bounded audit identifies one explicit option-mapping failure and four semantic-support failures.

Bounded audit examples: `q_001`, `q_006`, `q_010`, `q_017`, `q_031`.

## Freeform and tables

All 26 fresh freeform fields are nonempty. Fresh obtains 16/26 exact matches versus cached-exact 23/26; nine normalized outputs differ from cached, one correct output differs only in formatting, and assembly introduced zero renderer effects.

All 11 repaired tables match the official schema. Four row sets and seven cell-metric outcomes differ from cached-exact. No table rows changed from fresh raw output to the reassembled final; the repair added the missing schema to 11 tables without changing row values. Fresh has three fully correct tables versus ten cached-exact.

Bounded freeform examples: `q_001`, `q_010`, `q_017`, `q_020`, `q_021`. Table row-set differences: `q_022`, `q_025`, `q_029`, `q_056`.

## Earliest divergence

Across all 55 queries, the earliest primary category is:

- fresh API semantic/evidence generation: 30;
- table extraction: 4;
- base MC: 2;
- no fresh-versus-cached divergence: 19.

No query first diverges at paper retrieval, freeform serialization, final serialization, or the repaired table schema. Cached-correct-to-fresh-incorrect metric contributors are 0 paper queries, 3 evidence exact-set queries, 11 MC queries, 7 freeform queries, and 7 fully-correct-table queries.

The complete query-level taxonomy is in `records/FRESH_FAILURE_TAXONOMY.jsonl`; aggregate evidence and bounded examples are in `records/FRESH_VS_CACHED_DIVERGENCE.json`.
