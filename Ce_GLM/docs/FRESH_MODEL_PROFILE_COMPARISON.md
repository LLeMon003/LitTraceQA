# Fresh model-profile comparison

Status: `AUTHORITATIVE_FRESH_PROFILE_SELECTION`

This document selects a single recommended model profile for a new
blank-runtime-cache execution. It uses only aggregate, post-freeze evaluator
metrics from completed fresh runs. It does not put gold values, evaluator
output, query identifiers, historical predictions, or per-record decisions
into generation code, prompts, retrieval, or runtime configuration.

## Comparable complete fresh runs

Both rows below were generated from absent runtime roots with 55 raw records,
no historical cache or prediction input, no gold/evaluator input during
generation, and one hash-locked post-freeze evaluation. The model choice was
explicitly fixed for the V3.2 run. The V4 result is the previously verified
Version 2-compatible blank-cache profile.

| Profile | Prediction SHA-256 | Paper F1 | Evidence F1 | MC | Freeform | Table row F1 | Table-cell macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FRESH_AUTHORITATIVE_V4` (`DeepSeek-V4-Flash`) | `F1148B43EB0D2FB4A7EE1E6453FBB46089BDC7FAA47E8DA9C8744E5E2A94DF8E` | 1.000000 | 0.537994 | 28/41 | 19/26 | 0.865657 | 0.588384 |
| V3.2 C1 experimental (`DeepSeek-V3.2`) | `AFD88C3E187A86B2584319BDCF96195FB68E8529ADBCAEE6FD1EF2D59C3259D1` | 1.000000 | 0.581025 | 26/41 | 17/26 | 0.850649 | 0.515152 |
| V4 Pro fresh trial (`DeepSeek-V4-Pro`) | `D6E51C83B8262CCB76C1AD46612555A6479A6E84F38B1624B555B42BE7CF3A35` | 1.000000 | 0.561227 | 27/41 | 18/26 | 0.524747 | 0.452020 |

## Selection

`FRESH_AUTHORITATIVE_V4` is the recommended fresh profile. It wins the
user-prioritized MC and Freeform measures by two records each, and improves
both Table measures. V3.2 improves Evidence F1 by `0.043030`, but that single
gain does not outweigh the primary-family and table regressions.

The synthetic-only V4 Pro qualification passed its JSON and grounding smoke,
and its subsequent full fresh run completed the same zero-cache → freeze →
hash-verified replay → one-evaluation chain. Its one permitted aggregate
evaluation is nevertheless a regression on MC, Freeform, Table row, and
Table-cell macro versus V4 Flash. It is retained as a reproducible negative
fresh comparison, not a retry target and not the recommended profile.

The exact fresh command is documented in
[`FRESH_ZERO_CACHE_REPRODUCTION.md`](FRESH_ZERO_CACHE_REPRODUCTION.md). It
must use an absent runtime root, `--text-model deepseek-ai/DeepSeek-V4-Flash`,
hash verification, no resume directory, then freeze/replay and one separate
hash-locked evaluation.

## Honest upper-bound context

The sealed cache-exact diagnostic remains separately classified
`VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`: Evidence F1
`0.634646`, MC `38/41`, Freeform `23/26`, and Table-cell macro `0.954545`.
It is useful as a diagnostic ceiling, but is not an input to this model
selection and cannot be represented as a fresh provider result.

The V4 profile narrows none of those gaps by cache substitution; it is simply
the strongest verified fresh option currently available. A new provider run is
not guaranteed to recreate its SHA or metrics, even with temperature zero.

## Implications for further work

- Do not inject diagnostic predictions, gold, evaluator output, or
  query-specific logic into the fresh route.
- Future source-only improvements must clear provenance and fresh-process
  gates before a new full 55-record evaluation.
- A proposal that cannot demonstrate source grounding must preserve the parent
  field rather than fabricate an answer, citation, or table cell.
