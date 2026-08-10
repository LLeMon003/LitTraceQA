# Fresh zero-cache reproduction result

Status: `AUTHORITATIVE_FRESH_GENERATION_WITH_REPAIRED_DETERMINISTIC_CONTINUATION`

This report records the first completed empty-runtime-cache generation in this
repository. It is not a cache-exact replay and must not be compared as though
it were one.

## Provenance and boundaries

- Canonical raw generation began in the absent runtime root
  `VER3_FRESH_ZERO_CACHE_REPRODUCTION_003`, with `--mode raw-fresh`, hash
  verification, and no `--resume-run`.
- All 55 raw records were newly generated. No historical prediction, sealed
  cache, raw response, or gold value was an input to generation.
- Central credentials remained process-local. No credential, raw provider
  response, or evaluator/gold content was written to Git.
- The official evaluator ran exactly once after a separate hash lock and a
  contract-only check passed.

## Bounded infrastructure and repair record

| Attempt | Outcome | Evaluation use |
| --- | --- | --- |
| `001` | Noncanonical heartbeat-recovery run; preserved and excluded. | None |
| `002` | Stalled safely at 26/55 after the 1,200-second bound. | None |
| `003` | Generated 55/55. A missing active packaged entry point blocked only deterministic downstream processing. | One locked evaluation after repair |

The missing entry point was `scripts/run_source_grounded_generic.py`. It was
restored as the generic, source-root-driven implementation. The repair was
tested before use. The repaired source-grounded and typed-MC stages consumed
only the current `003` outputs, used neither `--evaluate` nor a gold argument,
and made zero evaluator calls. This is therefore fresh model generation with a
current-workspace deterministic continuation repair—not a historical replay.

## Locked evaluation evidence

- Fresh prediction SHA-256:
  `F1148B43EB0D2FB4A7EE1E6453FBB46089BDC7FAA47E8DA9C8744E5E2A94DF8E`
- Evaluation lock and result: external-only under
  `littraceqa_runtime/VER3_FRESH_ZERO_CACHE_REPRODUCTION_003/`.
- Lock contract: 55 unique prediction IDs, evaluator ID alignment, required
  answer fields, and immutable prediction/gold/evaluator hashes all verified.
- Result envelope: `AUTHORITATIVE`, invocation count `1`, inputs unchanged,
  empty evaluator stderr.

## Authoritative fresh metrics

| Metric | Fresh result |
| --- | ---: |
| Paper F1 | 1.000000 |
| Evidence F1 | 0.537994 |
| Multiple-choice accuracy | 0.682927 (28/41) |
| Freeform exact match | 0.730769 (19/26) |
| Table row F1 | 0.865657 |
| Table-cell macro | 0.588384 |
| Table-cell micro | 0.555556 |

For context only, the sealed cache-exact diagnostic reports Evidence F1
`0.634646`, MC `38/41`, Freeform `23/26`, and Table-cell macro `0.954545`.
Those are not reproduced by this fresh run and must remain labelled
`VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`.

## Validation and repository state

- Focused checks: `6 passed, 1 skipped`; the skipped historical test requires
  an intentionally external official-lock fixture and now skips rather than
  breaking fresh-clone collection.
- `git diff --check` passed.
- Raw model responses, secrets, official gold, and evaluator output remain
  outside Git.

## Conclusion

The GitHub repository can now execute a true blank-runtime-cache model run
using the centralized resolver and explicit hash-locked external inputs. It
cannot honestly promise the historical cache-exact score from a new provider
generation. The metrics above are the valid reproducible fresh result for this
completed run.
