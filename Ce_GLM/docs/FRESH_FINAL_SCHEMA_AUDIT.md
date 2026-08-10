# Fresh final schema audit

Created: 2026-07-18T01:57:13.604712+00:00

## Evaluator input finding

The completed evaluator run was pointed at `..\littraceqa_baseline_Ver.2\inputs\validation_inputs.jsonl` as `--gold`. That file is an input/question file, not a gold/evaluator-contract file: it has no `answer`, no `gold_papers`/`papers`, and no `evidence`. The resulting Phase 7 metrics are therefore structurally invalid as fresh-generation performance.

## Aggregate schema summary

### original_input

- Records/unique IDs: 55 / 55
- SHA-256: `8A16713478FACB0D54D28B4959A02B37FE584B8665AEEF245083A8329BFAC858`
- Papers present/nonempty: 0 / 0
- Evidence present/nonempty: 0 / 0
- MC letters: 0
- Freeform answers: 55
- Table answers present/nonempty: 0 / 0
- Table rows/cells: 0 / 0

### option_aware_input

- Records/unique IDs: 55 / 55
- SHA-256: `322078587C34F2283B10229CCB20CC897890825C21F0515DB9872D94504B2F81`
- Papers present/nonempty: 0 / 0
- Evidence present/nonempty: 0 / 0
- MC letters: 0
- Freeform answers: 55
- Table answers present/nonempty: 0 / 0
- Table rows/cells: 0 / 0

### fresh_prediction

- Records/unique IDs: 55 / 55
- SHA-256: `4791DD77091EA3A581E94019C1C77A962E64E729D65C53F29C6FDE65096E68BF`
- Papers present/nonempty: 55 / 55
- Evidence present/nonempty: 55 / 55
- MC letters: 30
- Freeform answers: 55
- Table answers present/nonempty: 11 / 11
- Table rows/cells: 44 / 74

### cached_exact_dev

- Records/unique IDs: 55 / 55
- SHA-256: `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`
- Papers present/nonempty: 55 / 55
- Evidence present/nonempty: 55 / 55
- MC letters: 41
- Freeform answers: 55
- Table answers present/nonempty: 11 / 11
- Table rows/cells: 45 / 76
