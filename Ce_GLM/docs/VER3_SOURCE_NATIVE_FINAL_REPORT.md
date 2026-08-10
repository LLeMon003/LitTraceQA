# VER3 Source-Native Challenger final report

## Result

Classification: `SOURCE_NATIVE_CHALLENGER_EXHAUSTED`.

No evaluator-facing candidate was created, frozen, evaluated, or replayed. Official evaluator invocations: zero.

## Workspace, inputs, and policy

The active code workspace and all current artifacts were created under the relocated `D:\Projects\GroundLM 2026` root. Historical OneDrive strings in the registry remain audit metadata; no active OneDrive path was opened or used. The workspace began at frozen production parent tag `v2.3.1-mc33`, commit `0b2b2a252f86758a38945396e85578c3026c594d`; the supplied frozen-parent prediction SHA is `98C9AF8031E0CA4CEF59E21554A53660740C2C28703DE5E49D725A0E5A0A6597`.

The apparent public development file was verified to be the same 55-record official gold and was excluded. A mechanically recoverable benchmark was instead built from source structures only: 13,644 records, locked split IDs, SHA `D4B876A9E3C4FD9585864C9F69EDB4B978DAA793490A3CE093A789702F92048E`. It contains no held-out query IDs or answers.

Verified immutable infrastructure: table-ledger archive `A83373FE99F638CD0E5F15FD3E18309F0A671BA715ED1B6AAE8EB366EEAE347A`; facts ledger `7B4AE67394987FA7760FA762365B7828E430FFCA23E438317997C442DD633DEE`; Docling index `D488354DCB102740BDB1C6798063D144078E7AB59B910529DC22C1E5D41257B1`.

## Architectures and gates

1. Cycle 1 deterministic structured source solver: failed its fixed development gate at 0.7913669065 holdout exact match; duplicate records were fail-closed.
2. Cycle 2 hybrid provenance-coalescing retrieval: passed synthetic mechanics at 1.0 exact match, but its frozen router selected 0/55 input-only held-out records. Its audit SHA is `93305C4BE2AA4FC41B32DB3C042636FFC6DC3A121D9E3926E1C1BCF0C08FEF8E`; all records retained the parent.
3. Cycle 3 local-model constrained extractor: the configured remote credential was unavailable. The local `qwen3-vl:4b` alternative timed out twice (180 and 90 seconds) before returning an output, so no response could pass deterministic source grounding.

The router never used parent correctness, gold answers, evaluator feedback, or manual per-record choices. Parent Paper and evaluator-facing Evidence were never modified. API/model usage was zero successful calls; local-model attempts were two failed timeouts. No proposal was accepted.

## Reproducibility and release

Focused tests passed before Cycle 2 and all source changes are checkpointed in `de0213e` and `ec4fe09`; no remote is configured or used. Runtime/development artifacts remain outside Git. A final clean code-only archive and manifest accompany this report. Because no candidate exists, unrelated-field preservation, candidate SHA, official metrics, and clean-clone prediction replay are not applicable.

Recommended next direction: obtain a centrally resolved model credential or a faster capable local model, then reopen only Cycle 3 with the same frozen benchmark, router policy, parent-paper firewall, and one-record bounded smoke before any candidate decision.
