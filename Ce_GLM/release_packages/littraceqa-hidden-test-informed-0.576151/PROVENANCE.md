# Provenance and evaluation boundary

This release freezes the exact LitTraceQA `test` prediction that received an organizer-reported composite score of `0.576151` on 2026-08-07. The prediction SHA-256 is `844C6248E8A353783BE7600050BBB247A7716931B24C3C2C10FD173B04CE6914`.

The artifact is a fail-closed composition of five frozen prediction files:

- serialization parent: `B_blank_invalid_recovery_001`;
- multiple choice: `MC40_round5_verified`;
- paper selection: `P70_round8_minimal_document_identity`;
- evidence: `E73_round8_minimal_exact`;
- table answer: `TABLE20_round8_strict_verified`.

Only the designated component fields are copied. All other fields remain byte-semantically inherited from the B parent, and evidence outside the selected paper set is rejected.

## Disclosure

This is a hidden-test-informed, manually reviewed silver artifact. It is not official gold, not an untouched blind-test result, and not evidence of out-of-sample generalization. The organizer exposed only aggregate metrics, not per-query gold answers. The included prediction is suitable for exact submission reproduction and for explicitly test-informed research, but it must remain isolated from claims or experiments that require an unseen test set.

## Dataset boundary

The LitTraceQA files are pinned to Hugging Face revision `bd35dc14cf0483e0ffa51fa2a54d2689c13f9845`. Test inputs, paper metadata, PDFs, caches, API responses, and organizer-internal data are not committed. The optional downloader retrieves only the public files needed for local schema validation and verifies their SHA-256 hashes before use.

LitTraceQA annotations and benchmark files are identified by the dataset authors as CC BY-NC 4.0. Consult the upstream repository for the authoritative terms.
