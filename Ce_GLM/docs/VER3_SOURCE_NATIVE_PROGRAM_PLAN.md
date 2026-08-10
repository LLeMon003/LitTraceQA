# VER3 Source-Native Challenger: locked plan

## Scope and data policy

`LitTraceQA/data/validation.jsonl` is official gold despite its public-development label in the task README. It is forbidden for this program. The frozen 55-record production prediction, official evaluator, official gold, Version 2.3 repositories, evaluated candidates, and completed Ver3 workspaces are immutable and excluded from development.

The only permitted development signal is a new, synthetic benchmark whose labels are deterministically recovered from hash-verified, paper-scoped source structures. Synthetic records must not contain a held-out query ID, answer, gold-paper set, evidence locator, or prediction field. A source record may yield at most one synthetic task. Stable synthetic IDs are assigned from the source object UID and task recipe. The benchmark split is locked by SHA-256 of that ID: 70% train (`00`–`B2`), 15% calibration (`B3`–`D8`), and 15% internal holdout (`D9`–`FF`). The resulting ID lists and their SHA-256 values are frozen before tuning.

## Paths and source acquisition

All current roots are relative to the relocated project root: code workspace, `littraceqa_experiment_artifacts/VER3_SOURCE_NATIVE_CHALLENGER_001`, `littraceqa_runtime/VER3_SOURCE_NATIVE_CHALLENGER_001`, `littraceqa_source_native_assets/VER3_SOURCE_NATIVE_CHALLENGER_001`, `littraceqa_development_outputs/VER3_SOURCE_NATIVE_CHALLENGER_001`, and `littraceqa_release_candidates/Ver3/VER3_SOURCE_NATIVE_CHALLENGER_001`.

Assets are resolved by manifest identity, expected filename, producer stage, size, and SHA-256 under the relocated root. Historical OneDrive strings remain unchanged as audit fields and are never opened. Existing Docling objects, the table-cell provenance ledger, and figure/equation facts are preferred. New paper-native files may be acquired only for a paper already permitted by its metadata; every acquisition is external-only and must have a manifest with origin, identifier, path, size, SHA-256, parser version, policy status, and reproduction role.

## Architecture and cycles

The common pipeline is answer-shape analysis → paper-constrained object retrieval → provenance bundle → family solver → deterministic completeness/contradiction checks → calibrated confidence → locked parent/challenger router. Parent paper and evaluator-facing Evidence stay frozen for any candidate. The first candidate may change answers only.

1. **Cycle 1:** deterministic structured solver over provenance-complete table cells and figure/equation facts; it performs exact lookup, numeric comparison, count, membership, and unit-aware normalization.
2. **Cycle 2:** hybrid lexical plus source-graph retrieval, with constrained extraction from a bounded evidence bundle.
3. **Cycle 3:** model-assisted extraction only when every proposed value is deterministically grounded to source objects and survives the same validator.

An evidence graph is JSON-round-trippable and deterministically serialized. Every node and edge records its source UID, source hash, type, location, and ambiguity status. LLM-suggested relations are not accepted without deterministic validation.

## Locked calibration, routing, and gates

Confidence is calibrated only on the synthetic calibration split. The router may select a challenger answer only when its family has calibrated confidence at least 0.90, complete required evidence, a unique validated answer, and zero unresolved contradictions; otherwise it retains the frozen parent. It never reads parent correctness, official query IDs, official answers, or evaluator feedback.

Before internal-holdout scoring, Cycle 1 must have at least 80 synthetic records across at least two answer recipes, answer exact-match at least 0.85, provenance validity 1.00, calibration ECE at most 0.10, and no recipe below 0.75. Cycle 2 must improve exact match by at least 0.03 and four records over Cycle 1, preserve provenance validity 1.00, and not regress a recipe by more than 0.02. Cycle 3 has the same improvement/non-regression requirements over the best earlier passing cycle, plus 1.00 deterministic grounding. Any cycle with fewer than 80 records or fewer than two recipes fails its gate. Gates will not be lowered.

Only a passed cycle may produce one 55-record candidate, exactly once, from the frozen prediction. That candidate must retain Paper and evaluator-facing Evidence, preserve unrelated fields byte-for-byte, record every routing choice, and be frozen and hashed before a single official evaluation. No candidate is created merely from an evidence-graph or retrieval result.

## Stop conditions

Stop with `BLOCKED_NO_PERMITTED_DEVELOPMENT_SIGNAL` if no credible mechanically recoverable benchmark can be formed. Stop with `SOURCE_NATIVE_CHALLENGER_EXHAUSTED` after all three architecture cycles fail their locked gates. Other permitted terminal states are immutable-input/policy violation, genuinely unavailable essential assets after bounded relocation, or a non-recoverable external blocker. No candidate has been created or evaluated at this stage.
