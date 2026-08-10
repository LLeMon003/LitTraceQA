# VER3 Target-Aligned Model Challenger: locked plan

## Boundaries

The frozen parent prediction, official gold, evaluator, Version 2.3 repositories, completed experiments, and historical artifacts are immutable. Generation may read only input-only held-out questions as aggregate distribution observations and may use the frozen parent only as a fallback after a candidate gate. Parent Paper, evaluator-facing Evidence, and all Table answers are fixed. Only router-selected MC or Freeform fields could ever change.

## Development protocol

`VER3_TARGET_ALIGNED_DEV_BENCHMARK_001` will be mechanically labeled from source-bearing tables, facts, and document objects. Each record has source hashes, object IDs, template, answer family/shape, operator, verification trace, difficulty features, split, and record hash. Split assignment is deterministic, grouped by paper and source-object family to prevent source leakage: train 70%, calibration 15%, holdout 15% by stable SHA-256 group bucket. Split IDs and hashes are frozen before model scoring.

The benchmark must cover MC and Freeform; numeric and textual answers; direct/structured extraction; comparison; negation or EXCEPT; and multi-object reasoning. Official answers, parent correctness, evaluator feedback, and manually assigned expected answers are excluded.

## Retrieval and model architectures

Retrieval first reuses the hash-verified Docling/fact/table sources, producing paper-constrained provenance bundles. It must reach answer-bearing recall >= 0.85 overall, no principal benchmark family below 0.70, and complete provenance for answer-bearing objects. At most two material refinements follow a failure.

Architecture 1: structured bundle -> constrained JSON extraction -> deterministic value/span grounding -> contradiction/type validation -> confidence features.

Architecture 2, only after Architecture 1 failure: family-specific decomposed bundles -> constrained intermediate operation -> deterministic extraction/computation -> grounding/contradiction validation -> confidence features.

The centralized resolver and remote structured text model are mandatory for primary model calls. Calls are batched one to two; raw responses are external-only. Model timeout/transport failures receive at most two bounded repairs per defect.

## Calibration and gates

Calibration uses only the calibration split. The frozen router threshold is chosen before holdout scoring and requires confidence >= 0.90, complete evidence, unique validated answer, zero unresolved contradiction, and family-level calibration support.

Each architecture must have at least 100 accepted holdout proposals across two answer families and three operators, selective exact-match >= 0.85, 100% grounding, zero unsupported accepted answers, reproducible output, and no catastrophic family regression. These thresholds will not be weakened.

## Candidate and stop gates

Only a passing architecture may receive one input-only held-out run. A candidate requires >=3 selected, fully grounded MC/Freeform proposals, exact preservation of Paper/Evidence/Table/unrelated fields, and exactly 55 unique records. It is frozen and hashed before one evaluator call. Fewer than three selections stops without evaluation.

Terminal outcomes are one evaluated/replayed candidate, both architectures failing their development gates, all model channels unavailable after repairs, no compliant benchmark, immutable-input/policy violation, or genuine non-recoverable blocker. No research outcome is implied by this planning document.
