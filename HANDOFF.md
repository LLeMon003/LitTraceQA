# LitTraceQA Handoff

## Purpose

Improve the post-selection evidence-to-answer path without reading official answers during inference. The target design is a grounded four-level hierarchy:

- **L0**: immutable parser records with official `source_type`, locator, text, structured table data, and optional figure crop.
- **L1**: bounded local windows that retain L0 references, heading path, nearby explanatory text, and object-specific context.
- **L2**: contextual triples/evidence cards. A factual relation is usable only when it carries exact L0 quotes or verified figure-crop evidence.
- **L3**: navigation metadata, paper/section summaries, entity-to-triple edges, candidate paths, and unresolved claims. L3 is never proof.

The immediate objective is to reduce generation-context noise while preserving a reverse path from every answer fact to L0. Frozen Qwen rerank results should be reused; do not rerun Qwen as a default path.

## Workspace And Runtime

- Workspace: `/mnt/d/codex_proj/workspace/littraceqa`
- Conda Python: `/home/lemon/miniconda3/envs/littraceqa/bin/python`
- Main package: `pdf_docling_rerank_selection_generation_pipeline`
- Active configuration: `pdf_docling_rerank_selection_generation_pipeline/.env`
- Do not expose credentials from `.env` in logs, replies, patches, or documentation.

## Model Decision

The user approved the two-model plan. The current operational configuration uses the approved Qwen VL model for all L2 calls because it handled structured scientific tables reliably within the short timeout:

- `Qwen/Qwen3-VL-8B-Instruct`: textual triple extraction, visual crop triples, and sufficiency gate.
- `deepseek-ai/DeepSeek-V4-Flash`: remains appropriate for cheap text-only work such as HyDE, but complex table-to-JSON extraction timed out in small online tests and is not used for L2 triples.

This is not an approval to rerun the expensive `Qwen/Qwen3-VL-Reranker-8B`. Its existing scores remain the default quality path.

## Completed Work

### Grounded contextual triples

Implemented in `pdf_docling_rerank_selection_generation_pipeline/contextual_triples.py`:

- `l1_evidence_windows` with source-aware table, figure, equation, and citation context.
- `l2_contextual_triples` with `support_refs`, exact `support_quotes`, `l1_window_ids`, and card IDs.
- Deterministic high-precision table comparison relation extraction for the observed `Ours` vs baseline/table condition pattern.
- LLM relation verification. It only accepts a relation if its cited quote occurs exactly in L0 and its entities, values, and qualifiers are grounded.
- Fixed an important bug: LLM refinement no longer overwrites a deterministic relation emitted from the same L1 window.
- Separate visual crop extraction and visual verification. Image data is never mixed into text triple batches.
- L3 triple navigation graph with multiple candidate paths retained rather than a single early path.

### Sufficiency and prompt safety

- A cheap structural precheck only decides whether a semantic sufficiency call is possible; it never asserts semantic sufficiency.
- The semantic sufficiency response may request bounded L1/L0 expansion by known triple IDs only.
- Expansion is capped and keeps raw provenance. L0 expansion now includes the anchor record rather than every neighbor to control prompt growth.
- Keyed prompt projection supports accepted triples and requested L1/L0 expansions through card keys, not exposed evidence references or locators.

### Tests and validation before this handoff

Earlier explicit tests passed:

```bash
/home/lemon/miniconda3/envs/littraceqa/bin/python -m unittest \
  pdf_docling_rerank_selection_generation_pipeline.test_contextual_triples \
  pdf_docling_rerank_selection_generation_pipeline.test_evidence_hierarchy \
  pdf_docling_rerank_selection_generation_pipeline.test_prompt_grounding
```

The final small navigation patch was made after the previous pass. Re-run the same command before any model call.

`py_compile` and `git diff --check` had passed before the final patch; rerun them as part of the first next-session check.

## Artifacts And Findings

### Frozen input hierarchy

`outputs/pdf_docling_reselect_qwen_union_b150_b250_b500_inputonly_hierarchy_extractive/evidence_hierarchy.jsonl`

This is the frozen hierarchy used as the source for contextual-triple experiments. It preserves the previous Qwen-selection result, but its L0 catalog is built only from selected anchors plus immediate local neighbors.

### Deterministic full run

`outputs/pdf_docling_reselect_qwen_union_b150_b250_b500_inputonly_contextual_triples_extractive/`

- 55 queries
- 1,168 L1 windows and 1,169 candidate triples
- only one high-confidence deterministic table relation
- therefore deterministic rules are a supplement, not a complete L2 solution

### LLM smoke run

`outputs/pdf_docling_contextual_triples_llm_smoke3/`

- Queries `q_001`, `q_002`, `q_003`
- The table comparison query produced a grounded LLM triple and a deterministic table triple; its sufficiency gate recognized the needed comparison.
- The hardware and bibliography questions produced no usable relation. Diagnosis showed the required raw records were absent from that run's L0 catalog, not that Qwen failed to interpret an available record.

## Current Change: Deterministic Navigation Supplement

The current unvalidated patch adds a small additive route in `evidence_hierarchy.py`:

- `build_l0_l1_l3(..., question=...)` now accepts the public question.
- `_deterministic_navigation_records()` identifies papers named by a strong title/alias match in the question.
- For a question such as `the 24th reference`, it admits only the matching `citation_context` record with the same `citation_id`.
- Otherwise it admits at most four lexical parser records per named paper, and only when actual query terms occur in the record.
- Added records receive `role="deterministic_navigation"` and `navigation_reasons`, so they remain auditable and distinguishable from frozen Qwen anchors.
- `build_evidence_hierarchy.py` now passes the public question into the hierarchy builder.

Why this exists: the frozen hierarchy omitted the raw hardware record for `q_002` and the target bibliography record for `q_003`; no later L1/L2 model can recover records that were never present in L0.

This route is intentionally not a hard prefilter and does not call the Qwen reranker. It must be evaluated separately because it changes the candidate L0 set after frozen selection.

## Current State / Blockers

1. Three-query smoke completed successfully; the next required verification is an official evaluation over a broader fixed split and then all 55 queries.
2. Existing contextual-triple artifacts cannot gain missing L0 records by calling `enrich_contextual_triples.py` alone. Rebuild the hierarchy from frozen selected contexts and processed records first.
3. The deterministic navigation supplement changes the post-selection L0 set. Measure it separately from frozen-Qwen selection and do not report it as an unchanged selector result.
4. No official full-set evaluation has been run for the navigation supplement or triple-first generation. Do not claim a benchmark improvement yet.

## Latest Smoke Result (2026-07-28)

Artifacts:

```text
outputs/pdf_docling_contextual_triples_navigation_smoke3_v4/
outputs/pdf_docling_contextual_triples_navigation_verified_reuse_smoke3_v4/
outputs/pdf_docling_contextual_triples_triplefirst_generation_smoke3/
```

- Rebuilt L0/L1 includes the exact S-RAG hardware sentence and EasySpec Reference 24 entry.
- `q_002` uses a deterministic `experiments -> uses -> a single NVIDIA RTX 4090 GPU` relation, grounded to its L0 text record.
- `q_003` uses a deterministic `Reference 24 -> cites -> Freda Shi` relation, constrained to the paper explicitly named in the question; duplicate reference number 24 from other candidate papers is excluded.
- Existing Qwen triple JSON batches were reused through `--reuse-raw-output` only after an exact L1-window fingerprint match. No triple rerank call was repeated.
- The sufficiency gate marked all three queries sufficient once trusted triples were forced into its fixed eight-triple budget.
- Triple-first keyed generation returned grounded outputs for all three smoke queries. Prompt sizes were 10,949, 5,919, and 5,812 characters respectively. The generation prompt contained no raw `E...` evidence references; outputs cite C-keys and the runtime restores L0 locators.
- `errors.jsonl` in the generation smoke contains `evidence_ref_echo_resolved` informational events, not generation failures. Treat this as a reporting-quality issue if aggregate error counts are used.

## Latest Full Evaluation (2026-07-28)

Full artifacts:

```text
outputs/pdf_docling_contextual_triples_navigation_full_v1/
outputs/pdf_docling_contextual_triples_verified_full_v1/
outputs/pdf_docling_contextual_triples_triplefirst_generation_full_v1/
outputs/docling_qwen_hyde_final_over_060_direct_triple_overlay_v1/
```

- Full hierarchy: 55 queries, 1,309 contextual L1 windows. The additive navigation route added 315 lexical and one explicit citation L0 reasons; it produced 108 L1 windows without an existing rich L2-card seed.
- Full L2 enrichment: 12 verified LLM relations, one deterministic table relation, one deterministic hardware relation, one deterministic citation relation, and 38 verified figure-crop relations. Eleven queries passed the semantic sufficiency gate and used triple-first prompt mode; the other 44 used keyed-card fallback.
- Triple-first generation alone was not a replacement for the broad context strategy: official evidence F1 was 0.3496, MC accuracy 0.0, and freeform EM 0.1923. Do not use it as a standalone final artifact.
- `assemble_direct_triple_hybrid.py` applies a conservative overlay to the existing best artifact only when the public contract is freeform-only, the gate is sufficient, at least one relation is trusted, and posthoc grounding marks the answer directly supported. It replaced exactly two predictions without reading gold data.
- The overlay's official local evaluator metrics are: paper F1 0.9573, evidence precision/recall/F1 0.4529/0.4616/0.4320, MC 0.5610, freeform EM 0.3846, table row F1 0.7002, and table macro/micro cell accuracy 0.4535/0.3704. This improves the prior best's evidence F1 (0.4139) and freeform EM (0.3462) without regressing the other reported metrics.

## Recommended Next Steps

1. Validate the source patch:

```bash
/home/lemon/miniconda3/envs/littraceqa/bin/python -m unittest \
  pdf_docling_rerank_selection_generation_pipeline.test_contextual_triples \
  pdf_docling_rerank_selection_generation_pipeline.test_evidence_hierarchy \
  pdf_docling_rerank_selection_generation_pipeline.test_prompt_grounding
/home/lemon/miniconda3/envs/littraceqa/bin/python -m py_compile \
  pdf_docling_rerank_selection_generation_pipeline/evidence_hierarchy.py \
  pdf_docling_rerank_selection_generation_pipeline/build_evidence_hierarchy.py \
  pdf_docling_rerank_selection_generation_pipeline/contextual_triples.py
git diff --check
```

2. Locate the exact frozen selected-context and candidate-paper inputs used to produce the hierarchy. The likely sources are:

```text
outputs/pdf_docling_reselect_qwen_union_b150_b250_b500_inputonly/selected_symbolic_contexts.debug.jsonl
outputs/docling_qwen_hyde_paper_selection_only/candidate_papers.jsonl
```

Verify their query IDs and candidate set match before use. Use the same Docling version as the frozen selection when possible; current parser roots include several versions under `processed_pdfs/pdf_docling_symbolic/`.

3. Rebuild only `q_001,q_002,q_003` into a fresh output, with `--mode extractive --no-contextual-triples` only if separately testing base hierarchy; normal smoke should leave contextual triples enabled. Then inspect whether L0 contains:

- the S-RAG hardware sentence for `q_002`
- citation ID 24 for `q_003`

4. Run `enrich_contextual_triples.py --mode verified_llm` on that rebuilt smoke hierarchy. It should use its JSON cache, so retries do not repeat successful model requests.

5. Only after the smoke demonstrates correct L0/L1 inclusion, implement a triple-first prompt policy:

- if the sufficiency gate is accepted, expose accepted L2 triples plus only their supporting cards and requested expansions;
- if the gate rejects or fails, retain the broader existing keyed fallback;
- add an audit measuring actual L0 references serialized into the final prompt against gold *only offline*, never in inference.

6. Run official evaluation only after the prompt audit passes. Report at least: L0 anchor retention, accepted triple count, sufficiency decisions, actual prompt anchor recall, final evidence F1, freeform EM, MC accuracy, and table metrics.

## Important Constraints And Pitfalls

- Never use official `validation.jsonl` answers or evidence at inference time. It is allowed only for offline diagnosis/evaluation.
- Do not describe an L2 triple as evidence unless its verification status is `deterministic_table_relation`, `verified_llm_relation`, or `visual_crop_verified_relation`.
- Do not allow LLM output to invent a locator. Restore locators only from L0 after generation.
- For tables, retain header path, row labels, condition rows, and footnotes with the selected values. Plain flattened table text is insufficient.
- For figures, use the actual crop only when `crop_path` exists and independently verify the visual relation. Text captions are not proof of a visual fact.
- Citation questions require special handling of `citation_id`; matching the first number found in generic prose is unsafe.
- No hard BM25/E5 prefilter before Qwen on the main quality route. Cheap lexical routing is allowed here only as an additive, explicitly audited navigation supplement.
- The repository has unrelated user modifications and untracked directories. Do not reset, checkout, or delete them.
- The root package directories are untracked, so `git diff` may not show all changed source files. Use direct file inspection and targeted tests.
