# LitTraceQA Docling-Qwen System Design

## 1. Purpose and Boundaries

This document describes the current `pdf_docling_rerank_selection_generation_pipeline` implementation and the final public-development prediction artifact. The system answers scholarly-document questions with traceable paper and evidence outputs. It is designed around the following invariants:

1. Retrieval, extraction, ranking, generation, and evaluation are separately materialized and reusable.
2. Every submitted evidence item is restored from an exact parser record; the generator never invents a page, source type, or locator.
3. L2 compression may shorten the reasoning path, but never replaces the L0 proof path.
4. Gold answers are not an input to retrieval, extraction, selection, compression, or generation. The public gold file is used only by the evaluator after predictions exist.
5. Qwen reranking is a full-candidate quality path. Sparse prefiltering is an optional acceleration experiment, not the default quality path.

The system accepts `official_dev/data/validation_inputs.jsonl`, candidate metadata, PDFs, and cached parser artifacts. It produces evaluator-compatible `predictions.jsonl` records containing `gold_papers`, `evidence`, and typed answer fields.

## 2. End-to-End Architecture

```mermaid
flowchart LR
    I[validation_inputs.jsonl] --> R[Metadata retrieval]
    R --> P[Candidate papers]
    P --> D[PDF cache and availability audit]
    D --> X[Docling transcription]
    X --> S[Symbolic standardization]
    S --> U[Record-aware units]
    U --> Q[BM25 routing plus full Qwen rerank]
    Q --> K[Coverage-aware package selector]
    K --> H[L0-L3 evidence hierarchy]
    H --> G[Keyed L2 generation]
    G --> A[Post-hoc grounding and parser normalization]
    A --> O[predictions.jsonl]
    O --> E[Official evaluator]
```

The normal online path is implemented by `run_pipeline.py`. Cached-stage utilities permit independently rebuilding L0-L3, regenerating answers, reselecting with stored Qwen scores, and combining fully evaluated predictions without rerunning upstream modules.

## 3. Input and Output Contracts

### 3.1 Public input

Each input query supplies:

- `query_id`
- `task_family`: `single_paper` or `multi_paper`
- `primary_evidence_type`
- `question`
- `answer_types`: any subset of `freeform`, `multiple_choice`, and `table`

The public input deliberately omits answer values. Sanitized answer contracts may supply MC option text and table schema without exposing gold values.

### 3.2 Canonical symbolic record

The extraction boundary normalizes every retained PDF object to a record with at least:

```json
{
  "paper_id": "acl2025_00005",
  "global_record_id": "acl2025_00005::p006_r0170",
  "page": 6,
  "record_type": "table",
  "source_type": "table",
  "label": "Table 3",
  "locator": {"page": 6, "table_id": "Table 3"},
  "text": "...",
  "section_id": "sec_005",
  "section_title": "Results",
  "document_order": 170,
  "crop_path": null
}
```

Official source types are restricted to `text_span`, `table`, `figure`, `equation_algorithm`, and `citation_context`.

### 3.3 Prediction contract

The final prediction uses:

```json
{
  "query_id": "q_001",
  "gold_papers": [{"paper_id": "acl2025_00005"}],
  "evidence": [{
    "paper_id": "acl2025_00005",
    "source_type": "table",
    "locator": {"page": 6, "table_id": "Table 3"}
  }],
  "answer": {
    "freeform": {"text": "..."},
    "multiple_choice": {"gold": "C"},
    "table": {"rows": []}
  }
}
```

The normalizer removes answer types not requested by the public contract, canonicalizes MC keys and table schema, and reconstructs evidence exclusively from allowed selected records.

## 4. Retrieval and PDF Acquisition

### 4.1 Metadata retrieval

`metadata_index.py` indexes official paper metadata and retrieves candidate papers using alias-aware BM25/hybrid matching. Query decomposition can produce subqueries; `RETRIEVAL_SUBQUERY_TOP_K` bounds their per-route contribution. Candidate counts are task-family aware:

- single-paper: `SINGLE_PAPER_TOP_K_PAPERS` (default 5)
- multi-paper: `MULTI_PAPER_TOP_K_PAPERS` (default 12)

The artifact is a flat `candidate_papers.jsonl`, sorted by rank for each query. It is a cache boundary: later stages can use `--candidate-papers-input` without repeating metadata retrieval.

### 4.2 PDF cache

`pdf_cache.py`, `pdf_downloader.py`, and `proceedings_fallbacks.py` obtain and audit candidate PDFs. `pdf_availability.jsonl` records availability and prevents extraction failures from being confused with retrieval failures. OpenReview behavior is governed by `PDF_OPENREVIEW_POLICY`; the default uses proceedings first and avoids unnecessary direct OpenReview retrieval.

## 5. Docling Extraction and Structural Standardization

### 5.1 Transcription backend

The production backend is Docling (`LITTRACEQA_TRANSCRIPTION_BACKEND=docling`). It provides document text, tables, figures, formulas, page provenance, and structured table cells. PyMuPDF remains available as a fallback/backend experiment.

Docling output is persisted per paper under:

```text
processed_pdfs/<cache>/<backend>/<paper_id>/
  raw_docling_output/
  symbolic_records.runtime.jsonl
  symbolic_records.debug.jsonl
  artifact_status.json
```

`runtime` is the compact retrieval/generation source. `debug` retains bounding boxes, standardization rules, raw backend types, and crop metadata for auditing.

### 5.2 Source typing and locators

`transcription_backends/standardizer.py` uses structural Docling types before text heuristics:

- Docling `table` becomes `table`.
- Docling `picture`/`image` becomes `figure`.
- Docling `formula`/`equation` becomes `equation_algorithm`.
- A code block is an algorithm only when it carries an `Algorithm n` label.
- Prose mentioning `Equation (n)` remains `text_span`; it is not promoted into a false object.

Label parsing creates canonical locators such as `table_id`, `figure_id`, `equation_id`, `algorithm_id`, and `citation_id`. Source type and locator are never inferred from answer text.

### 5.3 Citation contexts

Within References/Bibliography sections, numbered entries become `citation_context` records. The standardizer preserves the printed reference number and merges obvious continuation paragraphs, including cross-column/page continuations. This prevents a single bibliography entry from splitting into unrelated records and gives citation questions stable `citation_id` locators.

### 5.4 Sections and text

Only recognized short, common first-level headings are emitted as section records. Ordinary table or paragraph text cannot become a section solely because of its layout. Text paragraphs remain `text_span`, with section path and document order attached for packaging and L1 context.

### 5.5 Object crops

Figures retain Docling-exported images when available. The optional `attach_docling_table_crops.py` utility renders a table crop from Docling's page-level bbox and attaches it to matching L0 table records. It converts Docling bottom-left provenance to PyMuPDF top-left clipping coordinates and attaches a crop only after verifying that a file exists.

Equation/algorithm crops are supported by the PyMuPDF extractor. Crops are supplemental visual inputs: L0 text/structured content and official locators remain the proof source.

## 6. Full Qwen Relevance and Package Selection

### 6.1 Record-aware units

`section_relevance.py` converts canonical records into units:

- Text is packed into token-bounded chunks with configurable overlap.
- Table, figure, equation/algorithm, and citation records create atomic object units.
- Each object unit includes bounded neighboring text so captions, explanations, definitions, and local conditions are available.

The canonical representation has a versioned text/image projection and cache key. Duplicated global records, overlapping text units, and repeated crop images are deduplicated while preserving a one-to-many provenance map.

### 6.2 Routing and scoring

BM25 only builds route candidates: original query, requested modality, explicit locator, and optional multi-paper claims. Reciprocal-rank fusion forms a union. The quality path then sends every canonical unit in that union to Qwen reranking; no BM25/E5 hard prefilter is permitted by default.

Qwen text and visual projections are scored separately (`qwen_text`, `qwen_visual`). A weak layout/locator signal ranks labels and explicit object requests. The selector fuses ranks rather than linearly mixing incomparable scores.

The rerank cache key includes query, instruction version, canonical representation version, canonical text, image hash, artifact version, and model version. This permits selection experiments to reuse Qwen scores without reissuing model requests.

### 6.3 Packages and coverage-aware selection

`evidence_packages.py` and `symbolic_context_selector.py` convert scored units into packages. A package is atomic for context budgeting and may contain an object plus its necessary explanatory neighboring records. Package selection balances:

- Qwen relevance and route support
- requested source type and explicit locator match
- paper diversity for multi-paper tasks
- page diversity and duplicate suppression
- text support around objects
- character and package budgets

The selector records every choice in `section_relevance.trace.jsonl` and `selected_symbolic_contexts.debug.jsonl`. `SINGLE_PAPER_EVIDENCE_PACKAGE_BUDGET` and `MULTI_PAPER_EVIDENCE_PACKAGE_BUDGET` are upper bounds. Adaptive stop may finish earlier only after minimum coverage requirements are met.

## 7. L0-L3 Evidence Hierarchy

`build_evidence_hierarchy.py` is a post-selection stage. It does not invoke metadata retrieval, PDF extraction, or Qwen reranking and can safely run from frozen selected contexts.

| Level | Content | Permitted role |
| --- | --- | --- |
| L0 | Exact parser record, original locator, text, crop path | Sole proof source and final evidence restoration |
| L1 | Heading path, bounded neighboring records, caption/header/footnote context | Disambiguation and condition recovery |
| L2 | Extractive or verified evidence cards, proposition, entities, values, conditions, support refs | Compact factual context for generation |
| L3 | Paper metadata, section navigation, entity mentions, unresolved claims | Navigation only; never factual proof |

### 7.1 L2 cards

The extractive fallback creates query-aware propositions by selecting sentences or table fragments from L0. Verified LLM cards are allowed only when exact support quotes, entities, numbers, and conditions can be traced to L0. Tables preserve caption, headers, row labels, cells, units, and relevant footnotes rather than flattening into unsupported prose.

### 7.2 Keyed projection

For keyed generation, the prompt receives opaque keys:

- `Cxxx`: L2 card key
- `Rxxxx`: runtime record key
- `Pxx`: paper key
- `Sxxx`: section key
- `IMGxxx`: attached crop key

Raw L0 text, full locators, and evidence refs are not serialized into the keyed prompt. The runtime alone maintains `C -> R -> L0 evidence_ref`. This avoids emitting a fabricated locator while retaining a reversible proof path.

## 8. Generation, Grounding, and Normalization

### 8.1 Answer generation

`generate_from_cached_selection.py` supports legacy evidence ledgers and keyed L2 mode. The production keyed prompt contains L2 cards, micro propositions, compact navigation, answer contract, candidate paper IDs, and optional figure/table/equation crops. Prompt fitting enforces a serialized-character ceiling rather than a proxy estimate. Image slots prioritize the query's primary evidence type.

The Qwen answer client retries transport and rate-limit failures, records raw responses, and applies a narrow JSON suffix repair only for missing structural closing delimiters. It never repairs semantic content.

### 8.2 Echo and post-hoc grounding

The generator must emit `claim_to_support_keys` and, for table rows, a support key. The post-hoc gate:

1. Resolves `Cxxx` to exact L0 `evidence_ref` values.
2. Restores `paper_id`, `source_type`, and locator only from those records.
3. Rejects support keys outside the visible keyed hierarchy.
4. Rejects unsupported generated freeform facts and table values when they are absent from their resolved L0 support.
5. Normalizes MC options and table rows against the sanitized public answer contract.

The optional raw-L0 refinement runner is fail-soft: a failed verification preserves the prior prediction. It is not allowed to remove a contract-required answer type merely because a redacted contract omitted public `answer_types` metadata.

### 8.3 Provenance lock

`generation_provenance.json` hashes public inference inputs, candidate cache, hierarchy, sanitized answer contract, prompt parameters, and artifact versions. It rejects `validation.jsonl` as a generation input. Resume refuses an incompatible provenance manifest.

## 9. Evaluation and Diagnostics

### 9.1 Extraction and selection evaluation

`extraction_selection_evaluation.py` measures the pre-generation pipeline:

- `R_paper_candidate`
- `R_extraction_page`
- `R_extraction_source`
- `R_extraction_locator`
- relaxed and strict parser recall
- `R_selection_given_extracted`
- `R_selection_over_gold`

Strict matching requires paper, page, source type, and object/citation locator agreement. Failure reports distinguish absent paper, source type mismatch, page/locator mismatch, and extracted-but-not-selected evidence.

### 9.2 Official evaluator

`official_dev/scripts/evaluate.py` reports 11 public metrics:

1. paper precision/recall/F1 macro
2. evidence precision/recall/F1 macro
3. multiple-choice accuracy
4. freeform exact match
5. table row F1 macro
6. table cell accuracy macro and micro

There is no organizer-defined scalar aggregate. The final benchmark artifact additionally records an explicitly labeled unweighted mean of these 11 metrics; it is not an official metric.

## 10. Final Public-Development Artifact

The current frozen best artifact is:

```text
outputs/docling_qwen_hyde_final_over_060/
  predictions.jsonl
  evaluation_summary.json
```

Its answer/evidence fields come from the fully evaluated `ledger65k_refs` prediction. A deterministic paper-only completion is applied for multi-paper queries: if the base prediction contains fewer than five papers, candidate ranks 1 through 4 are unioned; single-paper predictions remain unchanged. The candidate cache had already achieved full gold-paper candidate recall before this completion. No answer text or gold evidence is used by the assembly.

Public-development metrics recorded in `evaluation_summary.json` are:

| Metric | Value |
| --- | ---: |
| Paper precision / recall / F1 | 0.9406 / 0.9934 / 0.9573 |
| Evidence precision / recall / F1 | 0.4347 / 0.4434 / 0.4139 |
| MC accuracy | 0.5610 |
| Freeform exact match | 0.3462 |
| Table row F1 | 0.7002 |
| Table cell macro / micro | 0.4535 / 0.3704 |
| Explicit 11-metric mean | 0.6013 |

## 11. Configuration Surface

All concrete defaults and allowed option sets are in `.env.example`; `config.py` is the runtime authority. The major groups are:

| Group | Representative settings | Role |
| --- | --- | --- |
| API/model | `ANSWER_*`, `PARSER_*`, `LLMRERANK_*` | Provider, model, timeout, retries, temperature |
| extraction | `LITTRACEQA_TRANSCRIPTION_BACKEND`, `DOCLING_DO_OCR`, `STRUCTURED_CACHE_POLICY` | Backend and artifact reuse |
| retrieval | `RETRIEVAL_*`, `SINGLE_PAPER_TOP_K_PAPERS`, `MULTI_PAPER_TOP_K_PAPERS` | Metadata candidates and decomposition |
| routing | `PAGE_ROUTING_*`, `PDF_NATIVE_TEXT_*` | Parse-page selection and expansion |
| relevance | `SECTION_RELEVANCE_*`, `LLMRERANK_*` | Unit shape, full-Qwen scoring, cache behavior |
| packages | `*_EVIDENCE_PACKAGE_*`, `MULTI_PAPER_*` | Coverage, diversity, package and character budgets |
| claims | `MULTI_PAPER_HYDE_*`, `PAPER_CONDITIONED_CLAIMS_*` | Optional routing-only multi-paper claims |
| hierarchy | `EVIDENCE_HIERARCHY_*` | L1/L2/L3 limits, cards, images, micro index, refinement |
| generation | `VLM2_CONTEXT_MODE`, `GENERATION_*` | Text/cropped-image context and resilient JSON generation |

Any experiment must write to a new output directory. A changed selection implementation may overwrite only its own current-stage artifacts; it must not silently invalidate a rerank cache from a different canonical representation version.

## 12. Operational Recipes

### 12.1 Full upstream pipeline through selection

```bash
conda activate littraceqa
python -m pdf_docling_rerank_selection_generation_pipeline \
  --official-dir official_dev \
  --output-dir outputs/<run_name> \
  --pdf-output-dir raw_pdfs \
  --processed-output-dir processed_pdfs/<cache_name> \
  --extract-all-pages \
  --enable-figure-crops \
  --enable-table-crops \
  --enable-equation-crops \
  --skip-generation \
  --show-progress \
  --env-path pdf_docling_rerank_selection_generation_pipeline/.env
```

### 12.2 Build L0-L3 from frozen selection

```bash
python -m pdf_docling_rerank_selection_generation_pipeline.build_evidence_hierarchy \
  --official-dir official_dev \
  --selected-contexts-input outputs/<selection>/selected_symbolic_contexts.debug.jsonl \
  --candidate-papers-input outputs/<retrieval>/candidate_papers.jsonl \
  --processed-output-dir processed_pdfs/<cache_name>/docling \
  --output-dir outputs/<hierarchy> \
  --mode extractive \
  --env-path pdf_docling_rerank_selection_generation_pipeline/.env
```

### 12.3 Keyed L2 generation from cache

```bash
python -m pdf_docling_rerank_selection_generation_pipeline.generate_from_cached_selection \
  --official-dir official_dev \
  --selected-contexts-input outputs/<selection>/selected_symbolic_contexts.debug.jsonl \
  --candidate-papers-input outputs/<retrieval>/candidate_papers.jsonl \
  --answer-contracts-input outputs/<contracts>/answer_contracts.redacted.jsonl \
  --hierarchy-input outputs/<hierarchy>/evidence_hierarchy.jsonl \
  --hierarchy-prompt-mode keyed \
  --output-dir outputs/<generation> \
  --max-prompt-chars 80000 \
  --max-images 4 \
  --env-path pdf_docling_rerank_selection_generation_pipeline/.env
```

### 12.4 Official evaluation

```bash
python official_dev/scripts/evaluate.py \
  --gold official_dev/data/validation.jsonl \
  --pred outputs/<run_name>/predictions.jsonl
```

## 13. Safety and Change-Control Rules

1. Never pass `validation.jsonl` to generation, reranking, or extraction code.
2. Treat official evaluation as a terminal measurement, not as an inference input.
3. Keep L0 provenance when deduplicating, projecting, packaging, summarizing, or attaching crops.
4. Do not replace a strict locator with a same-page package or a nearby caption.
5. Do not make source type or locator depend on the generated answer.
6. Record model responses, normalizer errors, cache version, prompt parameters, and source artifact hashes for every generation run.
7. Run package-qualified tests with `python -m unittest <module...>`; direct discovery from inside the package breaks relative imports.

