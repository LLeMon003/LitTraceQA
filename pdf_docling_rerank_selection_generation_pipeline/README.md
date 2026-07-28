# Docling Rerank Selection Generation Pipeline

Complete architecture, data contracts, provenance rules, configuration groups,
and the current frozen public-development result are documented in
[`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md).

This baseline parses candidate PDFs with Docling/PyMuPDF, scores all canonical
record-aware units with BM25 and Qwen reranking, then sends evidence packages
rather than bare chunks to VLM-2.

Run:

```bash
conda activate littraceqa
python -m pdf_docling_rerank_selection_generation_pipeline \
  --official-dir official_dev \
  --output-dir outputs/pdf_docling_rerank_selection_generation_pipeline \
  --processed-output-dir processed_pdfs/pdf_docling_rerank_selection_generation_pipeline \
  --pdf-output-dir raw_pdfs \
  --max-queries 3
```

Dry extraction/context smoke test without VLM-2:

```bash
python -m pdf_docling_rerank_selection_generation_pipeline \
  --official-dir official_dev \
  --output-dir outputs/pdf_docling_rerank_selection_generation_pipeline_smoke \
  --processed-output-dir processed_pdfs/pdf_docling_rerank_selection_generation_pipeline \
  --pdf-output-dir raw_pdfs \
  --max-queries 3 \
  --skip-generation
```

Known limitations:

- Image-only pages and scanned tables still need OCR fallback.
- Figure records contain captions and optional crops, but no visual reasoning.
- Complex vector figures and dense equations are heuristic.
- Inline math remains `text_span`.
- Full answer accuracy still depends on metadata retrieval, symbolic selection,
  and VLM-2.
# Selection Design

`section_relevance` is now a scoring backend, not the final context selector:

- all canonical record-aware units are Qwen scored; sparse Qwen prefiltering is
  disabled on this path;
- text records are packaged with neighboring paragraphs, and visual/object
  records retain their caption/context and crop provenance;
- BM25 routes the original query, requested modality, explicit locator, and,
  for multi-paper queries, optional HyDE claims; RRF forms a candidate union;
- Qwen text/image projections remain separate `qwen_text` and `qwen_visual`
  tracks; a rule-based `layout_score` ranks labels and requested modalities.
  The tracks contribute ranks to RRF and are never linearly mixed;
- Qwen relevance plus explicit modality, claim, page-diversity, and cross-paper
  coverage chooses packages within configured package and character budgets.

The character budget is package-atomic: a table, figure, equation, or citation
package is included whole or skipped, never truncated record-by-record.

`*_EVIDENCE_PACKAGE_BUDGET` is `K_max`, not a mandatory output count. The
selector first reaches the configured minimum package and multi-paper coverage,
then stops when remaining candidates add no new requested modality or paper.
Its trace records `adaptive_stop_reason`, package/character budgets, selected
modalities, and separate score-track availability.

`MULTI_PAPER_HYDE_ENABLED=true` only generates additional value-masked routing
queries. It never fuses scores or changes the Qwen relevance signal.

`VLM2_CONTEXT_SELECTION_MODE` is intentionally limited to `section_relevance`
for this pipeline. Older page/record budget modes belong to the predecessor
baseline and are not part of this configuration surface.

The selector trace records canonical Qwen projections, package candidate union,
package selection, and HyDE routing audit. The evaluator reports paper recall,
evidence recall, all-support query recall, modality recall, and context-budget
recall instead of treating a fixed Top-K as the primary metric.

Defaults and all experiment variables are documented in `.env.example`.

## Auditable L0-L3 Compression

`build_evidence_hierarchy` is a post-selection experiment: it reuses frozen
Qwen-selected records and does not call retrieval, Docling, or the reranker.
It writes an auditable hierarchy for every query:

- `L0`: exact parser records and official locators;
- `L1`: heading paths and bounded, provenance-preserving neighbors;
- `L2`: verified proposition cards plus a bounded extractive micro-evidence
  index, each with a raw `support_ref`;
- `L3`: paper/section/entity navigation only, never sole factual support.

Build a full extractive hierarchy from a frozen selection cache:

```bash
python -m pdf_docling_rerank_selection_generation_pipeline.build_evidence_hierarchy \
  --official-dir official_dev \
  --selected-contexts-input outputs/pdf_docling_rerank_selection_page_audit_b150/selected_symbolic_contexts.debug.jsonl \
  --candidate-papers-input outputs/docling_qwen_hyde_paper_retrieval_only/candidate_papers.jsonl \
  --processed-output-dir processed_pdfs/pdf_docling_symbolic/docling_v1.5/docling \
  --output-dir outputs/pdf_docling_rerank_selection_page_audit_b150_hierarchy_extractive \
  --env-path pdf_docling_rerank_selection_generation_pipeline/.env \
  --mode extractive --dry-run
```

Generate from the hierarchy without rerunning upstream modules:

```bash
python -m pdf_docling_rerank_selection_generation_pipeline.generate_from_cached_selection \
  --official-dir official_dev \
  --selected-contexts-input outputs/pdf_docling_rerank_selection_page_audit_b150/selected_symbolic_contexts.debug.jsonl \
  --candidate-papers-input outputs/docling_qwen_hyde_paper_retrieval_only/candidate_papers.jsonl \
  --hierarchy-input outputs/pdf_docling_rerank_selection_page_audit_b150_hierarchy_extractive/evidence_hierarchy.jsonl \
  --output-dir outputs/pdf_docling_rerank_selection_page_audit_b150_hierarchy_micro_prediction \
  --env-path pdf_docling_rerank_selection_generation_pipeline/.env \
  --max-prompt-chars 65000
```

`verified_llm` may be used for card construction. It is guarded by exact
support-quote, numeric/entity, and lexical-overlap checks; rejected cards fall
back to extractive cards. The final generator only accepts evidence locators
visible through an L2 `support_ref`, including parser-expanded table plans.
