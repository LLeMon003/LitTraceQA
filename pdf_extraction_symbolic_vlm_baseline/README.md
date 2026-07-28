# PDF Extraction Symbolic VLM Baseline

This baseline removes the VLM-1 page-image transcription stage used by
`pdf_vlm_symbolic_vlm_baseline`. It parses every available candidate paper page
with PyMuPDF, writes symbolic runtime/debug records, then reuses the existing
symbolic context selector and VLM-2 answer generator.

Run:

```bash
conda activate littraceqa
python -m pdf_extraction_symbolic_vlm_baseline.run_pdf_extraction_symbolic_vlm_baseline \
  --official-dir official_dev \
  --output-dir outputs/pdf_extraction_symbolic_vlm_baseline \
  --processed-output-dir processed_pdfs/pdf_extraction_symbolic \
  --pdf-output-dir raw_pdfs \
  --max-queries 3
```

Dry extraction/context smoke test without VLM-2:

```bash
python -m pdf_extraction_symbolic_vlm_baseline.run_pdf_extraction_symbolic_vlm_baseline \
  --official-dir official_dev \
  --output-dir outputs/pdf_extraction_symbolic_vlm_baseline_smoke \
  --processed-output-dir processed_pdfs/pdf_extraction_symbolic \
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
# Experimental Multi-Paper HyDE

`MULTI_PAPER_HYDE_ENABLED=true` adds a removable relevance-only fusion step
after Qwen section reranking and before the existing fixed section Top-K.
Single-paper queries and disabled runs keep the original path.

HyDE uses `deepseek-ai/DeepSeek-V4-Flash` to generate validated, value-masked
JSON claims. E5 scores bounded text chunks and object text projections for each
claim. Per-claim unit scores are min-max normalized, locally max-pooled to
sections, and fused with min-max-normalized Qwen scores:

```text
fused = w_original * normalized_qwen + w_hyde * normalized_hyde
```

Generation and E5 document embeddings are cached under
`<processed-output-dir>/.multi_paper_hyde_cache/`. The complete audit is stored
inside each multi-paper row of `section_relevance.trace.jsonl`; generated claims
are never added to selected evidence or answer prompts.

Defaults and all experiment variables are documented in `.env.example`.

## Record-Aware Section Retrieval

`SECTION_RELEVANCE_UNIT_MODE=record_aware` keeps extracted records intact while
ranking. Text records are packed into bounded units; tables, figures, equations,
algorithms, and citation records remain atomic object units with configurable
neighboring text. A section score is pooled from its top units and returned as a
bounded bonus to every unit in that section. Qwen reranking first uses BM25 to
prefilter units, while explicit object locators such as `Table 2`, `Figure 4`,
`Equation 6`, or `24th reference` are always admitted to reranking.

Each `section_relevance.trace.jsonl` row records local relevance, section
relevance, bonuses, final relevance, and both local and final rank. The old
`LLMRERANK_SECTION_CHUNK_*` settings apply only when
`SECTION_RELEVANCE_UNIT_MODE=token_chunks`.
