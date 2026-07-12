# Multi-Paper Data Flow: `pdf_vlm_symbolic_vlm_baseline`

This document simulates the current multi-paper flow after task-family budgeting and hybrid span page ranking.
All examples are schematic. They describe runtime data shape and boundaries, not gold labels.

## High-Level Flow

```text
official_dev/data/validation_inputs.jsonl
+ sanitized answer constraints from validation.jsonl
+ paper_metadata.jsonl
-> multi-paper hybrid metadata retrieval
-> optional query decomposition
-> optional topic-profile expansion only when explicitly enabled
-> local PDF cache / proceedings-first PDF resolution
-> native-text extraction for all candidate papers
-> query-global page pool
-> multi-paper hybrid span page scoring
-> fixed-budget selected pages
-> VLM-1 symbolic parsing for selected page images
-> processed_pdfs durable symbolic store
-> symbolic context selector
-> evidence packets / prompt-facing symbolic context
-> VLM-2 answer generation
-> parser normalization
-> official predictions.jsonl
```

VLM-1 sees rendered selected page images. VLM-2 sees selected symbolic evidence only. Retrieval scores, page scores,
selector scores, local PDF paths, bbox, parser confidence, internal record ids, official gold papers, official gold
evidence, and official gold answers are not exposed to any VLM.

## 1. Query Input

`validation_inputs.jsonl` provides query-facing fields:

```json
{
  "query_id": "q_multi_001",
  "benchmark": "LitTraceQA",
  "task_family": "multi_paper",
  "primary_evidence_type": "text_span",
  "question": "Across the methods that report training efficiency improvements, which papers attribute the improvement to cache reuse or prompt reuse, and what evidence supports each claim?",
  "answer_types": ["freeform"]
}
```

`validation.jsonl` may be read only for safe answer-shape constraints, such as available multiple-choice options or
table schema. The system must not expose official `gold_papers`, official `evidence`, or gold answer values to either
VLM.

## 2. Multi-Paper Retrieval

Default budget:

```text
MULTI_PAPER_TOP_K_PAPERS=12
MULTI_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=3
effective_top_p_pages = 12 * 3 = 36
```

`candidate_papers.jsonl`:

```json
{
  "query_id": "q_multi_001",
  "task_family": "multi_paper",
  "task_family_bucket": "multi_paper",
  "effective_top_k_papers": 12,
  "effective_page_routing_top_pages_per_candidate": 3,
  "effective_top_p_pages": 36,
  "retrieval_enable_query_decomposition": true,
  "topic_expansion": null,
  "candidates": [
    {
      "rank": 1,
      "paper_id": "acl2025_01001",
      "title": "Reusable Prompt Caches for Efficient Long-Context Inference",
      "abstract": "We study prompt reuse and cache reuse for reducing inference cost...",
      "venue": "ACL",
      "year": 2025,
      "score": 241.6,
      "retrieval_method": "hybrid_alias",
      "retrieval_score_components": {
        "title_bm25": 21.4,
        "abstract_bm25": 74.8,
        "full_bm25": 82.2,
        "method_substring_boost": 60.0,
        "weighted_total": 241.6
      }
    },
    {
      "rank": 2,
      "paper_id": "neurips2025_02002",
      "title": "Prompt Reuse for Memory-Efficient Reasoning",
      "abstract": "The proposed system reuses previous prompts and intermediate states...",
      "venue": "NeurIPS",
      "year": 2025,
      "score": 226.1,
      "retrieval_method": "hybrid_alias"
    },
    {
      "rank": 3,
      "paper_id": "iclr2025_03003",
      "title": "Fast Decoding with Context Caching",
      "abstract": "We cache context representations to reduce repeated computation...",
      "venue": "ICLR",
      "year": 2025,
      "score": 214.9,
      "retrieval_method": "hybrid_alias"
    }
  ]
}
```

Retrieval scores are audit-only. They are used before page routing and are not sent to VLM-1 or VLM-2.

## 3. PDF Availability

`pdf_availability.jsonl`:

```json
[
  {
    "query_id": "q_multi_001",
    "paper_id": "acl2025_01001",
    "available": true,
    "local_path": "raw_pdfs/acl2025_01001.pdf",
    "source": "cache"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "neurips2025_02002",
    "available": true,
    "local_path": "raw_pdfs/neurips2025_02002.pdf",
    "source": "proceedings"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "iclr2025_03003",
    "available": true,
    "local_path": "raw_pdfs/iclr2025_03003.pdf",
    "source": "cache"
  }
]
```

Local paths are runtime audit fields only. They are not prompt content.

## 4. Native-Text Page Pool

Every available candidate paper contributes native text pages to a query-global pool.

`native_page_text.jsonl`:

```json
[
  {
    "query_id": "q_multi_001",
    "paper_id": "acl2025_01001",
    "candidate_rank": 1,
    "page": 4,
    "native_text_char_count": 3180,
    "has_native_text": true,
    "extraction_method": "pymupdf_text"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "neurips2025_02002",
    "candidate_rank": 2,
    "page": 7,
    "native_text_char_count": 2864,
    "has_native_text": true,
    "extraction_method": "pymupdf_text"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "iclr2025_03003",
    "candidate_rank": 3,
    "page": 5,
    "native_text_char_count": 2951,
    "has_native_text": true,
    "extraction_method": "pymupdf_text"
  }
]
```

Native text is used for page ranking only. It is not submitted as final evidence and is not directly sent to VLM-2.

## 5. Multi-Paper Hybrid Span Page Ranking

For multi-paper queries, the current baseline uses hybrid span scoring:

```text
page_i = {chunk_i1, chunk_i2, ..., chunk_im}

s_ij = alpha * normalized_BM25(query, chunk_ij)
     + (1 - alpha) * local_TF-IDF_cosine(query, chunk_ij)

S_span(page_i) = log_mean_exp({s_ij}, gamma)

S_final(page_i) = S_span(page_i) + normalized_current_policy_page_score(page_i)
```

Default controls:

```env
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ENABLED=true
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ALPHA=0.75
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_GAMMA=4
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_CHUNK_MAX_CHARS=700
```

`global_page_ranking.jsonl`:

```json
{
  "query_id": "q_multi_001",
  "ranking_source": "native_text",
  "ranking_method": "global_native_text_bm25_rules",
  "page_ranking_bonus_enabled": true,
  "page_ranking_multi_text_span_hybrid_enabled": true,
  "page_ranking_multi_text_span_hybrid_alpha": 0.75,
  "page_ranking_multi_text_span_hybrid_gamma": 4.0,
  "page_ranking_multi_text_span_hybrid_chunk_max_chars": 700,
  "page_routing_strategy": "global_ranked_pages",
  "top_k_papers": 12,
  "total_candidate_pages": 144,
  "ranked_pages": [
    {
      "global_page_rank": 1,
      "paper_id": "acl2025_01001",
      "candidate_rank": 1,
      "page": 4,
      "score": 2.182431,
      "score_components": {
        "native_text_bm25": 13.4,
        "query_overlap": 2.1,
        "candidate_rank_prior": 0.0,
        "primary_evidence_type_boost": 0.9,
        "label_match_boost": 0.0,
        "section_heading_boost": 0.4,
        "page_position_prior": 0.2,
        "target_locator_bonus": 0.0,
        "query_hint_bonus": 1.0,
        "multi_text_span_hybrid_enabled": true,
        "multi_text_span_hybrid_score": 1.231224,
        "multi_text_span_current_policy_score_norm": 0.951207
      },
      "selected_for_initial_parse": true
    },
    {
      "global_page_rank": 2,
      "paper_id": "neurips2025_02002",
      "candidate_rank": 2,
      "page": 7,
      "score": 2.041009,
      "selected_for_initial_parse": true
    },
    {
      "global_page_rank": 3,
      "paper_id": "iclr2025_03003",
      "candidate_rank": 3,
      "page": 5,
      "score": 1.994551,
      "selected_for_initial_parse": true
    }
  ],
  "selected_pages_initial": [
    {"paper_id": "acl2025_01001", "page": 4, "global_page_rank": 1, "candidate_rank": 1, "selection_rank": 1},
    {"paper_id": "neurips2025_02002", "page": 7, "global_page_rank": 2, "candidate_rank": 2, "selection_rank": 2},
    {"paper_id": "iclr2025_03003", "page": 5, "global_page_rank": 3, "candidate_rank": 3, "selection_rank": 3}
  ],
  "selected_pages_final": [
    {"paper_id": "acl2025_01001", "page": 4, "global_page_rank": 1, "candidate_rank": 1, "selection_rank": 1},
    {"paper_id": "neurips2025_02002", "page": 7, "global_page_rank": 2, "candidate_rank": 2, "selection_rank": 2},
    {"paper_id": "iclr2025_03003", "page": 5, "global_page_rank": 3, "candidate_rank": 3, "selection_rank": 3}
  ]
}
```

Important multi-paper behavior:

- The selected page budget remains fixed.
- Pages are ranked in one query-global pool.
- There is no per-paper minimum page allocation by default.
- `candidate_rank_prior` is zero for multi-paper queries.
- `label_match_boost` is zero.
- Page scores are audit fields only.

## 6. VLM-1 Symbolic Parsing

Only selected pages are rendered and parsed by VLM-1.

`symbolic_records.runtime.jsonl`:

```json
[
  {
    "query_id": "q_multi_001",
    "paper_id": "acl2025_01001",
    "page": 4,
    "record_id": "p004_r0002",
    "record_type": "paragraph",
    "source_type": "text_span",
    "label": "",
    "text": "Cache reuse avoids recomputing prompt states and reduces inference latency by 31% in long-context settings.",
    "locator": {"page": 4},
    "page_status": "complete"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "neurips2025_02002",
    "page": 7,
    "record_id": "p007_r0005",
    "record_type": "paragraph",
    "source_type": "text_span",
    "label": "",
    "text": "Prompt reuse preserves previously computed states, reducing memory movement during multi-turn reasoning.",
    "locator": {"page": 7},
    "page_status": "complete"
  },
  {
    "query_id": "q_multi_001",
    "paper_id": "iclr2025_03003",
    "page": 5,
    "record_id": "p005_r0004",
    "record_type": "table",
    "source_type": "table",
    "label": "Table 2",
    "text": "Method | Latency Reduction | Memory Saving\nContext caching | 28.4 | 19.7",
    "locator": {"page": 5, "table_id": "Table 2"},
    "page_status": "complete"
  }
]
```

The durable store is organized per parser model and paper:

```text
processed_pdfs/vlm_symbolic_runs/<run_or_cache_name>/<parser_model_slug>/<paper_id>/
  artifact_status.json
  symbolic_records.runtime.jsonl
  symbolic_records.debug.jsonl
  symbolic_index.json
  page_records/
  page_status/
  page_images/
```

## 7. Symbolic Context Selection And Evidence Packets

The selector ranks symbolic records and builds prompt-facing evidence. Debug output may keep scores and record ids;
prompt output removes them.

`selected_symbolic_contexts.debug.jsonl`:

```json
{
  "query_id": "q_multi_001",
  "selection_method": "symbolic_lexical_bm25_without_embedding",
  "source_type_distribution": {"text_span": 18, "table": 6, "figure": 2},
  "selected_records": [
    {
      "paper_id": "acl2025_01001",
      "page": 4,
      "record_id": "p004_r0002",
      "source_type": "text_span",
      "score": 42.7,
      "text": "Cache reuse avoids recomputing prompt states..."
    },
    {
      "paper_id": "neurips2025_02002",
      "page": 7,
      "record_id": "p007_r0005",
      "source_type": "text_span",
      "score": 39.1,
      "text": "Prompt reuse preserves previously computed states..."
    }
  ]
}
```

`selected_symbolic_contexts.prompt.jsonl`:

```json
{
  "query_id": "q_multi_001",
  "selected_evidence": [
    {
      "paper_id": "acl2025_01001",
      "page": 4,
      "source_type": "text_span",
      "label": "",
      "grounding_label": {"type": "page", "value": 4},
      "text": "Cache reuse avoids recomputing prompt states and reduces inference latency by 31% in long-context settings."
    },
    {
      "paper_id": "neurips2025_02002",
      "page": 7,
      "source_type": "text_span",
      "label": "",
      "grounding_label": {"type": "page", "value": 7},
      "text": "Prompt reuse preserves previously computed states, reducing memory movement during multi-turn reasoning."
    },
    {
      "paper_id": "iclr2025_03003",
      "page": 5,
      "source_type": "table",
      "label": "Table 2",
      "grounding_label": {"type": "table_id", "value": "Table 2"},
      "text": "Method | Latency Reduction | Memory Saving\nContext caching | 28.4 | 19.7"
    }
  ],
  "has_partial_artifacts": false,
  "attached_image_refs": []
}
```

## 8. VLM-2 Input

VLM-2 receives candidate metadata and selected symbolic evidence, but not ranking scores.

```json
{
  "query_id": "q_multi_001",
  "task_family": "multi_paper",
  "primary_evidence_type": "text_span",
  "question": "Across the methods that report training efficiency improvements, which papers attribute the improvement to cache reuse or prompt reuse, and what evidence supports each claim?",
  "answer_contract": {
    "answer_types": ["freeform"],
    "freeform": {"output_rule": "produce concise text only if freeform is listed in answer_types"}
  },
  "multi_paper_contract": {
    "instruction": "Identify all papers that contribute evidence to the final answer. Do not output only the single most relevant paper if multiple papers support the answer."
  },
  "candidate_papers": [
    {"paper_id": "acl2025_01001", "title": "Reusable Prompt Caches for Efficient Long-Context Inference", "abstract": "..."},
    {"paper_id": "neurips2025_02002", "title": "Prompt Reuse for Memory-Efficient Reasoning", "abstract": "..."},
    {"paper_id": "iclr2025_03003", "title": "Fast Decoding with Context Caching", "abstract": "..."}
  ],
  "selected_evidence": [
    {"paper_id": "acl2025_01001", "page": 4, "source_type": "text_span", "text": "Cache reuse avoids recomputing prompt states..."},
    {"paper_id": "neurips2025_02002", "page": 7, "source_type": "text_span", "text": "Prompt reuse preserves previously computed states..."},
    {"paper_id": "iclr2025_03003", "page": 5, "source_type": "table", "label": "Table 2", "text": "Method | Latency Reduction | Memory Saving..."}
  ]
}
```

The internal response may include contributing papers for parser-side recovery and auditing:

```json
{
  "query_id": "q_multi_001",
  "contributing_papers": [
    {
      "paper_id": "acl2025_01001",
      "supporting_evidence": [{"page": 4, "source_type": "text_span"}],
      "contribution": "reports cache reuse as the source of reduced recomputation"
    },
    {
      "paper_id": "neurips2025_02002",
      "supporting_evidence": [{"page": 7, "source_type": "text_span"}],
      "contribution": "reports prompt reuse as the source of memory movement reduction"
    }
  ],
  "answer": {
    "freeform": {
      "text": "Cache reuse and prompt reuse are both reported as efficiency mechanisms: acl2025_01001 attributes latency reduction to avoiding repeated prompt-state computation, while neurips2025_02002 attributes memory savings to reusing previously computed prompt states."
    }
  }
}
```

## 9. Official Prediction

The official prediction keeps only official-schema fields:

```json
{
  "query_id": "q_multi_001",
  "gold_papers": [
    {"paper_id": "acl2025_01001"},
    {"paper_id": "neurips2025_02002"}
  ],
  "evidence": [
    {
      "paper_id": "acl2025_01001",
      "source_type": "text_span",
      "locator": {"page": 4}
    },
    {
      "paper_id": "neurips2025_02002",
      "source_type": "text_span",
      "locator": {"page": 7}
    }
  ],
  "answer": {
    "freeform": {
      "text": "acl2025_01001 reports cache reuse to avoid recomputing prompt states; neurips2025_02002 reports prompt reuse to reduce memory movement."
    }
  }
}
```

## 10. Evaluation Boundaries

For multi-paper page-selection diagnostics, use page-only evaluators before VLM runs:

```text
retrieved_paper_recall
retrieved_page_recall
R_selector
R_parser
R_context
```

For any full downstream run, report:

```text
VLM calls
input pages/images per query
input context length
output length
latency
estimated cost
official evaluator delta
```

Do not improve multi-paper performance by simply increasing top-k, selected page budget, context length, retry count,
or model size without a supported pilot diagnosis.
