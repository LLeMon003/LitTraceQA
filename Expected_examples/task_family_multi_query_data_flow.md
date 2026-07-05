# Multi-Paper Query Data Flow: task_family Budget v5

This document simulates the current `pdf_vlm_symbolic_vlm_baseline` data flow for a multi-paper query.

Current budget rule:

```text
task_family: multi_paper
task_family_bucket: multi_paper
effective_top_k_papers: 12
effective_page_routing_top_pages_per_candidate: 3
effective_top_p_pages: 36
```

Important boundary:

```text
The system retrieves more papers for multi-paper tasks.
It does not allocate exactly 3 pages per paper.
It ranks all candidate pages globally and selects the top 36 pages.
VLM-2 receives selected symbolic evidence together in one query-level answer call.
```

Assumed configuration:

```env
TASK_FAMILY_BUDGET_ENABLED=true
SINGLE_PAPER_TOP_K_PAPERS=5
SINGLE_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=5
MULTI_PAPER_TOP_K_PAPERS=12
MULTI_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=3
PAGE_ROUTING_ENABLED=true
PAGE_ROUTING_SOURCE=native_text
PAGE_ROUTING_METHOD=global_native_text_bm25_rules
PAGE_ROUTING_PARSE_BATCH_SIZE=16
SYMBOLIC_ARTIFACT_VERSION=v5_eval_grounded_minimal_symbolic
VLM2_CONTEXT_MODE=text_only
VLM2_INCLUDE_PARSE_CONFIDENCE=false
```

## 1. Official Input

Input row:

```json
{
  "query_id": "q_multi_020",
  "benchmark": "LitTraceQA",
  "task_family": "multi_paper",
  "primary_evidence_type": "citation_context",
  "question": "Which NAACL 2025 papers explicitly mention or reference MCTS?",
  "answer_types": ["freeform"]
}
```

The runner maps the task to:

```json
{
  "task_family": "multi_paper",
  "task_family_bucket": "multi_paper",
  "effective_top_k_papers": 12,
  "effective_page_routing_top_pages_per_candidate": 3,
  "effective_top_p_pages": 36
}
```

## 2. Metadata Retrieval

Output file:

```text
outputs/<run>/candidate_papers.jsonl
```

Example row:

```json
{
  "query_id": "q_multi_020",
  "question": "Which NAACL 2025 papers explicitly mention or reference MCTS?",
  "task_family": "multi_paper",
  "task_family_bucket": "multi_paper",
  "effective_top_k_papers": 12,
  "effective_page_routing_top_pages_per_candidate": 3,
  "effective_top_p_pages": 36,
  "candidates": [
    {
      "rank": 1,
      "paper_id": "naacl2025_01001",
      "title": "Planning with Monte Carlo Tree Search for Language Agents",
      "abstract": "We use MCTS for planning...",
      "venue": "NAACL",
      "year": 2025,
      "score": 420.5
    },
    {
      "rank": 2,
      "paper_id": "naacl2025_01002",
      "title": "Tree Search Guided Reasoning",
      "abstract": "The method references Monte Carlo Tree Search...",
      "venue": "NAACL",
      "year": 2025,
      "score": 389.1
    },
    {
      "rank": 12,
      "paper_id": "naacl2025_01012",
      "title": "Surveying Search-Augmented LLMs",
      "abstract": "We compare beam search, MCTS, and graph search...",
      "venue": "NAACL",
      "year": 2025,
      "score": 178.4
    }
  ]
}
```

The higher top-k is intended to improve recall for multi-paper answers.

## 3. PDF Availability

Output file:

```text
outputs/<run>/pdf_availability.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "paper_id": "naacl2025_01001",
  "available": true,
  "local_path": "raw_pdfs/naacl2025_01001.pdf",
  "source": "cache"
}
```

OpenReview handling remains proceedings-first. Direct OpenReview access is skipped by default under:

```env
PDF_OPENREVIEW_POLICY=proceedings_first_skip_direct_openreview
```

## 4. Native Text Scan

All pages from all 12 available candidate PDFs are scanned with PyMuPDF native text.

Output file:

```text
outputs/<run>/native_page_text.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "candidate_rank": 2,
  "paper_id": "naacl2025_01002",
  "page": 9,
  "native_text": "Related Work. Monte Carlo Tree Search (MCTS) has been used...",
  "native_text_char_count": 1844,
  "has_native_text": true,
  "extraction_method": "pymupdf_text",
  "paper_title": "Tree Search Guided Reasoning"
}
```

Native text is an internal routing signal. It is not VLM-2 evidence.

## 5. Global Page Pool

If the 12 candidate PDFs contain 240 total pages, all 240 page rows enter one query-level pool.

Output file:

```text
outputs/<run>/global_page_pool.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "paper_id": "naacl2025_01002",
  "candidate_rank": 2,
  "page": 9,
  "native_text": "Related Work. Monte Carlo Tree Search (MCTS) has been used...",
  "native_text_char_count": 1844,
  "has_native_text": true
}
```

## 6. Global Page Ranking

The global ranker scores all pages together and selects 36 pages.

Output file:

```text
outputs/<run>/global_page_ranking.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "ranking_source": "native_text",
  "ranking_method": "global_native_text_bm25_rules",
  "top_k_papers": 12,
  "total_candidate_pages": 240,
  "selected_pages_initial": [
    {"paper_id": "naacl2025_01001", "page": 3, "global_page_rank": 1, "candidate_rank": 1},
    {"paper_id": "naacl2025_01002", "page": 9, "global_page_rank": 2, "candidate_rank": 2},
    {"paper_id": "naacl2025_01005", "page": 12, "global_page_rank": 3, "candidate_rank": 5},
    {"paper_id": "naacl2025_01001", "page": 8, "global_page_rank": 4, "candidate_rank": 1}
  ],
  "fallback_reason": ""
}
```

The 36 pages can come from fewer than 12 papers if those pages rank highest. There is no per-paper fixed quota.

## 7. Global Page Parse Plan

Output file:

```text
outputs/<run>/global_page_parse_plan.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "task_family": "multi_paper",
  "task_family_bucket": "multi_paper",
  "top_k_papers": 12,
  "page_routing_top_pages_per_candidate": 3,
  "total_candidate_pages": 240,
  "initial_top_p_global": 36,
  "max_pages_global": 36,
  "initial_selected_pages": [
    {"paper_id": "naacl2025_01001", "page": 3, "global_page_rank": 1, "candidate_rank": 1}
  ],
  "expanded_pages": [],
  "final_parsed_pages": [
    {"paper_id": "naacl2025_01001", "page": 3, "global_page_rank": 1, "candidate_rank": 1}
  ],
  "skipped_pages_count": 204
}
```

`PAGE_ROUTING_PARSE_BATCH_SIZE=16` schedules the 36 selected pages as three parser batches:

```text
batch 1: 16 pages
batch 2: 16 pages
batch 3: 4 pages
```

The batch size is not a semantic budget.

## 8. Page Rendering for VLM-1

Only selected pages are rendered.

Output file:

```text
outputs/<run>/page_rendering_artifacts.jsonl
```

Example:

```json
{
  "paper_id": "naacl2025_01002",
  "page": 9,
  "image_path": "processed_pdfs/vlm_symbolic/Qwen_Qwen3-VL-8B-Instruct/naacl2025_01002/rendered_pages/page_009.jpg",
  "dpi": 160,
  "format": "jpg"
}
```

The full page image is never sent to VLM-2.

## 9. VLM-1 Parser

Output file:

```text
outputs/<run>/raw_vlm_parser_responses.jsonl
```

Example raw response wrapper:

```json
{
  "paper_id": "naacl2025_01002",
  "page": 9,
  "pass_index": 1,
  "attempt": 1,
  "content": "{\n  \"records\": [\n    {\"kind\": \"citation_context\", \"text\": \"Monte Carlo Tree Search (MCTS) has been used for planning...\", \"label\": null}\n  ],\n  \"coverage\": {\"needs_continuation\": false},\n  \"warnings\": []\n}"
}
```

VLM-1 does not receive retrieval scores or page ranking scores.

## 10. Symbolic Records

Runtime output:

```text
outputs/<run>/symbolic_records.runtime.jsonl
```

Example rows from multiple papers:

```json
{
  "paper_id": "naacl2025_01001",
  "page": 3,
  "record_id": "naacl2025_01001_p003_r004",
  "record_type": "citation_context",
  "source_type": "citation_context",
  "text": "We adopt Monte Carlo Tree Search for action planning...",
  "locator": {"page": 3},
  "reading_order": 4,
  "page_status": "complete"
}
```

```json
{
  "paper_id": "naacl2025_01002",
  "page": 9,
  "record_id": "naacl2025_01002_p009_r002",
  "record_type": "citation_context",
  "source_type": "citation_context",
  "text": "MCTS is discussed as a search baseline in prior work...",
  "locator": {"page": 9},
  "reading_order": 2,
  "page_status": "complete"
}
```

## 11. Context Selection Across Papers

The selector ranks symbolic records across all parsed pages from all candidate papers.

Output file:

```text
outputs/<run>/selected_symbolic_contexts.prompt.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "selected_evidence": [
    {
      "paper_id": "naacl2025_01001",
      "page": 3,
      "source_type": "citation_context",
      "locator": {"page": 3},
      "text": "We adopt Monte Carlo Tree Search for action planning..."
    },
    {
      "paper_id": "naacl2025_01002",
      "page": 9,
      "source_type": "citation_context",
      "locator": {"page": 9},
      "text": "MCTS is discussed as a search baseline in prior work..."
    },
    {
      "paper_id": "naacl2025_01005",
      "page": 12,
      "source_type": "text_span",
      "locator": {"page": 12},
      "text": "The paper explicitly references Monte Carlo Tree Search."
    }
  ],
  "has_partial_artifacts": false,
  "attached_image_refs": []
}
```

Audit-only fields removed before VLM-2:

```text
global_record_id
record_type
record_id
label
image_path
score
bbox_1000
vlm_parse_confidence
```

## 12. VLM-2 Answer Call

VLM-2 receives one query-level prompt with selected evidence records from multiple papers.

VLM-2 does not know the retrieval score or selector score. It only sees answer-facing evidence:

```json
{
  "paper_id": "naacl2025_01001",
  "page": 3,
  "source_type": "citation_context",
  "locator": {"page": 3},
  "text": "We adopt Monte Carlo Tree Search for action planning..."
}
```

## 13. Prediction

Output file:

```text
outputs/<run>/predictions.jsonl
```

Example:

```json
{
  "query_id": "q_multi_020",
  "gold_papers": [
    {"paper_id": "naacl2025_01001"},
    {"paper_id": "naacl2025_01002"},
    {"paper_id": "naacl2025_01005"}
  ],
  "evidence": [
    {"paper_id": "naacl2025_01001", "source_type": "citation_context", "locator": {"page": 3}},
    {"paper_id": "naacl2025_01002", "source_type": "citation_context", "locator": {"page": 9}},
    {"paper_id": "naacl2025_01005", "source_type": "text_span", "locator": {"page": 12}}
  ],
  "answer": {
    "freeform": {
      "text": "The NAACL 2025 papers that explicitly mention or reference MCTS are naacl2025_01001, naacl2025_01002, and naacl2025_01005."
    }
  }
}
```

## Summary

```text
multi-paper query
→ retrieve 12 candidate papers
→ scan native text for all candidate PDF pages
→ rank all pages globally
→ parse top 36 selected pages with VLM-1
→ aggregate symbolic records across papers
→ select compact query-level evidence
→ call VLM-2 once
```
