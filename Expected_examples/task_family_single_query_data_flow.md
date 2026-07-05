# Single-Paper Query Data Flow: task_family Budget v5

This document simulates the current `pdf_vlm_symbolic_vlm_baseline` data flow for a single-paper-style query.

Current budget rule:

```text
task_family: hidden_source_single_paper
task_family_bucket: single_paper
effective_top_k_papers: 5
effective_page_routing_top_pages_per_candidate: 5
effective_top_p_pages: 25
```

Important boundary:

```text
top-p is query-level global page budget.
It is not a per-paper quota.
VLM-1 receives selected rendered page images.
VLM-2 receives one query-level selected symbolic evidence list.
VLM-2 does not receive full PDF pages or page images.
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
  "query_id": "q_single_001",
  "benchmark": "LitTraceQA",
  "task_family": "hidden_source_single_paper",
  "primary_evidence_type": "table",
  "question": "In the ExampleNet paper, what is the reported F1 score in Table 3?",
  "answer_types": ["freeform", "multiple_choice"]
}
```

The runner maps the task to:

```json
{
  "task_family": "hidden_source_single_paper",
  "task_family_bucket": "single_paper",
  "effective_top_k_papers": 5,
  "effective_page_routing_top_pages_per_candidate": 5,
  "effective_top_p_pages": 25
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
  "query_id": "q_single_001",
  "question": "In the ExampleNet paper, what is the reported F1 score in Table 3?",
  "task_family": "hidden_source_single_paper",
  "task_family_bucket": "single_paper",
  "effective_top_k_papers": 5,
  "effective_page_routing_top_pages_per_candidate": 5,
  "effective_top_p_pages": 25,
  "candidates": [
    {
      "rank": 1,
      "paper_id": "acl2025_10001",
      "title": "ExampleNet: Efficient Table Reasoning",
      "abstract": "We report table reasoning results including Table 3...",
      "venue": "ACL",
      "year": 2025,
      "score": 312.7,
      "score_components": {
        "title_bm25": 132.4,
        "abstract_bm25": 45.9,
        "alias_bm25": 80.0,
        "title_contains_method_boost": 54.4
      }
    },
    {
      "rank": 2,
      "paper_id": "emnlp2025_20002",
      "title": "Compact Evidence Selection",
      "abstract": "A retrieval and evidence selection system...",
      "venue": "EMNLP",
      "year": 2025,
      "score": 201.6
    }
  ]
}
```

Retrieval scores are saved for audit. They are not passed to VLM-1 or VLM-2.

## 3. PDF Availability

Output file:

```text
outputs/<run>/pdf_availability.jsonl
```

Example row:

```json
{
  "query_id": "q_single_001",
  "paper_id": "acl2025_10001",
  "available": true,
  "local_path": "raw_pdfs/acl2025_10001.pdf",
  "source": "cache"
}
```

`local_path` is internal. It is not sent to VLM-2.

## 4. Native Text Scan for Page Routing

Native text is extracted from all available candidate PDFs.

Output file:

```text
outputs/<run>/native_page_text.jsonl
```

Example rows:

```json
{
  "query_id": "q_single_001",
  "candidate_rank": 1,
  "paper_id": "acl2025_10001",
  "page": 6,
  "native_text": "Table 3 reports F1 scores on NaturalQ...",
  "native_text_char_count": 2140,
  "has_native_text": true,
  "extraction_method": "pymupdf_text",
  "paper_title": "ExampleNet: Efficient Table Reasoning"
}
```

Native text is used only for page routing. It is not directly submitted as final evidence.

## 5. Global Page Pool

All pages from the 5 candidate papers are merged into one query-level pool.

Output file:

```text
outputs/<run>/global_page_pool.jsonl
```

Example:

```json
{
  "query_id": "q_single_001",
  "paper_id": "acl2025_10001",
  "candidate_rank": 1,
  "page": 6,
  "native_text": "Table 3 reports F1 scores on NaturalQ...",
  "native_text_char_count": 2140,
  "has_native_text": true
}
```

If the five candidate PDFs contain 90 pages total, the pool has 90 page rows.

## 6. Global Page Ranking

Output file:

```text
outputs/<run>/global_page_ranking.jsonl
```

The ranker selects the first 25 pages globally:

```json
{
  "query_id": "q_single_001",
  "ranking_source": "native_text",
  "ranking_method": "global_native_text_bm25_rules",
  "top_k_papers": 5,
  "total_candidate_pages": 90,
  "selected_pages_initial": [
    {"paper_id": "acl2025_10001", "page": 6, "global_page_rank": 1, "candidate_rank": 1},
    {"paper_id": "acl2025_10001", "page": 5, "global_page_rank": 2, "candidate_rank": 1},
    {"paper_id": "emnlp2025_20002", "page": 8, "global_page_rank": 3, "candidate_rank": 2}
  ],
  "fallback_reason": ""
}
```

The selected 25 pages can be unevenly distributed across candidate papers.

## 7. Global Page Parse Plan

Output file:

```text
outputs/<run>/global_page_parse_plan.jsonl
```

Example:

```json
{
  "query_id": "q_single_001",
  "task_family": "hidden_source_single_paper",
  "task_family_bucket": "single_paper",
  "top_k_papers": 5,
  "page_routing_top_pages_per_candidate": 5,
  "total_candidate_pages": 90,
  "initial_top_p_global": 25,
  "max_pages_global": 25,
  "initial_selected_pages": [
    {"paper_id": "acl2025_10001", "page": 6, "global_page_rank": 1, "candidate_rank": 1}
  ],
  "expanded_pages": [],
  "final_parsed_pages": [
    {"paper_id": "acl2025_10001", "page": 6, "global_page_rank": 1, "candidate_rank": 1}
  ],
  "skipped_pages_count": 65
}
```

`PAGE_ROUTING_PARSE_BATCH_SIZE=16` means the 25 selected pages are scheduled in two parser batches: 16 pages, then 9 pages. It is a scheduling batch size, not a semantic page limit.

## 8. Page Rendering

Selected pages are rendered to images for VLM-1.

Output file:

```text
outputs/<run>/page_rendering_artifacts.jsonl
```

Example:

```json
{
  "paper_id": "acl2025_10001",
  "page": 6,
  "image_path": "processed_pdfs/vlm_symbolic/Qwen_Qwen3-VL-8B-Instruct/acl2025_10001/rendered_pages/page_006.jpg",
  "dpi": 160,
  "format": "jpg"
}
```

The full rendered page image is used by VLM-1 only.

## 9. VLM-1 Raw Parser Response

Output file:

```text
outputs/<run>/raw_vlm_parser_responses.jsonl
```

VLM-1 receives one selected rendered page image and emits minimal symbolic JSON:

```json
{
  "paper_id": "acl2025_10001",
  "page": 6,
  "pass_index": 1,
  "attempt": 1,
  "content": "{\n  \"records\": [\n    {\"kind\": \"table\", \"text\": \"Table 3 ... F1 72.4 ...\", \"label\": \"Table 3\"}\n  ],\n  \"coverage\": {\"needs_continuation\": false},\n  \"warnings\": []\n}"
}
```

## 10. Symbolic Records

Runtime records:

```text
outputs/<run>/symbolic_records.runtime.jsonl
```

Example runtime row:

```json
{
  "paper_id": "acl2025_10001",
  "page": 6,
  "record_id": "acl2025_10001_p006_r001",
  "record_type": "table",
  "source_type": "table",
  "text": "Table 3 ... F1 72.4 ...",
  "locator": {"page": 6, "table_id": "Table 3"},
  "reading_order": 1,
  "page_status": "complete"
}
```

Debug records keep audit-only fields such as `label`, `image_path`, `score`, and validation status. Those are not prompt fields for VLM-2.

## 11. Context Selection

Output files:

```text
outputs/<run>/selected_symbolic_contexts.debug.jsonl
outputs/<run>/selected_symbolic_contexts.prompt.jsonl
```

Prompt projection:

```json
{
  "query_id": "q_single_001",
  "selected_evidence": [
    {
      "paper_id": "acl2025_10001",
      "page": 6,
      "source_type": "table",
      "locator": {"page": 6, "table_id": "Table 3"},
      "text": "Table 3 ... F1 72.4 ..."
    }
  ],
  "has_partial_artifacts": false,
  "attached_image_refs": []
}
```

Removed before VLM-2:

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

VLM-2 receives one query-level prompt containing selected evidence records from all selected pages and candidate papers.

It does not receive:

```text
full PDF
full page image
all parsed symbolic records
retrieval scores
page ranking scores
selector scores
local image paths
```

VLM-2 output is normalized into:

```text
outputs/<run>/predictions.jsonl
```

Example prediction:

```json
{
  "query_id": "q_single_001",
  "gold_papers": [{"paper_id": "acl2025_10001"}],
  "evidence": [
    {
      "paper_id": "acl2025_10001",
      "source_type": "table",
      "locator": {"page": 6, "table_id": "Table 3"}
    }
  ],
  "answer": {
    "freeform": {"text": "72.4"},
    "multiple_choice": {"gold": "A"}
  }
}
```

## Summary

```text
single-paper query
→ retrieve 5 candidate papers
→ rank all candidate pages globally
→ parse top 25 selected pages with VLM-1
→ aggregate symbolic records
→ select compact query-level evidence
→ call VLM-2 once
```
