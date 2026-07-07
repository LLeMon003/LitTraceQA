# Current Data Flow: `pdf_vlm_symbolic_vlm_baseline`

This document summarizes the current data flow after the v5 symbolic store, VLM-2 answer-contract fixes, and metadata-only paper-selection experiment.

## High-Level Flow

```text
official_dev/data/validation_inputs.jsonl
+ sanitized answer constraints from validation.jsonl
+ paper_metadata.jsonl
-> hybrid metadata retrieval
-> optional query decomposition for multi-paper retrieval
-> optional topic-profile expansion for opt-in ablation only
-> PDF cache and proceedings-first source resolution
-> native-text page routing
-> selected pages rendered as images
-> VLM-1 minimal symbolic parsing
-> processed_pdfs durable symbolic store
-> symbolic context selector
-> VLM-2 answer generation from selected symbolic evidence
-> parser normalization and official predictions.jsonl
```

VLM-2 does not receive full PDF pages, native PDFs, URLs, local PDF paths, retrieval scores, selector scores, bbox, parser confidence, or internal record ids. If images are ever used by VLM-2, they must be selected evidence crops, not full page images.

## 1. Official Inputs

`validation_inputs.jsonl` provides query-facing fields:

```json
{
  "query_id": "q_001",
  "benchmark": "LitTraceQA",
  "task_family": "hidden_source_single_paper",
  "primary_evidence_type": "table",
  "question": "Among the two prompt compression methods, how much does 500xCompressor outperform ICAE on the NaturalQ benchmark in terms of F1 score under the 500-to-1 compression setting?",
  "answer_types": ["freeform", "multiple_choice"]
}
```

`validation.jsonl` is used only to extract safe answer-shape constraints:

```json
{
  "answer": {
    "multiple_choice": {
      "options": {
        "A": "11.12",
        "B": "1.60",
        "C": "14.70",
        "D": "15.66"
      }
    },
    "table": {
      "schema": [
        {"name": "Method", "type": "string", "is_row_key": true},
        {"name": "2-step FID on CIFAR-10", "type": "number", "is_row_key": false}
      ]
    }
  }
}
```

The system must not expose `answer.*.gold`, `gold_papers`, or `evidence` from `validation.jsonl` to any VLM.

The sanitized VLM-2 answer contract is:

```json
{
  "query_id": "q_001",
  "answer_types": ["freeform", "multiple_choice"],
  "multiple_choice": {
    "options": [
      {"key": "A", "text": "11.12"},
      {"key": "B", "text": "1.60"},
      {"key": "C", "text": "14.70"},
      {"key": "D", "text": "15.66"}
    ],
    "output_field": "gold",
    "options_available": true
  },
  "table": {
    "table_schema": null
  },
  "freeform": {
    "output_rule": "produce concise text only if freeform is listed in answer_types"
  }
}
```

## 2. Metadata Retrieval

The default retrieval method is `hybrid_alias`.

Generic components:

```text
title BM25
abstract BM25
full metadata BM25
alias/method-name matching
venue/year hint
title exact/substring boost
```

For multi-paper queries, optional query decomposition can run per method-like mention and merge candidates. This is generic and does not use gold labels.

Ce-style topic profiles are now available only as explicit opt-in ablation:

```bash
RETRIEVAL_ENABLE_TOPIC_EXPANSION=true
```

Topic profiles are hand-written task/dev-set-oriented retrieval hints. They can reproduce a high paper-retrieval upper bound, but should not be treated as the default generic retrieval baseline.

`candidate_papers.jsonl`:

```json
{
  "query_id": "q_001",
  "task_family": "hidden_source_single_paper",
  "task_family_bucket": "single_paper",
  "effective_top_k_papers": 5,
  "effective_page_routing_top_pages_per_candidate": 5,
  "effective_top_p_pages": 25,
  "topic_expansion": null,
  "candidates": [
    {
      "rank": 1,
      "paper_id": "acl2025_00005",
      "title": "500x Compressor: Generalized Prompt Compression for Large Language Models",
      "abstract": "...",
      "venue": "ACL",
      "year": 2025,
      "score": 259.508924,
      "retrieval_method": "hybrid_alias",
      "retrieval_score_components": {
        "title_bm25": 13.83,
        "abstract_bm25": 67.07,
        "full_bm25": 72.72,
        "alias_bm25": 0.0,
        "method_substring_boost": 90.0,
        "weighted_total": 259.508924
      }
    }
  ]
}
```

Retrieval scores are audit data only and are not sent to VLM-1 or VLM-2.

## 3. PDF Availability And Page Routing

The PDF resolver uses local cache first and proceedings mirrors before direct OpenReview access.

`pdf_availability.jsonl`:

```json
{
  "query_id": "q_001",
  "paper_id": "acl2025_00005",
  "available": true,
  "local_path": "raw_pdfs/acl2025_00005.pdf",
  "source": "cache"
}
```

Native PDF text is used for page routing only:

```json
{
  "query_id": "q_001",
  "paper_id": "acl2025_00005",
  "page": 6,
  "native_text_char_count": 2140,
  "has_native_text": true,
  "extraction_method": "pymupdf_text"
}
```

`global_page_ranking.jsonl`:

```json
{
  "query_id": "q_001",
  "ranking_source": "native_text",
  "ranking_method": "global_native_text_bm25_rules",
  "total_candidate_pages": 90,
  "selected_pages_initial": [
    {"paper_id": "acl2025_00005", "page": 6, "global_page_rank": 1, "candidate_rank": 1},
    {"paper_id": "acl2025_00005", "page": 5, "global_page_rank": 2, "candidate_rank": 1}
  ]
}
```

For task-family budgeting:

```text
single-paper default: top-k papers = 5, average ranked pages per paper = 5
multi-paper default:  top-k papers = 12, average ranked pages per paper = 3
```

The top-p page budget is query-global. It is computed from top-k and pages-per-candidate; it is not a strict per-paper quota.

## 4. VLM-1 And Symbolic Store

VLM-1 receives rendered selected page images and outputs minimal evaluator-grounded symbolic records. It should not output bbox for all records. Figure localization may create figure crop paths as debug/runtime support, but bbox is not part of the VLM-2 prompt.

Run-level `symbolic_records.runtime.jsonl`:

```json
{
  "paper_id": "acl2025_00005",
  "page": 6,
  "record_id": "p006_r0003",
  "record_type": "table",
  "source_type": "table",
  "label": "Table 4",
  "text": "Dataset Length Eval. Metrics ... Absolute Δ 18.99 18.45 14.70 ...",
  "locator": {"page": 6, "table_id": "Table 4"},
  "page_status": "complete",
  "figure_crop_path": null
}
```

Durable processed store layout:

```text
processed_pdfs/vlm_symbolic_runs/<run_or_cache_name>/<parser_model_slug>/<paper_id>/
  artifact_status.json
  symbolic_records.runtime.jsonl
  symbolic_records.debug.jsonl
  symbolic_index.json
  page_records/
    page_006.records.runtime.jsonl
    page_006.records.debug.jsonl
  page_status/
    page_006.status.json
  page_images/
    page_006.jpg
  page_006/
    figure_crops/
      Figure_3_p006_r0009.jpg
```

`page_status/page_006.status.json`:

```json
{
  "paper_id": "acl2025_00005",
  "page": 6,
  "parser_model": "Qwen/Qwen3-VL-8B-Instruct",
  "artifact_version": "v5_eval_grounded_minimal_symbolic",
  "parser_mode": "text_first_symbolic_transcription",
  "page_status": "complete",
  "valid_record_count": 12,
  "rejected_record_count": 0,
  "created_at": "2026-07-07T04:29:00Z"
}
```

## 5. Symbolic Context Selector

The selector reads symbolic records and creates two views.

`selected_symbolic_contexts.debug.jsonl` keeps audit fields:

```json
{
  "query_id": "q_001",
  "selection_method": "symbolic_lexical_bm25_without_embedding",
  "source_type_distribution": {"table": 6, "text_span": 12, "figure": 3},
  "selected_records": [
    {
      "paper_id": "acl2025_00005",
      "page": 6,
      "record_id": "p006_r0003",
      "record_type": "table",
      "source_type": "table",
      "label": "Table 4",
      "score": 49.43,
      "text": "Dataset Length Eval. Metrics ...",
      "locator": {"page": 6, "table_id": "Table 4"},
      "image_path": "processed_pdfs/.../page_images/page_006.jpg"
    }
  ]
}
```

`selected_symbolic_contexts.prompt.jsonl` is the VLM-2-facing projection:

```json
{
  "query_id": "q_001",
  "selected_evidence": [
    {
      "paper_id": "acl2025_00005",
      "page": 6,
      "source_type": "table",
      "label": "Table 4",
      "grounding_label": {"type": "table_id", "value": "Table 4"},
      "text": "Dataset Length Eval. Metrics ... Absolute Δ 18.99 18.45 14.70 ..."
    }
  ],
  "has_partial_artifacts": false,
  "attached_image_refs": []
}
```

The prompt projection intentionally removes retrieval scores, selector scores, bbox, parser confidence, record ids, local paths, and full page images.

## 6. VLM-2 Input And Prediction

VLM-2 receives:

```json
{
  "query_id": "q_001",
  "task_family": "hidden_source_single_paper",
  "primary_evidence_type": "table",
  "question": "...",
  "answer_contract": {"answer_types": ["freeform", "multiple_choice"], "...": "..."},
  "required_answer_fields": ["freeform", "multiple_choice"],
  "required_answer_shape": {
    "freeform": {"text": "<concise answer text>"},
    "multiple_choice": {"gold": "<one of ['A', 'B', 'C', 'D']>"}
  },
  "candidate_papers": [
    {"paper_id": "acl2025_00005", "title": "...", "abstract": "..."}
  ],
  "selected_evidence": [
    {"paper_id": "acl2025_00005", "page": 6, "source_type": "table", "label": "Table 4", "text": "..."}
  ]
}
```

`predictions.jsonl`:

```json
{
  "query_id": "q_001",
  "gold_papers": [{"paper_id": "acl2025_00005"}],
  "evidence": [
    {
      "paper_id": "acl2025_00005",
      "source_type": "table",
      "locator": {"page": 6, "table_id": "Table 4"}
    }
  ],
  "answer": {
    "freeform": {"text": "14.70"},
    "multiple_choice": {"gold": "C"}
  }
}
```

The parser normalizes:

```text
bare string freeform -> {"text": "..."}
missing freeform with valid MC key -> fill freeform from option text
invalid evidence locator -> remove or repair locator, do not delete the answer
```

## 7. Metadata-Only V2

Metadata-only v2 is not direct top-k submission. Its flow is:

```text
top-k metadata candidates
-> VLM-2 sees only title/abstract/authors/venue/year
-> VLM-2 selects prediction.gold_papers
-> evidence = []
-> answer fields = empty values
```

`metadata_selection_prompts.jsonl`:

```json
{
  "query_id": "q_001",
  "messages": [
    {"role": "system", "content": "You are a LitTraceQA metadata-only paper selection model..."},
    {"role": "user", "content": "INPUT: {question, answer_types, candidate_papers...}"}
  ]
}
```

`raw_vlm_metadata_selection.jsonl`:

```json
{
  "query_id": "q_001",
  "content": "{\"query_id\":\"q_001\",\"gold_papers\":[{\"paper_id\":\"acl2025_00005\"}]}",
  "raw_response": {"choices": [{"message": {"content": "..."}}]}
}
```

`metadata-only` prediction:

```json
{
  "query_id": "q_001",
  "gold_papers": [{"paper_id": "acl2025_00005"}],
  "evidence": [],
  "answer": {
    "freeform": {"text": ""},
    "multiple_choice": {"gold": ""}
  }
}
```

Latest evaluated metadata-only v2:

```json
{
  "paper_precision_macro": 0.7445454545454546,
  "paper_recall_macro": 0.5636363636363636,
  "paper_f1_macro": 0.6009090909090908
}
```

## 8. Current Finding

The pure symbolic layer has a large effect on downstream context quality. VLM-1 plus symbolic validation makes the pipeline auditable, but VLM-2 currently depends heavily on selected symbolic text as its main context. This creates several bottlenecks:

```text
page routing miss -> VLM-1 never sees the right page
VLM-1 transcription loss -> symbolic record lacks the needed value
symbolic selector miss -> VLM-2 never sees the useful record
structured text distortion -> VLM-2 reasons over lossy symbolic context
```

The next baseline should investigate a different role for symbolic data:

```text
VLM-1 + symbolic layer = structured hints / anchors / provenance
not the only primary context source
```

In that next design, VLM-2 should receive richer original evidence context, while symbolic records provide hints such as candidate paper ids, page ids, source type, labels, table/figure/equation ids, and selected snippets for grounding.
