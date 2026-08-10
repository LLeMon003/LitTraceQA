# Final assembly trace

Created: 2026-07-18T01:57:54.750787+00:00

## Field survival matrix

| Stage | Records | Papers | Evidence | Freeform | Table | MC present | MC letter |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_input | 55 | 0 | 0 | 0 | 0 | 0 | 0 |
| option_aware_input | 55 | 0 | 0 | 0 | 0 | 0 | 0 |
| raw_generation_log | 55 | 55 | 55 | 26 | 11 | 41 | 0 |
| raw_stage_00 | 55 | 55 | 55 | 26 | 11 | 41 | 0 |
| option_aware_mc | 55 | 55 | 55 | 26 | 11 | 41 | 23 |
| evidence_safe_cleanup | 55 | 55 | 55 | 26 | 11 | 41 | 23 |
| source_grounded_mc | 55 | 55 | 55 | 26 | 11 | 41 | 28 |
| typed_mc | 55 | 55 | 55 | 26 | 11 | 41 | 30 |
| final_prediction | 55 | 55 | 55 | 26 | 11 | 41 | 30 |

## Earliest field-loss summary

- papers_present_nonempty: no count drop across traced prediction stages
- evidence_present_nonempty: no count drop across traced prediction stages
- freeform_present_nonempty: no count drop across traced prediction stages
- table_present_structural: no count drop across traced prediction stages
- mc_present: no count drop across traced prediction stages
- mc_letter_nonempty: no count drop across traced prediction stages

## Bounded schema examples

[
  {
    "query_id": "q_001",
    "top_keys": [
      "query_id",
      "gold_papers",
      "evidence",
      "answer"
    ],
    "answer_keys": [
      "freeform",
      "multiple_choice"
    ],
    "evidence_count": 1,
    "gold_papers_count": 1,
    "freeform_type": "dict",
    "table_keys": [],
    "mc_keys": [
      "gold"
    ]
  },
  {
    "query_id": "q_002",
    "top_keys": [
      "query_id",
      "gold_papers",
      "evidence",
      "answer"
    ],
    "answer_keys": [
      "freeform"
    ],
    "evidence_count": 1,
    "gold_papers_count": 1,
    "freeform_type": "dict",
    "table_keys": [],
    "mc_keys": []
  },
  {
    "query_id": "q_003",
    "top_keys": [
      "query_id",
      "gold_papers",
      "evidence",
      "answer"
    ],
    "answer_keys": [
      "freeform"
    ],
    "evidence_count": 1,
    "gold_papers_count": 1,
    "freeform_type": "dict",
    "table_keys": [],
    "mc_keys": []
  },
  {
    "query_id": "q_004",
    "top_keys": [
      "query_id",
      "gold_papers",
      "evidence",
      "answer"
    ],
    "answer_keys": [
      "freeform",
      "multiple_choice"
    ],
    "evidence_count": 1,
    "gold_papers_count": 1,
    "freeform_type": "dict",
    "table_keys": [],
    "mc_keys": [
      "gold"
    ]
  },
  {
    "query_id": "q_005",
    "top_keys": [
      "query_id",
      "gold_papers",
      "evidence",
      "answer"
    ],
    "answer_keys": [
      "freeform"
    ],
    "evidence_count": 1,
    "gold_papers_count": 1,
    "freeform_type": "dict",
    "table_keys": [],
    "mc_keys": []
  }
]