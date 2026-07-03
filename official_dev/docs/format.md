# LitTraceQA Format

This page documents the public development files and submission format.

## Files

```text
data/validation.jsonl          # gold development set
data/validation_inputs.jsonl   # input-only file; table questions include table_schema
data/sample_submission.jsonl   # empty submission template
data/paper_metadata.jsonl      # searchable paper metadata pool
schema/littraceqa.schema.json  # machine-readable gold sample schema
```

All files are JSON Lines unless otherwise noted.

## Paper Metadata Pool

All questions are scoped to the released paper metadata pool in
`data/paper_metadata.jsonl`. Participants should retrieve papers from this pool
and submit canonical `paper_id` values.

`paper_metadata.jsonl` contains `paper_id`, `title`, `authors`, `abstract`,
`venue`, `year`, and available source identifiers or URLs. Local cache fields
and PDF download bookkeeping are not included.

## Input Records

`validation_inputs.jsonl` contains the fields participants need for local dry
runs:

| Field | Description |
|---|---|
| `query_id` | Stable question identifier. |
| `benchmark` | Always `LitTraceQA`. |
| `task_family` | `hidden_source_single_paper` or `multi_paper`. |
| `primary_evidence_type` | Main evidence type. |
| `question` | Research question shown to the system. |
| `answer_types` | Requested answer components: `freeform`, `multiple_choice`, and/or `table`. |
| `table_schema` | Present only for table-answer questions. Use it for submitted rows. |

## Gold Records

`validation.jsonl` additionally includes:

| Field | Description |
|---|---|
| `gold_papers` | Gold relevant-paper set. |
| `evidence` | Gold supporting evidence locations. |
| `answer` | Gold answer keyed by answer type. |

## Submission Format

For each question, submit one JSON object per line:

```json
{
  "query_id": "q_001",
  "gold_papers": [
    {"paper_id": "acl2025_00005"}
  ],
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

For table questions, `validation_inputs.jsonl` provides a `table_schema`.
Submit table rows under `answer.table.rows`. Repeating the schema in the
submission is optional; the scorer aligns rows using the provided schema.

```json
{
  "query_id": "q_table_example",
  "gold_papers": [
    {"paper_id": "iclr2025_03463"}
  ],
  "evidence": [
    {
      "paper_id": "iclr2025_03463",
      "source_type": "table",
      "locator": {"page": 7, "table_id": "Table 1"}
    }
  ],
  "answer": {
    "table": {
      "rows": [
        {"method": "TCM", "fid": 2.05},
        {"method": "IMM", "fid": 1.98}
      ]
    }
  }
}
```

Table rows should use exactly the column names declared in `table_schema`.
Numeric cells should be JSON numbers, not strings with units or percent signs.
Use `null` only when the value is genuinely missing or unreported.

## Evidence Locations

The shared task evaluates evidence grounding at a coarse, reproducible level:
paper page and, when applicable, table or figure number.

| Evidence type | Expected coarse locator |
|---|---|
| `table` | `page`, `table_id` |
| `figure` | `page`, `figure_id` |
| `text_span` | `page` |
| `equation_algorithm` | `page` |
| `citation_context` | `page` |

Additional fields such as `row`, `column`, `section`, or `region` may appear in
the gold development data as helpful detail, but systems are not expected to
reproduce that fine-grained detail for the shared-task evidence score.
