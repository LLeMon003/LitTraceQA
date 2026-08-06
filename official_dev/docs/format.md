# LitTraceQA Format

This page documents the public development files and submission format.

## Files

```text
data/validation.jsonl          # gold development set
data/validation_inputs.jsonl   # input-only file; MC/table questions include answer schemas
data/sample_submission.jsonl   # valid placeholder submission template
data/test.jsonl
data/test_sample_submission.jsonl
data/test_extra.jsonl
data/test_extra_sample_submission.jsonl
data/paper_metadata.jsonl      # searchable paper metadata pool
schema/input.schema.json       # participant input records
schema/submission.schema.json  # participant output records
schema/littraceqa.schema.json  # public gold development records
scripts/validate_submission.py # completed-output validator
```

All files are JSON Lines unless otherwise noted.
The sample submission files contain schema-valid placeholder predictions; replace
the placeholder papers, evidence, and answers with model outputs before upload.

## Paper Metadata Pool

All questions are scoped to the released paper metadata pool in
`data/paper_metadata.jsonl`. Participants should retrieve papers from this pool
and submit canonical `paper_id` values.

`paper_metadata.jsonl` contains `paper_id`, `title`, `authors`, `abstract`,
`venue`, `year`, and available source identifiers or URLs. Local cache fields
and PDF download bookkeeping are not included.

## Input Records

`validation_inputs.jsonl`, `test.jsonl`, and
`test_extra.jsonl` contain the fields participants need for
local dry runs or online submission:

| Field | Description |
|---|---|
| `query_id` | Stable identifier. Challenge splits use opaque IDs in the form `ltqa_<16 hex characters>`. |
| `benchmark` | Always `LitTraceQA`. |
| `question` | Research question shown to the system. |
| `answer_types` | Requested answer components: `freeform`, `multiple_choice`, and/or `table`. |
| `multiple_choice_options` | Present only for multiple-choice questions. List of `{label, text}` option objects. |
| `table_schema` | Present only for table-answer questions. Use it for submitted rows. |

The input-only challenge files do not include gold papers, evidence, or gold
answers. Hidden labels are evaluated organizer-side through the submission
website.

## Gold Records

`validation.jsonl` additionally includes:

| Field | Description |
|---|---|
| `gold_papers` | Gold relevant-paper set. |
| `evidence` | Gold supporting evidence locations. |
| `answer` | Gold answer keyed by answer type. |

Gold records for `test` and `test_extra` are
not distributed publicly.

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

`gold_papers` is the benchmark's canonical field name for predicted relevant
papers. Submission paper objects contain only `paper_id`; do not repeat titles,
venues, or years.

For `test`, submissions must include predicted `gold_papers`,
predicted `evidence`, and predicted `answer`. For
`test_extra`, submissions include predicted `gold_papers` and
predicted `answer`; evidence may be omitted because that split does not score
evidence grounding.

Every released input row must have exactly one prediction row, identified by
the unchanged `query_id`. Do not add predictions for other splits to the same
file. Multiple-choice outputs use one of the letters present in that row's
`multiple_choice_options`; the number of options is not assumed to be four.
Query IDs do not encode the annotation source, generation batch, or task type.

Example input options:

```json
"multiple_choice_options": [
  {"label": "A", "text": "First option"},
  {"label": "B", "text": "Second option"}
]
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
| `text_span` | `page`; `section` is accepted when a page is unavailable |
| `equation_algorithm` | `page`, plus `equation_id` or `algorithm_id` when available |
| `citation_context` | `page` and/or `citation_id` |

Additional fields such as `row`, `column`, `section`, or `region` may appear in
the gold development data as helpful detail, but systems are not expected to
reproduce that fine-grained detail for the shared-task evidence score.

## Validate Before Upload

From the dataset repository root:

```bash
python scripts/validate_submission.py \
  --input data/test.jsonl \
  --pred my_test_predictions.jsonl

python scripts/validate_submission.py \
  --input data/test_extra.jsonl \
  --pred my_test_extra_predictions.jsonl
```

The first command requires evidence. The second infers that evidence is
optional. Both commands require complete query coverage and use
`data/paper_metadata.jsonl` to reject unknown paper IDs.
