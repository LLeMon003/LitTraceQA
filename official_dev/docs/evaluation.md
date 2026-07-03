# LitTraceQA Evaluation

The public evaluator is intended for local development on the validation split:

```bash
python scripts/evaluate.py \
  --gold data/validation.jsonl \
  --pred data/sample_submission.jsonl
```

For a real submission, replace `data/sample_submission.jsonl` with your
prediction file.

The script prints:

```json
{
  "metrics": {},
  "details": {}
}
```

`metrics` contains comparable scores. `details` contains counts and missing or
extra prediction IDs.

## Retrieval

Paper retrieval is evaluated over canonical `paper_id` sets.

Reported metrics:

- `paper_precision_macro`
- `paper_recall_macro`
- `paper_f1_macro`

Each question receives precision/recall/F1, then scores are macro-averaged.

## Evidence Grounding

Evidence is evaluated by coarse exact match. Each evidence item is normalized to:

```text
(paper_id, source_type, page, object_id)
```

`object_id` is `table_id` for table evidence, `figure_id` for figure evidence,
and empty for other evidence types. Fine-grained fields such as `row`,
`column`, `region`, or `section` are ignored by the public evaluator.

Reported metrics:

- `evidence_precision_macro`
- `evidence_recall_macro`
- `evidence_f1_macro`

## Answers

Multiple choice:

- `multiple_choice_accuracy`

Freeform:

- `freeform_exact_match`

Table:

- `table_row_f1_macro`: row-key precision/recall/F1 macro average.
- `table_cell_accuracy_macro`: per-question cell accuracy, then macro average.
- `table_cell_accuracy_micro`: all table cells pooled together.

Table rows are aligned using the row-key columns from the provided
`table_schema`. Cell accuracy is computed over non-row-key columns after row
alignment.

The hidden test evaluation may include additional organizer-side checks, but it
will use the same high-level output components.
