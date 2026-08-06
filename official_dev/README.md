---
pretty_name: LitTraceQA
language:
- en
license: cc-by-nc-4.0
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-retrieval
task_ids:
- open-domain-qa
- multiple-choice-qa
tags:
- scientific-papers
- evidence-grounding
- paper-retrieval
- multi-document-qa
- table-question-answering
- shared-task
- workshop
configs:
- config_name: inputs
  data_files:
  - split: validation
    path: data/validation_inputs.jsonl
  - split: test
    path: data/test.jsonl
  - split: test_extra
    path: data/test_extra.jsonl
- config_name: validation_gold
  data_files:
  - split: validation
    path: data/validation.jsonl
---

# LitTraceQA

LitTraceQA is a workshop shared-task dataset for literature-grounded question
answering. Systems retrieve the relevant paper or papers and produce the
requested answer format; the human-verified hidden split additionally evaluates
coarse evidence locations.

The public repository contains development gold plus input-only online
evaluation files. Hidden gold labels are kept organizer-side and are evaluated
through the submission website.

## Task

Given a research question and requested `answer_types`, return:

1. relevant paper IDs from the released metadata pool;
2. supporting evidence locations when the split evaluates evidence;
3. the final answer in the requested format.

All questions are scoped to `data/paper_metadata.jsonl`. Submissions should use
canonical `paper_id` values from that file.

Released challenge `query_id` values are stable opaque identifiers and do not
encode source files or generation batches.

## Splits

| Split | Rows | Purpose |
|---|---:|---|
| `validation` | 55 | Public development set with gold retrieval, evidence, and answers. |
| `test` | 71 | Required online leaderboard input split. Gold is hidden; paper, evidence, and answer are evaluated. |
| `test_extra` | 4,901 | Optional large online evaluation input split. Gold is hidden and not human-verified; paper and answer are evaluated. Results on this split are diagnostic and not required for the main leaderboard. |

Public `validation` task families:

| Family | Count |
|---|---:|
| `hidden_source_single_paper` | 26 |
| `multi_paper` | 29 |

Public `validation` primary evidence types:

| Type | Count |
|---|---:|
| `table` | 17 |
| `figure` | 15 |
| `text_span` | 9 |
| `citation_context` | 7 |
| `equation_algorithm` | 7 |

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
scripts/evaluate.py            # local development evaluator
scripts/validate_submission.py # participant output validator
schema/input.schema.json       # machine-readable input schema
schema/submission.schema.json  # machine-readable submission schema
schema/littraceqa.schema.json  # public gold development schema
```

Detailed docs:

- [Data and submission format](docs/format.md)
- [Evaluation metrics](docs/evaluation.md)

## Quick Start

Run the local evaluator on the placeholder validation submission:

```bash
python scripts/evaluate.py \
  --gold data/validation.jsonl \
  --pred data/sample_submission.jsonl
```

For a real run, replace `data/sample_submission.jsonl` with your prediction
file. See [Data and submission format](docs/format.md) for the expected JSONL
format.

Validate a completed challenge submission before upload:

```bash
python scripts/validate_submission.py \
  --input data/test.jsonl \
  --pred my_test_predictions.jsonl
```

For `test_extra`, use `data/test_extra.jsonl`; evidence is optional for that
split. The validator checks query coverage, answer shape, table columns,
multiple-choice labels, and paper IDs against the released metadata pool.

## Intended Use

The `validation` split is for local development, prompt engineering, loader
debugging, and local validation. The two input-only challenge splits are for
online submission. The optional unverified split is not used for the required
leaderboard because its labels have not been manually screened.

## License Notice

The LitTraceQA annotations and benchmark files are released under CC BY-NC 4.0.
Paper metadata remains subject to the original publishers' terms. PDFs are not
redistributed in this dataset. See [`LICENSE.md`](LICENSE.md).
