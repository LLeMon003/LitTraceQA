---
pretty_name: LitTraceQA
language:
- en
license: cc-by-nc-4.0
size_categories:
- n<1K
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
- config_name: default
  data_files:
  - split: validation
    path: data/validation.jsonl
---

# LitTraceQA

LitTraceQA is a workshop shared-task development set for literature-grounded
question answering. Systems must retrieve the relevant paper or papers, identify
coarse evidence locations, and produce the requested answer format.

This public split is intentionally small. It is for challenge participants to
inspect the task format, build loaders, tune prompts, and run local validation
before submitting to the hidden evaluation set.

## Task

Given a research question and requested `answer_types`, return:

1. relevant paper IDs from the released metadata pool;
2. supporting evidence locations in those papers;
3. the final answer in the requested format.

All questions are scoped to `data/paper_metadata.jsonl`. Submissions should use
canonical `paper_id` values from that file.

## Public Split

| Split | Rows | Purpose |
|---|---:|---|
| `validation` | 55 | Public development set with gold retrieval, evidence, and answers. |

Task families:

| Family | Count |
|---|---:|
| `hidden_source_single_paper` | 26 |
| `multi_paper` | 29 |

Primary evidence types:

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
data/validation_inputs.jsonl   # input-only file; table questions include table_schema
data/sample_submission.jsonl   # empty submission template
data/paper_metadata.jsonl      # searchable paper metadata pool
scripts/evaluate.py            # local development evaluator
schema/littraceqa.schema.json  # machine-readable gold sample schema
```

Detailed docs:

- [Data and submission format](docs/format.md)
- [Evaluation metrics](docs/evaluation.md)

## Quick Start

Run the local evaluator on the empty sample submission:

```bash
python scripts/evaluate.py \
  --gold data/validation.jsonl \
  --pred data/sample_submission.jsonl
```

For a real run, replace `data/sample_submission.jsonl` with your prediction
file. See [Data and submission format](docs/format.md) for the expected JSONL
format.

## Intended Use

This public split is for workshop challenge development, prompt engineering,
loader debugging, and local validation. It should not be treated as the hidden
test set or as an estimate of final leaderboard performance.

## License Notice

The LitTraceQA annotations and benchmark files are released under CC BY-NC 4.0.
Paper metadata remains subject to the original publishers' terms. PDFs are not
redistributed in this dataset. See [`LICENSE.md`](LICENSE.md).
