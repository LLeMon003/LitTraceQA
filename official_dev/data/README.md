# Data Directory

LitTraceQA public development files:

```text
validation.jsonl                              # gold development set
validation_inputs.jsonl                       # input-only copy for participant dry runs
sample_submission.jsonl                       # validation submission template
test.jsonl                  # required online evaluation inputs
test_sample_submission.jsonl       # required online evaluation template
test_extra.jsonl              # optional large online evaluation inputs
test_extra_sample_submission.jsonl   # optional online evaluation template
paper_metadata.jsonl                          # searchable candidate paper metadata pool
```

The `validation` split is a small public development set for workshop challenge
participants. The `test` split is the required input-only
online leaderboard split. The `test_extra` split is a larger,
input-only diagnostic split; submissions for it are not required for the main
leaderboard.

Each JSONL file has one sample per line. Do not place private gold files, PDF
caches, raw paper PDFs, or annotation-workflow scratch files in this directory.

Questions are scoped to the papers listed in `paper_metadata.jsonl`. Use the
canonical `paper_id` values from that file when submitting retrieved papers.

Machine-readable schemas:

```text
../schema/input.schema.json
../schema/submission.schema.json
../schema/littraceqa.schema.json
```
