# Locked official evaluation contract

The authoritative fresh evaluation is locked by `records/OFFICIAL_EVALUATION_LOCK.json`. Prediction, gold, and evaluator paths and SHA-256 values must all match before execution. The wrapper also requires 55 unique aligned query IDs; complete reference answer, paper, evidence, and answer-type fields; evaluator-compatible table schemas; and the required fresh prediction fields.

The wrapper rejects raw question inputs and prediction checkpoints as gold, hash drift, count or ID mismatches, duplicate IDs, missing reference fields, and table-contract mismatches. It verifies that prediction, gold, and evaluator are unchanged after the subprocess and refuses a second invocation once the authoritative result exists.

Command:

```powershell
& python scripts\run_locked_official_evaluation.py --lock records\OFFICIAL_EVALUATION_LOCK.json
```

Ten focused contract tests passed. They invoked the evaluator zero times. The first implementation cycle exposed only an overly specific expected error-message assertion in the test harness; the second cycle passed completely.
