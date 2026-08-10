# Version 2.3 reproduction guide

Run commands from the Version 2.3 workspace with Python 3.12.

## Deterministic DEV reproduction

```powershell
python scripts\run_ver2_reproduction.py `
  --mode cached-exact `
  --output outputs\cached_exact\predictions.jsonl `
  --verify-hashes
```

The run begins at the verified cache boundary and records every stage as `VERIFIED_CACHE_INPUT`, `CACHE_HIT`, `EXECUTED`, or `NOT_APPLICABLE`. It never accepts the final DEV target, validation gold, or evaluator output as a producing input.

The accepted option-aware stage must use `littraceqa_baseline_uq_experiments/outputs/document_object_index.jsonl` SHA-256 `222C63DEDBBC40FE6A441D6EDD76B996812D777DA9C214E4724192BA1CEE9DE6`. The later full-Docling index changes the q_041 confidence gate and does not reproduce the accepted checkpoint.

Expected final SHA-256: `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`.

## Production reproduction

```powershell
python scripts\run_ver2_reproduction.py `
  --mode production-cached-exact `
  --output outputs\production_exact\predictions.jsonl `
  --verify-hashes
```

Expected final SHA-256: `076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8`.

## Partial runs and resume

Use `--start-stage`, `--stop-stage`, and `--resume-manifest` for bounded recovery. A non-initial start requires a prior manifest. Every resumed output is rehashed before it is classified as `CACHE_HIT`.

## JSONL transport and contracts

```powershell
python -m unittest -v tests.test_jsonl_io
python scripts\run_component_contracts.py
```

The transport is strict UTF-8, preserves U+2028/U+2029 and semantic identity, emits one object per physical line, and reports line/column parser errors.

## Raw-fresh mode

```powershell
python scripts\run_ver2_reproduction.py `
  --mode raw-fresh `
  --output outputs\fresh_api\predictions.jsonl `
  --verify-hashes
```

This command transmits validation questions, ordered options, retrieved paper/PDF context, and existing Version 2 prompts to SiliconFlow. Gold, frozen answers, expected letters, evaluator feedback, and Version 3 corrections are prohibited.

The completed run used resumable execution and write-isolated caches:

```powershell
& python scripts\run_ver2_reproduction.py `
  --mode raw-fresh `
  --resume-run outputs\fresh_api_manual_20260718_010651 `
  --output outputs\fresh_api_manual_20260718_010651\predictions.jsonl `
  --verify-hashes `
  --stall-timeout-sec 1200 `
  --heartbeat-interval-sec 30
```

It reused 30 verified records and generated only 25 missing records. The evaluator-compatible offline assembly is frozen at `outputs/fresh_api_manual_20260718_010651/predictions_reassembled.jsonl`, SHA-256 `A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281`.

## Evaluation

Freeze predictions before evaluation. Do not pass gold to any producing stage.

The official DEV gold is `..\LitTraceQA\data\validation.jsonl`, SHA-256 `DAA5ED246C00A5E4BB571843BAA985B6256700DA8A7AE5695BD642DFD4298E41`. Use the locked wrapper:

```powershell
& python scripts\run_locked_official_evaluation.py --lock records\OFFICIAL_EVALUATION_LOCK.json
```

The wrapper verifies the prediction, gold, and evaluator hashes; checks 55 unique aligned IDs and answer/table contracts; rejects raw question inputs and prediction checkpoints as gold; verifies input immutability; and refuses a second authoritative invocation.

Status labels: the earlier question-input-as-gold evaluation is `INVALID`; the cached-prediction-as-gold evaluation is `DIAGNOSTIC_ONLY`; `records/OFFICIAL_FRESH_EVALUATION.json` is `AUTHORITATIVE`.
