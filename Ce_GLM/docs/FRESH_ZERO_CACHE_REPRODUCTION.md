# Fresh zero-runtime-cache reproduction

This is the only supported path for a new raw generation. It is intentionally different from `cached-exact`, which begins from a sealed historical cache boundary.

## What this run proves

A successful `raw-fresh` run starts with an empty output/runtime directory, downloads or derives its runtime PDF caches anew, calls the configured provider, and executes the deterministic finishing stages. It does not consume the sealed cache-exact bundle, prior raw responses, or a `--resume-run` directory.

It does **not** promise the historical cache-exact prediction SHA or metrics. Provider generation remains an external model operation; the published cache-exact result has an explicitly unavailable producer edge.

## Required external asset releases

The code-only Git repository intentionally excludes datasets, indexes, paper files, credentials, raw model responses, evaluator gold, and runtime caches. Before a fresh run, supply two independently installed roots:

- `--release-root`: the Version 2 source release;
- `--source-root`: the option-aware source release.

Their exact nine immutable pre-generation inputs and SHA-256 values are listed in `configs/fresh_reproduction_assets.json`. The runner checks them before API work when `--verify-hashes` is given. The manifest excludes the historical cache boundary and official evaluation artifacts.

## Credential contract

Do not create a project `.env` and do not pass a key on the command line. The runner resolves the existing centralized credential chain in this order:

1. inherited provider variables;
2. `LITTRACEQA_SECRETS_FILE`;
3. `%USERPROFILE%\.littraceqa\secrets.env`.

It places a resolved credential only into the spawned raw-generation process environment when `SILICONFLOW_API_KEY` is not already inherited. No value is printed, persisted in a manifest, or committed.

## Non-answering preflight

```powershell
python scripts\run_ver2_reproduction.py `
  --mode raw-fresh `
  --output D:\runtime\fresh_001\predictions.jsonl `
  --release-root D:\external\littraceqa_baseline_Ver.2 `
  --source-root D:\external\littraceqa_baseline_uq_experiments `
  --preflight
```

`ready` means the assets match and a credential is available. It makes no API, model, evaluator, cache-exact, or answer-producing call.

## Fresh execution

Use a new, empty output root for every zero-cache attempt:

`configs/profiles/fresh_authoritative_v4.yaml` is the currently recommended
fresh profile. Its model selection is based only on completed aggregate fresh
evaluations, not on a query-level rule, prompt content, or runtime evaluator
feedback. `DeepSeek-V3.2` remains an experimental Evidence-focused profile;
its lower MC, Freeform, and Table results do not justify using it as the
default for this score-prioritized route. See
`docs/FRESH_MODEL_PROFILE_COMPARISON.md` for the disclosed comparison.

```powershell
python scripts\run_ver2_reproduction.py `
  --mode raw-fresh `
  --output D:\runtime\fresh_001\predictions.jsonl `
  --release-root D:\external\littraceqa_baseline_Ver.2 `
  --source-root D:\external\littraceqa_baseline_uq_experiments `
  --text-model deepseek-ai/DeepSeek-V4-Flash `
  --verify-hashes `
  --stall-timeout-sec 1200 `
  --heartbeat-interval-sec 30
```

Do not add `--resume-run` when validating zero-runtime-cache behavior. Evaluation is a separate, hash-locked operation requiring independently provisioned official evaluator and gold artifacts; it must never provide labels to generation.

## Freeze and exact replay of a fresh result

After a completed fresh run has produced exactly 55 records, it may be frozen
as its own cache-exact artifact. This makes the *same fresh prediction*
byte-for-byte replayable; it does not use, approximate, or inherit the score of
the separate historical cache-exact diagnostic bundle.

```powershell
python scripts\freeze_fresh_cache_exact.py freeze `
  --prediction D:\runtime\fresh_001\predictions.jsonl `
  --cache-dir D:\runtime\fresh_001\frozen_cache_exact

python scripts\freeze_fresh_cache_exact.py replay `
  --cache-dir D:\runtime\fresh_001\frozen_cache_exact `
  --output D:\runtime\fresh_001\replayed\predictions.jsonl
```

Both commands require exactly 55 unique query IDs and verify the prediction
SHA-256. They make no provider request, read no official gold, and run no
evaluator. The manifest label is `FRESH_FROZEN_CACHE_EXACT_REPLAY`; it must
never be reported as `VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`.

## One hash-locked evaluation of the replay

Only after the replay above succeeds, separately provision the official gold
and evaluator. Create a new lock and result path for this fresh run; neither
may already exist. The evaluator receives the replayed prediction, never the
raw generation file.

```powershell
python scripts\prepare_fresh_evaluation_lock.py `
  --prediction D:\runtime\fresh_001\replayed\predictions.jsonl `
  --gold D:\external\LitTraceQA\data\validation.jsonl `
  --evaluator D:\external\littraceqa_baseline_Ver.2\evaluation\evaluate.py `
  --python $((Get-Command python).Source) `
  --result D:\runtime\fresh_001\authoritative_fresh_evaluation.json `
  --lock D:\runtime\fresh_001\fresh_evaluation.lock.json

python scripts\run_locked_official_evaluation.py `
  --lock D:\runtime\fresh_001\fresh_evaluation.lock.json
```

The preparer only hashes and validates the supplied files; it does not invoke
the evaluator. The evaluator wrapper verifies the prediction, gold, and
evaluator hashes before and after its one invocation, rejects input files used
as gold, and refuses a second result at the same path. Evaluation metrics are
for this fresh frozen-replay prediction only and cannot be substituted with
the separate sealed cache-exact diagnostic metrics.

### Optional automatic post-run handoff

After the raw runner has created its run manifest, the repository watcher can
perform the same freeze, replay, verification, and one locked evaluation
without an external script. It still waits for `status=completed` and never
evaluates the raw prediction directly.

```powershell
python scripts\watch_fresh_evaluation_once.py `
  --run-root D:\runtime\fresh_001 `
  --repo-root . `
  --gold D:\external\LitTraceQA\data\validation.jsonl `
  --evaluator D:\external\littraceqa_baseline_Ver.2\evaluation\evaluate.py `
  --python $((Get-Command python).Source) `
  --freeze-cache-dir D:\runtime\fresh_001\frozen_cache_exact `
  --frozen-replay D:\runtime\fresh_001\replayed\predictions.jsonl
```

The watcher refuses an existing lock, result, or status file, and does not
overwrite a cache or replay output. Its compact status contains no answer,
raw provider response, credential, or gold value.
