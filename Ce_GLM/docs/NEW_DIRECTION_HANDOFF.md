# Ver3 new-direction handoff

## Objective

Build a repo that begins from an empty runtime cache, runs all 55 records fresh, then freezes, byte-exactly replays, and evaluates that fresh output. Improve legitimately without official-gold leakage, historic prediction/cache inputs, query-specific rules, or false claims that a diagnostic replay is fresh generation.

Repository: `GroundLM-Ver3-GitHub-Reproduction_001`; branch: `ver3-cache-exact-complete-solution-001`; remote: `constantine617/GroundLM-Ver3`; last pushed commit at handoff: `94bca7e`. Preserve the user-owned untracked directories `package_extracted/` and `reproduction_output_001/`.

## Verified fresh chain

1. `scripts/run_ver2_reproduction.py --mode raw-fresh` starts from an absent root, hash-checks nine immutable source inputs, resolves credentials centrally, calls the provider, and generates 55 records.
2. `scripts/freeze_fresh_cache_exact.py` accepts only 55 unique IDs and freezes that fresh prediction.
3. Its `replay` operation verifies a byte-exact SHA and labels the output `FRESH_FROZEN_CACHE_EXACT_REPLAY`.
4. `scripts/watch_fresh_evaluation_once.py` can automate completion → freeze → replay → receipt validation → one hash-locked evaluator invocation.

See [FRESH_ZERO_CACHE_REPRODUCTION.md](FRESH_ZERO_CACHE_REPRODUCTION.md). Resolver precedence is inherited provider variables, `LITTRACEQA_SECRETS_FILE`, then `%USERPROFILE%\.littraceqa\secrets.env`; no project `.env` is allowed.

## Aggregate scores

| Run | Paper F1 | Evidence F1 | MC | Freeform | Table-cell macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best valid fresh: V4 Flash | 1.000000 | 0.537994 | 28/41 | 19/26 | 0.588384 |
| Fresh V3.2 | 1.000000 | 0.581025 | 26/41 | 17/26 | 0.515152 |
| Fresh V4 Pro | 1.000000 | 0.561227 | 27/41 | 18/26 | 0.452020 |
| Historical sealed diagnostic, **not fresh** | 1.000000 | 0.634646 | 38/41 | 23/26 | 0.954545 |

`configs/profiles/fresh_authoritative_v4.yaml` is the current recommended fresh profile. Its selection uses only completed aggregate fresh results. Details are in [FRESH_MODEL_PROFILE_COMPARISON.md](FRESH_MODEL_PROFILE_COMPARISON.md).

## Critical boundary

The high score is `VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION`. It deterministically replays a sealed historical cache boundary; the producer edge before that boundary is unavailable. The replay uses neither gold nor provider calls, but it is not evidence that an empty-cache provider run can reproduce the score.

Never use the sealed diagnostic bundle, its checkpoints, historical prediction/cache, raw responses, official evaluator output, official gold, or query-specific feedback as a fresh generator, prompt, router, threshold, or candidate input. Fresh freeze/replay only repeats the newly generated fresh prediction; it cannot improve it.

## Model-channel evidence

- V4 Flash and V4 Pro pass a tiny synthetic native-JSON/quote smoke.
- V4 Pro completed one full fresh run but regressed against V4 Flash.
- V4 Flash source-evidence workload failed twice with external-only `APITimeoutError`: 500-token and 160-token output caps, both at 45 seconds.
- V4 Pro source-evidence workload also failed with `APITimeoutError`: 160-token cap, 120 seconds.
- These are not path, credential, cache, gold, evaluator, or leakage failures. Tiny JSON transport works; the current provider/model route does not complete the strict source-evidence workload under its bounded contract.

`scripts/run_target_model_architecture1.py` now supports explicit `--model`, `--timeout-seconds`, `--max-tokens`, disabled SDK retries, atomic batch persistence, and resume. It remains synthetic-development-only and uses the central resolver.

## Reusable development infrastructure

- Target-aligned synthetic benchmark: `VER3_TARGET_ALIGNED_DEV_BENCHMARK_001__r01`, SHA-256 `909FB6CB30A15D2FDB600B4E53095E7CD1F569D5DC6942A05969BA06679D52A2`.
- It is mechanically source-derived and contains no official answer/gold fields.
- Its source-grounded retrieval gate passed with answer-bearing recall `0.982998118940819`.
- The model challenger accepts only bundle-whitelisted object IDs, exact cited quotes, source-supported values, and valid confidence. Keep raw responses external.

Previous source-native, table selective-repair, Freeform slot-recovery, MC structured-solver, evidence-reranking, and target-aligned model lines are preserved negative/exhausted results. Do not stack a rejected candidate into a new baseline.

## Requirements for the next direction

- Create a new registered, collision-safe workspace and external artifact roots; preserve prior completed experiments.
- Do not modify Version 2.3, frozen parent/predictions, official evaluator/gold, historical reports, or immutable manifests.
- Never access old OneDrive workspaces; old strings are historical audit metadata only.
- Treat the 55-record validation gold as held out: no development scoring, prompts, retrieval/routing tuning, threshold selection, or error analysis.
- Use only sanctioned development data or a mechanically verifiable source-derived synthetic benchmark for development.
- Lock development gate, model, configuration, threshold, and router before any held-out candidate. Candidate starts from frozen parent, preserves unrelated fields, freezes before one evaluator invocation, and is replayed cleanly.
- Keep credentials, raw prompts/responses, runtime outputs, caches, PDFs, official inputs/gold, and evaluator outputs outside Git.

## Recommended next step

Do not repeat the exhausted V4 Flash/V4 Pro source-evidence request configurations. First choose a materially different model channel or architecture. It must pass: resolver check → tiny synthetic JSON smoke → two-record synthetic source-evidence request with strict JSON and deterministic grounding. Only then scale on the locked synthetic benchmark.

Possible directions: a provider/model with reliable structured-context latency, a properly installed local text model that emits strict JSON, or a materially different deterministic source-native solver. Do not merely raise timeouts, reduce declared evidence below contract, or retry identical failed calls.

Before new work, run focused tests and `git diff --check`. Prior fresh-chain tests:

```powershell
python -m pytest tests/test_freeze_fresh_cache_exact.py tests/test_watch_fresh_evaluation_once.py tests/test_fresh_authoritative_v4_profile.py -q
```

No secret has been printed or committed.
