# LitTraceQA Version 2.3 full-code reproduction

Version 2.3 provides three deliberately separate reproducibility claims:

1. cached-exact DEV and production replay are byte-exact;
2. the fresh SiliconFlow pipeline completed from raw generation through accepted Version 2 post-processing;
3. the fresh output was evaluated once with hash-verified official DEV gold and is materially below the cached-exact metrics.

Primary classification: `FRESH_PIPELINE_COMPLETE_METRICS_DIVERGENT`. Deterministic classification: `EXACT_DETERMINISTIC_REPRODUCTION_COMPLETE`.

## Frozen results

- Cached-exact DEV SHA-256: `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364`
- Cached-exact production SHA-256: `076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8`
- Fresh reassembled SHA-256: `A3B9140444573CF1DA529091C653B9B7EC9CDAC61D453FA2724BCB1134442281`
- Official gold SHA-256: `DAA5ED246C00A5E4BB571843BAA985B6256700DA8A7AE5695BD642DFD4298E41`
- Official evaluator SHA-256: `9410B51E86FC1EA382565D376016152FD52A74FA5D1F9358E36AF54711D8895F`

Fresh authoritative metrics: Paper F1 1.0; Evidence P/R/F1 0.618254/0.568687/0.581560; MC 27/41; Freeform 16/26; Table row F1 0.875397; Table cells 14/27 micro.

## Command sequence

Run from this directory with Python 3.12.

### A. Cached-exact DEV

```powershell
& python scripts\run_ver2_reproduction.py --mode cached-exact --output outputs\cached_exact\predictions.jsonl --verify-hashes
```

### B. Cached-exact production

```powershell
& python scripts\run_ver2_reproduction.py --mode production-cached-exact --output outputs\production_exact\predictions.jsonl --verify-hashes
```

### C. Fresh API with resume

```powershell
& python scripts\run_ver2_reproduction.py --mode raw-fresh --resume-run outputs\fresh_api_manual_20260718_010651 --output outputs\fresh_api_manual_20260718_010651\predictions.jsonl --verify-hashes --stall-timeout-sec 1200 --heartbeat-interval-sec 30
```

This transmits questions, ordered options, retrieved paper/PDF context, and Version 2 prompts to SiliconFlow. It must never transmit official gold, evaluator feedback, checkpoint answers, or Version 3 corrections. The recorded run reused 30 completed records and generated only the remaining 25.

### D. Locked official evaluation

```powershell
& python scripts\run_locked_official_evaluation.py --lock records\OFFICIAL_EVALUATION_LOCK.json
```

The wrapper verifies prediction, gold, and evaluator hashes and rejects raw inputs or predictions as gold. It refuses a repeated invocation after the authoritative result exists.

### Fresh-result freeze/replay (no score substitution)

For a completed `raw-fresh` result, use `scripts\freeze_fresh_cache_exact.py freeze` and then `replay` to reproduce that same 55-record prediction by hash without another API call. This is distinct from modes `cached-exact` and `production-cached-exact`, which replay the sealed historical diagnostic caches. See `docs\FRESH_ZERO_CACHE_REPRODUCTION.md` for the command contract.

## Evaluation labels

- Question-input-as-gold result: `INVALID`.
- Cached-prediction-as-gold result: `DIAGNOSTIC_ONLY`.
- Locked `LitTraceQA/data/validation.jsonl` result: `AUTHORITATIVE`.

## Scope and limitations

The raw-to-`modular_composed_label_locator_predictions.jsonl` historical producer edge remains unavailable. That artifact is a verified cache boundary, not an inferred lineage edge. Cached exactness therefore proves deterministic reconstruction below the boundary; it does not prove reconstruction of the historical cache producer. Fresh generation is a separate empirical run and its lower metrics do not invalidate deterministic replay.

See `docs/REPRODUCTION_GUIDE.md`, `docs/REPRODUCTION_READINESS.md`, and `docs/FRESH_VS_CACHED_DIVERGENCE.md` for the audit details.
