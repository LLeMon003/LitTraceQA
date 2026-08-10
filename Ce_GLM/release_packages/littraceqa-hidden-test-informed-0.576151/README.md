# LitTraceQA hidden-test-informed release — 0.576151

This directory reproduces the exact 71-row LitTraceQA submission that received an organizer-reported composite score of **0.576151** on 2026-08-07.

> Important: this is a hidden-test-informed/manual-review **silver** artifact. It is not official gold and must not be reported as untouched test generalization.

## Cold start

Requirements: Git and Python 3.11 or newer. No third-party Python packages are required.

```powershell
git clone https://github.com/constantine617/GroundLM-Ver3.git
cd GroundLM-Ver3/release_packages/littraceqa-hidden-test-informed-0.576151
python scripts/build.py
python scripts/verify_release.py --prediction build/test_predictions.jsonl
```

Expected result:

```text
status: PASS
rows: 71
SHA-256: 844C6248E8A353783BE7600050BBB247A7716931B24C3C2C10FD173B04CE6914
```

The scored file is also frozen at `scored_output/test_predictions.jsonl`. A rebuilt file is accepted only if it is byte-identical to that artifact.

On Windows, the same offline workflow is available as:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_reproduce.ps1
```

## Optional official schema validation

The official test inputs are intentionally not committed. To download the public validator inputs from the hash-pinned Hugging Face revision and run the organizer's local schema validator:

```powershell
python scripts/fetch_official_snapshot.py
python scripts/verify_release.py --prediction build/test_predictions.jsonl --official-snapshot .cache/official_snapshot
```

The pinned dataset revision is `bd35dc14cf0483e0ffa51fa2a54d2689c13f9845`. The downloader checks every file hash before making it available.

## Frozen composition

| Component | Frozen artifact | Role |
|---|---|---|
| Parent | `B_blank_invalid_recovery_001` | row order and serialization parent |
| MC | `MC40_round5_verified` | multiple-choice field |
| Paper | `P70_round8_minimal_document_identity` | selected papers |
| Evidence | `E73_round8_minimal_exact` | evidence tuples |
| Table | `TABLE20_round8_strict_verified` | table answers |

The composer copies only those designated fields, preserves the B parent elsewhere, rejects query-order drift, and rejects evidence outside the selected paper set.

## Official aggregate metrics

| Metric | Score |
|---|---:|
| Composite | **0.576151** |
| Paper P / R / F1 | 0.7441 / 0.8509 / 0.7340 |
| Evidence P / R / F1 | 0.4271 / 0.4601 / 0.4236 |
| Multiple-choice accuracy | **1.0000** |
| Table row F1 | 0.4469 |
| Table cell accuracy macro / micro | 0.2659 / 0.3103 |

These are user-reported organizer aggregates. No item-level golden answers were available. See `records/official_score.json` and `PROVENANCE.md` for the complete boundary.

## Verification levels

- `python scripts/build.py`: verifies every frozen input hash, rebuilds the output, and requires the scored SHA-256.
- `python scripts/verify_release.py`: independently checks composition, ordering, containment, and byte identity.
- `python -m unittest discover -s tests -v`: performs a clean temporary-directory rebuild and verification.
- `fetch_official_snapshot.py` plus `--official-snapshot`: additionally runs the pinned official schema validator.
