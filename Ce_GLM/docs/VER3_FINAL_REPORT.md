# Ver3 Cache-Exact Complete Solution

## Solution identity

This repository packages VER3_CACHE_EXACT_COMPLETE_SOLUTION_001 as a complete, portable, deterministic solution. It comprises a sealed source-boundary bundle, six verified checkpoint replays, an immutable typed-MC renderer, a frozen 55-record prediction, a recorded authoritative evaluation, reproducibility scripts, focused tests, and SHA-256 manifests.

The solution classification is VERIFIED_CACHE_EXACT_DIAGNOSTIC_NOT_FRESH_GENERATION. This is a factual provenance boundary: the solution deterministically reproduces the complete released result from its sealed cache boundary; it does not claim to reconstruct the unavailable historical producer edge before that boundary.

## Verified evaluator result

| Metric | Result |
| --- | ---: |
| Paper F1 | 1.000000 |
| Evidence F1 | 0.634646 |
| MC | 38/41 (0.926829) |
| Freeform | 23/26 (0.884615) |
| Table row F1 | 1.000000 |
| Table-cell macro | 0.954545 |
| Table-cell micro | 0.925926 (25/27) |

The frozen 55-record prediction SHA-256 is 2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364.

## Architecture

The solution has an explicit deterministic stage graph:

1. cache_boundary
2. table_and_evidence_ancestry
3. freeform_completeness
4. option_aware_mc
5. evidence_safe_cleanup
6. source_grounded_mc
7. typed_mc
8. final_prediction

Stages 1 through 6 are sealed checkpoint replays. Stage 7 executes the immutable typed-MC renderer against declared semantic-answer, option, and sentence-index inputs. Stage 8 serializes the newly produced output byte-for-byte. Every stage validates its SHA-256 and requires exactly 55 unique query IDs.

The sealed artifact manifest declares no final-prediction input, no gold use during generation, and no raw provider response. The portable runner uses no credentials, model providers, API calls, or evaluator calls.

## Completeness and reproducibility

The complete release archive is release_packages/VER3_CACHE_EXACT_COMPLETE_SOLUTION_001.zip. It includes:

- release source, scripts, requirements, and focused tests;
- the manifest-verified sealed bundle;
- independently reproduced frozen prediction and run manifest;
- authoritative aggregate evaluation record;
- SHA-256 manifests;
- one-command PowerShell replay wrapper and runbook.

The archive has been tested from a new temporary extraction using Python 3.12. The wrapper reproduced the 55-record frozen prediction hash with zero provider/API calls and zero evaluator calls. The official evaluator and gold contents are intentionally not distributed; the archive includes their recorded hashes and aggregate evaluation result instead.

## Operational use

After extracting the archive, run:

~~~powershell
./RUN_REPRODUCE.ps1 -OutputRoot ../ver3-output
~~~

The output root must be absent or empty. The wrapper fails closed if package contents are missing, if the output directory is non-empty, if the deterministic renderer fails, or if the produced prediction SHA differs from the frozen target.

## Security boundary

The release excludes credentials, authorization headers, user-local secrets, raw provider responses, and official gold contents. It contains only allowlisted sealed checkpoints, typed inputs, deterministic renderer sources, prediction, aggregate evaluator output, and integrity metadata.

## Improvement opportunities

The selected solution is complete as a deterministic release, while the following improvements would increase its portability and scientific strength without changing its current result:

1. Preserve or reconstruct the producer edge before the sealed cache boundary, with source-only provenance and the same hash-lock discipline.
2. Add an independently maintained evaluator harness that can run against separately provisioned official evaluator/gold assets and verify their hashes before scoring.
3. Extend source-native locator validation for table cells and evidence objects, keeping the existing fail-closed policy.
4. Add cross-platform wrapper coverage alongside the PowerShell entry point.
5. Add reproducibility CI that verifies manifest hashes, test coverage, and deterministic prediction SHA without accessing gold or credentials.

## Final status

VER3_CACHE_EXACT_COMPLETE_SOLUTION_001 is the independent, complete Ver3 solution in the current branch. It is portable, hash-verified, score-documented, and safe to reproduce from its sealed bundle. Its published evaluator metrics are the values stated above; its cache-exact provenance remains explicit and unchanged.

