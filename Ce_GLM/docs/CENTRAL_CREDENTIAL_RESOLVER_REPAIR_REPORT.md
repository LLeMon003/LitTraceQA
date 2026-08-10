# Central Credential Resolver Repair

## Scope

This repair touched only the active Ver3 workspace. No Version 2.3 source, frozen prediction, gold, evaluator, historical artifact, benchmark, retrieval, candidate, or evaluator workflow was changed or run.

## Files inspected and changed

- Inspected: the historical generic resolver and its tests (read-only reference), current local-model smoke utility, active provider environment references, and provider example configuration.
- Added: `src/credential_resolver.py`, `tests/test_credential_resolver.py`, and `scripts/run_central_provider_smoke.py`.

The new smoke utility is the active Ver3 provider entry point. Current inherited Version 2.3 scripts remain untouched by policy.

## Final precedence contract

1. Non-empty inherited `SILICONFLOW_API_KEY`, `SILICONFLOW_TOKEN`, or `SILICONFLOW_KEY`.
2. File named by `LITTRACEQA_SECRETS_FILE`.
3. `%USERPROFILE%\.littraceqa\secrets.env`.

The resolver does not require or consult a project-local `.env`, does not mutate process environment variables, and returns only a non-secret status: presence, variable name, source category, endpoint, and model. Missing credentials raise a fixed non-secret diagnostic.

## Validation

- Focused resolver tests: 5 passed. They cover inherited precedence, configured-file resolution, default-file resolution, missing file/key behavior, and absence of values in status/errors/representation.
- Credential status: resolved from `configured_secret_file` without displaying a value.
- Authenticated non-answering SiliconFlow `/models` smoke: passed.
- Tiny structured request: passed; the returned JSON `label` was deterministically verified as an exact value in the supplied minimal snippet. Raw provider response is external-only at `littraceqa_runtime/VER3_SOURCE_NATIVE_CHALLENGER_001/credential_resolver_tiny_smoke_001`.
- Fresh-process smoke: passed after explicitly removing inherited provider credential variables; it resolved the configured central file without manual export.
- `git diff --check` and Python compilation passed.

No secret, authorization header, API key, or response body was printed, committed, or written into repository artifacts.

## Outcome

The prior missing-credential failure was a provider-runner configuration defect, not a migration/path issue. The central credential chain is now ready for `VER3_TARGET_ALIGNED_MODEL_CHALLENGER_001`; that next research Goal was not started by this repair.
