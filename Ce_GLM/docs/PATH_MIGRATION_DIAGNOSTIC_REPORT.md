# Path Migration and Runtime Diagnostic

Scope: bounded diagnosis only; no challenger, benchmark, retrieval, candidate, or evaluator work was run.

## Active workspace

The sole active worktree is `D:\Projects\GroundLM 2026\littraceqa_baseline_Ver3__VER3_SOURCE_NATIVE_CHALLENGER_001` on branch `ver3-source-native-challenger-001` at `1b7895f`. It is clean. All six current roots in the registry exist under the authoritative project root. The code archive hash matches its release manifest: `7AFBDA181E80A74B6E77BBBDA72E949338C44FC8A5637CF612E463C4CB926A2C`.

## Path findings

| Classification | Location | Finding | Action |
|---|---|---|---|
| Active valid relocated path | registry entry `VER3_SOURCE_NATIVE_CHALLENGER_001` | Code, artifact, runtime, source-asset, development-output, and release roots all resolve under `D:\Projects\GroundLM 2026`. | Verified. |
| Active broken metadata | registry keys `status`, `latest_checkpoint`, `prediction_sha256`, `protected_or_finalized` | Entry still described active planning despite the completed exhausted program. | Safely repaired to `SOURCE_NATIVE_CHALLENGER_EXHAUSTED`, final checkpoint, frozen-parent hash, and protected state. |
| Historical audit metadata | seven other registry records | Old OneDrive paths occur only in completed historical run records. | Preserved unchanged; never opened. |
| Historical audit metadata | `SESSION_CONTEXT.md`, program plan, final report | OneDrive text documents historical migration constraints, not an active resolver. | Preserved unchanged. |
| Active valid relocated path | release manifest and current scripts | Current active paths are relative/root-derived or supplied explicitly under the relocated root. No stale drive, virtualenv executable, or Ollama executable path is hard-coded. | Verified. |

All required inspected assets exist locally: input-only file, Docling index, fact ledger, synthetic benchmark manifest, frozen parent prediction, and evaluator. No essential migrated artifact is missing. The source-PDF collection was not needed by this diagnostic; any absent optional PDF representation is not the cause of the prior credential/model failure.

## Credential and provider chain

`%USERPROFILE%\.littraceqa\secrets.env` exists. `LITTRACEQA_SECRETS_FILE` is set and resolves to an existing target. The target contains a SiliconFlow credential key, but the current process does not inherit `SILICONFLOW_API_KEY` or `SILICONFLOW_TOKEN`.

The active/current provider scripts use environment-only credential lookup; they do not load the central secrets-file pointer. This is an active runtime configuration gap, not a stale path. Endpoint `https://api.siliconflow.cn/v1` is reachable: unauthenticated connectivity returned 401 and the non-answering authenticated `/models` smoke returned 200. No response body, key, header, or secret substring was logged.

The local Ollama installation is valid: `qwen3-vl:4b` is installed, exposes completion/vision capabilities, and is idle. The earlier timeout was therefore model execution performance under the bounded prompt, not a stale Ollama path or missing model installation.

## Cause assessment and recommended action

The previous failures were not caused by stale OneDrive paths, incomplete artifact migration, missing index/ledger/benchmark/prediction/evaluator assets, or a broken local-model path.

The missing-credential diagnosis was caused by process inheritance/configuration: a valid central secrets file was present but active provider code looked only at inherited environment variables. The local-model failure was a bounded inference timeout. Before any next Goal, add or route all new provider runners through a centralized resolver that honors `LITTRACEQA_SECRETS_FILE` and the default secrets file without logging values; then run one non-answering authenticated provider smoke and one short grounded model smoke with a strict timeout. Do not modify Version 2.3, frozen predictions, gold, evaluator, or historical artifacts.
