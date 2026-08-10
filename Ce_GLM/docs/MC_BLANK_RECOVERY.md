# Multiple-choice blank recovery experiment

`scripts/run_mc_blank_recovery.py` implements the no-gold experiment `V23_MC_BLANK_RECOVERY_001`.

The runner has three explicit stages:

1. `prepare` hashes the immutable fresh prediction, ordered-option input, and full-Docling object index; identifies only blank multiple-choice fields; resolves selected-paper source objects; and writes prompt manifests.
2. `run` calls `deepseek-ai/DeepSeek-V3.2` at temperature zero, validates that every response selects one supplied option letter, and persists each decision atomically for resume.
3. `freeze` requires complete decision coverage, refuses to overwrite any nonblank multiple-choice answer, writes the complete candidate prediction, and records its pre-evaluation hash.

Official reference answers and evaluator feedback are not inputs to prompt preparation or generation. Evaluation must occur only after `MC_CANDIDATE_FREEZE.json` exists.

The Version 2.3 baseline and historical outputs remain immutable. All generated experiment artifacts belong outside the repository in a versioned artifact root.
