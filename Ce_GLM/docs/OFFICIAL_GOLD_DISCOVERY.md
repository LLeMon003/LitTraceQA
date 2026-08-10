# Official gold discovery

Status: `VERIFIED_OFFICIAL_GOLD`.

The accepted typed-MC evaluation record names the exact gold and evaluator paths. Its result matches the accepted Version 2 DEV metrics (Paper F1 1.0, Evidence F1 0.6346464646464647, MC 38/41, Freeform 23/26, Table row F1 1.0, Table cells 25/27). The associated freeze manifest pins the accepted prediction SHA-256.

- Gold: `<WORKSPACE_PARENT>\LitTraceQA\data\validation.jsonl`
- Gold SHA-256: `DAA5ED246C00A5E4BB571843BAA985B6256700DA8A7AE5695BD642DFD4298E41`
- Evaluator: `<WORKSPACE_PARENT>\LitTraceQA\scripts\evaluate.py`
- Evaluator SHA-256: `9410B51E86FC1EA382565D376016152FD52A74FA5D1F9358E36AF54711D8895F`
- Historical proof: `littraceqa_baseline_uq_experiments\outputs\typed_mc_replay\official_evaluation.json`

The gold was inspected by a bounded UTF-8 JSONL schema/count/hash audit. It has 55 records, 55 unique aligned IDs, reference answers, answer types, gold papers, evidence, and 11 evaluator-compatible table references. No answer values were emitted. It is neither the raw question input nor a prediction checkpoint.

Only the first discovery strategy was needed. The earlier search missed the authoritative sibling `LitTraceQA` repository path.
