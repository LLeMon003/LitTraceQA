Experiment ID:
V2.3_API_REPAIR_CAUSAL_ABLATION_AND_RELEASE_PACKAGING

External API status:
Verified with bounded SiliconFlow requests.
Previous blocker:
Network-route transition between explicit localhost proxy and TUN transparent routing.
Successful smoke test:
current-c0/q_006, reused as experiment record.
Current model:
deepseek-ai/DeepSeek-V4-Flash
Stronger model:
deepseek-ai/DeepSeek-V3.2
Pilot records:
12
API calls:
35 successful generation outputs; within 42-record ceiling.
Option-aware effect:
Analyzed on matched structural outputs only.
Localization effect:
Analyzed on 11 matched C0/C1 pairs after separating network missingness.
Stronger-model effect:
Analyzed on 11 matched C1/stronger-C1 pairs.
Stability result:
Not triggered.
Model decision:
STRONGER_MODEL_ONLY_FOR_SELECTED_FAMILIES
Cached-exact status:
Byte-exact DEV and production replay complete.
Fresh-profile status:
Historical fresh complete; recommended experimental profile documented separately.
Official evaluation status:
Authoritative locked official-gold evaluation complete for reassembled fresh prediction.
Code-only bundle:
dist/littraceqa_baseline_Ver.2.3_reproduction_code_only.zip
Internal bundle:
dist/littraceqa_baseline_Ver.2.3_reproduction_internal.zip
Clean-room validation:
See records/CLEAN_ROOM_VALIDATION.json.
Security audit:
See records/RELEASE_SECURITY_AUDIT.json.
Immutable verification:
See records/FINAL_IMMUTABILITY_RESULT.json.
Remaining blocker:
One current-c1 pilot pair remains infrastructure-missing; historical raw-to-base-cache producer edge remains disclosed.
Next recommendation:
Use cached-exact profiles for deterministic reproduction; use recommended fresh experimental profile only as a documented pilot-level improvement path.
