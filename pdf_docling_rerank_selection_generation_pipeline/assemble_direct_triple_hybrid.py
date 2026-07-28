"""Conservatively apply direct, grounded triple answers to a frozen baseline.

This is an answer-stage overlay, not a gold-aware selector.  It only replaces
a baseline prediction when the public answer contract is freeform-only and the
triple hierarchy's semantic gate accepted a directly grounded answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_io import find_official_file, read_jsonl, write_jsonl


TRUSTED_STATUSES = {
    "deterministic_table_relation",
    "deterministic_citation_relation",
    "deterministic_hardware_relation",
    "verified_llm_relation",
    "visual_crop_verified_relation",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay direct triple-grounded freeform answers onto a frozen baseline.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--triple-predictions", required=True)
    parser.add_argument("--triple-internal-predictions", required=True)
    parser.add_argument("--triple-hierarchy", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _by_query(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("query_id") or ""): row for row in read_jsonl(path) if str(row.get("query_id") or "")}


def _eligible(hierarchy: dict[str, Any], internal: dict[str, Any], triple: dict[str, Any]) -> bool:
    gate = hierarchy.get("triple_sufficiency") if isinstance(hierarchy.get("triple_sufficiency"), dict) else {}
    if not gate.get("sufficient") or hierarchy.get("triple_prompt_policy") != "accepted_triples_first":
        return False
    if not any(
        str((row.get("verification") or {}).get("status") or "") in TRUSTED_STATUSES
        for row in hierarchy.get("l2_contextual_triples") or [] if isinstance(row, dict)
    ):
        return False
    answer = triple.get("answer") if isinstance(triple.get("answer"), dict) else {}
    text = ((answer.get("freeform") or {}).get("text") if isinstance(answer.get("freeform"), dict) else "")
    if not str(text or "").strip() or not (triple.get("evidence") or []):
        return False
    audits = internal.get("posthoc_grounding_audit") if isinstance(internal.get("posthoc_grounding_audit"), list) else []
    return any(item.get("field") == "answer.freeform.text" and item.get("status") == "directly_grounded" for item in audits if isinstance(item, dict))


def main() -> int:
    args = _args()
    inputs = _by_query(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    base = _by_query(args.base_predictions)
    triples = _by_query(args.triple_predictions)
    internal = _by_query(args.triple_internal_predictions)
    hierarchies = {
        str(row.get("query_id") or ""): dict(row.get("hierarchy") or {})
        for row in read_jsonl(args.triple_hierarchy) if isinstance(row.get("hierarchy"), dict)
    }
    rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for query_id, row in base.items():
        merged = dict(row)
        contract = inputs.get(query_id) or {}
        triple = triples.get(query_id) or {}
        if list(contract.get("answer_types") or []) == ["freeform"] and _eligible(hierarchies.get(query_id) or {}, internal.get(query_id) or {}, triple):
            merged["answer"] = triple["answer"]
            merged["evidence"] = list(triple.get("evidence") or [])
            changes.append({"query_id": query_id, "reason": "trusted_direct_triple_freeform", "evidence_count": len(merged["evidence"])})
        rows.append(merged)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", rows)
    write_jsonl(output / "overlay_changes.jsonl", changes)
    (output / "provenance.json").write_text(json.dumps({
        "base_predictions": str(args.base_predictions), "triple_predictions": str(args.triple_predictions),
        "triple_hierarchy": str(args.triple_hierarchy), "change_count": len(changes),
        "policy": "freeform-only + sufficient gate + trusted triple + posthoc direct grounding",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": len(rows), "changes": len(changes), "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
