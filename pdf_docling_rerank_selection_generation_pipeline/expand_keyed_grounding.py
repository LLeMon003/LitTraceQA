"""Expand a keyed prediction's exact citations without changing its answer.

This is a post-generation coverage experiment.  It consumes only public query
metadata, an existing prediction, its internal C-key echo, and the frozen L2
hierarchy.  No validation gold answer, gold evidence, or gold papers are read.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data_io import find_official_file, read_jsonl, write_jsonl
from .evidence_hierarchy import keyed_hierarchy_prompt_projection
from .symbolic_schema import canonicalize_locator, to_official_source_type


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand grounded evidence by one frozen L2 card per query claim.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--predictions-input", required=True)
    parser.add_argument("--internal-predictions-input", required=True)
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cards-single", type=int, default=6)
    parser.add_argument("--max-cards-multi", type=int, default=12)
    parser.add_argument("--expand-papers", choices=["none", "multi", "all"], default="multi")
    return parser.parse_args()


def _official_evidence(record: dict[str, Any]) -> dict[str, Any] | None:
    source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
    paper_id = str(record.get("paper_id") or "")
    if not source_type or not paper_id:
        return None
    locator = canonicalize_locator(source_type, record.get("locator"), record.get("label"))
    if not locator:
        return None
    return {"paper_id": paper_id, "source_type": source_type, "locator": locator}


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("paper_id") or ""),
        str(item.get("source_type") or ""),
        json.dumps(item.get("locator") or {}, ensure_ascii=False, sort_keys=True),
    )


def _valid_card_keys(internal: dict[str, Any], projection: dict[str, Any]) -> list[str]:
    valid = set((projection.get("_card_support_keys") or {}).keys())
    output: list[str] = []
    mapping = internal.get("claim_to_support_keys") if isinstance(internal.get("claim_to_support_keys"), dict) else {}
    for raw_keys in mapping.values():
        for raw_key in raw_keys if isinstance(raw_keys, list) else []:
            key = str(raw_key).strip()
            if key in valid and key not in output:
                output.append(key)
    return output


def expand_prediction(
    prediction: dict[str, Any],
    internal: dict[str, Any],
    hierarchy: dict[str, Any],
    *,
    task_family: str,
    max_cards: int,
    expand_papers: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a coherent prediction with provenance-only citation expansion."""
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    card_map = projection.get("_card_support_keys") or {}
    key_index = projection.get("_key_index") or {}
    cards_by_claim: dict[str, list[str]] = defaultdict(list)
    for card in projection.get("l2_cards") or []:
        key = str(card.get("key") or "")
        if key not in card_map:
            continue
        for claim_id in card.get("claim_ids") or []:
            if key not in cards_by_claim[str(claim_id)]:
                cards_by_claim[str(claim_id)].append(key)

    card_keys = _valid_card_keys(internal, projection)
    mapped_claim_ids = {
        str(claim_id)
        for claim_id, raw_keys in (internal.get("claim_to_support_keys") or {}).items()
        if isinstance(raw_keys, list) and any(str(key).strip() in card_map for key in raw_keys)
    }
    # Every L2 card below was constructed from a frozen selected record and is
    # explicitly assigned to a query claim.  Adding only the first such card
    # gives each claim one independently traceable proof path without flooding
    # the official evidence list with the whole selected corpus.
    for claim in hierarchy.get("query_claims") or []:
        claim_id = str(claim.get("claim_id") or "") if isinstance(claim, dict) else ""
        if claim_id in mapped_claim_ids:
            continue
        for key in cards_by_claim.get(claim_id, []):
            if key not in card_keys:
                card_keys.append(key)
                break
        if max_cards > 0 and len(card_keys) >= max_cards:
            break
    if max_cards > 0:
        card_keys = card_keys[:max_cards]

    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    refs: list[str] = []
    for key in card_keys:
        for record_key in card_map.get(key) or []:
            ref = str((key_index.get(record_key) or {}).get("evidence_ref") or "")
            if ref and ref not in refs:
                refs.append(ref)

    output = copy.deepcopy(prediction)
    evidence = [item for item in output.get("evidence") or [] if isinstance(item, dict)]
    seen = {_evidence_key(item) for item in evidence}
    added = 0
    added_papers: list[str] = []
    for ref in refs:
        item = _official_evidence(catalog.get(ref) or {})
        if not item or _evidence_key(item) in seen:
            continue
        evidence.append(item)
        seen.add(_evidence_key(item))
        added += 1
        paper_id = str(item["paper_id"])
        if paper_id not in added_papers:
            added_papers.append(paper_id)
    output["evidence"] = evidence

    is_multi = "multi" in str(task_family or "").lower()
    if expand_papers == "all" or (expand_papers == "multi" and is_multi):
        papers = [item for item in output.get("gold_papers") or [] if isinstance(item, dict) and item.get("paper_id")]
        known = {str(item.get("paper_id")) for item in papers}
        for paper_id in added_papers:
            if paper_id not in known:
                papers.append({"paper_id": paper_id})
                known.add(paper_id)
        output["gold_papers"] = papers
    return output, {
        "query_id": output.get("query_id"),
        "task_family": task_family,
            "existing_card_count": len(_valid_card_keys(internal, projection)),
        "mapped_claim_count": len(mapped_claim_ids),
        "expanded_card_count": len(card_keys),
        "resolved_l0_count": len(refs),
        "added_evidence_count": added,
        "added_paper_count": len(added_papers),
    }


def main() -> int:
    args = _args()
    inputs = {str(row.get("query_id") or ""): row for row in read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))}
    predictions = {str(row.get("query_id") or ""): row for row in read_jsonl(args.predictions_input)}
    internal = {str(row.get("query_id") or ""): row for row in read_jsonl(args.internal_predictions_input)}
    hierarchies = {
        str(row.get("query_id") or ""): row.get("hierarchy")
        for row in read_jsonl(args.hierarchy_input)
        if isinstance(row.get("hierarchy"), dict)
    }
    output_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for query_id, prediction in predictions.items():
        sample = inputs.get(query_id) or {}
        hierarchy = hierarchies.get(query_id)
        current = internal.get(query_id)
        if not hierarchy or not current:
            output_rows.append(prediction)
            audit.append({"query_id": query_id, "status": "unchanged_missing_internal_or_hierarchy"})
            continue
        is_multi = "multi" in str(sample.get("task_family") or "").lower()
        expanded, row_audit = expand_prediction(
            prediction,
            current,
            hierarchy,
            task_family=str(sample.get("task_family") or ""),
            max_cards=args.max_cards_multi if is_multi else args.max_cards_single,
            expand_papers=args.expand_papers,
        )
        output_rows.append(expanded)
        audit.append({"status": "expanded", **row_audit})
    output = Path(args.output_dir)
    write_jsonl(output / "predictions.jsonl", output_rows)
    write_jsonl(output / "grounding_expansion_audit.jsonl", audit)
    print(json.dumps({"predictions": len(output_rows), "expanded": sum(row.get("status") == "expanded" for row in audit), "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
