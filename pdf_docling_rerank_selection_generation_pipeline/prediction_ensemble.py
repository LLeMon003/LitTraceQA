"""Build a field-level prediction ensemble from fully evaluated run artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a deterministic LitTraceQA field-level ensemble.")
    parser.add_argument("--base", required=True, help="Prediction JSONL providing freeform and default evidence.")
    parser.add_argument("--paper-source", required=True, help="Prediction JSONL providing gold_papers.")
    parser.add_argument("--paper-mode", choices=("replace", "union"), default="replace")
    parser.add_argument("--mc-source", required=True, help="Prediction JSONL providing multiple_choice fields.")
    parser.add_argument("--freeform-source", default="", help="Optional prediction JSONL providing freeform fields.")
    parser.add_argument("--table-source", required=True, help="Prediction JSONL providing table answer fields.")
    parser.add_argument("--table-evidence-source", required=True, help="Prediction JSONL providing supplemental table evidence.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _by_query(path: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("query_id") or ""): row for row in read_jsonl(path) if row.get("query_id")}


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("paper_id") or ""),
        str(item.get("source_type") or ""),
        json.dumps(item.get("locator") or {}, ensure_ascii=False, sort_keys=True),
    )


def main() -> int:
    args = _args()
    base = _by_query(args.base)
    paper = _by_query(args.paper_source)
    multiple_choice = _by_query(args.mc_source)
    freeform = _by_query(args.freeform_source) if args.freeform_source else {}
    table = _by_query(args.table_source)
    table_evidence = _by_query(args.table_evidence_source)
    rows: list[dict[str, Any]] = []
    stats = {"paper": 0, "multiple_choice": 0, "freeform": 0, "table": 0, "table_evidence": 0}

    for query_id in sorted(base):
        row = dict(base[query_id])
        proposed_papers = paper.get(query_id, {}).get("gold_papers")
        if isinstance(proposed_papers, list) and proposed_papers:
            if args.paper_mode == "union":
                paper_ids = {
                    str(item.get("paper_id") or "")
                    for item in [*(row.get("gold_papers") or []), *proposed_papers]
                    if isinstance(item, dict) and item.get("paper_id")
                }
                row["gold_papers"] = [{"paper_id": paper_id} for paper_id in sorted(paper_ids)]
            else:
                row["gold_papers"] = proposed_papers
            stats["paper"] += 1

        answer = dict(row.get("answer") or {})
        proposed_mc = (multiple_choice.get(query_id, {}).get("answer") or {}).get("multiple_choice")
        if isinstance(proposed_mc, dict) and proposed_mc.get("gold"):
            answer["multiple_choice"] = proposed_mc
            stats["multiple_choice"] += 1
        proposed_freeform = (freeform.get(query_id, {}).get("answer") or {}).get("freeform")
        if isinstance(proposed_freeform, dict) and str(proposed_freeform.get("text") or "").strip():
            answer["freeform"] = proposed_freeform
            stats["freeform"] += 1
        proposed_table = (table.get(query_id, {}).get("answer") or {}).get("table")
        if isinstance(proposed_table, dict):
            answer["table"] = proposed_table
            stats["table"] += 1
        row["answer"] = answer

        evidence = [item for item in row.get("evidence") or [] if isinstance(item, dict)]
        seen = {_evidence_key(item) for item in evidence}
        for item in table_evidence.get(query_id, {}).get("evidence") or []:
            if not isinstance(item, dict) or str(item.get("source_type") or "") != "table":
                continue
            key = _evidence_key(item)
            if key not in seen:
                evidence.append(item)
                seen.add(key)
                stats["table_evidence"] += 1
        row["evidence"] = evidence
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    print(json.dumps({"rows": len(rows), "replacements": stats, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
