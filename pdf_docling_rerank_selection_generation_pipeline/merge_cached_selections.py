"""Union frozen selection artifacts without reranking or dropping provenance."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge cached symbolic selections as a provenance-preserving union.")
    parser.add_argument("--selected-contexts-input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or record.get("record_id") or "")


def merge(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Keep the first projection of each logical record and record every route."""
    by_query: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        source_name = path.parent.name
        for row in read_jsonl(path):
            query_id = str(row.get("query_id") or "")
            if query_id:
                by_query[query_id].append((source_name, row))
    merged: list[dict[str, Any]] = []
    for query_id in sorted(by_query):
        records: list[dict[str, Any]] = []
        sources_by_record: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        source_summaries: list[dict[str, Any]] = []
        for source_name, row in by_query[query_id]:
            source_summaries.append({
                "source": source_name,
                "selection_method": row.get("selection_method"),
                "package_selection": row.get("package_selection") or {},
            })
            for raw in row.get("selected_records") or []:
                if not isinstance(raw, dict):
                    continue
                identifier = _record_id(raw)
                if not identifier:
                    continue
                sources_by_record[identifier].append(source_name)
                if identifier in seen:
                    continue
                seen.add(identifier)
                record = dict(raw)
                record["cached_selection_sources"] = [source_name]
                records.append(record)
        for record in records:
            record["cached_selection_sources"] = list(dict.fromkeys(sources_by_record[_record_id(record)]))
        merged.append({
            "query_id": query_id,
            "selection_method": "cached_selection_provenance_union",
            "selected_record_count": len(records),
            "selected_records": records,
            "selected_context_groups": [],
            "package_selection": {
                "mode": "union",
                "selection_source_count": len(source_summaries),
                "selection_sources": source_summaries,
                "selected_record_count": len(records),
            },
        })
    return merged


def main() -> int:
    args = _args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = merge(args.selected_contexts_input)
    write_jsonl(output / "selected_symbolic_contexts.debug.jsonl", rows)
    print(json.dumps({"queries": len(rows), "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
