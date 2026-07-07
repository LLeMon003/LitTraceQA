from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..data_io import find_official_file, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper retrieval top-k recall against LitTraceQA validation gold papers.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--candidates", required=True, help="Path to candidate_papers.jsonl from metadata-only retrieval.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k cutoff. Defaults to all saved candidates per query.")
    parser.add_argument(
        "--candidate-field",
        choices=["candidates", "selection_candidates"],
        default="candidates",
        help="Candidate list field to evaluate. Use candidates for retrieval; selection_candidates for VLM-2 selection pool diagnostics.",
    )
    parser.add_argument("--include-details", action="store_true", help="Print per-query details in stdout.")
    parser.add_argument("--details-out", default=None, help="Optional path to write per-query retrieval recall details as JSONL.")
    return parser.parse_args()


def _paper_ids(items: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if not isinstance(items, list):
        return ids
    for item in items:
        paper_id = item.get("paper_id") if isinstance(item, dict) else item
        paper_id = str(paper_id or "")
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            ids.append(paper_id)
    return ids


def _gold_by_query(official_dir: str | Path) -> dict[str, list[str]]:
    rows = read_jsonl(find_official_file(official_dir, "validation.jsonl"))
    return {
        str(row.get("query_id") or ""): _paper_ids(row.get("gold_papers") or row.get("papers") or [])
        for row in rows
        if row.get("query_id")
    }


def _candidate_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_query: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            by_query[query_id] = row
    return by_query


def evaluate_retrieval(
    *,
    official_dir: str | Path,
    candidates_path: str | Path,
    top_k: int | None = None,
    candidate_field: str = "candidates",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = _gold_by_query(official_dir)
    candidate_by_id = _candidate_rows(candidates_path)
    details: list[dict[str, Any]] = []
    recalls: list[float] = []
    full_hits = 0
    any_hits = 0
    missing_candidate_rows: list[str] = []

    for query_id, gold_ids in gold_by_id.items():
        row = candidate_by_id.get(query_id)
        if row is None:
            missing_candidate_rows.append(query_id)
            candidate_ids: list[str] = []
            effective_top_k = 0
        else:
            all_candidate_ids = _paper_ids(row.get(candidate_field) or [])
            candidate_ids = all_candidate_ids[:top_k] if top_k is not None else all_candidate_ids
            effective_top_k = len(candidate_ids)
        gold_set = set(gold_ids)
        candidate_set = set(candidate_ids)
        matched = sorted(gold_set & candidate_set)
        recall = len(matched) / len(gold_set) if gold_set else 1.0
        recalls.append(recall)
        if gold_set and gold_set <= candidate_set:
            full_hits += 1
        if matched:
            any_hits += 1
        details.append(
            {
                "query_id": query_id,
                "gold_paper_ids": gold_ids,
                "candidate_paper_ids": candidate_ids,
                "matched_gold_paper_ids": matched,
                "missing_gold_paper_ids": [paper_id for paper_id in gold_ids if paper_id not in candidate_set],
                "retrieval_recall": recall,
                "full_gold_covered": bool(gold_set and gold_set <= candidate_set),
                "any_gold_hit": bool(matched),
                "effective_top_k": effective_top_k,
            }
        )

    total = len(gold_by_id)
    metrics = {
        "query_count": total,
        "candidate_row_count": len(candidate_by_id),
        "missing_candidate_row_count": len(missing_candidate_rows),
        "missing_candidate_rows": missing_candidate_rows,
        "candidate_field": candidate_field,
        "top_k": top_k if top_k is not None else "all_saved",
        "paper_retrieval_recall_macro": round(mean(recalls), 6) if recalls else 0.0,
        "full_gold_coverage_rate": round(full_hits / max(1, total), 6),
        "any_gold_hit_rate": round(any_hits / max(1, total), 6),
    }
    return metrics, details


def main() -> int:
    args = parse_args()
    metrics, details = evaluate_retrieval(
        official_dir=args.official_dir,
        candidates_path=args.candidates,
        top_k=args.top_k,
        candidate_field=args.candidate_field,
    )
    if args.details_out:
        output = Path(args.details_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n", encoding="utf-8")
    payload: dict[str, Any] = {"metrics": metrics}
    if args.include_details:
        payload["details"] = details
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
