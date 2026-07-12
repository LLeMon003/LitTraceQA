from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..data_io import find_official_file, read_jsonl


PAGE_FIELD_CHOICES = {
    "auto",
    "final_parsed_pages",
    "initial_selected_pages",
    "expanded_pages",
    "selected_pages_final",
    "selected_pages_initial",
    "ranked_pages",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether pages containing official LitTraceQA validation gold evidence "
            "are covered by ranked/selected page retrieval outputs."
        )
    )
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument(
        "--pages",
        required=True,
        help="Path to global_page_parse_plan.jsonl or global_page_ranking.jsonl.",
    )
    parser.add_argument(
        "--page-field",
        choices=sorted(PAGE_FIELD_CHOICES),
        default="auto",
        help=(
            "Page list field to evaluate. auto prefers final_parsed_pages, then selected_pages_final, "
            "initial_selected_pages, selected_pages_initial, and finally ranked_pages."
        ),
    )
    parser.add_argument(
        "--ranked-mode",
        choices=["selected", "all"],
        default="selected",
        help="When evaluating ranked_pages, use only selected_for_initial_parse pages or all ranked pages.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional cutoff applied to the evaluated page list after field/mode selection.",
    )
    parser.add_argument("--include-details", action="store_true", help="Print per-query details in stdout.")
    parser.add_argument("--details-out", default=None, help="Optional path to write per-query page recall details as JSONL.")
    return parser.parse_args()


def _as_page(value: Any) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _gold_evidence_by_query(official_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(find_official_file(official_dir, "validation.jsonl"))
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            continue
        evidence_rows: list[dict[str, Any]] = []
        for index, evidence in enumerate(row.get("evidence") or [], start=1):
            if not isinstance(evidence, dict):
                continue
            locator = evidence.get("locator") if isinstance(evidence.get("locator"), dict) else {}
            paper_id = str(evidence.get("paper_id") or "")
            page = _as_page(locator.get("page"))
            if not paper_id or page is None:
                continue
            evidence_rows.append(
                {
                    "evidence_index": index,
                    "evidence_id": evidence.get("evidence_id"),
                    "paper_id": paper_id,
                    "page": page,
                    "source_type": evidence.get("source_type"),
                    "locator": locator,
                }
            )
        by_query[query_id] = evidence_rows
    return by_query


def _page_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_query: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            by_query[query_id] = row
    return by_query


def _auto_page_field(row: dict[str, Any]) -> str:
    for field in (
        "final_parsed_pages",
        "selected_pages_final",
        "initial_selected_pages",
        "selected_pages_initial",
        "ranked_pages",
        "expanded_pages",
    ):
        if isinstance(row.get(field), list):
            return field
    return "ranked_pages"


def _selected_pages(
    row: dict[str, Any] | None,
    *,
    page_field: str,
    ranked_mode: str,
    top_k: int | None,
) -> tuple[list[dict[str, Any]], str]:
    if not row:
        return [], page_field
    resolved_field = _auto_page_field(row) if page_field == "auto" else page_field
    pages = row.get(resolved_field)
    if not isinstance(pages, list):
        return [], resolved_field
    selected: list[dict[str, Any]] = []
    for item in pages:
        if not isinstance(item, dict):
            continue
        if resolved_field == "ranked_pages" and ranked_mode == "selected" and not item.get("selected_for_initial_parse"):
            continue
        paper_id = str(item.get("paper_id") or "")
        page = _as_page(item.get("page"))
        if not paper_id or page is None:
            continue
        selected.append(item)
    if top_k is not None:
        selected = selected[: max(0, top_k)]
    return selected, resolved_field


def evaluate_page_retrieval(
    *,
    official_dir: str | Path,
    pages_path: str | Path,
    page_field: str = "auto",
    ranked_mode: str = "selected",
    top_k: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = _gold_evidence_by_query(official_dir)
    page_by_id = _page_rows(pages_path)
    details: list[dict[str, Any]] = []
    query_recalls: list[float] = []
    total_gold_evidence = 0
    matched_gold_evidence = 0
    full_hits = 0
    any_hits = 0
    missing_page_rows: list[str] = []
    field_counts: dict[str, int] = {}

    for query_id, gold_evidence in gold_by_id.items():
        row = page_by_id.get(query_id)
        if row is None:
            missing_page_rows.append(query_id)
        selected_pages, resolved_field = _selected_pages(row, page_field=page_field, ranked_mode=ranked_mode, top_k=top_k)
        field_counts[resolved_field] = field_counts.get(resolved_field, 0) + 1
        selected_keys = {
            (str(page.get("paper_id") or ""), int(page.get("page") or 0))
            for page in selected_pages
            if page.get("paper_id") and _as_page(page.get("page")) is not None
        }
        matched: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for evidence in gold_evidence:
            key = (str(evidence.get("paper_id") or ""), int(evidence.get("page") or 0))
            if key in selected_keys:
                matched.append(evidence)
            else:
                missing.append(evidence)

        gold_count = len(gold_evidence)
        matched_count = len(matched)
        total_gold_evidence += gold_count
        matched_gold_evidence += matched_count
        recall = matched_count / gold_count if gold_count else 1.0
        query_recalls.append(recall)
        if gold_count and matched_count == gold_count:
            full_hits += 1
        if matched_count:
            any_hits += 1
        details.append(
            {
                "query_id": query_id,
                "page_field": resolved_field,
                "ranked_mode": ranked_mode if resolved_field == "ranked_pages" else None,
                "top_k": top_k if top_k is not None else "all_selected",
                "gold_evidence_count": gold_count,
                "matched_gold_evidence_count": matched_count,
                "gold_evidence_page_recall": recall,
                "selected_page_count": len(selected_pages),
                "selected_pages": [
                    {
                        "paper_id": str(page.get("paper_id") or ""),
                        "page": int(page.get("page") or 0),
                        "global_page_rank": page.get("global_page_rank"),
                        "candidate_rank": page.get("candidate_rank"),
                        "selection_rank": page.get("selection_rank"),
                    }
                    for page in selected_pages
                ],
                "matched_gold_evidence": matched,
                "missing_gold_evidence": missing,
            }
        )

    query_count = len(gold_by_id)
    metrics = {
        "query_count": query_count,
        "page_row_count": len(page_by_id),
        "missing_page_row_count": len(missing_page_rows),
        "missing_page_rows": missing_page_rows,
        "page_field": page_field,
        "resolved_page_field_counts": field_counts,
        "ranked_mode": ranked_mode,
        "top_k": top_k if top_k is not None else "all_selected",
        "gold_evidence_count": total_gold_evidence,
        "matched_gold_evidence_count": matched_gold_evidence,
        "gold_evidence_page_recall_micro": round(matched_gold_evidence / max(1, total_gold_evidence), 6),
        "gold_evidence_page_recall_macro": round(mean(query_recalls), 6) if query_recalls else 0.0,
        "full_gold_evidence_page_coverage_rate": round(full_hits / max(1, query_count), 6),
        "any_gold_evidence_page_hit_rate": round(any_hits / max(1, query_count), 6),
    }
    return metrics, details


def main() -> int:
    args = parse_args()
    metrics, details = evaluate_page_retrieval(
        official_dir=args.official_dir,
        pages_path=args.pages,
        page_field=args.page_field,
        ranked_mode=args.ranked_mode,
        top_k=args.top_k,
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
