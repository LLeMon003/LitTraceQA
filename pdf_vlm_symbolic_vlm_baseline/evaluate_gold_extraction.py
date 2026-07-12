from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .data_io import find_official_file, read_jsonl
from .source_type_hints import infer_source_type_hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate gold evidence survival before VLM-2: paper retrieval, page ranking/selection, "
            "parser coverage, and final symbolic context retention."
        )
    )
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", required=True, help="Baseline output directory to evaluate.")
    parser.add_argument(
        "--selected-page-field",
        choices=["final_parsed_pages", "initial_selected_pages", "selected_pages_final", "selected_pages_initial"],
        default="final_parsed_pages",
        help="Which selected/parsed page list to treat as the pages that survived page selection.",
    )
    parser.add_argument("--include-details", action="store_true", help="Print per-query and per-evidence details.")
    parser.add_argument("--details-out", default=None, help="Optional path to write per-evidence details as JSONL.")
    parser.add_argument("--use-source-type-hints", action="store_true", help="Treat source_type_hints as compact alternate typed records for parser/context coverage diagnostics.")
    return parser.parse_args()


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _as_page(value: Any) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tokens(value: Any) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(value or "")) if len(token) > 1}


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


def _row_by_query(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row.get("query_id") or ""): row for row in rows if row.get("query_id")}


def _gold_rows(official_dir: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = read_jsonl(find_official_file(official_dir, "validation.jsonl"))
    gold_by_query: dict[str, dict[str, Any]] = {}
    evidence_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            continue
        gold_by_query[query_id] = row
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
                    "query_id": query_id,
                    "evidence_index": index,
                    "evidence_id": evidence.get("evidence_id"),
                    "paper_id": paper_id,
                    "page": page,
                    "source_type": str(evidence.get("source_type") or ""),
                    "evidence_text_or_value": str(evidence.get("evidence_text_or_value") or ""),
                    "locator": locator,
                }
            )
        evidence_by_query[query_id] = evidence_rows
    return gold_by_query, evidence_by_query


def _candidate_papers(row: dict[str, Any] | None) -> set[str]:
    if not row:
        return set()
    return set(_paper_ids(row.get("candidates") or []))


def _page_keys_from_list(items: Any, *, only_selected_ranked: bool = False) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not isinstance(items, list):
        return keys
    for item in items:
        if not isinstance(item, dict):
            continue
        if only_selected_ranked and not item.get("selected_for_initial_parse"):
            continue
        paper_id = str(item.get("paper_id") or "")
        page = _as_page(item.get("page"))
        if paper_id and page is not None:
            keys.add((paper_id, page))
    return keys


def _ranked_page_keys(row: dict[str, Any] | None) -> set[tuple[str, int]]:
    if not row:
        return set()
    return _page_keys_from_list(row.get("ranked_pages") or [])


def _selected_page_keys(row: dict[str, Any] | None, field: str) -> set[tuple[str, int]]:
    if not row:
        return set()
    if isinstance(row.get(field), list):
        return _page_keys_from_list(row.get(field))
    for fallback in ("final_parsed_pages", "selected_pages_final", "initial_selected_pages", "selected_pages_initial"):
        if isinstance(row.get(fallback), list):
            return _page_keys_from_list(row.get(fallback))
    return set()


def _records_by_page(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_page
    for record in read_jsonl(path):
        paper_id = str(record.get("paper_id") or "")
        page = _as_page(record.get("page") or (record.get("locator") or {}).get("page"))
        if paper_id and page is not None:
            by_page[(paper_id, page)].append(record)
    return by_page


def _selected_records_by_query(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return by_query
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        records = row.get("selected_records") or row.get("selected_records_debug") or []
        by_query[query_id] = records if isinstance(records, list) else []
    return by_query


def _record_label_values(record: dict[str, Any]) -> list[str]:
    values = [
        record.get("label"),
        (record.get("locator") or {}).get("table_id") if isinstance(record.get("locator"), dict) else None,
        (record.get("locator") or {}).get("figure_id") if isinstance(record.get("locator"), dict) else None,
        (record.get("locator") or {}).get("equation_id") if isinstance(record.get("locator"), dict) else None,
        (record.get("locator") or {}).get("algorithm_id") if isinstance(record.get("locator"), dict) else None,
        (record.get("locator") or {}).get("citation_id") if isinstance(record.get("locator"), dict) else None,
        (record.get("locator") or {}).get("reference_id") if isinstance(record.get("locator"), dict) else None,
    ]
    grounding = record.get("grounding_label")
    if isinstance(grounding, dict):
        values.append(grounding.get("value"))
    return [str(value) for value in values if value is not None and str(value)]


def _record_text(record: dict[str, Any]) -> str:
    parts = [str(record.get("text") or "")]
    parts.extend(_record_label_values(record))
    return " ".join(parts)


def _id_match(gold_value: Any, record: dict[str, Any]) -> bool:
    target = _norm(gold_value)
    if not target:
        return False
    for value in _record_label_values(record):
        if _norm(value) == target:
            return True
    return target in _norm(_record_text(record))


def _citation_match(gold_value: Any, record: dict[str, Any]) -> bool:
    target = str(gold_value or "").strip()
    if not target:
        return False
    if _id_match(target, record):
        return True
    return bool(re.search(rf"(?:^|\n|\s)\[{re.escape(target)}\]\s+", str(record.get("text") or "")))


def _text_match(gold_text: str, record: dict[str, Any]) -> bool:
    gold = str(gold_text or "").strip()
    text = str(record.get("text") or "")
    if not gold:
        return True
    if _norm(gold) and _norm(gold) in _norm(text):
        return True
    gold_tokens = _tokens(gold)
    record_tokens = _tokens(text)
    if not gold_tokens or not record_tokens:
        return False
    overlap = len(gold_tokens & record_tokens)
    return overlap >= min(5, len(gold_tokens)) or overlap / max(1, len(gold_tokens)) >= 0.55


def _matches_gold_evidence(evidence: dict[str, Any], record: dict[str, Any]) -> bool:
    if str(record.get("paper_id") or "") != str(evidence.get("paper_id") or ""):
        return False
    if _as_page(record.get("page") or (record.get("locator") or {}).get("page")) != int(evidence.get("page") or 0):
        return False

    source_type = str(evidence.get("source_type") or "")
    record_source = str(record.get("source_type") or "")
    locator = evidence.get("locator") if isinstance(evidence.get("locator"), dict) else {}
    if source_type and record_source and source_type != record_source:
        # Table/figure/equation/citation evidence should survive as same source type.
        if source_type != "text_span":
            return False

    if source_type == "table":
        return _id_match(locator.get("table_id"), record) or (record_source == "table" and _text_match(evidence.get("evidence_text_or_value", ""), record))
    if source_type == "figure":
        return _id_match(locator.get("figure_id"), record) or record_source == "figure"
    if source_type == "citation_context":
        return _citation_match(locator.get("citation_id") or locator.get("reference_id"), record) or _text_match(evidence.get("evidence_text_or_value", ""), record)
    if source_type == "equation_algorithm":
        return (
            _id_match(locator.get("algorithm_id"), record)
            or _id_match(locator.get("equation_id"), record)
            or (record_source == "equation_algorithm" and _text_match(evidence.get("evidence_text_or_value", ""), record))
        )
    return _text_match(evidence.get("evidence_text_or_value", ""), record)


def _hint_virtual_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    virtual: list[dict[str, Any]] = []
    hints = record.get("source_type_hints")
    if not isinstance(hints, list):
        hints = infer_source_type_hints(record)
    for hint in hints or []:
        if not isinstance(hint, dict):
            continue
        copied = dict(record)
        copied["source_type"] = str(hint.get("source_type") or copied.get("source_type") or "")
        if hint.get("label"):
            copied["label"] = hint.get("label")
        locator = dict(copied.get("locator") or {})
        hint_locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
        locator.update({key: value for key, value in hint_locator.items() if value not in {None, ""}})
        copied["locator"] = locator
        copied["global_record_id"] = f"{record.get('global_record_id') or record.get('record_id')}__hint_{copied['source_type']}"
        copied.pop("source_type_hints", None)
        virtual.append(copied)
    return virtual


def _find_matching_records(evidence: dict[str, Any], records: list[dict[str, Any]], *, use_source_type_hints: bool = False) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if _matches_gold_evidence(evidence, record):
            matches.append(record)
            continue
        if use_source_type_hints:
            matches.extend(virtual for virtual in _hint_virtual_records(record) if _matches_gold_evidence(evidence, virtual))
    return matches


def _ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def _mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def evaluate_gold_extraction(
    *,
    official_dir: str | Path,
    output_dir: str | Path,
    selected_page_field: str = "final_parsed_pages",
    use_source_type_hints: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = Path(output_dir)
    gold_by_query, evidence_by_query = _gold_rows(official_dir)
    candidates_by_query = _row_by_query(output / "candidate_papers.jsonl")
    ranking_by_query = _row_by_query(output / "global_page_ranking.jsonl")
    parse_plan_by_query = _row_by_query(output / "global_page_parse_plan.jsonl")
    runtime_by_page = _records_by_page(output / "symbolic_records.runtime.jsonl")
    selected_records_by_query = _selected_records_by_query(output / "selected_symbolic_contexts.debug.jsonl")

    evidence_details: list[dict[str, Any]] = []
    query_details: list[dict[str, Any]] = []

    counters = Counter()
    macro = defaultdict(list)

    for query_id, gold_row in gold_by_query.items():
        gold_papers = set(_paper_ids(gold_row.get("gold_papers") or gold_row.get("papers") or []))
        candidate_papers = _candidate_papers(candidates_by_query.get(query_id))
        ranked_pages = _ranked_page_keys(ranking_by_query.get(query_id))
        selected_pages = _selected_page_keys(parse_plan_by_query.get(query_id), selected_page_field)
        selected_records = selected_records_by_query.get(query_id, [])

        matched_gold_papers = gold_papers & candidate_papers
        counters["gold_papers"] += len(gold_papers)
        counters["matched_gold_papers"] += len(matched_gold_papers)
        paper_recall = len(matched_gold_papers) / len(gold_papers) if gold_papers else 1.0
        macro["paper_recall"].append(paper_recall)

        q_counts = Counter()
        for evidence in evidence_by_query.get(query_id, []):
            page_key = (str(evidence["paper_id"]), int(evidence["page"]))
            page_ranked = page_key in ranked_pages
            page_selected = page_key in selected_pages
            page_records = runtime_by_page.get(page_key, [])
            parser_page_has_records = bool(page_records)
            parser_matching_records = _find_matching_records(evidence, page_records, use_source_type_hints=use_source_type_hints) if page_selected else []
            context_matching_records = _find_matching_records(evidence, selected_records, use_source_type_hints=use_source_type_hints)
            parser_survived = bool(parser_matching_records)
            context_retained = bool(context_matching_records)

            counters["gold_evidence"] += 1
            q_counts["gold_evidence"] += 1
            if page_ranked:
                counters["gold_evidence_in_ranked_pool"] += 1
                q_counts["gold_evidence_in_ranked_pool"] += 1
            if page_selected:
                counters["gold_evidence_page_selected"] += 1
                q_counts["gold_evidence_page_selected"] += 1
            if page_ranked and page_selected:
                counters["selector_hits"] += 1
                q_counts["selector_hits"] += 1
            if page_selected:
                counters["parser_denominator_selected_gold_evidence"] += 1
                q_counts["parser_denominator_selected_gold_evidence"] += 1
                if parser_page_has_records:
                    counters["selected_gold_pages_with_any_symbolic_records"] += 1
                if parser_survived:
                    counters["parser_hits"] += 1
                    q_counts["parser_hits"] += 1
            if page_selected and parser_survived:
                counters["context_denominator_parser_survived_gold_evidence"] += 1
                q_counts["context_denominator_parser_survived_gold_evidence"] += 1
                if context_retained:
                    counters["context_hits"] += 1
                    q_counts["context_hits"] += 1

            evidence_details.append(
                {
                    "query_id": query_id,
                    "evidence_index": evidence["evidence_index"],
                    "evidence_id": evidence.get("evidence_id"),
                    "paper_id": evidence["paper_id"],
                    "page": evidence["page"],
                    "source_type": evidence.get("source_type"),
                    "gold_page_in_ranked_pool": page_ranked,
                    "gold_page_selected": page_selected,
                    "parser_page_has_any_records": parser_page_has_records,
                    "parser_matching_record_count": len(parser_matching_records),
                    "parser_survived": parser_survived,
                    "context_matching_record_count": len(context_matching_records),
                    "context_retained": context_retained,
                    "matching_runtime_record_ids": [record.get("global_record_id") or record.get("record_id") for record in parser_matching_records[:10]],
                    "matching_context_record_ids": [record.get("global_record_id") or record.get("record_id") for record in context_matching_records[:10]],
                    "locator": evidence.get("locator"),
                    "evidence_text_or_value": evidence.get("evidence_text_or_value"),
                }
            )

        ranked_den = q_counts["gold_evidence_in_ranked_pool"]
        selected_den = q_counts["parser_denominator_selected_gold_evidence"]
        parser_den = q_counts["context_denominator_parser_survived_gold_evidence"]
        macro["rank_pool_recall"].append(q_counts["gold_evidence_in_ranked_pool"] / q_counts["gold_evidence"] if q_counts["gold_evidence"] else 1.0)
        macro["selector_recall"].append(q_counts["selector_hits"] / ranked_den if ranked_den else 0.0)
        macro["parser_coverage"].append(q_counts["parser_hits"] / selected_den if selected_den else 0.0)
        macro["context_retention"].append(q_counts["context_hits"] / parser_den if parser_den else 0.0)
        query_details.append(
            {
                "query_id": query_id,
                "gold_paper_recall": paper_recall,
                "gold_evidence_count": q_counts["gold_evidence"],
                "gold_evidence_in_ranked_pool": q_counts["gold_evidence_in_ranked_pool"],
                "gold_evidence_page_selected": q_counts["gold_evidence_page_selected"],
                "selector_recall": q_counts["selector_hits"] / ranked_den if ranked_den else 0.0,
                "parser_coverage": q_counts["parser_hits"] / selected_den if selected_den else 0.0,
                "context_retention": q_counts["context_hits"] / parser_den if parser_den else 0.0,
            }
        )

    # Unique-page companion metrics. These are useful because several evidence rows
    # can share one paper/page, especially table cells.
    unique_ranked = set()
    unique_selected = set()
    unique_gold_pages = set()
    for detail in evidence_details:
        key = (detail["query_id"], detail["paper_id"], detail["page"])
        unique_gold_pages.add(key)
        if detail["gold_page_in_ranked_pool"]:
            unique_ranked.add(key)
        if detail["gold_page_selected"]:
            unique_selected.add(key)

    paper_recall_micro = _ratio(counters["matched_gold_papers"], counters["gold_papers"])
    paper_recall_macro = _mean(macro["paper_recall"])
    page_rank_recall_micro = _ratio(counters["gold_evidence_in_ranked_pool"], counters["gold_evidence"])
    page_rank_recall_macro = _mean(macro["rank_pool_recall"])
    selector_recall_micro = _ratio(counters["selector_hits"], counters["gold_evidence_in_ranked_pool"])
    selector_recall_macro = _mean(macro["selector_recall"])
    parser_coverage_micro = _ratio(counters["parser_hits"], counters["parser_denominator_selected_gold_evidence"])
    parser_coverage_macro = _mean(macro["parser_coverage"])
    context_retention_micro = _ratio(counters["context_hits"], counters["context_denominator_parser_survived_gold_evidence"])
    context_retention_macro = _mean(macro["context_retention"])

    metrics = {
        "query_count": len(gold_by_query),
        "output_dir": str(output),
        "selected_page_field": selected_page_field,
        "use_source_type_hints": use_source_type_hints,
        "retrieved_paper_recall_micro": paper_recall_micro,
        "retrieved_paper_recall_macro": paper_recall_macro,
        "retrieved_page_recall_micro": page_rank_recall_micro,
        "retrieved_page_recall_macro": page_rank_recall_macro,
        "R_selector_micro": selector_recall_micro,
        "R_selector_macro": selector_recall_macro,
        "R_parser_micro": parser_coverage_micro,
        "R_parser_macro": parser_coverage_macro,
        "R_context_micro": context_retention_micro,
        "R_context_macro": context_retention_macro,
        "gold_paper_recall_micro": paper_recall_micro,
        "gold_paper_recall_macro": paper_recall_macro,
        "gold_evidence_count": counters["gold_evidence"],
        "gold_evidence_rank_pool_recall_micro": page_rank_recall_micro,
        "gold_evidence_rank_pool_recall_macro": page_rank_recall_macro,
        "selector_recall_micro": selector_recall_micro,
        "selector_recall_macro": selector_recall_macro,
        "parser_coverage_micro": parser_coverage_micro,
        "parser_coverage_macro": parser_coverage_macro,
        "selected_gold_evidence_pages_with_any_symbolic_records_micro": _ratio(
            counters["selected_gold_pages_with_any_symbolic_records"],
            counters["parser_denominator_selected_gold_evidence"],
        ),
        "context_retention_micro": context_retention_micro,
        "context_retention_macro": context_retention_macro,
        "end_to_end_pre_vlm2_gold_evidence_recall_micro": _ratio(counters["context_hits"], counters["gold_evidence"]),
        "unique_gold_page_count": len(unique_gold_pages),
        "unique_gold_page_rank_pool_recall": _ratio(len(unique_ranked), len(unique_gold_pages)),
        "unique_gold_page_selector_recall": _ratio(len(unique_ranked & unique_selected), len(unique_ranked)),
        "counts": dict(counters),
    }
    return {"metrics": metrics, "query_details": query_details}, evidence_details


def main() -> int:
    args = parse_args()
    summary, evidence_details = evaluate_gold_extraction(
        official_dir=args.official_dir,
        output_dir=args.output_dir,
        selected_page_field=args.selected_page_field,
        use_source_type_hints=args.use_source_type_hints,
    )
    if args.details_out:
        output = Path(args.details_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in evidence_details) + "\n", encoding="utf-8")
    payload: dict[str, Any] = {"metrics": summary["metrics"]}
    if args.include_details:
        payload["query_details"] = summary["query_details"]
        payload["evidence_details"] = evidence_details
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
