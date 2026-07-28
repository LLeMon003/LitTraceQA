from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .evidence_matcher import OFFICIAL_SOURCE_TYPES, normalize_source_type


FAILURE_STAGES = [
    "selected",
    "extracted_but_not_selected",
    "locator_not_extracted",
    "source_type_not_extracted",
    "page_not_extracted",
    "pdf_not_available",
    "paper_not_retrieved",
    "unknown_failure",
]


def paper_source(metadata: dict[str, Any] | None, paper_id: str = "") -> str:
    row = metadata or {}
    fields = " ".join(str(row.get(key) or "") for key in ("venue", "conference", "source", "source_url", "pdf_url", "openreview_id", "arxiv_id", "paper_id"))
    lowered = fields.lower()
    if row.get("openreview_id") or "openreview.net" in lowered:
        return "OpenReview"
    if row.get("arxiv_id") or "arxiv.org" in lowered:
        return "arXiv"
    for label in ("ACL", "EMNLP", "NAACL", "ICML", "NeurIPS", "ICLR", "CVPR", "ICCV", "ECCV"):
        if label.lower() in lowered:
            return label
    prefix = str(paper_id or row.get("paper_id") or "").split("_", 1)[0].lower()
    prefix_map = {
        "acl2025": "ACL",
        "emnlp2025": "EMNLP",
        "naacl2025": "NAACL",
        "icml2025": "ICML",
        "neurips2025": "NeurIPS",
        "iclr2025": "ICLR",
        "cvpr2025": "CVPR",
        "iccv2025": "ICCV",
        "eccv2025": "ECCV",
    }
    if prefix in prefix_map:
        return prefix_map[prefix]
    for key in ("source_url", "pdf_url"):
        domain = urlparse(str(row.get(key) or "")).netloc.lower()
        if domain:
            return domain
    return "unknown"


def _ratio(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 6)


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    extracted_den = sum(1 for row in rows if row.get("locator_extracted"))
    selected_count = sum(1 for row in rows if row.get("selected"))
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row.get("query_id") or "")].append(row)
    union_available = [row for row in rows if row.get("candidate_union_available")]
    union_extracted_den = sum(1 for row in union_available if row.get("locator_extracted"))
    return {
        "gold_count": total,
        "R_paper_candidate": _ratio(sum(1 for row in rows if row.get("candidate_paper_hit")), total),
        "R_extraction_page": _ratio(sum(1 for row in rows if row.get("page_extracted")), total),
        "R_extraction_source": _ratio(sum(1 for row in rows if row.get("source_type_extracted")), total),
        "R_extraction_locator": _ratio(sum(1 for row in rows if row.get("locator_extracted")), total),
        "R_parser_relaxed": _ratio(sum(1 for row in rows if row.get("source_type_extracted")), total),
        "R_parser_strict": _ratio(sum(1 for row in rows if row.get("locator_extracted")), total),
        "R_selection_over_gold": _ratio(selected_count, total),
        "R_selection_given_extracted": _ratio(selected_count, extracted_den),
        "R_pre_answer_context": _ratio(selected_count, total),
        "R_evidence_selected": _ratio(selected_count, total),
        "R_context_budget_given_extracted": _ratio(selected_count, extracted_den),
        "R_all_support_query": _ratio(
            sum(1 for query_rows in by_query.values() if query_rows and all(row.get("selected") for row in query_rows)),
            len(by_query),
        ),
        "R_candidate_union_over_gold": _ratio(sum(1 for row in union_available if row.get("candidate_union_selected")), len(union_available)) if union_available else None,
        "R_candidate_union_given_extracted": _ratio(sum(1 for row in union_available if row.get("candidate_union_selected")), union_extracted_den) if union_available else None,
        "failure_stage_distribution": dict(Counter(str(row.get("failure_stage") or "unknown_failure") for row in rows)),
    }


def breakdown_by(rows: list[dict[str, Any]], key: str, *, include_all_source_types: bool = False) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = normalize_source_type(row.get(key)) if key == "gold_source_type" else str(row.get(key) or "unknown")
        grouped[value].append(row)
    result = {name: metrics_from_rows(items) for name, items in sorted(grouped.items())}
    if include_all_source_types:
        for source_type in sorted(OFFICIAL_SOURCE_TYPES):
            result.setdefault(source_type, metrics_from_rows([]))
    return result


def write_summary_markdown(path: str | Path, summary: dict[str, Any], examples: list[dict[str, Any]] | None = None) -> None:
    lines = [
        "# Extraction Selection Evaluation",
        "",
        "## Aggregate",
        "```json",
        json.dumps(summary.get("aggregate", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Failure Stage Distribution",
        "```json",
        json.dumps(summary.get("failure_stage_distribution", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## By Source Type",
        "```json",
        json.dumps(summary.get("by_source_type", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## By Paper Source",
        "```json",
        json.dumps(summary.get("by_paper_source", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    if examples:
        lines.extend(["", "## Representative Failure Cases"])
        for item in examples[:20]:
            lines.append(f"- `{item.get('query_id')}` {item.get('failure_stage')}: {item.get('gold_paper_id')} p.{item.get('gold_page')} {item.get('gold_source_type')} {item.get('gold_locator')}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_examples_markdown(path: str | Path, failures: list[dict[str, Any]]) -> None:
    lines = ["# Representative Failure Cases", ""]
    for item in failures[:50]:
        lines.extend(
            [
                f"## {item.get('query_id')} / {item.get('failure_stage')}",
                "",
                f"- paper_id: `{item.get('gold_paper_id')}`",
                f"- page: `{item.get('gold_page')}`",
                f"- source_type: `{item.get('gold_source_type')}`",
                f"- locator: `{json.dumps(item.get('gold_locator'), ensure_ascii=False)}`",
                f"- reason: {item.get('failure_reason')}",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
