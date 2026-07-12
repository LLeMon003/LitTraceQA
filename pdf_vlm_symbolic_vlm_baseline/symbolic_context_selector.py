from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from .metadata_index import BM25Okapi, tokenize
from .symbolic_schema import (
    HEADER_FOOTER_RECORD_TYPES,
    OFFICIAL_EVIDENCE_SOURCE_TYPES,
    VISUAL_RECORD_TYPES,
    grounding_label_from_record,
    to_official_source_type,
)
from .source_type_hints import infer_source_type_hints


SOURCE_TYPE_ORDER = {
    "table": 0,
    "figure": 1,
    "equation_algorithm": 2,
    "text_span": 3,
    "citation_context": 4,
}

TYPE_BOOSTS = {
    "table": {"table": 2.0},
    "figure": {"figure": 2.0},
    "equation_algorithm": {"equation": 2.0},
    "citation_context": {"citation_context": 2.0},
    "text_span": {"paragraph": 1.2},
}

TABLE_ID_RE = re.compile(r"\btable\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)
FIGURE_ID_RE = re.compile(r"\b(?:figure|fig\.)\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)
ALGORITHM_ID_RE = re.compile(r"\balgorithm\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)


def _normalized_object_id(prefix: str, value: Any) -> str:
    text = str(value or "").strip().rstrip(".,;:")
    if not text:
        return ""
    if text.lower().startswith(prefix.lower()):
        return text
    return f"{prefix} {text}"


def _question_targets(query: str) -> dict[str, Any]:
    lower = str(query or "").lower()
    table_match = TABLE_ID_RE.search(query or "")
    figure_match = FIGURE_ID_RE.search(query or "")
    algorithm_match = ALGORITHM_ID_RE.search(query or "")
    equation_match = re.search(r"\b(?:equation|eq\.)\s*\(?([A-Za-z0-9.\-]+)\)?", query or "", re.IGNORECASE)
    reference_id: int | None = None
    for pattern in (
        r"\b(?:reference|ref\.)\s*\[?(\d{1,3})\]?",
        r"\b(\d{1,3})(?:st|nd|rd|th)\s+reference\b",
    ):
        match = re.search(pattern, lower)
        if match:
            try:
                reference_id = int(match.group(1))
                break
            except ValueError:
                pass
    return {
        "table_id": _normalized_object_id("Table", table_match.group(1) if table_match else ""),
        "figure_id": _normalized_object_id("Figure", figure_match.group(1) if figure_match else ""),
        "algorithm_id": _normalized_object_id("Algorithm", algorithm_match.group(1) if algorithm_match else ""),
        "equation_id": str(equation_match.group(1)).strip().rstrip(".,;:") if equation_match else "",
        "reference_id": reference_id,
        "last_reference": bool(re.search(r"\blast\s+reference\b|\bindex\s+of\s+the\s+last\s+reference\b", lower)),
        "subfigure_count": bool(re.search(r"\bhow many\s+(?:subfigures|sub-figures|panels)\b|\bnumber of\s+(?:subfigures|sub-figures|panels)\b", lower)),
        "hardware": bool(re.search(r"\bhardware\b|\bgpu\b|\bconfigure\b|\bconfiguration\b", lower)),
    }


def _object_mention(prefix: str, value: str) -> re.Pattern[str] | None:
    if not value:
        return None
    suffix = re.sub(rf"^{re.escape(prefix)}\s*", "", value, flags=re.IGNORECASE).strip()
    if not suffix:
        return None
    return re.compile(rf"\b{re.escape(prefix)}\s*{re.escape(suffix)}\b", re.IGNORECASE)


def _record_source_type_bonus(
    *,
    source_type: str,
    primary_evidence_type: str | None,
    query: str,
    text: str,
    label: Any,
    targets: dict[str, Any],
) -> float:
    """Small record-level target bonus.

    Page ranking uses large bonuses to decide which pages are worth parsing.
    Here the page has already been selected, so bonuses are intentionally small:
    they should help truncation prefer likely evidence without eliminating
    diverse support records needed for VLM-2 global reasoning.
    """

    bonus = 0.0
    combined = " ".join([str(label or ""), text or ""])
    primary = to_official_source_type(source_type=primary_evidence_type) or str(primary_evidence_type or "")
    if source_type == primary:
        bonus += 0.35
    if source_type == "table":
        table_pattern = _object_mention("Table", str(targets.get("table_id") or ""))
        if table_pattern and table_pattern.search(combined):
            bonus += 3.0
        elif re.search(r"\btable\b", combined, re.IGNORECASE):
            bonus += 0.45
        if re.search(r"\|.+\|", text or ""):
            bonus += 0.35
    elif source_type == "figure":
        figure_pattern = _object_mention("Figure", str(targets.get("figure_id") or ""))
        if figure_pattern and figure_pattern.search(combined):
            bonus += 3.0
        elif re.search(r"\b(?:figure|fig\.)\b", combined, re.IGNORECASE):
            bonus += 0.45
        if targets.get("subfigure_count") and re.search(r"\([a-z]\)", text or ""):
            bonus += 0.7
    elif source_type == "citation_context":
        target_ref = targets.get("reference_id")
        if isinstance(target_ref, int) and re.search(rf"(?:^|\n|\s)\[{target_ref}\]\s+", text or ""):
            bonus += 3.0
        if targets.get("last_reference") and re.search(r"(?:^|\n|\s)\[\d{1,3}\]\s+", text or ""):
            bonus += 1.1
        if re.search(r"\breferences\b", text or "", re.IGNORECASE):
            bonus += 0.8
    elif source_type == "equation_algorithm":
        algorithm_pattern = _object_mention("Algorithm", str(targets.get("algorithm_id") or ""))
        equation_id = str(targets.get("equation_id") or "")
        if algorithm_pattern and algorithm_pattern.search(combined):
            bonus += 3.0
        if equation_id and re.search(rf"\(\s*{re.escape(equation_id)}\s*\)", combined):
            bonus += 3.0
        elif re.search(r"\b(?:algorithm|equation|loss|objective)\b|=", combined, re.IGNORECASE):
            bonus += 0.6
    elif source_type == "text_span":
        if targets.get("hardware") and re.search(r"\b(?:gpu|rtx|a100|h100|cuda|nvidia)\b", text or "", re.IGNORECASE):
            bonus += 2.0
        if re.search(r"\b(?:implementation|experiment|setup|configuration|hardware|optimizer|learning rate|batch size|epoch|overhead|efficiency)\b", text or "", re.IGNORECASE):
            bonus += 0.45
    if re.search(r"\b(?:answer|result|performance|score|accuracy|f1|nrmse|ap|success rate)\b", query, re.IGNORECASE) and re.search(r"[-+]?\d+(?:\.\d+)?%?", text or ""):
        bonus += 0.25
    return bonus


def _image_path_for(record: dict[str, Any], processed_root: Path, parser_model_slug: str) -> str:
    return str(processed_root / parser_model_slug / str(record.get("paper_id")) / "page_images" / f"page_{int(record.get('page') or 0):03d}.jpg")


def _query_needs_header_footer(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ["header", "footer", "page number", "running head"])


def _record_image_ref(record: dict[str, Any]) -> str:
    return f"attached_record_{str(record.get('record_id') or 'record')}"


def _is_header_footer_record(record: dict[str, Any]) -> bool:
    return str(record.get("record_type") or "").strip() in HEADER_FOOTER_RECORD_TYPES


def project_context_for_vlm2(record: dict[str, Any], mode: str) -> dict[str, Any]:
    projected = {
        "paper_id": record.get("paper_id"),
        "page": record.get("page"),
        "source_type": record.get("source_type"),
        "label": record.get("label"),
        "grounding_label": record.get("grounding_label"),
        "text": record.get("text"),
    }
    if projected["grounding_label"] is None:
        projected.pop("grounding_label", None)
    if record.get("source_type_hints"):
        projected["source_type_hints"] = record.get("source_type_hints")
    if mode == "cropped_image":
        projected["image_ref"] = _record_image_ref(record)
    return projected


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or f"{record.get('paper_id')}::{record.get('page')}::{record.get('record_id')}::{record.get('text')}")


def _add_ranked(
    selected: list[dict[str, Any]],
    seen: set[str],
    records: list[dict[str, Any]],
    limit: int,
) -> None:
    if limit <= 0:
        return
    for record in records:
        if len(selected) >= limit:
            break
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)


def _select_with_type_budgets(
    ranked: list[dict[str, Any]],
    total_budget: int,
    primary_evidence_type: str | None,
    primary_min: int,
    support_text_min: int,
    context_types_enabled: bool,
    per_type_budget: int,
) -> list[dict[str, Any]]:
    total_budget = max(1, int(total_budget or 1))
    primary = to_official_source_type(source_type=primary_evidence_type) or str(primary_evidence_type or "")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    if primary in OFFICIAL_EVIDENCE_SOURCE_TYPES:
        _add_ranked(selected, seen, [r for r in ranked if r.get("source_type") == primary], min(primary_min, total_budget))
    if len(selected) < total_budget:
        _add_ranked(selected, seen, [r for r in ranked if r.get("source_type") == "text_span"], min(total_budget, len(selected) + support_text_min))
    if context_types_enabled and len(selected) < total_budget:
        for source_type in ("table", "figure", "equation_algorithm", "citation_context", "text_span"):
            _add_ranked(
                selected,
                seen,
                [r for r in ranked if r.get("source_type") == source_type],
                min(total_budget, len(selected) + max(0, per_type_budget)),
            )
            if len(selected) >= total_budget:
                break
    _add_ranked(selected, seen, ranked, total_budget)
    return selected[:total_budget]


def _record_order_key(record: dict[str, Any]) -> tuple[Any, ...]:
    try:
        page = int(record.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    return (
        -float(record.get("_candidate_bm25_score") or 0.0),
        str(record.get("paper_id") or ""),
        page,
        SOURCE_TYPE_ORDER.get(str(record.get("source_type") or ""), 99),
        str(record.get("record_id") or ""),
    )


def _limit_records(records: list[dict[str, Any]], max_records: int = 0, max_chars: int = 0) -> tuple[list[dict[str, Any]], bool]:
    limited = records
    truncated = False
    if max_records and max_records > 0 and len(limited) > max_records:
        limited = limited[:max_records]
        truncated = True
    if max_chars and max_chars > 0:
        char_total = 0
        char_limited: list[dict[str, Any]] = []
        for record in limited:
            text_len = len(str(record.get("text") or ""))
            if char_limited and char_total + text_len > max_chars:
                truncated = True
                break
            char_limited.append(record)
            char_total += text_len
        limited = char_limited
    return limited, truncated


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _has_context_limit(max_records: int = 0, max_chars: int = 0) -> bool:
    return (max_records is not None and max_records > 0) or (max_chars is not None and max_chars > 0)


def select_symbolic_contexts(
    query: str,
    candidate_records: list[dict[str, Any]],
    processed_root: str | Path,
    parser_model_slug: str,
    top_n_records: int = 24,
    top_n_visual_records: int = 6,
    primary_evidence_type: str | None = None,
    query_id: str | None = None,
    vlm2_context_mode: str = "text_only",
    include_parse_confidence: bool = True,
    evidence_total_budget: int | None = None,
    primary_evidence_min: int = 6,
    support_text_min: int = 4,
    context_types_enabled: bool = True,
    context_type_budget_per_type: int = 3,
    context_selection_mode: str = "page_all_symbolic",
    max_context_records: int = 0,
    max_context_chars: int = 0,
    source_type_hints_enabled: bool = True,
) -> dict[str, Any]:
    allow_header_footer = _query_needs_header_footer(query)
    valid = [
        r
        for r in candidate_records
        if r.get("validation_status") != "rejected"
        and (allow_header_footer or not _is_header_footer_record(r))
    ]
    if not valid:
        return {
            "query_id": query_id,
            "selection_method": f"symbolic_{context_selection_mode}",
            "partial_artifacts_present": False,
            "partial_selected_record_count": 0,
            "prompt_context_mode": vlm2_context_mode,
            "selected_evidence": [],
            "selected_records_debug": [],
            "selected_records": [],
            "selected_visual_records": [],
            "source_type_distribution": {source_type: 0 for source_type in OFFICIAL_EVIDENCE_SOURCE_TYPES},
            "primary_evidence_type_count": 0,
            "supporting_evidence_count": 0,
            "grounding_label_hints_by_type": {},
            "context_selection_mode": str(context_selection_mode or "page_all_symbolic").strip().lower(),
            "context_truncated": False,
            "selected_record_count": 0,
        }
    query_tokens = tokenize(query)
    corpus = [tokenize(" ".join([str(r.get("text") or ""), str(r.get("label") or "")])) for r in valid]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(valid)
    boosts = TYPE_BOOSTS.get(primary_evidence_type or "", {})
    targets = _question_targets(query)
    processed = Path(processed_root)
    ranked: list[dict[str, Any]] = []
    for record, bm25_score in zip(valid, scores):
        source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
        if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            continue
        text = str(record.get("text") or "")
        source_bonus = _record_source_type_bonus(
            source_type=source_type,
            primary_evidence_type=primary_evidence_type,
            query=query,
            text=text,
            label=record.get("label"),
            targets=targets,
        )
        score = float(bm25_score)
        score += boosts.get(str(record.get("record_type")), 0.0)
        score += boosts.get(source_type, 0.0)
        score += source_bonus
        score += 0.15 * float(record.get("_candidate_bm25_score") or 0.0)
        record_type = str(record.get("record_type") or "")
        if str(record.get("_page_status") or record.get("page_status") or "") == "partial":
            score -= 0.5
        if record_type in HEADER_FOOTER_RECORD_TYPES:
            score -= 0.75
        if record_type == "citation_context" and primary_evidence_type != "citation_context":
            score -= 1.0
        if len(str(record.get("text") or "")) > 2500 and record_type == "citation_context" and primary_evidence_type != "citation_context":
            score -= 0.75
        selected = {
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "global_record_id": record.get("global_record_id"),
            "record_id": record.get("record_id"),
            "record_type": record.get("record_type"),
            "source_type": source_type,
            "label": record.get("label"),
            "grounding_label": grounding_label_from_record(source_type, record.get("label")),
            "score": round(score, 6),
            "source_type_specific_bonus": round(source_bonus, 6),
            "text": record.get("text"),
            "locator": record.get("locator") or {"page": record.get("page")},
            "image_path": _image_path_for(record, processed, parser_model_slug),
            "figure_crop_path": record.get("figure_crop_path"),
        }
        if source_type_hints_enabled:
            hints = infer_source_type_hints(selected)
            if hints:
                selected["source_type_hints"] = hints
        selected["_page_status"] = record.get("_page_status") or record.get("page_status")
        ranked.append(selected)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    normalized_mode = str(context_selection_mode or "page_all_symbolic").strip().lower()
    if normalized_mode in {"ranked_budget", "bm25_budget", "record_budget"}:
        selected_records_internal = _select_with_type_budgets(
            ranked,
            evidence_total_budget or top_n_records,
            primary_evidence_type,
            primary_evidence_min,
            support_text_min,
            context_types_enabled,
            context_type_budget_per_type,
        )
        selection_method = "symbolic_lexical_bm25_type_budget"
        context_truncated = False
    else:
        limit_enabled = _has_context_limit(max_context_records, max_context_chars)
        selected_records_internal = _dedupe_records(list(ranked)) if limit_enabled else _dedupe_records(sorted(ranked, key=_record_order_key))
        selected_records_internal, context_truncated = _limit_records(
            selected_records_internal,
            max_context_records,
            max_context_chars,
        )
        selection_method = "symbolic_page_all_records_relevance_limited" if limit_enabled else "symbolic_page_all_records_from_routed_pages"
    partial_count = sum(1 for r in selected_records_internal if r.get("_page_status") == "partial")
    selected_records_debug = []
    for rank, record in enumerate(selected_records_internal, start=1):
        debug_record = {k: v for k, v in record.items() if not k.startswith("_")}
        debug_record["page_status"] = record.get("_page_status") or "unknown"
        debug_record["selection_rank"] = rank
        debug_record["selection_method"] = selection_method
        selected_records_debug.append(debug_record)
    selected_evidence = [
        project_context_for_vlm2(
            record,
            vlm2_context_mode,
        )
        for record in selected_records_internal
    ]
    distribution = dict(Counter(str(r.get("source_type")) for r in selected_evidence if isinstance(r, dict)))
    for source_type in OFFICIAL_EVIDENCE_SOURCE_TYPES:
        distribution.setdefault(source_type, 0)
    primary_source = to_official_source_type(source_type=primary_evidence_type) or str(primary_evidence_type or "")
    primary_count = sum(1 for r in selected_evidence if r.get("source_type") == primary_source)
    grounding_label_hints_by_type = dict(
        Counter(
            str((r.get("grounding_label") or {}).get("type"))
            for r in selected_evidence
            if isinstance(r.get("grounding_label"), dict)
        )
    )
    visual_records = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in ranked
        if r.get("record_type") in VISUAL_RECORD_TYPES and Path(str(r.get("image_path"))).exists()
    ]
    return {
        "query_id": query_id,
        "selection_method": selection_method,
        "prompt_context_mode": vlm2_context_mode,
        "partial_artifacts_present": partial_count > 0,
        "partial_selected_record_count": partial_count,
        "has_partial_artifacts": partial_count > 0,
        "selected_evidence": selected_evidence,
        "selected_records_debug": selected_records_debug,
        "selected_records": selected_records_debug,
        "selected_visual_records": visual_records[:top_n_visual_records],
        "source_type_distribution": distribution,
        "primary_evidence_type_count": primary_count,
        "supporting_evidence_count": max(0, len(selected_evidence) - primary_count),
        "grounding_label_hints_by_type": grounding_label_hints_by_type,
        "context_selection_mode": normalized_mode,
        "context_truncated": context_truncated,
        "selected_record_count": len(selected_evidence),
    }
