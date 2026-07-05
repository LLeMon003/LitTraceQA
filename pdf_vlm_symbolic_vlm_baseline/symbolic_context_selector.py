from __future__ import annotations

from pathlib import Path
from typing import Any

from .metadata_index import BM25Okapi, tokenize
from .symbolic_schema import VISUAL_RECORD_TYPES


TYPE_BOOSTS = {
    "table": {"table": 2.0},
    "figure": {"figure": 2.0},
    "equation_algorithm": {"equation": 2.0},
    "citation_context": {"citation_context": 2.0},
    "text_span": {"paragraph": 1.2},
}


def _image_path_for(record: dict[str, Any], processed_root: Path, parser_model_slug: str) -> str:
    return str(processed_root / parser_model_slug / str(record.get("paper_id")) / "page_images" / f"page_{int(record.get('page') or 0):03d}.jpg")


def _query_needs_header_footer(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ["header", "footer", "page number", "running head"])


def _record_image_ref(record: dict[str, Any]) -> str:
    return f"attached_record_{str(record.get('record_id') or 'record')}"


def project_context_for_vlm2(
    record: dict[str, Any],
    mode: str,
    include_confidence: bool = True,
) -> dict[str, Any]:
    projected = {
        "paper_id": record.get("paper_id"),
        "page": record.get("page"),
        "source_type": record.get("source_type"),
        "locator": record.get("locator") or {"page": record.get("page")},
        "text": record.get("text"),
    }
    if mode == "cropped_image":
        projected["image_ref"] = _record_image_ref(record)
    return projected


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
) -> dict[str, Any]:
    allow_header_footer = _query_needs_header_footer(query)
    valid = [
        r
        for r in candidate_records
        if r.get("validation_status") != "rejected"
        and (allow_header_footer or r.get("record_type") != "header_footer")
    ]
    if not valid:
        return {
            "query_id": query_id,
            "selection_method": "symbolic_lexical_bm25_without_embedding",
            "partial_artifacts_present": False,
            "partial_selected_record_count": 0,
            "prompt_context_mode": vlm2_context_mode,
            "selected_evidence": [],
            "selected_records_debug": [],
            "selected_records": [],
            "selected_visual_records": [],
        }
    query_tokens = tokenize(query)
    corpus = [tokenize(" ".join([str(r.get("text") or ""), str(r.get("label") or "")])) for r in valid]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(valid)
    boosts = TYPE_BOOSTS.get(primary_evidence_type or "", {})
    processed = Path(processed_root)
    ranked: list[dict[str, Any]] = []
    for record, bm25_score in zip(valid, scores):
        score = float(bm25_score)
        score += boosts.get(str(record.get("record_type")), 0.0)
        score += 0.15 * float(record.get("_candidate_bm25_score") or 0.0)
        record_type = str(record.get("record_type") or "")
        if str(record.get("_page_status") or record.get("page_status") or "") == "partial":
            score -= 0.5
        if record_type == "header_footer":
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
            "source_type": record.get("source_type"),
            "label": record.get("label"),
            "score": round(score, 6),
            "text": record.get("text"),
            "locator": record.get("locator") or {"page": record.get("page")},
            "image_path": _image_path_for(record, processed, parser_model_slug),
        }
        selected["_page_status"] = record.get("_page_status") or record.get("page_status")
        ranked.append(selected)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected_records_internal = ranked[:top_n_records]
    partial_count = sum(1 for r in selected_records_internal if r.get("_page_status") == "partial")
    selection_method = "symbolic_lexical_bm25_without_embedding"
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
            include_confidence=include_parse_confidence,
        )
        for record in selected_records_internal
    ]
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
    }
