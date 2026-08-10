from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Sequence

from .metadata_index import BM25Okapi, tokenize
from .evidence_packages import EvidencePackageConfig, build_packages, select_packages
from .multi_paper_hyde import MultiPaperHyDEConfig
from .paper_conditioned_claims import PaperConditionedClaimsConfig, generate_evidence_plan
from .section_relevance import SectionRelevanceConfig, query_for_relevance_mode, retrieve_section_relevance
from .symbolic_schema import (
    HEADER_FOOTER_RECORD_TYPES,
    OFFICIAL_EVIDENCE_SOURCE_TYPES,
    VISUAL_RECORD_TYPES,
    grounding_label_from_record,
    to_official_source_type,
)
from .source_type_hints import infer_source_type_hints
from .task_structure import as_source_types


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
    del processed_root, parser_model_slug
    source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
    keys = {
        "figure": ("figure_crop_path", "crop_path"),
        "table": ("table_crop_path", "crop_path"),
        "equation_algorithm": ("equation_algorithm_crop_path", "crop_path"),
    }.get(source_type, ())
    for key in keys:
        path = Path(str(record.get(key) or ""))
        if path.is_file():
            return str(path)
    return ""


def _query_needs_header_footer(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ["header", "footer", "page number", "running head"])


def _record_locator(record: dict[str, Any]) -> dict[str, Any]:
    locator = dict(record.get("locator") or {})
    locator.setdefault("page", record.get("page"))
    grounding = record.get("grounding_label")
    if isinstance(grounding, dict):
        key = str(grounding.get("type") or "")
        value = grounding.get("value")
        if key and value not in {None, ""}:
            locator.setdefault(key, value)
    return locator


def _is_header_footer_record(record: dict[str, Any]) -> bool:
    return str(record.get("record_type") or "").strip() in HEADER_FOOTER_RECORD_TYPES


def project_context_for_vlm2(record: dict[str, Any], mode: str) -> dict[str, Any]:
    projected = {
        "evidence_ref": record.get("evidence_ref"),
        "paper_id": record.get("paper_id"),
        "page": record.get("page"),
        "source_type": record.get("source_type"),
        "label": record.get("label"),
        "locator": _record_locator(record),
        "section_id": record.get("section_id"),
        "section_title": record.get("section_title"),
        "section_path": record.get("section_path"),
        "section_type": record.get("section_type"),
        "reading_order": record.get("reading_order"),
        "document_order": record.get("document_order"),
        "grounding_label": record.get("grounding_label"),
        "text": record.get("text"),
    }
    if projected["grounding_label"] is None:
        projected.pop("grounding_label", None)
    if record.get("source_type_hints"):
        projected["source_type_hints"] = record.get("source_type_hints")
    if projected["source_type"] == "table" and isinstance(record.get("table_structure"), dict):
        projected["table_structure"] = record.get("table_structure")
    if mode == "cropped_image" and record.get("image_ref"):
        projected["image_ref"] = record.get("image_ref")
    return projected


def _packet_record(record: dict[str, Any]) -> dict[str, Any]:
    locator = _record_locator(record)
    locator.pop("page", None)
    projected = {
        "evidence_ref": record.get("evidence_ref"),
        "page": record.get("page"),
        "source_type": record.get("source_type"),
        "label": record.get("label"),
        "locator": locator,
        "text": record.get("text"),
    }
    if record.get("image_ref"):
        projected["image_ref"] = record.get("image_ref")
    if record.get("source_type_hints"):
        projected["source_type_hints"] = record.get("source_type_hints")
    return projected


def _merge_record_defaults(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    defaults: dict[str, Any] = {}
    for key in ("page", "source_type"):
        values = {row.get(key) for row in rows}
        if len(values) == 1:
            defaults[key] = rows[0].get(key)
    compact = [{key: value for key, value in row.items() if key not in defaults} for row in rows]
    return defaults, compact


def _compact_chunk_packets(
    relevance: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id = {_record_key(record): record for record in records}
    packets: list[dict[str, Any]] = []
    emitted_refs: set[str] = set()
    for section in relevance["trace"].get("sections") or []:
        if not section.get("selected"):
            continue
        chunks = sorted(
            (chunk for chunk in section.get("chunks") or [] if chunk.get("selected_for_context")),
            key=lambda chunk: (
                -float(chunk.get("llmrerank_raw_score") or 0.0),
                int(chunk.get("chunk_index") or 0),
            ),
        )
        for chunk in chunks:
            chunk_records = [
                records_by_id[record_id]
                for record_id in chunk.get("record_ids") or []
                if record_id in records_by_id
                and str(records_by_id[record_id].get("evidence_ref") or "") not in emitted_refs
            ]
            if not chunk_records:
                continue
            for record in chunk_records:
                emitted_refs.add(str(record.get("evidence_ref") or ""))
            defaults, packet_records = _merge_record_defaults([_packet_record(record) for record in chunk_records])
            packets.append(
                {
                    "chunk_ref": f"{section.get('paper_id')}::{section.get('section_id')}::c{int(chunk.get('chunk_index') or 0):03d}",
                    "paper_id": section.get("paper_id"),
                    "section_id": section.get("section_id"),
                    "section_title": section.get("section_title"),
                    "section_type": section.get("section_type"),
                    "section_path": section.get("section_path"),
                    "record_defaults": defaults,
                    "records": packet_records,
                }
            )
    missing_records = [
        record
        for record in records
        if str(record.get("evidence_ref") or "") not in emitted_refs
    ]
    missing_by_section: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in missing_records:
        missing_by_section.setdefault(
            (str(record.get("paper_id") or ""), str(record.get("section_id") or "")),
            [],
        ).append(record)
    for (paper_id, section_id), section_records in missing_by_section.items():
        first = section_records[0]
        defaults, packet_records = _merge_record_defaults([_packet_record(record) for record in section_records])
        packets.append(
            {
                "chunk_ref": f"{paper_id}::{section_id}::unmapped",
                "paper_id": paper_id,
                "section_id": first.get("section_id"),
                "section_title": first.get("section_title"),
                "section_type": first.get("section_type"),
                "section_path": first.get("section_path"),
                "record_defaults": defaults,
                "records": packet_records,
            }
        )
    return packets


def _compact_package_packets(
    packages: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep package provenance while serializing each record under its own section."""
    records_by_id = {_record_key(record): record for record in records}
    packets: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for package in packages:
        package_records = [
            records_by_id[_record_key(record)]
            for record in package.get("records") or []
            if _record_key(record) in records_by_id and _record_key(record) not in emitted
        ]
        if not package_records:
            continue
        emitted.update(_record_key(record) for record in package_records)
        by_section: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in package_records:
            key = (str(record.get("paper_id") or ""), str(record.get("section_id") or ""))
            by_section.setdefault(key, []).append(record)
        for index, ((paper_id, section_id), section_records) in enumerate(by_section.items()):
            first = section_records[0]
            packet_paper_id = first.get("paper_id") or package.get("paper_id")
            packet_section_id = first.get("section_id") or package.get("section_id")
            defaults, packet_records = _merge_record_defaults([_packet_record(record) for record in section_records])
            packets.append(
                {
                    "chunk_ref": f"{package.get('package_id')}::s{index:02d}",
                    "package_id": package.get("package_id"),
                    "anchor_record_id": package.get("anchor_record_id"),
                    "package_source_type": package.get("source_type"),
                    "package_label": package.get("label"),
                    "paper_id": packet_paper_id,
                    "section_id": packet_section_id,
                    "section_title": first.get("section_title") or package.get("section_title"),
                    "section_type": first.get("section_type") or package.get("section_type"),
                    "section_path": first.get("section_path") or package.get("section_path"),
                    "record_defaults": defaults,
                    "records": packet_records,
                }
            )
    return packets


def audit_selected_context(
    selected_evidence: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    attachments: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_by_ref = {
        str(record.get("evidence_ref") or ""): record
        for record in selected_evidence
        if str(record.get("evidence_ref") or "")
    }
    if len(selected_by_ref) != len(selected_evidence):
        raise ValueError("prompt_audit:selected evidence refs must be present and unique")

    packet_by_ref: dict[str, dict[str, Any]] = {}
    for packet in packets:
        defaults = packet.get("record_defaults") if isinstance(packet.get("record_defaults"), dict) else {}
        inherited = {
            "paper_id": packet.get("paper_id"),
            "section_id": packet.get("section_id"),
            "section_title": packet.get("section_title"),
            "section_type": packet.get("section_type"),
            "section_path": packet.get("section_path"),
        }
        for raw_record in packet.get("records") or []:
            if not isinstance(raw_record, dict):
                raise ValueError("prompt_audit:packet record must be an object")
            record = {**inherited, **defaults, **raw_record}
            ref = str(record.get("evidence_ref") or "")
            if not ref or ref in packet_by_ref:
                raise ValueError("prompt_audit:packet evidence refs must be present and unique")
            locator = {"page": record.get("page"), **dict(record.get("locator") or {})}
            packet_by_ref[ref] = {**record, "locator": locator}

    if set(packet_by_ref) != set(selected_by_ref):
        missing = sorted(set(selected_by_ref) - set(packet_by_ref))
        extra = sorted(set(packet_by_ref) - set(selected_by_ref))
        raise ValueError(f"prompt_audit:packet refs differ from selected refs missing={missing} extra={extra}")
    for ref, selected in selected_by_ref.items():
        packet = packet_by_ref[ref]
        for key in (
            "paper_id", "page", "source_type", "label", "locator", "text",
            "section_id", "section_title", "section_type", "section_path",
            "reading_order", "document_order",
        ):
            if packet.get(key, selected.get(key)) != selected.get(key):
                raise ValueError(f"prompt_audit:{ref} field mismatch: {key}")

    attachment_by_ref = {str(item.get("image_ref") or ""): item for item in attachments}
    declared_image_refs = {
        str(record.get("image_ref") or "")
        for record in selected_evidence
        if str(record.get("image_ref") or "")
    }
    if declared_image_refs != set(attachment_by_ref):
        raise ValueError("prompt_audit:image refs differ from attached images")
    for image_ref, attachment in attachment_by_ref.items():
        if not Path(str(attachment.get("path") or "")).is_file():
            raise ValueError(f"prompt_audit:missing attached image: {image_ref}")
    return {
        "passed": True,
        "selected_record_count": len(selected_by_ref),
        "packet_record_count": len(packet_by_ref),
        "packet_count": len(packets),
        "attached_image_count": len(attachments),
    }


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
    preferred_source_types: Sequence[str],
    primary_min: int,
    support_text_min: int,
    context_types_enabled: bool,
    per_type_budget: int,
) -> list[dict[str, Any]]:
    total_budget = max(1, int(total_budget or 1))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    preferred = tuple(as_source_types(preferred_source_types))
    if preferred:
        per_type_min = max(1, int(primary_min) // max(1, len(preferred)))
        for source_type in preferred:
            remaining = max(0, total_budget - len(selected))
            if remaining <= 0:
                break
            _add_ranked(
                selected,
                seen,
                [r for r in ranked if r.get("source_type") == source_type],
                min(remaining, per_type_min),
            )
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


def _section_query_bonus(query: str, section_title: Any, section_path: Any, section_type: Any) -> tuple[float, float]:
    lower = str(query or "").lower()
    title_text = " ".join([str(section_title or ""), " ".join(str(item) for item in section_path or [])]).lower()
    title_bonus = 0.0
    type_bonus = 0.0
    for token in tokenize(lower):
        if len(token) >= 4 and token in title_text:
            title_bonus += 0.12
    section_kind = str(section_type or "").lower()
    if section_kind == "references" and re.search(r"\b(?:reference|citation|cited|bibliography|author)\b", lower):
        type_bonus += 1.4
    elif section_kind in {"method", "experiments", "results"} and re.search(r"\b(?:method|experiment|result|evaluation|performance|setup)\b", lower):
        type_bonus += 0.45
    elif section_kind == "appendix" and re.search(r"\b(?:appendix|supplementary)\b", lower):
        type_bonus += 0.7
    return min(title_bonus, 0.8), type_bonus


def _document_order(record: dict[str, Any]) -> int:
    try:
        return int(record.get("document_order") or 0)
    except (TypeError, ValueError):
        return 0


def _section_context_groups(
    anchors: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    mode: str,
    *,
    max_groups: int = 8,
    window: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_section: dict[str, list[dict[str, Any]]] = {}
    for record in ranked:
        section_id = str(record.get("section_id") or "")
        if section_id:
            by_section.setdefault(section_id, []).append(record)
    for records in by_section.values():
        records.sort(key=lambda item: (_document_order(item), str(item.get("record_id") or "")))

    groups: list[dict[str, Any]] = []
    expanded: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    expansion_mode = "local_window"
    if mode in {"section_full", "subsection_full", "section_aware_full"}:
        expansion_mode = "section_full"
    elif mode in {"record_only", "section_record_only"}:
        expansion_mode = "record_only"

    for anchor in anchors:
        section_id = str(anchor.get("section_id") or "")
        if not section_id or section_id in seen_sections:
            continue
        seen_sections.add(section_id)
        records = by_section.get(section_id, [])
        if expansion_mode == "record_only":
            group_records = [anchor]
        elif expansion_mode == "section_full":
            group_records = records
        else:
            order = _document_order(anchor)
            group_records = [record for record in records if abs(_document_order(record) - order) <= window]
            if anchor not in group_records:
                group_records.append(anchor)
                group_records.sort(key=lambda item: (_document_order(item), str(item.get("record_id") or "")))
        groups.append(
            {
                "section_id": section_id,
                "section_title": anchor.get("section_title"),
                "section_type": anchor.get("section_type"),
                "section_path": anchor.get("section_path"),
                "expansion_mode": expansion_mode,
                "anchor_record_id": anchor.get("global_record_id") or anchor.get("record_id"),
                "record_count": len(group_records),
                "records": [{k: v for k, v in record.items() if not k.startswith("_")} for record in group_records],
            }
        )
        expanded.extend(group_records)
        if len(groups) >= max_groups:
            break
    return groups, _dedupe_records(expanded or anchors)


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
    source_hint_query: str | None = None,
    top_n_records: int = 24,
    top_n_visual_records: int = 6,
    primary_evidence_type: str | None = None,
    preferred_source_types: Sequence[str] | None = None,
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
    section_relevance_config: SectionRelevanceConfig | None = None,
    section_relevance_top_k: int = 0,
    multi_paper_hyde_config: MultiPaperHyDEConfig | None = None,
    paper_conditioned_claims_config: PaperConditionedClaimsConfig | None = None,
    paper_local_bm25_route_mode: str = "disabled",
    is_multi_paper_task: bool = False,
    hyde_client: Any = None,
    candidate_paper_metadata: list[dict[str, Any]] | None = None,
    evidence_package_budget: int = 0,
    evidence_package_min_budget: int = 4,
    multi_paper_min_distinct_papers: int = 4,
    evidence_package_adaptive_stop: bool = True,
    multi_paper_modality_packages_per_paper: int = 2,
    multi_paper_supporting_text_packages_per_paper: int = 0,
    evidence_package_page_text_anchors_per_page: int = 0,
    evidence_package_max_context_chars: int = 0,
    evidence_package_rrf_k: int = 60,
    evidence_package_candidate_pool_per_route: int = 0,
    evidence_package_max_per_page: int = 2,
    slot_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = source_hint_query or query
    normalized_mode = str(context_selection_mode or "page_all_symbolic").strip().lower()
    preferred = tuple(
        as_source_types(preferred_source_types if preferred_source_types is not None else primary_evidence_type)
    )
    preferred_set = set(preferred)
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
            "context_selection_mode": normalized_mode,
            "context_truncated": False,
            "selected_record_count": 0,
            "selected_context_groups": [],
        }
    if normalized_mode == "section_relevance":
        configured_section_config = section_relevance_config or SectionRelevanceConfig()
        if configured_section_config.backend not in {"bm25", "llmrerank"}:
            raise ValueError("This pipeline supports SECTION_RELEVANCE_BACKEND=bm25 or llmrerank only.")
        if configured_section_config.llmrerank_failure_fallback not in {"none", "bm25"}:
            raise ValueError("This pipeline supports LLMRERANK_FAILURE_FALLBACK=none or bm25 only.")
        # The scorer has one quality path: score every canonical record-aware
        # unit before package routing. There is no sparse prefilter or Top-K.
        score_config = configured_section_config
        relevance = retrieve_section_relevance(
            query=query,
            candidate_records=valid,
            processed_root=processed_root,
            config=score_config,
            top_k_sections=0,
            query_id=query_id,
            primary_evidence_type=preferred[0] if preferred else None,
            candidate_paper_metadata=candidate_paper_metadata,
        )
        # Delayed to avoid the existing selector -> hierarchy -> slot module
        # dependency cycle during import.
        from .slot_generation import plan_package_routes, plan_paper_package_routes

        slot_package_routes = plan_package_routes(slot_plan, question)
        route_queries: list[str] = []
        slot_route_queries = [
            (route["query"], tuple(route["record_types"]))
            for route in slot_package_routes
        ]
        slot_paper_package_routes = plan_paper_package_routes(slot_plan, candidate_paper_metadata, question)
        slot_paper_route_queries = [
            (route["paper_id"], route["query"], tuple(route["record_types"]))
            for route in slot_paper_package_routes
        ]
        if paper_local_bm25_route_mode not in {"disabled", "original", "mask_method_aliases"}:
            raise ValueError("Unsupported PAPER_LOCAL_BM25_ROUTE_MODE.")
        paper_route_queries: list[tuple[str, str]] = []
        paper_local_route_queries: list[tuple[str, str]] = []
        hyde_audit: dict[str, Any] = {"enabled": False, "replaced_by": "evidence_planning"}
        paper_claim_config = paper_conditioned_claims_config or PaperConditionedClaimsConfig()
        paper_claim_audit: dict[str, Any] = {"enabled": False, "mode": "evidence_planning"}
        if is_multi_paper_task and paper_claim_config.enabled:
            try:
                plan, warnings, cache_hit = generate_evidence_plan(
                    query_id=query_id,
                    query=query,
                    primary_evidence_type=preferred[0] if preferred else None,
                    candidate_papers=candidate_paper_metadata or [],
                    config=paper_claim_config,
                    client=hyde_client,
                    cache_root=Path(processed_root) / ".multi_paper_hyde_cache",
                )
                paper_route_queries = [
                    (item["paper_id"], " ".join([*item["source_types"], item["retrieval_query"]]).strip())
                    for item in plan["plans"]
                ]
                paper_claim_audit = {
                    "enabled": True,
                    "mode": "evidence_planning",
                    "cross_paper": plan["cross_paper"],
                    "plan_count": len(plan["plans"]),
                    "cache_hit": cache_hit,
                    "warnings": warnings,
                }
            except Exception as exc:
                paper_claim_audit = {"enabled": True, "mode": "evidence_planning", "fallback": str(exc)}
        paper_local_bm25_audit: dict[str, Any] = {"enabled": False}
        if is_multi_paper_task and paper_local_bm25_route_mode != "disabled":
            route_query = query_for_relevance_mode(query, paper_local_bm25_route_mode)
            paper_ids = sorted({str(record.get("paper_id") or "") for record in valid if record.get("paper_id")})
            paper_local_route_queries = [(paper_id, route_query) for paper_id in paper_ids]
            paper_local_bm25_audit = {
                "enabled": True,
                "mode": paper_local_bm25_route_mode,
                "paper_count": len(paper_ids),
                "route_query": route_query,
            }
        package_config = EvidencePackageConfig(
            package_budget=max(1, int(evidence_package_budget or evidence_total_budget or top_n_records)),
            min_package_budget=max(1, int(evidence_package_min_budget)),
            min_distinct_papers=max(1, int(multi_paper_min_distinct_papers)),
            adaptive_stop=bool(evidence_package_adaptive_stop),
            modality_packages_per_paper=max(1, int(multi_paper_modality_packages_per_paper)),
            supporting_text_packages_per_paper=max(0, int(multi_paper_supporting_text_packages_per_paper)),
            page_text_anchors_per_page=max(0, int(evidence_package_page_text_anchors_per_page)),
            # Zero preserves the inherited generation-context default; a
            # negative value is an explicit unbounded audit-selection mode.
            max_context_chars=(
                evidence_package_max_context_chars
                if evidence_package_max_context_chars != 0
                else (max_context_chars or 80000)
            ),
            text_neighbors=max(0, int((section_relevance_config or SectionRelevanceConfig()).object_neighbor_records)),
            rrf_k=max(1, int(evidence_package_rrf_k)),
            candidate_pool_per_route=max(0, int(evidence_package_candidate_pool_per_route)),
            max_packages_per_page=max(1, int(evidence_package_max_per_page)),
        )
        package_result = select_packages(
            query=query,
            source_hint_query=question,
            packages=build_packages(valid, relevance["trace"], package_config),
            preferred_source_types=preferred,
            is_multi_paper_task=is_multi_paper_task,
            route_queries=route_queries,
            slot_route_queries=slot_route_queries,
            slot_paper_route_queries=slot_paper_route_queries,
            paper_route_queries=paper_route_queries,
            paper_local_route_queries=paper_local_route_queries,
            config=package_config,
        )
        relevance["trace"]["hyde"] = hyde_audit
        relevance["trace"]["paper_conditioned_claims"] = paper_claim_audit
        relevance["trace"]["paper_local_bm25_route"] = paper_local_bm25_audit
        relevance["trace"]["slot_package_routes"] = slot_package_routes
        relevance["trace"]["slot_paper_package_routes"] = slot_paper_package_routes
        relevance["trace"]["package_selection"] = package_result["trace"]
        processed = Path(processed_root)
        selected_records_internal: list[dict[str, Any]] = []
        for record in package_result["records"]:
            source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
            if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
                continue
            selected = dict(record)
            selected["source_type"] = source_type
            selected["grounding_label"] = grounding_label_from_record(source_type, record.get("label"))
            selected["image_path"] = _image_path_for(record, processed, parser_model_slug)
            if source_type_hints_enabled:
                hints = infer_source_type_hints(selected)
                if hints:
                    selected["source_type_hints"] = hints
            selected_records_internal.append(selected)
        attachments: list[dict[str, Any]] = []
        attachment_by_path: dict[str, dict[str, Any]] = {}
        for rank, record in enumerate(selected_records_internal, start=1):
            record["evidence_ref"] = f"E{rank:04d}"
            image_path = str(record.get("image_path") or "")
            if vlm2_context_mode != "cropped_image" or not image_path:
                continue
            attachment = attachment_by_path.get(image_path)
            if attachment is None:
                if len(attachments) >= max(0, int(top_n_visual_records)):
                    continue
                attachment = {
                    "image_ref": f"IMG{len(attachments) + 1:03d}",
                    "path": image_path,
                    "evidence_refs": [],
                }
                attachment_by_path[image_path] = attachment
                attachments.append(attachment)
            record["image_ref"] = attachment["image_ref"]
            attachment["evidence_refs"].append(record["evidence_ref"])
        selected_records_debug = []
        for rank, record in enumerate(selected_records_internal, start=1):
            debug = {key: value for key, value in record.items() if not key.startswith("_")}
            debug["page_status"] = record.get("_page_status") or record.get("page_status") or "unknown"
            debug["selection_rank"] = rank
            debug["selection_method"] = "evidence_package_coverage"
            selected_records_debug.append(debug)
        selected_evidence = [project_context_for_vlm2(record, vlm2_context_mode) for record in selected_records_internal]
        primary_source = preferred[0] if preferred else ""
        compact_chunk_packets = _compact_package_packets(package_result["packages"], selected_records_internal)
        prompt_audit = audit_selected_context(selected_evidence, compact_chunk_packets, attachments)
        distribution = dict(Counter(str(record.get("source_type")) for record in selected_evidence))
        for source_type in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            distribution.setdefault(source_type, 0)
        visual_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in selected_records_internal
            if record.get("record_type") in VISUAL_RECORD_TYPES
        ]
        selected_record_ids = {_record_key(record) for record in selected_records_internal}
        selected_section_summary = [
            {
                "package_id": package["package_id"],
                "anchor_record_id": package["anchor_record_id"],
                "paper_id": package["paper_id"],
                "page": package["page"],
                "section_id": package["section_id"],
                "section_title": package["section_title"],
                "section_type": package["section_type"],
                "source_type": package["source_type"],
                "label": package["label"],
                "qwen_score": package.get("qwen"),
                "qwen_text_score": package.get("qwen_text"),
                "qwen_visual_score": package.get("qwen_visual"),
                "layout_score": package.get("layout_score"),
                "rrf_score": package["rrf_score"],
                "expansion_mode": "evidence_package",
                "records": [
                    dict(record)
                    for record in package["records"]
                    if _record_key(record) in selected_record_ids
                ],
            }
            for package in package_result["packages"]
        ]
        return {
            "query_id": query_id,
            "selection_method": "evidence_package_coverage",
            "prompt_context_mode": vlm2_context_mode,
            "partial_artifacts_present": any(record.get("_page_status") == "partial" for record in selected_records_internal),
            "partial_selected_record_count": sum(record.get("_page_status") == "partial" for record in selected_records_internal),
            "has_partial_artifacts": any(record.get("_page_status") == "partial" for record in selected_records_internal),
            "selected_evidence": selected_evidence,
            "compact_chunk_packets": compact_chunk_packets,
            "attached_image_refs": [
                {"image_ref": item["image_ref"], "evidence_refs": item["evidence_refs"]}
                for item in attachments
            ],
            "attached_image_paths": [item["path"] for item in attachments],
            "prompt_audit": prompt_audit,
            "selected_records_debug": selected_records_debug,
            "selected_records": selected_records_debug,
            "selected_visual_records": visual_records[:top_n_visual_records],
            "source_type_distribution": distribution,
            "primary_evidence_type_count": sum(record.get("source_type") in preferred_set for record in selected_evidence),
            "supporting_evidence_count": max(0, len(selected_evidence) - sum(record.get("source_type") in preferred_set for record in selected_evidence)),
            "grounding_label_hints_by_type": dict(Counter(str((record.get("grounding_label") or {}).get("type")) for record in selected_evidence if record.get("grounding_label"))),
            "context_selection_mode": normalized_mode,
            "context_truncated": False,
            "selected_record_count": len(selected_evidence),
            "selected_context_groups": selected_section_summary,
            "targeted_candidate_paper_ids": list(dict.fromkeys(
                route["paper_id"] for route in slot_paper_package_routes
            )),
            "section_relevance_trace": relevance["trace"],
        }
    query_tokens = tokenize(query)
    corpus = [
        tokenize(
            " ".join(
                [
                    str(r.get("text") or ""),
                    str(r.get("label") or ""),
                    str(r.get("section_title") or ""),
                    " ".join(str(item) for item in (r.get("section_path") or [])),
                ]
            )
        )
        for r in valid
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(valid)
    boosts: dict[str, float] = {}
    for source_type in preferred:
        for record_type, boost in TYPE_BOOSTS.get(source_type, {}).items():
            boosts[record_type] = boosts.get(record_type, 0.0) + float(boost)
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
            query=query,
            text=text,
            label=record.get("label"),
            targets=targets,
        )
        section_title_bonus, section_type_bonus = _section_query_bonus(
            query,
            record.get("section_title"),
            record.get("section_path"),
            record.get("section_type"),
        )
        score = float(bm25_score)
        score += boosts.get(str(record.get("record_type")), 0.0)
        score += boosts.get(source_type, 0.0)
        score += source_bonus
        score += section_title_bonus + section_type_bonus
        score += 0.15 * float(record.get("_candidate_bm25_score") or 0.0)
        record_type = str(record.get("record_type") or "")
        if str(record.get("_page_status") or record.get("page_status") or "") == "partial":
            score -= 0.5
        if record_type in HEADER_FOOTER_RECORD_TYPES:
            score -= 0.75
        penalize_citation = bool(preferred) and "citation_context" not in preferred
        if record_type == "citation_context" and penalize_citation:
            score -= 1.0
        if len(str(record.get("text") or "")) > 2500 and record_type == "citation_context" and penalize_citation:
            score -= 0.75
        selected = {
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "global_record_id": record.get("global_record_id"),
            "record_id": record.get("record_id"),
            "record_type": record.get("record_type"),
            "source_type": source_type,
            "label": record.get("label"),
            "section_id": record.get("section_id"),
            "section_title": record.get("section_title"),
            "section_level": record.get("section_level"),
            "section_path": record.get("section_path"),
            "section_type": record.get("section_type"),
            "document_order": record.get("document_order"),
            "grounding_label": grounding_label_from_record(source_type, record.get("label")),
            "score": round(score, 6),
            "source_type_specific_bonus": round(source_bonus, 6),
            "section_title_match_bonus": round(section_title_bonus, 6),
            "section_type_bonus": round(section_type_bonus, 6),
            "candidate_paper_prior": round(0.15 * float(record.get("_candidate_bm25_score") or 0.0), 6),
            "text": record.get("text"),
            "locator": record.get("locator") or {"page": record.get("page")},
            "image_path": _image_path_for(record, processed, parser_model_slug),
            "crop_path": record.get("crop_path"),
            "table_crop_path": record.get("table_crop_path"),
            "figure_crop_path": record.get("figure_crop_path"),
            "equation_algorithm_crop_path": record.get("equation_algorithm_crop_path"),
        }
        if source_type_hints_enabled:
            hints = infer_source_type_hints(selected)
            if hints:
                selected["source_type_hints"] = hints
        selected["_page_status"] = record.get("_page_status") or record.get("page_status")
        ranked.append(selected)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected_context_groups: list[dict[str, Any]] = []
    section_aware_available = any(record.get("section_id") for record in ranked)
    if section_aware_available:
        anchor_budget = evidence_total_budget or top_n_records
        anchors = _select_with_type_budgets(
            ranked,
            anchor_budget,
            preferred,
            primary_evidence_min,
            support_text_min,
            context_types_enabled,
            context_type_budget_per_type,
        )
        selected_context_groups, expanded = _section_context_groups(
            anchors,
            sorted(ranked, key=lambda item: (_document_order(item), str(item.get("record_id") or ""))),
            normalized_mode,
            max_groups=max(1, min(8, anchor_budget)),
        )
        selected_records_internal, context_truncated = _limit_records(
            expanded,
            max_context_records or anchor_budget,
            max_context_chars,
        )
        selection_method = "symbolic_section_aware_anchor_expansion"
    elif normalized_mode in {"ranked_budget", "bm25_budget", "record_budget"}:
        selected_records_internal = _select_with_type_budgets(
            ranked,
            evidence_total_budget or top_n_records,
            preferred,
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
    primary_count = sum(1 for r in selected_evidence if r.get("source_type") in preferred_set)
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
        if r.get("record_type") in VISUAL_RECORD_TYPES
        and (
            Path(str(r.get("crop_path"))).exists()
            or Path(str(r.get("table_crop_path"))).exists()
            or Path(str(r.get("figure_crop_path"))).exists()
            or Path(str(r.get("equation_algorithm_crop_path"))).exists()
            or Path(str(r.get("image_path"))).exists()
        )
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
        "selected_context_groups": selected_context_groups,
    }
