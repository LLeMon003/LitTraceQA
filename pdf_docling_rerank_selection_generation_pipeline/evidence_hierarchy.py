"""Auditable L0-L3 evidence hierarchy used after frozen selection.

L0 is always an exact parser record with an official locator.  L1 provides
local, source-specific context while remaining a collection of L0 references.
L2 cards are query-aware propositions whose fields are accepted only when an
exact supporting quote is present.  L3 is deterministic navigation metadata;
it never becomes the sole support for an answer fact.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .metadata_index import tokenize
from .parser import extract_json_object
from .symbolic_context_selector import grounding_label_from_record
from .symbolic_schema import canonicalize_locator, to_official_source_type
from .table_structure import table_text_to_structure


HIERARCHY_VERSION = "l0_l1_l2_l3_v3_contextual_triples"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:%|x|×|[A-Za-z]+)?")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{1,}|[A-Z]{2,}[A-Z0-9_-]*)\b")
_MIXED_IDENTIFIER_RE = re.compile(r"\b[A-Za-z0-9_-]*\d+[A-Za-z][A-Za-z0-9_-]*\b")
_ENTITY_STOP = {"A", "An", "At", "For", "From", "In", "Of", "On", "The", "This", "That", "Table", "Figure", "Equation", "Algorithm"}


def _clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def _query_aware_extractive_proposition(record: dict[str, Any], query_terms: set[str], limit: int) -> str:
    """Return a short L2 extractive proposition, centered on query evidence.

    A prefix crop often contains running headers or scene-setting prose while
    the requested value is later in a paragraph.  This remains fully
    extractive: it only selects one original sentence/line and, if needed, a
    bounded window around a query term.
    """
    raw = str(record.get("text") or "").replace("\x00", " ")
    candidates = [re.sub(r"\s+", " ", value).strip() for value in _SENTENCE_RE.split(raw)]
    candidates = [value for value in candidates if len(value) >= 12]
    if not candidates:
        return _clip(raw, limit)
    label_terms = set(tokenize(str(record.get("label") or "")))
    def score(value: str, index: int) -> tuple[int, int, int, int]:
        terms = set(tokenize(value))
        overlap = len(query_terms.intersection(terms))
        label_overlap = len(label_terms.intersection(terms))
        # Numbers and named entities are often the requested answer, but do
        # not outweigh explicit query alignment.
        facts = len(_NUMBER_RE.findall(value)) + len(_ENTITY_RE.findall(value))
        boilerplate = int(bool(re.fullmatch(r"(?:\d+|[A-Za-z ]{0,24})", value)))
        return (overlap * 8 + label_overlap * 2 + min(facts, 4) - boilerplate * 6, overlap, facts, -index)
    sentence = max(enumerate(candidates), key=lambda item: score(item[1], item[0]))[1]
    if limit <= 0 or len(sentence) <= limit:
        return sentence
    lowered = sentence.lower()
    positions = [lowered.find(term.lower()) for term in query_terms if len(term) >= 3 and lowered.find(term.lower()) >= 0]
    if positions:
        position = min(positions)
        start = max(0, min(len(sentence) - limit, position - max(0, limit // 3)))
        end = min(len(sentence), start + limit)
        excerpt = sentence[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(sentence):
            excerpt = excerpt[: max(1, limit - 3)].rstrip() + "..."
        return excerpt
    # Preserve both ends for a non-lexical question such as "what happened?".
    head = max(1, int(limit * 0.72))
    tail = max(1, limit - head - 5)
    return sentence[:head].rstrip() + " ... " + sentence[-tail:].lstrip()


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or record.get("record_id") or "")


def _record_order(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(record.get("document_order") or record.get("reading_order") or 10**9),
        int(record.get("page") or 10**9),
        _record_id(record),
    )


def _is_context_candidate(record: dict[str, Any]) -> bool:
    return bool(str(record.get("text") or "").strip()) and to_official_source_type(
        record.get("record_type"), record.get("source_type")
    ) is not None


def load_processed_records(processed_root: str | Path | None, paper_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """Load local parser records, restoring crop fields from parser debug artifacts.

    Standardized runtime records intentionally omit backend payloads.  Their
    paired debug records retain local crop paths, and the main pipeline merges
    those fields before selection.  The hierarchy must use the same merge or
    visual evidence becomes text-only after frozen selection.
    """
    root = Path(processed_root) if processed_root else None
    result: dict[str, list[dict[str, Any]]] = {}
    if root is None or not root.is_dir():
        return result
    for paper_id in sorted({str(value) for value in paper_ids if str(value)}):
        paper_root = root / paper_id
        runtime_path = paper_root / "symbolic_records.runtime.jsonl"
        debug_path = paper_root / "symbolic_records.debug.jsonl"
        path = runtime_path if runtime_path.is_file() else debug_path
        if not path.is_file():
            continue
        debug_by_id: dict[str, dict[str, Any]] = {}
        if runtime_path.is_file() and debug_path.is_file():
            for line in debug_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and _record_id(row):
                    debug_by_id[_record_id(row)] = row
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and _is_context_candidate(row):
                debug = debug_by_id.get(_record_id(row)) or {}
                for key in ("crop_path", "table_crop_path", "figure_crop_path", "equation_algorithm_crop_path", "image_path"):
                    if debug.get(key):
                        row[key] = debug[key]
                rows.append(row)
        result[paper_id] = sorted(rows, key=_record_order)
    return result


def _local_neighbors(anchor: dict[str, Any], paper_records: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    if not paper_records:
        return []
    anchor_id = _record_id(anchor)
    anchor_order = _record_order(anchor)[0]
    anchor_section = str(anchor.get("section_id") or "")
    anchor_page = int(anchor.get("page") or 0)
    candidates = [record for record in paper_records if _record_id(record) and _record_id(record) != anchor_id]
    # Prefer adjacent prose in the same section, then same-page explanatory
    # material. Objects remain in L0 themselves; this only gives them context.
    candidates.sort(
        key=lambda record: (
            str(record.get("section_id") or "") != anchor_section,
            int(record.get("page") or 0) != anchor_page,
            abs(_record_order(record)[0] - anchor_order),
            _record_order(record),
        )
    )
    return candidates[: max(0, limit)]


def _l0_projection(record: dict[str, Any], *, role: str) -> dict[str, Any]:
    source_type = to_official_source_type(record.get("record_type"), record.get("source_type")) or "text_span"
    locator = canonicalize_locator(source_type, record.get("locator"), record.get("label"))
    locator.setdefault("page", record.get("page"))
    return {
        "paper_id": record.get("paper_id"),
        "global_record_id": _record_id(record),
        "page": record.get("page"),
        "source_type": source_type,
        "label": record.get("label"),
        "locator": locator,
        "section_id": record.get("section_id"),
        "section_title": record.get("section_title"),
        "section_type": record.get("section_type"),
        "section_path": record.get("section_path"),
        "document_order": record.get("document_order"),
        "reading_order": record.get("reading_order"),
        "selection_rank": record.get("selection_rank"),
        "cached_selection_sources": list(record.get("cached_selection_sources") or []),
        "text": str(record.get("text") or ""),
        "crop_path": record.get("crop_path") or record.get("image_path"),
        "table_structure": record.get("table_structure") if isinstance(record.get("table_structure"), dict) else None,
        "role": role,
    }


def _add_l0(catalog: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], record: dict[str, Any], *, role: str) -> dict[str, Any] | None:
    identifier = _record_id(record)
    if not identifier:
        return None
    existing = by_id.get(identifier)
    if existing:
        if role == "selected_anchor":
            existing["role"] = role
        navigation_reason = str(record.get("navigation_reason") or "")
        if navigation_reason:
            reasons = list(existing.get("navigation_reasons") or [])
            if navigation_reason not in reasons:
                existing["navigation_reasons"] = [*reasons, navigation_reason]
        return existing
    projected = _l0_projection(record, role=role)
    navigation_reason = str(record.get("navigation_reason") or "")
    if navigation_reason:
        projected["navigation_reasons"] = [navigation_reason]
    by_id[identifier] = projected
    catalog.append(projected)
    return projected


_NAVIGATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "cited", "does", "for", "from", "how", "in", "is", "it",
    "kind", "many", "of", "on", "paper", "reference", "the", "to", "used", "was", "what", "which", "who",
}
_ORDINAL_REFERENCE_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+reference\b", re.IGNORECASE)


def _paper_navigation_score(question: str, candidate: dict[str, Any]) -> int:
    """Return a conservative title-to-question identity score.

    This is a deterministic navigation route, not a replacement retrieval
    score.  An exact leading title alias (for example ``S-RAG`` or
    ``EasySpec``) is intentionally much stronger than generic word overlap.
    """
    title = str(candidate.get("title") or "")
    question_terms = {term for term in tokenize(question) if term not in _NAVIGATION_STOPWORDS}
    title_terms = {term for term in tokenize(title) if term not in _NAVIGATION_STOPWORDS}
    overlap = len(question_terms.intersection(title_terms))
    prefix = title.split(":", 1)[0]
    alias = re.sub(r"[^a-z0-9]+", "", prefix.lower())
    question_compact = re.sub(r"[^a-z0-9]+", "", question.lower())
    return overlap + (20 if len(alias) >= 4 and alias in question_compact else 0)


def _citation_number(record: dict[str, Any]) -> int | None:
    locator = record.get("locator") if isinstance(record.get("locator"), dict) else {}
    value = locator.get("citation_id")
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:^|\n|\s)\[?(\d{1,4})\]?[.)]?\s", str(record.get("text") or ""))
    return int(match.group(1)) if match else None


def _deterministic_navigation_records(
    question: str,
    candidates: list[dict[str, Any]],
    processed_records: dict[str, list[dict[str, Any]]],
    *,
    limit_per_paper: int = 4,
) -> list[dict[str, Any]]:
    """Find a few auditable L0 entry points for explicitly named papers.

    The frozen Qwen result remains the quality path.  This additive route is
    reserved for deterministic identifiers that Qwen package selection can
    miss, notably an ordinal bibliography reference and a title/alias named in
    the question.  Every returned item is still an unmodified parser record.
    """
    target_papers = [
        candidate for candidate in candidates
        if str(candidate.get("paper_id") or "") in processed_records and _paper_navigation_score(question, candidate) >= 3
    ]
    if not target_papers:
        return []
    question_terms = {term for term in tokenize(question) if term not in _NAVIGATION_STOPWORDS and len(term) >= 3}
    reference_match = _ORDINAL_REFERENCE_RE.search(question)
    reference_number = int(reference_match.group(1)) if reference_match else None
    routed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in sorted(target_papers, key=lambda item: (-_paper_navigation_score(question, item), str(item.get("paper_id") or ""))):
        paper_id = str(candidate.get("paper_id") or "")
        records = processed_records.get(paper_id) or []
        direct: list[tuple[int, dict[str, Any], str]] = []
        lexical: list[tuple[int, dict[str, Any], str]] = []
        for record in records:
            if not _is_context_candidate(record):
                continue
            source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
            if reference_number is not None and source_type == "citation_context" and _citation_number(record) == reference_number:
                direct.append((0, record, f"explicit_citation_id:{reference_number}"))
                continue
            text_terms = set(tokenize(" ".join([str(record.get("label") or ""), str(record.get("section_title") or ""), str(record.get("text") or "")])))
            overlap = len(question_terms.intersection(text_terms))
            if overlap:
                # Prefer the requested evidence type only when the textual
                # match exists; no type-only records are admitted.
                primary_bonus = 3 if source_type == "text_span" else 0
                lexical.append((-(overlap * 10 + primary_bonus), record, "lexical_text_navigation"))
        for _, record, reason in sorted(direct, key=lambda item: _record_order(item[1])):
            identifier = _record_id(record)
            if identifier and identifier not in seen:
                routed.append({**record, "navigation_reason": reason})
                seen.add(identifier)
        remaining = max(0, limit_per_paper - len(direct))
        for _, record, reason in sorted(lexical, key=lambda item: (item[0], _record_order(item[1])))[:remaining]:
            identifier = _record_id(record)
            if identifier and identifier not in seen:
                routed.append({**record, "navigation_reason": reason})
                seen.add(identifier)
    return routed


def _l1_for_anchor(anchor: dict[str, Any], neighbor_refs: list[str], catalog_by_ref: dict[str, dict[str, Any]], max_chars: int) -> dict[str, Any]:
    source_type = str(anchor.get("source_type") or "text_span")
    neighbors = [catalog_by_ref[ref] for ref in neighbor_refs if ref in catalog_by_ref]
    local = [_clip(item.get("text"), max_chars) for item in neighbors]
    base: dict[str, Any] = {
        "anchor_ref": anchor["evidence_ref"],
        "heading_path": anchor.get("section_path") or [anchor.get("section_title")] if anchor.get("section_title") else [],
        "neighbor_refs": neighbor_refs,
        "neighbor_excerpts": local,
    }
    text = str(anchor.get("text") or "")
    if source_type == "table":
        structure = anchor.get("table_structure") if isinstance(anchor.get("table_structure"), dict) else table_text_to_structure(text)
        if structure:
            base["table_context"] = {
                "caption": _clip(text.splitlines()[0] if text.splitlines() else "", max_chars),
                "header_rows": structure.get("header_rows", []),
                "columns": structure.get("columns", []),
                "rows": structure.get("rows", [])[:8],
                "parse_warnings": structure.get("parse_warnings", []),
            }
    elif source_type == "figure":
        base["figure_context"] = {
            "caption": _clip(text, max_chars),
            "has_crop": bool(anchor.get("crop_path")),
        }
    elif source_type == "equation_algorithm":
        base["equation_context"] = {
            "block": _clip(text, max_chars * 2),
            "label": anchor.get("label"),
        }
    elif source_type == "citation_context":
        base["citation_context"] = {"citation_text": _clip(text, max_chars)}
    return base


def _navigation(catalog: list[dict[str, Any]], candidates: list[dict[str, Any]], paper_chars: int) -> dict[str, Any]:
    candidate_by_id = {str(row.get("paper_id") or ""): row for row in candidates}
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in catalog:
        by_paper[str(record.get("paper_id") or "")].append(record)
        by_section[(str(record.get("paper_id") or ""), str(record.get("section_id") or ""))].append(record)
    papers = []
    for paper_id, records in by_paper.items():
        metadata = candidate_by_id.get(paper_id, {})
        papers.append({
            "paper_id": paper_id,
            "title": metadata.get("title"),
            "abstract_preview": _clip(metadata.get("abstract"), paper_chars),
            "source_type_counts": dict(sorted(Counter(str(record.get("source_type") or "") for record in records).items())),
        })
    sections = []
    for (paper_id, section_id), records in by_section.items():
        first = records[0]
        sections.append({
            "paper_id": paper_id,
            "section_id": section_id,
            "section_title": first.get("section_title"),
            "section_path": first.get("section_path"),
            "anchor_refs": [str(record.get("evidence_ref")) for record in records if record.get("role") == "selected_anchor"],
            "source_type_counts": dict(sorted(Counter(str(record.get("source_type") or "") for record in records).items())),
        })
    graph_edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in catalog:
        ref = str(record.get("evidence_ref") or "")
        for entity in _ENTITY_RE.findall(_clip(record.get("text"), 900))[:8]:
            key = (ref, entity)
            if ref and key not in seen:
                seen.add(key)
                graph_edges.append({"evidence_ref": ref, "entity": entity})
    return {"papers": papers, "sections": sections, "entity_mentions": graph_edges[:160]}


def build_l0_l1_l3(
    selected_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    processed_records: dict[str, list[dict[str, Any]]],
    *,
    question: str = "",
    l1_max_chars: int,
    l3_paper_chars: int,
    neighbor_limit: int = 2,
) -> dict[str, Any]:
    """Expand selected L0 anchors with bounded local raw context."""
    catalog: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    anchors: list[dict[str, Any]] = []
    processed_by_id = {
        _record_id(record): record
        for records in processed_records.values()
        for record in records
        if _record_id(record)
    }
    for selection_rank, raw in enumerate(selected_records, start=1):
        if not isinstance(raw, dict) or not _is_context_candidate(raw):
            continue
        raw = dict(raw)
        # Selector expansions may serialize the field as null. Treat that as
        # missing and preserve the frozen selected_records order rather than
        # silently pushing those records behind every ranked anchor.
        if not isinstance(raw.get("selection_rank"), int) or int(raw.get("selection_rank") or 0) <= 0:
            raw["selection_rank"] = selection_rank
        # Frozen Qwen selections are intentionally reusable across parser-only
        # changes.  Refresh the selected anchor from the current processed
        # artifact so a newly recovered table structure, crop, text, or
        # locator reaches L0/L2 without invalidating the score cache.
        current = processed_by_id.get(_record_id(raw))
        if current:
            selection_fields = {
                key: raw.get(key)
                for key in ("selection_rank", "cached_selection_sources", "qwen_score", "score", "rank")
                if raw.get(key) is not None
            }
            raw = {**raw, **current, **selection_fields}
        projected = _add_l0(catalog, by_id, raw, role="selected_anchor")
        if projected:
            anchors.append(projected)
    l1_pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for anchor in anchors:
        local = _local_neighbors(anchor, processed_records.get(str(anchor.get("paper_id") or ""), []), neighbor_limit)
        l1_pending.append((anchor, local))
        for record in local:
            _add_l0(catalog, by_id, record, role="l1_context")
    # Add only explicit-question navigation records after the frozen selected
    # anchors. They become independently auditable L0/L1 entries and cannot
    # silently replace a Qwen-selected package.
    for routed in _deterministic_navigation_records(question, candidates, processed_records):
        projected = _add_l0(catalog, by_id, routed, role="deterministic_navigation")
        if projected is None:
            continue
        local = _local_neighbors(routed, processed_records.get(str(routed.get("paper_id") or ""), []), neighbor_limit)
        l1_pending.append((projected, local))
        for record in local:
            _add_l0(catalog, by_id, record, role="l1_context")
    catalog.sort(key=_record_order)
    for index, record in enumerate(catalog, start=1):
        record["evidence_ref"] = f"E{index:04d}"
        record["grounding_label"] = grounding_label_from_record(str(record.get("source_type") or "text_span"), record.get("label"))
    ref_by_id = {str(record.get("global_record_id") or ""): str(record.get("evidence_ref") or "") for record in catalog}
    catalog_by_ref = {str(record.get("evidence_ref") or ""): record for record in catalog}
    l1 = []
    for anchor, local in l1_pending:
        anchor_ref = ref_by_id.get(str(anchor.get("global_record_id") or ""), "")
        if not anchor_ref:
            continue
        actual_anchor = catalog_by_ref[anchor_ref]
        refs = [ref_by_id.get(_record_id(record), "") for record in local]
        l1.append(_l1_for_anchor(actual_anchor, [ref for ref in refs if ref], catalog_by_ref, l1_max_chars))
    return {
        "version": HIERARCHY_VERSION,
        "l0_catalog": catalog,
        "selected_anchor_refs": [ref_by_id.get(str(anchor.get("global_record_id") or ""), "") for anchor in anchors],
        "l1_contexts": l1,
        "l3_navigation": _navigation(catalog, candidates, l3_paper_chars),
    }


def _extractive_claims(question: str, max_claims: int) -> list[dict[str, str]]:
    chunks = [part.strip(" ,;:.?") for part in re.split(r"(?:;|\?|\band\b|\bwhile\b)", question, flags=re.IGNORECASE) if part.strip()]
    claims = chunks[: max(1, max_claims)] or [question]
    return [{"claim_id": f"Q{index:02d}", "claim": claim} for index, claim in enumerate(claims, start=1)]


def _table_view(text: str, query_terms: set[str], structure: dict[str, Any] | None = None) -> dict[str, Any] | None:
    structure = structure if isinstance(structure, dict) else table_text_to_structure(text)
    if not structure:
        return None
    rows = list(structure.get("rows") or [])
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    active_group = ""
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        values = row.get("values") if isinstance(row.get("values"), dict) else {}
        row_text = " ".join(str(value) for value in values.values())
        if bool(row.get("row_section")):
            active_group = str(row.get("row_label") or row_text).strip()
        row["group_label"] = active_group
        row_overlap = len(query_terms.intersection(tokenize(row_text)))
        group_overlap = len(query_terms.intersection(tokenize(active_group)))
        salient_match = bool(re.search(r"\b(?:absolute|relative|delta|improvement|ours|ic ae)\b|∆", row_text, re.IGNORECASE))
        # A grouped table often repeats the same method under conditions such
        # as 1-step and 2-step.  The group is therefore part of the row's
        # meaning, not merely presentation: rank it ahead of an earlier row
        # with the same method name but a mismatched condition.
        score = row_overlap * 10 + group_overlap * 8 + int(salient_match)
        if score:
            ranked.append((score, index, row))
    if ranked:
        relevant = [row for _, _, row in sorted(ranked, key=lambda item: (-item[0], item[1]))[:8]]
    else:
        relevant = [dict(row, group_label="") for row in rows[: min(6, len(rows))]]
    return {
        "caption": _clip(structure.get("caption") or (text.splitlines()[0] if text.splitlines() else ""), 420),
        "header_rows": structure.get("header_rows") or [],
        "columns": structure.get("columns") or [],
        "rows": relevant[:8],
        "footnotes": list(structure.get("footnotes") or [])[:4],
    }


def _table_proposition(view: dict[str, Any], limit: int = 1100) -> str:
    """Serialize a table view as directly readable row/cell evidence.

    A caption-only card forces the answer model to rediscover the relationship
    between the separate header and row arrays.  Keep the full header path and
    every selected row's values together in the extractive proposition while
    leaving the richer table_view available for schema-aware generation.
    """
    columns = [str(column) for column in view.get("columns") or []]
    parts = [str(view.get("caption") or "").strip()]
    if columns:
        parts.append("Columns: " + " | ".join(columns))
    for row in view.get("rows") or []:
        if not isinstance(row, dict):
            continue
        values = row.get("values") if isinstance(row.get("values"), dict) else {}
        row_label = str(row.get("row_label") or "").strip()
        group_label = str(row.get("group_label") or "").strip()
        cells = [f"{column}={str(values.get(column) or '').strip()}" for column in columns if str(values.get(column) or '').strip()]
        if cells:
            prefix = f"Group {group_label}; " if group_label and group_label != row_label else ""
            parts.append(f"{prefix}Row {row_label}: " + "; ".join(cells))
    return _clip("\n".join(part for part in parts if part), limit)


def _rank_extractive_fragments(
    question: str,
    catalog: list[dict[str, Any]],
    l1_by_anchor: dict[str, dict[str, Any]],
    primary_evidence_type: str = "",
) -> list[dict[str, Any]]:
    query_terms = set(tokenize(question))
    fragments: list[dict[str, Any]] = []
    for record in catalog:
        ref = str(record.get("evidence_ref") or "")
        text = str(record.get("text") or "")
        source_type = str(record.get("source_type") or "")
        if source_type == "table":
            view = _table_view(text, query_terms, record.get("table_structure"))
            if view:
                table_text = " ".join([
                    str(view.get("caption") or ""),
                    " ".join(" ".join(row) for row in view.get("header_rows") or []),
                    " ".join(str(row.get("values") or "") for row in view.get("rows") or []),
                ])
                overlap = len(query_terms.intersection(tokenize(table_text)))
                # A query naming a metric, benchmark, or comparison often
                # needs the table's header path even when the desired row has
                # no lexical overlap. This is a bounded extractive route, not
                # a score replacement for frozen Qwen selection.
                table_bonus = 3 if query_terms.intersection({"f1", "score", "accuracy", "benchmark", "outperform", "how", "much"}) else 0
                if source_type == primary_evidence_type:
                    table_bonus += 3
                fragments.append({
                    "evidence_ref": ref,
                    "text": _table_proposition(view),
                    "score": overlap + table_bonus,
                    "table_view": view,
                })
                # A table already has a structure-preserving card.  Adding a
                # second raw-markdown sentence card duplicates the object and
                # crowds out independent papers under a fixed L2 card budget.
                continue
        for sentence in _SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if len(sentence) < 16:
                continue
            overlap = len(query_terms.intersection(tokenize(sentence)))
            label_bonus = 2 if any(term in str(record.get("label") or "").lower() for term in query_terms) else 0
            primary_bonus = 2 if source_type == primary_evidence_type else 0
            fragments.append({"evidence_ref": ref, "text": _clip(sentence, 420), "score": overlap + label_bonus + primary_bonus})
    return sorted(fragments, key=lambda row: (-int(row["score"]), len(str(row["text"])), str(row["evidence_ref"])))


def build_extractive_cards(
    question: str,
    hierarchy: dict[str, Any],
    max_claims: int,
    max_cards: int,
    primary_evidence_type: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    claims = _extractive_claims(question, max_claims)
    catalog = list(hierarchy.get("l0_catalog") or [])
    l1_by_anchor = {str(row.get("anchor_ref") or ""): row for row in hierarchy.get("l1_contexts") or []}
    fragments = _rank_extractive_fragments(question, catalog, l1_by_anchor, primary_evidence_type)
    cards: list[dict[str, Any]] = []
    for index, fragment in enumerate(fragments[: max(1, max_cards)], start=1):
        ref = str(fragment["evidence_ref"])
        record = next((row for row in catalog if str(row.get("evidence_ref") or "") == ref), {})
        cards.append({
            "card_id": f"C{index:03d}",
            "claim_ids": [claims[min(index - 1, len(claims) - 1)]["claim_id"]],
            "proposition": fragment["text"],
            "entities": _ENTITY_RE.findall(fragment["text"])[:6],
            "values": _NUMBER_RE.findall(fragment["text"])[:8],
            "conditions": [],
            "paper_id": record.get("paper_id"),
            "source_type": record.get("source_type"),
            "locator": record.get("locator"),
            "support_refs": [ref],
            "support_quotes": [{"evidence_ref": ref, "quote": fragment["text"]}],
            "l1_refs": list((l1_by_anchor.get(ref) or {}).get("neighbor_refs") or []),
            "table_view": fragment.get("table_view"),
            "verification": {"status": "extractive_fallback", "reason": "no_verified_llm_card"},
        })
    return cards, claims


def card_generation_messages(question: str, hierarchy: dict[str, Any], max_claims: int, max_cards: int, source_chars: int = 85000) -> list[dict[str, str]]:
    catalog = hierarchy.get("l0_catalog") or []
    l1 = hierarchy.get("l1_contexts") or []
    query_terms = set(tokenize(question))
    def card_priority(record: dict[str, Any]) -> tuple[int, int, int, int]:
        text = " ".join((str(record.get("label") or ""), str(record.get("text") or "")))
        overlap = len(query_terms.intersection(tokenize(text)))
        direct_object = int(bool(re.search(r"\b(?:table|figure|fig\.|equation|algorithm|reference)\s*\d+", text, re.IGNORECASE)))
        anchor = int(record.get("role") == "selected_anchor")
        return (-overlap, -direct_object, -anchor, int(record.get("selection_rank") or 10**8))
    compact_l0 = []
    used = 0
    for record in sorted(catalog, key=card_priority):
        limit = 2200 if record.get("source_type") == "table" else 900
        projected = {
            "evidence_ref": record.get("evidence_ref"), "source_type": record.get("source_type"),
            "label": record.get("label"), "text": _clip(record.get("text"), limit),
        }
        serialized = len(json.dumps(projected, ensure_ascii=False, separators=(",", ":")))
        if compact_l0 and source_chars > 0 and used + serialized > source_chars:
            continue
        compact_l0.append(projected)
        used += serialized
    prompt = {
        "question": question,
        "limits": {"max_claims": max_claims, "max_cards": max_cards},
        "l0_records": compact_l0,
        "l1_contexts": l1,
    }
    system = (
        "Create 2-8 atomic query claims and compact grounded cards; do not answer the question or use outside knowledge. "
        "Every card fact must occur in an exact contiguous support_quote from its support_ref. Keep table headers, rows, units, and footnotes with each value. "
        "Return JSON only."
    )
    user = (
        "Return {\"claims\":[{\"claim_id\":\"Q01\",\"claim\":\"...\"}],"
        "\"cards\":[{\"claim_ids\":[\"Q01\"],\"proposition\":\"...\",\"entities\":[],\"values\":[],"
        "\"conditions\":[],\"support_refs\":[\"E0001\"],\"support_quotes\":[{\"evidence_ref\":\"E0001\",\"quote\":\"exact contiguous quote from E0001\"}],\"l1_refs\":[]}]}\n"
        + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _canonical_terms(value: Any) -> set[str]:
    return {term.lower() for term in tokenize(str(value or "")) if len(term) > 1}


def _space_normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def verify_llm_cards(raw: dict[str, Any], hierarchy: dict[str, Any], max_claims: int, max_cards: int) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    claims_raw = raw.get("claims") if isinstance(raw.get("claims"), list) else []
    claims: list[dict[str, str]] = []
    for index, claim in enumerate(claims_raw[: max(1, max_claims)], start=1):
        if isinstance(claim, dict) and str(claim.get("claim") or "").strip():
            claims.append({"claim_id": str(claim.get("claim_id") or f"Q{index:02d}"), "claim": _clip(claim.get("claim"), 360)})
    if not claims:
        claims = _extractive_claims("", max_claims)
    claim_ids = {row["claim_id"] for row in claims}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw_card in enumerate((raw.get("cards") or [])[: max(1, max_cards)], start=1):
        if not isinstance(raw_card, dict):
            rejected.append({"index": index, "reason": "not_object"})
            continue
        refs = [str(ref) for ref in raw_card.get("support_refs") or [] if str(ref) in catalog]
        quotes = raw_card.get("support_quotes") if isinstance(raw_card.get("support_quotes"), list) else []
        quote_text_by_ref: dict[str, list[str]] = defaultdict(list)
        for quote in quotes:
            if isinstance(quote, dict):
                ref, text = str(quote.get("evidence_ref") or ""), str(quote.get("quote") or "").strip()
                if ref in catalog and text and _space_normalized(text) in _space_normalized(catalog[ref].get("text")):
                    quote_text_by_ref[ref].append(text)
            elif isinstance(quote, str) and len(refs) == 1 and _space_normalized(quote) in _space_normalized(catalog[refs[0]].get("text")):
                quote_text_by_ref[refs[0]].append(quote.strip())
        if not refs or any(ref not in quote_text_by_ref for ref in refs):
            rejected.append({"index": index, "reason": "missing_exact_support_quote", "support_refs": refs})
            continue
        source_text = " ".join(text for values in quote_text_by_ref.values() for text in values)
        proposition = _clip(raw_card.get("proposition"), 520)
        if not proposition:
            rejected.append({"index": index, "reason": "empty_proposition"})
            continue
        # A deterministic grounding gate avoids accepting a fluent paraphrase
        # whose named entities or values cannot be traced to the quoted source.
        source_terms = _canonical_terms(source_text)
        proposition_terms = _canonical_terms(proposition)
        numeric_ok = all(value in source_text for value in _NUMBER_RE.findall(proposition))
        salient = [term for term in [*_ENTITY_RE.findall(proposition), *_MIXED_IDENTIFIER_RE.findall(proposition)] if len(term) > 2 and term not in _ENTITY_STOP]
        entity_ok = all(term.lower() in source_text.lower() for term in salient)
        overlap = len(proposition_terms.intersection(source_terms)) / max(1, len(proposition_terms))
        if not numeric_ok or not entity_ok or overlap < 0.42:
            rejected.append({"index": index, "reason": "proposition_not_extractive_enough", "overlap": round(overlap, 3)})
            continue
        accepted.append({
            "card_id": f"C{len(accepted) + 1:03d}",
            "claim_ids": [str(value) for value in raw_card.get("claim_ids") or [] if str(value) in claim_ids] or [claims[0]["claim_id"]],
            "proposition": proposition,
            "entities": [str(value) for value in raw_card.get("entities") or [] if str(value) and str(value).lower() in source_text.lower()][:8],
            "values": [str(value) for value in raw_card.get("values") or [] if str(value) and str(value) in source_text][:10],
            "conditions": [str(value) for value in raw_card.get("conditions") or [] if str(value) and str(value).lower() in source_text.lower()][:6],
            "paper_id": catalog[refs[0]].get("paper_id"),
            "source_type": catalog[refs[0]].get("source_type"),
            "locator": catalog[refs[0]].get("locator"),
            "support_refs": refs,
            "support_quotes": [{"evidence_ref": ref, "quote": text} for ref, values in quote_text_by_ref.items() for text in values],
            "l1_refs": [str(value) for value in raw_card.get("l1_refs") or [] if str(value) in catalog][:4],
            "verification": {"status": "accepted_extractive_gate", "overlap": round(overlap, 3)},
        })
    return accepted, claims, rejected


def attach_cards(
    question: str,
    hierarchy: dict[str, Any],
    *,
    mode: str,
    max_claims: int,
    max_cards: int,
    primary_evidence_type: str = "",
    llm_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_cards, fallback_claims = build_extractive_cards(question, hierarchy, max_claims, max_cards, primary_evidence_type)
    rejected: list[dict[str, Any]] = []
    if mode == "verified_llm" and isinstance(llm_result, dict):
        cards, claims, rejected = verify_llm_cards(llm_result, hierarchy, max_claims, max_cards)
        if not cards:
            cards, claims = fallback_cards, fallback_claims
        else:
            # Ensure every generated query claim has at least one directly
            # grounded card. The fallback is extractive and never overwrites a
            # verified card.
            covered = {claim_id for card in cards for claim_id in card.get("claim_ids") or []}
            for fallback in fallback_cards:
                if len(cards) >= max_cards:
                    break
                if any(claim_id not in covered for claim_id in fallback.get("claim_ids") or []):
                    fallback = dict(fallback)
                    fallback["card_id"] = f"C{len(cards) + 1:03d}"
                    cards.append(fallback)
                    covered.update(fallback.get("claim_ids") or [])
    else:
        cards, claims = fallback_cards, fallback_claims
    hierarchy["l2_evidence_cards"] = cards[:max_cards]
    hierarchy["query_claims"] = claims[:max_claims]
    hierarchy["l3_navigation"]["unresolved_claims"] = [
        claim for claim in hierarchy["query_claims"]
        if claim["claim_id"] not in {claim_id for card in hierarchy["l2_evidence_cards"] for claim_id in card.get("claim_ids") or []}
    ]
    hierarchy["card_verification"] = {
        "mode": mode,
        "accepted_card_count": sum(1 for card in hierarchy["l2_evidence_cards"] if card.get("verification", {}).get("status") == "accepted_extractive_gate"),
        "fallback_card_count": sum(1 for card in hierarchy["l2_evidence_cards"] if card.get("verification", {}).get("status") == "extractive_fallback"),
        "rejected": rejected,
    }
    return hierarchy


def hierarchy_metrics(hierarchy: dict[str, Any]) -> dict[str, Any]:
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    cards = hierarchy.get("l2_evidence_cards") or []
    support_refs = {str(ref) for card in cards for ref in card.get("support_refs") or []}
    selected = {str(ref) for ref in hierarchy.get("selected_anchor_refs") or [] if str(ref)}
    raw_chars = sum(len(str(row.get("text") or "")) for row in catalog.values())
    card_chars = len(json.dumps({"cards": cards, "navigation": hierarchy.get("l3_navigation")}, ensure_ascii=False, separators=(",", ":")))
    return {
        "hierarchy_version": HIERARCHY_VERSION,
        "l0_record_count": len(catalog),
        "selected_anchor_count": len(selected),
        "l1_context_count": len(hierarchy.get("l1_contexts") or []),
        "l2_card_count": len(cards),
        "exact_anchor_retention": 1.0 if selected.issubset(catalog) else 0.0,
        "card_support_reference_validity": 1.0 if support_refs.issubset(catalog) else 0.0,
        "l0_raw_chars": raw_chars,
        "l2_l3_serialized_chars": card_chars,
        "compression_ratio": round(card_chars / max(1, raw_chars), 4),
    }


def hierarchy_prompt_projection(hierarchy: dict[str, Any], l1_max_chars: int = 420) -> dict[str, Any]:
    """Return the compact, answer-facing view without discarding audit L0."""
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    cards: list[dict[str, Any]] = []
    needed_l1_refs: set[str] = set()
    for card in hierarchy.get("l2_evidence_cards") or []:
        if not isinstance(card, dict):
            continue
        refs = [str(value) for value in card.get("support_refs") or [] if str(value) in catalog]
        l1_refs = [str(value) for value in card.get("l1_refs") or [] if str(value) in catalog]
        needed_l1_refs.update(l1_refs)
        cards.append({
            "card_id": card.get("card_id"),
            "claim_ids": card.get("claim_ids") or [],
            "proposition": card.get("proposition"),
            "entities": card.get("entities") or [],
            "values": card.get("values") or [],
            "conditions": card.get("conditions") or [],
            "paper_id": card.get("paper_id"),
            "source_type": card.get("source_type"),
            "locator": card.get("locator"),
            "support_refs": refs,
            "support_quotes": card.get("support_quotes") or [],
            "l1_refs": l1_refs,
            "table_view": card.get("table_view"),
        })
    l1_expansions = [
        {
            "evidence_ref": ref,
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "source_type": record.get("source_type"),
            "label": record.get("label"),
            "locator": record.get("locator"),
            "section_path": record.get("section_path"),
            "text": _clip(record.get("text"), l1_max_chars),
        }
        for ref, record in catalog.items() if ref in needed_l1_refs
    ]
    micro_budget = int(hierarchy.get("prompt_micro_index_chars") or 26000)
    micro_text_chars = int(hierarchy.get("prompt_micro_text_chars") or 240)
    anchors = [record for record in catalog.values() if record.get("role") == "selected_anchor"]
    anchors.sort(key=lambda record: (int(record.get("selection_rank") or 10**8), _record_order(record)))
    is_multi = "multi" in str(hierarchy.get("task_family") or "").lower()
    query_terms = set(tokenize(" ".join(str(claim.get("claim") or "") for claim in hierarchy.get("query_claims") or [])))
    micro_order = str(hierarchy.get("keyed_micro_order") or hierarchy.get("micro_order") or "selection")
    query_aware_order = micro_order in {"query_aware", "stability_query_aware"}
    stability_aware_order = micro_order == "stability_query_aware"
    priority_cache = hierarchy.setdefault("_micro_query_priority_cache", {}) if query_aware_order else {}
    def priority(record: dict[str, Any]) -> tuple[int, int, int, int, tuple[int, int, str]]:
        ref = str(record.get("evidence_ref") or "")
        cached = priority_cache.get(ref)
        if isinstance(cached, tuple) and len(cached) == 4:
            return cached
        text = " ".join((str(record.get("label") or ""), str(record.get("text") or "")))
        overlap = len(query_terms.intersection(tokenize(text)))
        primary_source = str(hierarchy.get("primary_evidence_type") or "")
        primary = int(str(record.get("source_type") or "") == primary_source)
        facts = min(3, len(_NUMBER_RE.findall(text)) + len(_ENTITY_RE.findall(text)))
        # A provenance union is an ensemble, not a fresh reranker score. When
        # prompt fitting must drop rows, records retained by multiple frozen
        # selection paths are the least speculative anchors to preserve.
        stability = len(set(str(value) for value in record.get("cached_selection_sources") or []))
        result = (
            -stability if stability_aware_order else 0,
            -overlap,
            -primary,
            -facts,
            _record_order(record),
        )
        if ref:
            priority_cache[ref] = result
        return result
    ordered: list[dict[str, Any]] = []
    if is_multi:
        seen_papers: set[str] = set()
        paper_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in anchors:
            paper_rows[str(record.get("paper_id") or "")].append(record)
        paper_order = sorted(paper_rows, key=lambda paper_id: min((int(item.get("selection_rank") or 10**8) for item in paper_rows[paper_id]), default=10**8))
        for paper_id in paper_order:
            rows = paper_rows[paper_id]
            best = min(rows, key=priority) if query_aware_order else rows[0]
            if paper_id and paper_id not in seen_papers:
                ordered.append(best)
                seen_papers.add(paper_id)
    remaining = [record for record in anchors if record not in ordered]
    ordered.extend(sorted(remaining, key=priority) if query_aware_order else remaining)
    micro_evidence: list[dict[str, Any]] = []
    used = 0
    for record in ordered:
        source_type = str(record.get("source_type") or "")
        # L2 micro rows complement richer cards/table views; letting every
        # table consume 3x the width collapses evidence coverage under a fixed
        # generation budget. Keep a small structural allowance instead.
        text_limit = micro_text_chars * (2 if source_type == "table" else 1 if source_type in {"figure", "equation_algorithm"} else 1)
        text = _query_aware_extractive_proposition(record, query_terms, text_limit)
        row = {
            "support_ref": record.get("evidence_ref"), "paper_id": record.get("paper_id"), "page": record.get("page"),
            "source_type": source_type, "label": record.get("label"), "extractive_proposition": text,
        }
        # The keyed prompt carries S-keys instead of full section paths. Do not
        # charge omitted L1/L3 text against its micro-card budget.
        if not bool(hierarchy.get("_keyed_micro_projection", False)):
            row["section_path"] = record.get("section_path")
        if source_type == "table" and bool(hierarchy.get("include_table_views", True)):
            view = _table_view(str(record.get("text") or ""), query_terms)
            if view:
                row_parts = []
                for table_row in (view.get("rows") or [])[:2]:
                    if not isinstance(table_row, dict):
                        continue
                    values = table_row.get("values") if isinstance(table_row.get("values"), dict) else {}
                    value_text = " | ".join(
                        f"{column}={value}"
                        for column, value in values.items()
                        if value is not None and str(value).strip()
                    )
                    if value_text:
                        row_parts.append(" ".join(part for part in (str(table_row.get("row_label") or ""), value_text) if part))
                row["table_view"] = view
                row["extractive_proposition"] = _clip(
                    " ".join([
                        str(view.get("caption") or ""),
                        " | ".join(str(column) for column in view.get("columns") or []),
                        *row_parts,
                    ]),
                    text_limit,
                )
        cost = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        if micro_evidence and micro_budget > 0 and used + cost > micro_budget:
            continue
        micro_evidence.append(row)
        used += cost
    raw_navigation = hierarchy.get("l3_navigation") or {}
    navigation = {
        "papers": raw_navigation.get("papers") or [],
        # L3 is navigation only. Do not serialize every selected locator here:
        # the cards already carry exact support refs, and repeated anchor lists
        # would recreate the original context-budget failure.
        "sections": [
            {
                "paper_id": section.get("paper_id"),
                "section_id": section.get("section_id"),
                "section_title": section.get("section_title"),
                "section_path": section.get("section_path"),
                "source_type_counts": section.get("source_type_counts") or {},
            }
            for section in raw_navigation.get("sections") or []
        ],
        "entity_mentions": list(raw_navigation.get("entity_mentions") or [])[:60],
        "unresolved_claims": raw_navigation.get("unresolved_claims") or [],
    }
    return {
        "version": hierarchy.get("version", HIERARCHY_VERSION),
        "query_claims": hierarchy.get("query_claims") or [],
        "l2_evidence_cards": cards,
        "l2_micro_evidence": micro_evidence,
        "l1_on_demand_expansions": l1_expansions,
        "l3_navigation": navigation,
        "image_map": hierarchy.get("image_map") or [],
        "grounding_invariant": "Every answer fact must be supported by an L2 card support_ref or L2 micro-evidence support_ref. L3 is navigation only; L1 is explanatory context only.",
    }


def keyed_hierarchy_prompt_projection(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """L2-only prompt view: no L0/L1 text, quotes, locators, or raw refs.

    Card keys are the only grounding identifiers visible to generation.  The
    local ``key_index`` is enough for the runtime to resolve them back to L0.
    """
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    record_keys = {ref: f"R{index:04d}" for index, ref in enumerate(sorted(catalog), start=1)}
    key_index = {
        record_keys[ref]: {
            "record_key": record_keys[ref], "paper_id": record.get("paper_id"), "section_id": record.get("section_id"),
            "page": record.get("page"), "source_type": record.get("source_type"), "object_label": record.get("label"),
            "l0_record_key": record.get("global_record_id"), "evidence_ref": ref,
        }
        for ref, record in catalog.items()
    }
    image_ref_by_evidence_ref = {
        str(evidence_ref): str(item.get("image_ref") or "")
        for item in hierarchy.get("image_map") or []
        if isinstance(item, dict)
        for evidence_ref in item.get("evidence_refs") or []
        if str(evidence_ref) and str(item.get("image_ref") or "")
    }
    cards: list[dict[str, Any]] = []
    accepted_triple_statuses = {
        "deterministic_table_relation", "deterministic_citation_relation", "deterministic_hardware_relation",
        "verified_llm_relation", "visual_crop_verified_relation",
    }

    table_view_rows = max(1, int(hierarchy.get("keyed_table_view_rows") or 2))
    table_view_limit = max(0, int(hierarchy.get("keyed_table_view_limit") or 4))

    def keyed_table_view(view: Any) -> dict[str, Any] | None:
        if not isinstance(view, dict):
            return None
        columns = [str(value) for value in view.get("columns") or []]
        rows = []
        for index, row in enumerate((view.get("rows") or [])[:table_view_rows], start=1):
            if not isinstance(row, dict):
                continue
            values = row.get("values") if isinstance(row.get("values"), dict) else {}
            rows.append([f"row_{index}", row.get("row_label"), [[f"C{col_index}", values.get(column)] for col_index, column in enumerate(columns, start=1)]])
        return {"caption": view.get("caption"), "headers": [[f"H{index}", column] for index, column in enumerate(columns, start=1)], "rows": rows}

    include_table_structure = bool(hierarchy.get("keyed_table_structure_enabled", False))
    table_view_count = 0
    for index, card in enumerate(hierarchy.get("l2_evidence_cards") or [], start=1):
        if not isinstance(card, dict):
            continue
        support_keys = [record_keys[ref] for ref in card.get("support_refs") or [] if ref in record_keys]
        if not support_keys:
            continue
        view = keyed_table_view(card.get("table_view")) if include_table_structure and table_view_count < table_view_limit else None
        table_view_count += bool(view)
        cards.append({
            "key": f"C{index:03d}", "claim_ids": card.get("claim_ids") or [], "proposition": card.get("proposition"),
            "entities": card.get("entities") or [], "values": card.get("values") or [], "conditions": card.get("conditions") or [],
            "table_view": view, "support_keys": support_keys, "verification": card.get("verification") or {},
        })
    # Micro propositions are L2 extractive compression, not L0 text. They are
    # separate cards so broad frozen-selection coverage can be retained without
    # giving the model raw locators or original record text.
    micro_hierarchy = dict(hierarchy)
    keyed_micro_budget = hierarchy.get("keyed_micro_index_chars")
    micro_hierarchy["prompt_micro_index_chars"] = 48000 if keyed_micro_budget is None else int(keyed_micro_budget)
    micro_hierarchy["prompt_micro_text_chars"] = int(hierarchy.get("keyed_micro_text_chars") or 180)
    # Rich table cards already reserve the bounded structural views. Micro
    # cards keep header/row/cell information in their proposition; materializing
    # full views here would consume a legacy-only budget but be dropped later.
    micro_hierarchy["include_table_views"] = False
    micro_hierarchy["_keyed_micro_projection"] = True
    for row in hierarchy_prompt_projection(micro_hierarchy).get("l2_micro_evidence") or []:
        ref = str(row.get("support_ref") or "")
        if ref not in record_keys:
            continue
        record_key = record_keys[ref]
        view = keyed_table_view(row.get("table_view")) if include_table_structure and table_view_count < table_view_limit else None
        table_view_count += bool(view)
        cards.append({
            "key": f"C{len(cards) + 1:03d}",
            "proposition": row.get("extractive_proposition"),
            "table_view": view,
            "support_keys": [record_key], "verification": {"status": "extractive_micro"},
        })
    # A verified triple can originate from an additive L1 navigation seed that
    # did not receive a legacy L2 card. Promote a compact, citeable support
    # card from its existing L0 proof rather than dropping the relation at the
    # generation boundary.
    for triple in hierarchy.get("l2_contextual_triples") or []:
        if not isinstance(triple, dict) or str((triple.get("verification") or {}).get("status") or "") not in accepted_triple_statuses:
            continue
        support_keys = [record_keys[ref] for ref in triple.get("support_refs") or [] if ref in record_keys]
        if not support_keys or any(set(support_keys).intersection(card.get("support_keys") or []) for card in cards):
            continue
        qualifiers = triple.get("qualifiers") if isinstance(triple.get("qualifiers"), dict) else {}
        qualifier_text = "; ".join(
            f"{key}=" + ", ".join(str(value) for value in values[:4])
            for key, values in qualifiers.items() if isinstance(values, list) and values
        )
        proposition = " ".join(str(triple.get(key) or "").strip() for key in ("subject", "predicate", "object") if str(triple.get(key) or "").strip())
        cards.append({
            "key": f"C{len(cards) + 1:03d}", "claim_ids": list(triple.get("claim_ids") or []),
            "proposition": _clip("; ".join(value for value in (proposition, qualifier_text) if value), 900),
            "entities": [triple.get("subject"), triple.get("object")], "values": [], "conditions": [], "table_view": None,
            "support_keys": support_keys, "triple_support": True,
            "verification": {"status": str((triple.get("verification") or {}).get("status") or "")},
        })
    referenced_record_keys = {record_key for values in (card["support_keys"] for card in cards) for record_key in values}
    # Expose the compact navigation contract alongside L2: opaque R/S/P keys
    # carry the required record/section/paper identities, while raw L0 text and
    # official locators remain runtime-only.
    paper_keys: dict[str, str] = {}
    section_keys: dict[tuple[str, str], str] = {}
    papers: list[list[str]] = []
    sections: list[list[str]] = []
    for record_key in sorted(referenced_record_keys):
        item = key_index[record_key]
        paper_id = str(item.get("paper_id") or "")
        paper_key = paper_keys.get(paper_id)
        if paper_key is None:
            paper_key = f"P{len(papers) + 1:02d}"
            paper_keys[paper_id] = paper_key
            papers.append([paper_key, paper_id])
        section_id = str(item.get("section_id") or "")
        section_key = section_keys.get((paper_id, section_id))
        if section_key is None:
            section_key = f"S{len(sections) + 1:03d}"
            section_keys[(paper_id, section_id)] = section_key
            sections.append([section_key, paper_key, section_id])
    navigation = {
        "schema": ["paper_key", "paper_id"],
        "papers": papers,
        "section_schema": ["section_key", "paper_key", "section_id"],
        "sections": sections,
    }
    def prompt_card(card: dict[str, Any]) -> dict[str, Any]:
        first_key = (card.get("support_keys") or [""])[0]
        item = key_index.get(first_key) or {}
        return {
            "key": card["key"],
            "record_key": first_key,
            "paper_key": paper_keys.get(str(item.get("paper_id") or ""), ""),
            "section_key": section_keys.get((str(item.get("paper_id") or ""), str(item.get("section_id") or "")), ""),
            "page": item.get("page"),
            "source_type": item.get("source_type"),
            "object_label": item.get("object_label"),
            "image_ref": image_ref_by_evidence_ref.get(str(item.get("evidence_ref") or "")),
            "claim_ids": card.get("claim_ids") or [],
            "proposition": card.get("proposition"),
            "table_view": card.get("table_view"),
        }

    rich_raw_cards = [card for card in cards if card.get("claim_ids") or card.get("table_view") or card.get("triple_support")]
    rich_cards = [prompt_card(card) for card in rich_raw_cards]
    micro_rows = [
        [
            card["key"],
            (card.get("support_keys") or [""])[0],
            paper_keys.get(str((key_index.get((card.get("support_keys") or [""])[0]) or {}).get("paper_id") or ""), ""),
            section_keys.get((
                str((key_index.get((card.get("support_keys") or [""])[0]) or {}).get("paper_id") or ""),
                str((key_index.get((card.get("support_keys") or [""])[0]) or {}).get("section_id") or ""),
            ), ""),
            (key_index.get((card.get("support_keys") or [""])[0]) or {}).get("page"),
            (key_index.get((card.get("support_keys") or [""])[0]) or {}).get("source_type"),
            (key_index.get((card.get("support_keys") or [""])[0]) or {}).get("object_label"),
            image_ref_by_evidence_ref.get(str((key_index.get((card.get("support_keys") or [""])[0]) or {}).get("evidence_ref") or "")),
            card.get("proposition"),
        ]
        for card in cards if card not in rich_raw_cards
    ]
    # Contextual triples are a compact fact/navigation layer. Expose only
    # relations that have a deterministic table proof, an exact L0 quote gate,
    # or a separately verified crop; extractive candidates remain runtime-only
    # navigation and cannot become answer facts by themselves.
    triple_rows = []
    for triple in hierarchy.get("l2_contextual_triples") or []:
        if not isinstance(triple, dict) or str((triple.get("verification") or {}).get("status") or "") not in accepted_triple_statuses:
            continue
        triple_record_keys = [record_keys[ref] for ref in triple.get("support_refs") or [] if ref in record_keys]
        triple_card_keys = [
            str(card.get("key") or "") for card in cards
            if set(triple_record_keys).intersection(card.get("support_keys") or [])
        ]
        if not triple_card_keys:
            continue
        qualifiers = triple.get("qualifiers") if isinstance(triple.get("qualifiers"), dict) else {}
        triple_rows.append([
            str(triple.get("triple_id") or ""), triple_card_keys[:4], _clip(triple.get("subject"), 120),
            _clip(triple.get("predicate"), 48), _clip(triple.get("object"), 220),
            {str(key): [_clip(value, 80) for value in values[:4]] for key, values in qualifiers.items() if isinstance(values, list)},
            str((triple.get("verification") or {}).get("status") or ""),
        ])
    if str(hierarchy.get("triple_prompt_policy") or "") == "accepted_triples_first" and triple_rows:
        allowed_keys = {str(key) for row in triple_rows for key in row[1]}
        cards = [card for card in cards if str(card.get("key") or "") in allowed_keys]
        rich_cards = [prompt_card(card) for card in cards]
        micro_rows = []
    expansions = hierarchy.get("sufficiency_expansions") if isinstance(hierarchy.get("sufficiency_expansions"), dict) else {}
    def expansion_card_keys(refs: Any) -> list[str]:
        expansion_record_keys = [record_keys[str(ref)] for ref in refs or [] if str(ref) in record_keys]
        return [
            str(card.get("key") or "") for card in cards
            if set(expansion_record_keys).intersection(card.get("support_keys") or [])
        ][:4]
    l1_expansion_rows = []
    for row in expansions.get("l1_rows") or []:
        if not isinstance(row, dict):
            continue
        support_card_keys = expansion_card_keys(row.get("support_refs"))
        if not support_card_keys:
            continue
        l1_expansion_rows.append([
            row.get("triple_id"), support_card_keys, _clip(row.get("anchor_quote"), 620),
            [_clip(value, 300) for value in (row.get("neighbor_quotes") or [])[:2]],
            list(row.get("table_lines") or [])[:8], row.get("object_context"),
        ])
    l0_expansion_rows = []
    for row in expansions.get("l0_rows") or []:
        if not isinstance(row, dict):
            continue
        support_card_keys = expansion_card_keys(row.get("support_refs"))
        if not support_card_keys:
            continue
        l0_expansion_rows.append([
            row.get("triple_id"), support_card_keys, row.get("source_type"), row.get("label"), _clip(row.get("text"), 1600),
        ])
    return {
        "version": hierarchy.get("version", HIERARCHY_VERSION), "mode": "keyed_l2_only",
        "query_claims": hierarchy.get("query_claims") or [], "l2_cards": rich_cards,
        "l2_triple_schema": ["triple_key", "support_card_keys", "subject", "predicate", "object", "qualifiers", "verification"],
        "l2_triple_rows": triple_rows,
        "l1_expansion_schema": ["triple_key", "support_card_keys", "anchor_quote", "neighbor_quotes", "table_lines", "object_context"],
        "l1_expansion_rows": l1_expansion_rows,
        "l0_expansion_schema": ["triple_key", "support_card_keys", "source_type", "label", "text"],
        "l0_expansion_rows": l0_expansion_rows,
        "l2_micro_schema": ["card_key", "record_key", "paper_key", "section_key", "page", "source_type", "object_label", "image_ref", "proposition"], "l2_micro_rows": micro_rows,
        "navigation_keys": navigation, "grounding_contract": "Return only Cxxx support keys. Cxxx -> Rxxx resolves at runtime to L0 provenance; R/S/P keys provide compact record, section, and paper navigation only.",
        # Not serialized by the prompt builder; retained for runtime only.
        "_key_index": key_index,
        "_card_support_keys": {str(card["key"]): list(card["support_keys"]) for card in cards},
        "_card_metadata": {
            str(card["key"]): {
                "proposition": str(card.get("proposition") or ""),
                "verification_status": str((card.get("verification") or {}).get("status") or ""),
                "source_type": str((key_index.get((card.get("support_keys") or [""])[0]) or {}).get("source_type") or ""),
            }
            for card in cards
        },
    }


def resolve_claim_support_keys(
    claim_to_support_keys: Any,
    hierarchy: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve generated C-keys to exact L0 evidence refs without guessing."""
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    card_map = projection["_card_support_keys"]
    key_index = projection["_key_index"]
    resolved: list[str] = []
    audit: list[dict[str, Any]] = []
    if not isinstance(claim_to_support_keys, dict):
        return resolved, [{"status": "invalid_claim_to_support_keys"}]
    for claim_id, raw_keys in claim_to_support_keys.items():
        keys = raw_keys if isinstance(raw_keys, list) else []
        keys = keys[:4]
        # Models occasionally render a valid C-key as C0012 rather than C012.
        # Normalize only the numeric zero padding; arbitrary identifiers still
        # fail the audit and cannot become evidence.
        valid_cards: list[str] = []
        for key in keys:
            raw_key = str(key).strip()
            match = re.fullmatch(r"C0*(\d+)", raw_key, re.IGNORECASE)
            canonical = f"C{int(match.group(1)):03d}" if match else raw_key
            if canonical in card_map and canonical not in valid_cards:
                valid_cards.append(canonical)
        refs: list[str] = []
        for card_key in valid_cards:
            for record_key in card_map[card_key]:
                ref = str(key_index[record_key]["evidence_ref"])
                if ref not in refs:
                    refs.append(ref)
                if ref not in resolved:
                    resolved.append(ref)
        audit.append({"claim_id": str(claim_id), "requested_keys": [str(key) for key in keys], "valid_card_keys": valid_cards, "evidence_refs": refs, "status": "grounded" if refs else "ungrounded"})
    return resolved, audit
