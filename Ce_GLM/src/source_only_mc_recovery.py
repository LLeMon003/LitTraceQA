"""Leakage-safe helpers for source-only multiple-choice blank recovery."""
from __future__ import annotations

import re
from typing import Any, Iterable


FORBIDDEN_INPUT_KEYS = {"gold", "answer_key", "expected_option", "expected_option_letter", "evaluator_label"}
STOP = {"a", "an", "and", "are", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "what", "which", "with"}


def terms(value: Any) -> set[str]:
    return {
        item
        for item in re.findall(r"[a-z0-9]+", str(value).lower())
        if (len(item) > 1 or any(character.isdigit() for character in item)) and item not in STOP
    }


def contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_INPUT_KEYS or contains_forbidden_key(child) for key, child in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_key(child) for child in value)
    return False


def selected_paper_ids(row: dict[str, Any]) -> set[str]:
    papers = row.get("gold_papers") if isinstance(row.get("gold_papers"), list) else row.get("papers")
    output: set[str] = set()
    for item in papers if isinstance(papers, list) else []:
        if isinstance(item, dict) and str(item.get("paper_id") or "").strip():
            output.add(str(item["paper_id"]).strip())
        elif isinstance(item, str) and item.strip():
            output.add(item.strip())
    return output


def blank_mc(row: dict[str, Any]) -> bool:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    multiple_choice = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else None
    return isinstance(multiple_choice, dict) and not str(multiple_choice.get("gold") or "").strip()


def index_object(row: dict[str, Any]) -> dict[str, Any] | None:
    """Project only allowed source fields; intentionally never reads query_ids."""
    object_id = str(row.get("object_uid") or "").strip()
    paper_id = str(row.get("paper_id") or "").strip()
    if not object_id or not paper_id:
        return None
    parts = [str(row.get(key) or "").strip() for key in ("normalized_text", "text", "cell_value")]
    content = " ".join(part for part in parts if part)
    if not content:
        return None
    return {
        "object_id": object_id,
        "paper_id": paper_id,
        "page": row.get("page"),
        "object_type": str(row.get("object_type") or ""),
        "object_label": str(row.get("object_label") or ""),
        "table_refs": [str(value) for value in row.get("table_refs", []) if str(value).strip()] if isinstance(row.get("table_refs"), list) else [],
        "figure_refs": [str(value) for value in row.get("figure_refs", []) if str(value).strip()] if isinstance(row.get("figure_refs"), list) else [],
        "algorithm_refs": [str(value) for value in row.get("algorithm_refs", []) if str(value).strip()] if isinstance(row.get("algorithm_refs"), list) else [],
        "content": content[:1600],
    }


def parent_locators(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Project existing evaluator-visible evidence locators without reading answers."""
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        return []
    locators: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        source_type = str(item.get("source_type") or "").strip().lower()
        locator = item.get("locator")
        if not paper_id or not source_type or not isinstance(locator, dict):
            continue
        page = locator.get("page")
        if not isinstance(page, int):
            continue
        locators.append(
            {
                "paper_id": paper_id,
                "source_type": source_type,
                "page": page,
                "table_id": str(locator.get("table_id") or "").strip(),
                "figure_id": str(locator.get("figure_id") or "").strip(),
                "algorithm_id": str(locator.get("algorithm_id") or "").strip(),
            }
        )
    return locators


def _contains_identifier(item: dict[str, Any], identifier: str) -> bool:
    if not identifier:
        return True
    value = identifier.lower()
    candidates = [str(item.get("object_label") or "").lower()]
    for key in ("table_refs", "figure_refs", "algorithm_refs"):
        candidates.extend(str(part).lower() for part in item.get(key, []))
    return any(value in candidate or candidate in value for candidate in candidates if candidate)


def locator_match(item: dict[str, Any], locator: dict[str, Any]) -> bool:
    """Require paper/page plus the declared source-object family when available."""
    if item["paper_id"] != locator["paper_id"] or item.get("page") != locator["page"]:
        return False
    object_type = item["object_type"].lower()
    source_type = locator["source_type"]
    if source_type == "table":
        return object_type in {"table", "table_row", "table_cell", "table_caption"} and _contains_identifier(item, locator["table_id"])
    if source_type == "figure":
        return object_type in {"figure_caption", "paragraph", "object_window"} and _contains_identifier(item, locator["figure_id"])
    if source_type == "equation_algorithm":
        identifier = locator["algorithm_id"]
        return object_type in {"equation_block", "equation_context", "algorithm_context", "paragraph"} and _contains_identifier(item, identifier)
    if source_type in {"text_span", "citation_context"}:
        return object_type in {"paragraph", "raw_block", "object_window", "section_header"}
    return False


def rank_bundle(question: str, options: dict[str, str], paper_ids: set[str], index_rows: Iterable[dict[str, Any]], limit: int = 12, locators: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    query = terms(question)
    for text in options.values():
        query.update(terms(text))
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    allowed_locators = list(locators)
    for raw in index_rows:
        item = index_object(raw)
        if item is None or item["paper_id"] not in paper_ids:
            continue
        score = len(query & terms(item["content"]))
        if item["object_type"].lower() == "table":
            score += 1
        if any(locator_match(item, locator) for locator in allowed_locators):
            score += 50
        if score:
            ranked.append((score, item["object_id"], item))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in ranked[:limit]]


def option_supports(option_text: str, cited: Iterable[dict[str, Any]]) -> bool:
    expected = terms(option_text)
    if not expected:
        return False
    cited_terms: set[str] = set()
    for item in cited:
        cited_terms.update(terms(item.get("content", "")))
    if expected <= cited_terms:
        return True
    return len(expected & cited_terms) >= min(2, len(expected))


def validate_proposal(value: Any, options: dict[str, str], bundle: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    letter = str(value.get("letter") or "").strip()
    citations = value.get("citations")
    if letter not in options or not isinstance(citations, list) or not citations:
        return None
    allowed = {str(item["object_id"]): item for item in bundle}
    citation_ids = [str(item) for item in citations]
    if len(citation_ids) != len(set(citation_ids)) or any(item not in allowed for item in citation_ids):
        return None
    cited = [allowed[item] for item in citation_ids]
    if not option_supports(options[letter], cited):
        return None
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return {"letter": letter, "citations": citation_ids, "confidence": confidence}


def proposal_status(value: Any, options: dict[str, str], bundle: list[dict[str, Any]]) -> str:
    """Return a non-answer diagnostic for rejected model output."""
    if not isinstance(value, dict):
        return "MALFORMED_JSON"
    letter = str(value.get("letter") or "").strip()
    citations = value.get("citations")
    if not letter and (not isinstance(citations, list) or not citations):
        return "MODEL_ABSTAIN"
    if letter not in options:
        return "INVALID_OPTION"
    if not isinstance(citations, list) or not citations:
        return "NO_CITATIONS"
    allowed = {str(item["object_id"]): item for item in bundle}
    citation_ids = [str(item) for item in citations]
    if len(citation_ids) != len(set(citation_ids)):
        return "DUPLICATE_CITATION"
    if any(item not in allowed for item in citation_ids):
        return "OUT_OF_BUNDLE_CITATION"
    if not option_supports(options[letter], [allowed[item] for item in citation_ids]):
        return "NO_OPTION_TEXT_SUPPORT"
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        return "INVALID_CONFIDENCE"
    return "INVALID_CONFIDENCE" if not 0.0 <= confidence <= 1.0 else "REJECTED"
