"""Slot contracts for staged, provenance-preserving answer generation.

Slots describe evidence needs, never facts.  Qwen may plan and extract values,
but this module binds every accepted value to visible C-keys before composition.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from .evidence_hierarchy import keyed_hierarchy_prompt_projection
from .metadata_index import tokenize
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


SLOT_PLAN_VERSION = "v1_qwen_slot_contract"
_ROLES = {"direct_answer", "condition", "contributor"}
_OPERATIONS = {"direct", "difference", "maximum", "minimum", "count", "list"}
_OPERATION_ALIASES = {"literal_extraction": "direct", "extract": "direct", "comparison": "difference"}
_SOURCE_TYPE_ALIASES = {"text": "text_span", "paragraph": "text_span", "citation": "citation_context", "equation": "equation_algorithm", "algorithm": "equation_algorithm"}
_STATUSES = {"supported", "partial", "unsupported", "conflict", "unreadable"}
_FOCUS_STOPWORDS = {"a", "an", "and", "are", "as", "by", "does", "do", "for", "from", "has", "have", "in", "is", "it", "kind", "of", "on", "the", "to", "under", "used", "use", "was", "were", "what", "which", "who", "with"}
_TITLE_ROUTING_STOPWORDS = _FOCUS_STOPWORDS | {"algorithm", "author", "citation", "cited", "equation", "figure", "first", "framework", "method", "model", "paper", "performance", "reference", "result", "score", "table"}
_CONDITION_STOPWORDS = _FOCUS_STOPWORDS | {
    "average", "benchmark", "compression", "condition", "configuration", "dataset", "metric",
    "percentage", "performance", "rate", "ratio", "result", "results", "score", "scores", "setting", "value",
}


def _clip(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _strings(value: Any, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clip(item, 160)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _fallback_focus(question: Any) -> list[str]:
    """Keep the interrogative target when Qwen returns an empty slot plan."""
    text = _clip(question, 800)
    match = re.search(r"\b(?:what|which|who|how)\b\s+(.+)", text, re.IGNORECASE)
    focus = match.group(1) if match else text
    # The leading question syntax is not an evidence condition. Keep the
    # remaining lexical target as a routing hint, never as an asserted fact.
    focus = re.sub(r"^(?:kind of|is|are|many|much)\s+", "", focus, flags=re.IGNORECASE)
    terms = [term for term in tokenize(focus) if term not in _FOCUS_STOPWORDS]
    return [" ".join(terms[:12])] if terms else []


def _explicit_locator_terms(question: str) -> set[str]:
    labels: set[str] = set()
    for kind, value in re.findall(r"\b(table|figure|fig\.?|equation|eq\.?|algorithm)\s*\(?\s*(\d{1,4})\s*\)?", question, re.IGNORECASE):
        normalized = "figure" if kind.lower().startswith("fig") else "equation" if kind.lower().startswith("eq") else kind.lower()
        labels.add(f"{normalized} {value}")
    for value in re.findall(r"\b(\d{1,4})(?:st|nd|rd|th)\s+(?:reference|citation)\b", question, re.IGNORECASE):
        labels.add(f"reference {value}")
    return labels


def _focus_routing_terms(value: Any) -> set[str]:
    """Use a short shared prefix only for query-focus lexical routing."""
    terms = set(tokenize(str(value or "")))
    return terms | {term[:7] for term in terms if len(term) >= 8}


def _named_candidate_papers(question: str, candidates: list[dict[str, Any]] | None) -> set[str]:
    """Route only unambiguous literal title mentions; never infer a paper."""
    question_terms = {term for term in tokenize(question) if len(term) >= 4 and term not in _TITLE_ROUTING_STOPWORDS}
    scored: list[tuple[int, str]] = []
    for row in candidates or []:
        paper_id = str(row.get("paper_id") or "")
        title_terms = {term for term in tokenize(str(row.get("title") or "")) if len(term) >= 4 and term not in _TITLE_ROUTING_STOPWORDS}
        overlap = len(question_terms.intersection(title_terms))
        if paper_id and overlap:
            scored.append((overlap, paper_id))
    if not scored:
        return set()
    best = max(score for score, _ in scored)
    winners = {paper_id for score, paper_id in scored if score == best}
    # One literal distinctive title token (EasySpec, DynaPipe) is sufficient
    # only when unique. Multi-paper questions remain unconstrained on ties.
    return winners if best >= 2 or len(winners) == 1 else set()


def _fallback_operation(question: Any) -> str:
    text = str(question or "").lower()
    if re.search(r"\b(?:how many|number of|count)\b", text):
        return "count"
    if re.search(r"\b(?:by how much|outperform|difference|delta|∆)\b", text):
        return "difference"
    if re.search(r"\b(?:highest|largest|maximum|best)\b", text):
        return "maximum"
    if re.search(r"\b(?:lowest|smallest|minimum)\b", text):
        return "minimum"
    return "direct"


def _default_plan(sample: dict[str, Any]) -> dict[str, Any]:
    primary = to_official_source_type(source_type=sample.get("primary_evidence_type"))
    return {
        "version": SLOT_PLAN_VERSION,
        "slots": [{
            "id": "S001", "role": "direct_answer", "operation": _fallback_operation(sample.get("question")),
            "paper_scope": [], "required_source_types": [primary] if primary else [],
            "entities": _fallback_focus(sample.get("question")), "required_conditions": [],
        }],
        "relations": [],
        "requires_cross_paper_synthesis": "multi" in str(sample.get("task_family") or "").lower(),
        "fallback": True,
    }


def _paper_identity_conditions(conditions: list[str], candidates: list[dict[str, Any]] | None) -> set[str]:
    """Drop conditions that merely restate a candidate paper's title/identity.

    Paper identity is established from evidence-bound C-keys, never from a
    planned evidence condition.  "DynaPipe paper" is routing metadata, not a
    measurable condition such as a dataset, metric, or setting, and forcing the
    extractor to match it produces spurious `missing_conditions`.
    """
    dropped: set[str] = set()
    titles = [str(row.get("title") or "") for row in candidates or [] if isinstance(row, dict)]
    if not titles:
        return dropped
    title_term_sets = [
        {term for term in tokenize(title) if len(term) >= 4 and term not in _TITLE_ROUTING_STOPWORDS}
        for title in titles
    ]
    for condition in conditions:
        normalized = re.sub(r"\s+", " ", condition.lower()).replace(" paper", "").strip()
        terms = {term for term in tokenize(normalized) if len(term) >= 4 and term not in _TITLE_ROUTING_STOPWORDS}
        if not terms:
            continue
        if any(terms.issubset(title_terms) for title_terms in title_term_sets if title_terms):
            dropped.add(condition)
    return dropped


def slot_plan_messages(sample: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    """Ask Qwen to decompose the public query/contract, not paper metadata."""
    payload = {
        "question": sample.get("question"), "task_family": sample.get("task_family"),
        "primary_evidence_type": sample.get("primary_evidence_type"), "answer_contract": contract,
    }
    system = (
        "Plan evidence requirements, not an answer. Do not infer facts, values, pages, locators, paper identities, or relevance. "
        "Split only independently verifiable answer requirements. You MUST return at least one slot. Return JSON only."
        " required_conditions must be measurable evidence conditions (dataset, metric, setting, configuration); "
        "never use a paper name, paper identity, or title as a condition."
    )
    user = (
        "Use literal values from this schema: role is direct_answer|condition|contributor; operation is direct|difference|maximum|minimum|count|list; "
        "required_source_types use text_span|table|figure|equation_algorithm|citation_context. Never return an empty slots array. Example: "
        "{\"slots\":[{\"id\":\"S001\",\"role\":\"direct_answer\",\"operation\":\"difference\","
        "\"required_source_types\":[\"table\"],\"entities\":[\"Method A\",\"Method B\"],"
        "\"required_conditions\":[\"F1\",\"Dataset X\"]}],\"relations\":[\"compare\"],"
        "\"requires_cross_paper_synthesis\":false}.\nINPUT:"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_slot_plan(
    raw: Any,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Accept only candidate-bound, source-typed, fact-free slot requirements."""
    value = raw if isinstance(raw, dict) else {}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index, row in enumerate(value.get("slots") or [], start=1):
        if not isinstance(row, dict) or len(accepted) >= 16:
            audit.append({"status": "slot_rejected_invalid", "index": index})
            continue
        source_types = []
        for raw_source in _strings(row.get("required_source_types"), 5):
            source = _SOURCE_TYPE_ALIASES.get(raw_source.lower(), raw_source)
            if source in OFFICIAL_EVIDENCE_SOURCE_TYPES and source not in source_types:
                source_types.append(source)
        primary = to_official_source_type(source_type=sample.get("primary_evidence_type"))
        if not source_types and primary:
            source_types = [primary]
        # Slot planning replaces metadata evidence planning. Paper identity is
        # established later from evidence-bound C-keys, never guessed here.
        requested_papers = _strings(row.get("paper_scope"), 12)
        role = str(row.get("role") or "direct_answer")
        raw_operation = str(row.get("operation") or "direct").lower()
        operation = _OPERATION_ALIASES.get(raw_operation, raw_operation)
        if role not in _ROLES or operation not in _OPERATIONS or not source_types:
            audit.append({"status": "slot_rejected_schema", "index": index, "role": role, "operation": operation})
            continue
        requested_conditions = _strings(row.get("required_conditions"))
        dropped_conditions = _paper_identity_conditions(requested_conditions, candidates)
        slot = {
            "id": f"S{len(accepted) + 1:03d}", "role": role, "operation": operation,
            "paper_scope": [], "required_source_types": source_types,
            "entities": _strings(row.get("entities")),
            "required_conditions": [
                condition
                for condition in requested_conditions
                if condition not in dropped_conditions
            ],
        }
        accepted.append(slot)
        audit.append(
            {
                "status": "slot_accepted",
                "slot_id": slot["id"],
                "paper_scope_removed": bool(requested_papers),
                "paper_identity_conditions_dropped": sorted(dropped_conditions),
            }
        )
    if not accepted:
        fallback = _default_plan(sample)
        return fallback, [*audit, {"status": "slot_plan_fallback"}]
    return {
        "version": SLOT_PLAN_VERSION, "slots": accepted,
        "relations": _strings(value.get("relations"), 8),
        "requires_cross_paper_synthesis": bool(value.get("requires_cross_paper_synthesis")),
        "fallback": False,
    }, audit


def _card_text(card: dict[str, Any]) -> str:
    return json.dumps(card, ensure_ascii=False, separators=(",", ":"))


def _readable_record(record: dict[str, Any]) -> str:
    """Make object records easy to read without inventing a physical schema."""
    source_type = str(record.get("source_type") or "text_span")
    label = _clip(record.get("label"), 120)
    text = " // ".join(_clip(line, 1200) for line in str(record.get("text") or "").splitlines() if _clip(line, 1200))
    prefix = {
        "table": "TABLE", "equation_algorithm": "EQUATION", "figure": "FIGURE", "citation_context": "CITATION",
    }.get(source_type, "TEXT")
    return _clip(" // ".join(part for part in (f"{prefix} {label}".strip(), text) if part), 4200 if source_type == "table" else 2200)


def _condition_terms(text: Any) -> set[str]:
    """Normalized condition terms for cross-operand consistency checks."""
    normalized = re.sub(r"->|→|—|–", "to", str(text or "").lower())
    return {
        term
        for term in re.findall(r"[a-z0-9]+", normalized)
        if len(term) >= 2 and term not in _CONDITION_STOPWORDS
    }


_CITATION_NOISE_PREFIXES = (
    "question:", "justification:", "the answer na", "guidelines:", "do the main claims",
    "1. claims", "2. limitations", "3. ethics", "4. reproducibility",
)
_BIBLIOGRAPHY_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _is_genuine_bibliography(text: str) -> bool:
    """Heuristic: bibliography entries are author-led with a year; checklist and
    table/figure caption noise that Docling mislabels as citations is excluded."""
    lowered = str(text or "").lstrip().lower()
    if lowered.startswith(_CITATION_NOISE_PREFIXES):
        return False
    if re.match(r"^(table|figure)\s*\d", lowered):
        return False
    if "computation and communication complexity" in lowered:
        return False
    if _BIBLIOGRAPHY_YEAR_RE.search(lowered):
        return True
    # Bibliography entries are author-led ("Surname, I., ..."); some parsed
    # records drop the year while keeping the author list.
    return bool(re.match(r"^[A-Z][A-Za-z'´`\-]+,?\s+[A-Z]", str(text or "").lstrip()))


def _ref_to_card_keys(hierarchy: dict[str, Any], refs: Iterable[str]) -> list[str]:
    """Map evidence refs to visible C-keys (C -> R -> L0)."""
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    card_map = projection.get("_card_support_keys") or {}
    key_index = projection.get("_key_index") or {}
    ref_set = set(str(ref) for ref in refs)
    keys: list[str] = []
    for card_key, record_keys in card_map.items():
        for record_key in record_keys:
            ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
            if ref in ref_set and card_key not in keys:
                keys.append(str(card_key))
    return keys


def deterministic_count_extraction(
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    question: str,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Deterministic answers for count-style questions computable from L0 records.

    Only high-precision patterns are handled; anything else returns None so the
    model path runs.  The returned extraction is already validated (the caller
    must not run it through validate_slot_extraction again).
    """
    hierarchy = ensure_slot_cards(hierarchy)
    lowered = str(question or "").lower()
    catalog = [record for record in hierarchy.get("l0_catalog") or [] if isinstance(record, dict)]
    paper_ids = _named_candidate_papers(str(question or ""), candidates)

    def supported(value: Any, refs: Iterable[str]) -> dict[str, Any]:
        keys = _ref_to_card_keys(hierarchy, refs)
        return {
            "slot_id": str(slot.get("id") or "S001"),
            "status": "supported",
            "value": str(value),
            "conditions": {},
            "support_keys": keys,
            "evidence_values": [],
            "missing_conditions": [],
            "reported_value": None,
            "deterministic": True,
            "validation": {
                "type_ok": bool(keys),
                "numeric_ok": True,
                "evidence_values_ok": True,
                "model_status": "deterministic",
                "derived_value": False,
                "reported_difference": False,
                "reported_difference_source": "",
                "visual_supported": False,
                "visual_keys": [],
                "condition_ok": True,
                "same_table_operands": False,
            },
        }

    # A) parentheses inside an equation: count "(" in the displayed body,
    # excluding the trailing equation-number parens "(N)".
    if "parenthes" in lowered:
        match = re.search(r"\bequation\s+(\d{1,3})\b", lowered)
        if match:
            target = f"equation {match.group(1)}"
            for record in catalog:
                if record.get("source_type") != "equation_algorithm":
                    continue
                if paper_ids and str(record.get("paper_id") or "") not in paper_ids:
                    continue
                label = str(record.get("label") or "").lower()
                if target not in label:
                    continue
                text = str(record.get("text") or "")
                if "was not emitted" in text or not re.search(r"[=λ∑∑]", text):
                    continue
                count = text.count("(")
                if re.search(r"\(\s*\d{1,3}\s*\)\s*$", text):
                    count -= 1
                if count >= 0:
                    return supported(count, [record.get("evidence_ref")])

    # B) last reference / reference index of a paper.
    if "last reference" in lowered or "reference index" in lowered or "index of the last reference" in lowered:
        refs = [
            record for record in catalog
            if record.get("source_type") == "citation_context"
            and (not paper_ids or str(record.get("paper_id") or "") in paper_ids)
            and _is_genuine_bibliography(str(record.get("text") or ""))
        ]
        ids = []
        for record in refs:
            loc = record.get("locator") or {}
            raw = str(loc.get("citation_id") or "")
            try:
                ids.append(int(raw.replace("#", "")))
            except ValueError:
                continue
        # Only trust the parser's citation numbering when it is exactly 1..N
        # with no duplicates.  q_005's parser output has duplicated and
        # out-of-range ids (max 269, 70 unique) so the true last reference is
        # not derivable; emitting a wrong max would be worse than abstaining.
        if ids and len(set(ids)) == max(ids) == len(ids):
            best = max(
                refs,
                key=lambda record: int(str((record.get("locator") or {}).get("citation_id") or "0").replace("#", "")),
            )
            return supported(max(ids), [best.get("evidence_ref")])

    # C) references that include <Surname> as an author.
    author_match = re.search(r"\binclude\s+([A-Z][A-Za-z\-]+)\s+as an author\b", str(question or ""))
    if author_match:
        surname = author_match.group(1).lower()
        refs = [
            record for record in catalog
            if record.get("source_type") == "citation_context"
            and (not paper_ids or str(record.get("paper_id") or "") in paper_ids)
            and _is_genuine_bibliography(str(record.get("text") or ""))
            and surname in str(record.get("text") or "").lower()
        ]
        if refs:
            return supported(len(refs), [record.get("evidence_ref") for record in refs])

    # D) number of papers cited in a named section (author-year citations).
    section_match = re.search(r"cited in\s+(?:the\s+)?([a-zA-Z ]+?)\s+section", lowered)
    if section_match:
        section_token = section_match.group(1).strip().lower()
        texts: list[str] = []
        refs: list[str] = []
        for record in catalog:
            if paper_ids and str(record.get("paper_id") or "") not in paper_ids:
                continue
            section = " ".join(str(value) for value in (record.get("section_title"), record.get("section_id"))).lower()
            if section_token not in section:
                continue
            texts.append(str(record.get("text") or ""))
            if record.get("evidence_ref"):
                refs.append(str(record.get("evidence_ref")))
        cited: set[tuple[str, str]] = set()
        for text in texts:
            for content in re.findall(r"\(([^()]*?)\)", text):
                for part in content.split(";"):
                    match = re.search(r"(.+?),\s*((?:19|20)\d{2})$", part.strip())
                    if match:
                        author = match.group(1).strip()
                        if author:
                            cited.add((author, match.group(2)))
        if cited:
            return supported(len(cited), refs)
    return None


def ensure_slot_cards(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Give every reservoir L0 record an extractive C-card for slot reading.

    Existing L2 cards are kept. New cards carry no inferred relation and make
    b160 available to the slot reader without changing package membership.

    A support ref on a rich L2 card is not necessarily readable evidence: one
    anchor card may point to many neighbors while serializing only its anchor.
    Therefore each L0 record receives its own card even when it is already a
    support ref elsewhere. This retains all package evidence without treating
    the L2 graph as the proof source.
    """
    updated = dict(hierarchy)
    cards = [dict(card) for card in hierarchy.get("l2_evidence_cards") or [] if isinstance(card, dict)]
    for record in hierarchy.get("l0_catalog") or []:
        if not isinstance(record, dict):
            continue
        ref = str(record.get("evidence_ref") or "")
        if not ref:
            continue
        proposition = _readable_record(record)
        if not str(proposition).strip():
            continue
        cards.append({
            "support_refs": [ref], "proposition": proposition,
            "verification": {"status": "extractive_slot_projection"},
        })
    updated["l2_evidence_cards"] = cards
    return updated


def slot_cards(
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    question: str,
    limit: int = 24,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically expose cards likely to satisfy one requirement."""
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    question_terms = set(tokenize(question))
    focus_terms = _focus_routing_terms(" ".join([*slot.get("entities", []), *slot.get("required_conditions", [])]))
    locator_terms = _explicit_locator_terms(question)
    types = set(slot.get("required_source_types") or [])
    papers = set(slot.get("paper_scope") or [])
    metadata = projection.get("_card_metadata") or {}
    key_index = projection.get("_key_index") or {}
    card_support = projection.get("_card_support_keys") or {}
    catalog = {str(record.get("evidence_ref") or ""): record for record in hierarchy.get("l0_catalog") or [] if isinstance(record, dict)}
    paper_ids_by_key = {str(key): str(value) for key, value in (projection.get("navigation_keys") or {}).get("papers") or []}
    named_papers = _named_candidate_papers(question, candidates)
    available_cards = [dict(card) for card in projection.get("l2_cards") or [] if isinstance(card, dict)]
    # Keyed micro rows are already extractive L0 projections with immutable
    # C/R mappings. They are the reservoir's readable route when a rich card
    # only serialized an anchor but retained another record as a support ref.
    for row in projection.get("l2_micro_rows") or []:
        if not isinstance(row, list) or len(row) < 9:
            continue
        available_cards.append({
            "key": row[0], "record_key": row[1], "paper_key": row[2], "section_key": row[3],
            "page": row[4], "source_type": row[5], "object_label": row[6],
            "image_ref": row[7], "proposition": row[8],
        })
    rows = []
    for card in available_cards:
        key = str(card.get("key") or "")
        item = dict(card)
        item_text = _card_text(item)
        item_type = str((metadata.get(key) or {}).get("source_type") or item.get("source_type") or "")
        item_paper = str(item.get("paper_id") or "")
        label = str(item.get("object_label") or (metadata.get(key) or {}).get("object_label") or "")
        # The required source type dominates the packet so a figure/table/
        # equation/citation slot always surfaces its own object cards before
        # nearby prose; text context cards may still rank after them.
        score = 100 * int(not types or item_type in types) + 2 * int(not papers or item_paper in papers)
        item_paper_id = paper_ids_by_key.get(str(item.get("paper_key") or ""), item_paper)
        score += 1000 * int(bool(named_papers) and item_paper_id in named_papers)
        item_terms = set(tokenize(item_text))
        score += len(question_terms.intersection(item_terms))
        score += 12 * len(focus_terms.intersection(_focus_routing_terms(item_text)))
        normalized_label = re.sub(r"\s+", " ", label.lower()).replace("fig.", "figure").replace("eq.", "equation")
        score += 200 * int(any(target in normalized_label for target in locator_terms))
        record_key = str(item.get("record_key") or next(iter(card_support.get(key) or []), ""))
        evidence_ref = str((key_index.get(record_key) or {}).get("evidence_ref") or "")
        try:
            selection_rank = int((catalog.get(evidence_ref) or {}).get("selection_rank") or 0)
        except (TypeError, ValueError):
            selection_rank = 0
        # The slot packet is a reader over frozen Qwen-selected packages. A
        # strong cached rank must not be erased by shallow lexical mismatch.
        score += max(0, 24 - selection_rank) if selection_rank else 0
        rows.append((score, key, item))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in rows[:max(1, limit)]]


def slot_extraction_messages(
    sample: dict[str, Any],
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    attached_images: list[dict[str, Any]] | None = None,
    table_schema: Any = None,
    card_limit: int = 24,
) -> list[dict[str, str]]:
    cards = slot_cards(slot, hierarchy, str(sample.get("question") or ""), candidates=candidates, limit=card_limit)
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    paper_ids = {str(key): str(value) for key, value in (projection.get("navigation_keys") or {}).get("papers") or []}
    titles = {str(row.get("paper_id") or ""): _clip(row.get("title"), 280) for row in candidates or []}
    paper_navigation = [
        {"paper_key": key, "title": titles[paper_id]}
        for key, paper_id in paper_ids.items()
        if key in {str(card.get("paper_key") or "") for card in cards} and titles.get(paper_id)
    ]
    image_index = [
        {"image_index": index, "support_card_keys": _strings(row.get("support_card_keys"), 4)}
        for index, row in enumerate(attached_images or [], start=1)
    ]
    payload = {
        "question": sample.get("question"), "slot": slot, "paper_navigation": paper_navigation,
        "cards": cards, "attached_images": image_index,
    }
    if table_schema:
        payload["table_schema"] = table_schema
    system = (
        "Extract one slot value only from supplied C-key cards. Do not answer from outside knowledge. "
        "Paper navigation titles are routing metadata, not factual evidence: use them only to disambiguate a paper named in the question. "
        "Attached images are ordered by image_index and are evidence only for their listed support_card_keys. "
        "For a visual count, label, or plotted value, inspect its linked image before returning unsupported. "
        "When the slot requires a figure, enumerate the visible panel labels (such as (a), (b), (c)) and count every chart/plot "
        "panel inside the figure; report the count and the panel labels you observed. Do not return unsupported when a linked "
        "image is present: if part of the figure is ambiguous, return partial with missing_conditions describing exactly what "
        "could not be determined. At least one support key must come from the slot's required_source_types; prefer cards whose "
        "source_type matches the required type (table, figure, equation_algorithm, citation_context, text_span). "
        "A value may be supported only by copied C-keys. Return JSON only."
    )
    table_instruction = (
        " When a table_schema is provided, return rows in \"table_rows\":[{\"values\":{<exact schema column>:<cell>},"
        "\"support_keys\":[\"C001\"]}] instead of a single value (set \"value\":null); one object per row, "
        "copying cell values verbatim from the evidence, and each row must carry its own support keys. "
        "The first schema column is the row label (method/model/paper name): every row MUST include it. "
        "Do NOT return bare numbers in evidence_values for a table slot; a supported table slot must contain at "
        "least one table_rows entry, otherwise return partial with missing_conditions."
        if table_schema
        else ""
    )
    user = (
        "Return {\"slot_id\":\"S001\",\"status\":\"supported|partial|unsupported|conflict|unreadable\","
        "\"value\":\"supported value or null\",\"conditions\":{\"name\":\"evidence value\"},"
        "\"support_keys\":[\"C001\"],\"evidence_values\":[{\"name\":\"operand\",\"value\":\"41.36\",\"support_keys\":[\"C001\"]}],"
        "\"reported_value\":\"explicitly reported result or null\",\"missing_conditions\":[\"missing condition\"]}."
        + table_instruction
        + " "
        "For operation=difference, provide exactly two evidence_values in minuend then subtrahend order, and each evidence "
        "value MUST carry a conditions object identifying its dataset, metric, and compression/setting (for example "
        "{\"dataset\":\"NaturalQ\",\"metric\":\"F1\",\"setting\":\"500 to 1\"}). Both operands must come from the same dataset, "
        "metric, and setting, differing only in the compared method or system; otherwise return partial with missing_conditions. "
        "When the same evidence explicitly reports the requested difference (for example an Absolute Delta row), copy it into reported_value exactly; otherwise leave reported_value null. Do not calculate the difference yourself.\nINPUT:"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def slot_image_attachments(
    sample: dict[str, Any],
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    max_images: int = 0,
) -> list[dict[str, Any]]:
    """Return valid crops and their C-card bindings for one slot.

    A VLM image has the same evidence scope as the cards used for extraction;
    it must not inherit unrelated attachments from a global prompt preview.
    """
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    card_support = projection.get("_card_support_keys") or {}
    key_index = projection.get("_key_index") or {}
    catalog = {
        str(record.get("evidence_ref") or ""): record
        for record in hierarchy.get("l0_catalog") or []
        if isinstance(record, dict)
    }
    explicit_object_count = len(_explicit_locator_terms(str(sample.get("question") or "")))
    image_limit = min(max_images, explicit_object_count) if max_images > 0 and explicit_object_count else max_images
    attachments: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for card in slot_cards(slot, hierarchy, str(sample.get("question") or ""), candidates=candidates):
        key = str(card.get("key") or "")
        record_keys = card_support.get(key) or [str(card.get("record_key") or "")]
        for record_key in record_keys:
            ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
            record = catalog.get(ref) or {}
            if str(record.get("source_type") or "") not in {"figure", "table", "equation_algorithm"}:
                continue
            path = str(record.get("crop_path") or record.get("image_path") or "")
            if not path or not Path(path).is_file():
                continue
            attachment = by_path.get(path)
            if attachment is None:
                attachment = {"path": path, "support_card_keys": []}
                attachments.append(attachment)
                by_path[path] = attachment
            if key and key not in attachment["support_card_keys"]:
                attachment["support_card_keys"].append(key)
            if image_limit > 0 and len(attachments) >= image_limit:
                return attachments
    return attachments


def slot_image_paths(
    sample: dict[str, Any],
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    max_images: int = 0,
) -> list[str]:
    """Compatibility helper for callers that only need attachment paths."""
    return [
        str(row["path"])
        for row in slot_image_attachments(sample, slot, hierarchy, candidates, max_images)
    ]


def _numbers(value: Any) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", str(value or ""))


def _value_is_numeric(value: Any) -> bool:
    """Whether a slot value is a standalone number (not an identifier or prose)."""
    text = str(value or "").strip()
    return bool(_numbers(text)) and not re.search(r"[A-Za-z]", re.sub(r"[%x×]", "", text))


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f") if normalized != normalized.to_integral() else str(normalized.quantize(Decimal("1")))


def _evidence_values(raw: Any, card_map: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    accepted: list[dict[str, Any]] = []
    valid = True
    for item in (raw if isinstance(raw, list) else [])[:8]:
        if not isinstance(item, dict):
            valid = False
            continue
        keys = [key for key in _strings(item.get("support_keys"), 4) if key in card_map]
        value = item.get("value")
        if not keys or value in (None, ""):
            valid = False
            continue
        accepted.append({
            "name": _clip(item.get("name"), 160),
            "value": value,
            "support_keys": keys,
            **(
                {"conditions": item["conditions"]}
                if isinstance(item.get("conditions"), dict) and item["conditions"]
                else {}
            ),
        })
    return accepted, valid


def _table_reported_difference(
    evidence_values: list[dict[str, Any]],
    keys: list[str],
    metadata: dict[str, Any],
    card_map: dict[str, Any],
    key_index: dict[str, Any],
    catalog: dict[str, Any],
) -> str | None:
    """Read an explicit same-column absolute-delta cell from a bound table card.

    Displayed operands are often rounded, so an author-reported delta is more
    faithful than subtracting two rounded cells. This recognizes only a table
    row explicitly labeled as an absolute delta and only its shared operand
    column; it never infers a new row, column, or support key.
    """
    if len(evidence_values) != 2:
        return None
    operands = [str(item.get("value") or "").strip() for item in evidence_values]
    if not all(operands):
        return None
    for key in keys:
        card = metadata.get(key) or {}
        blocks = [str(card.get("proposition") or "")] if str(card.get("source_type") or "") == "table" else []
        for record_key in card_map.get(key) or []:
            ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
            record = catalog.get(ref) or {}
            if str(record.get("source_type") or "") == "table":
                blocks.append(str(record.get("text") or ""))
        for block in blocks:
            rows = [
                [cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in block.splitlines()
                if len(line.strip().strip("|").split("|")) >= 2
            ]
            candidates: list[tuple[int, int, str]] = []
            for column in range(max((len(row) for row in rows), default=0)):
                positions = [
                    [index for index, row in enumerate(rows) if column < len(row) and row[column] == operand]
                    for operand in operands
                ]
                if not all(positions):
                    continue
                # Match the explicit delta immediately following this pair of
                # rows. That prevents a prior compression setting's delta from
                # being reused merely because it shares the metric column.
                after_pair = max(max(found) for found in positions)
                for index, row in enumerate(rows[after_pair + 1:], start=after_pair + 1):
                    label = row[0].lower().replace("∆", "delta").replace("Δ", "delta")
                    if "absolute" in label and "delta" in label and column < len(row) and _numbers(row[column]):
                        candidates.append((index - after_pair, column, row[column]))
                        break
            if candidates:
                return min(candidates, key=lambda item: (item[0], item[1]))[2]
    return None


def _has_attached_crop(
    key: str,
    card_map: dict[str, Any],
    key_index: dict[str, Any],
    catalog: dict[str, Any],
) -> bool:
    """Whether a support card resolves to an L0 record with a real crop file."""
    for record_key in card_map.get(key) or []:
        ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
        record = catalog.get(ref) or {}
        path = str(record.get("crop_path") or record.get("image_path") or "")
        if path and Path(path).is_file():
            return True
    return False


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _rows_from_evidence_values(
    evidence_values: list[dict[str, Any]],
    table_schema: Any,
) -> list[dict[str, Any]]:
    """Reconstruct table rows from flat evidence_values pairs.

    The extractor may emit row data as consecutive evidence_values whose names
    match the public schema columns (for example "Method Name" then "Training
    Objective Equation ID"); a repeated first column starts a new row.
    """
    columns: list[str] = []
    for column in table_schema or []:
        name = column.get("name") or column.get("column") or column.get("key") if isinstance(column, dict) else column
        text = str(name or "").strip()
        if text:
            columns.append(text)
    if not columns:
        return []
    by_normalized = {_normalize_column_name(column): column for column in columns}
    first = columns[0]
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in evidence_values:
        name = _normalize_column_name(item.get("name"))
        column = by_normalized.get(name)
        if column is None:
            continue
        if column == first and current is not None:
            rows.append(current)
            current = None
        if current is None:
            current = {"values": {}, "support_keys": []}
        current["values"][column] = item.get("value")
        for key in _strings(item.get("support_keys"), 4):
            if key not in current["support_keys"]:
                current["support_keys"].append(key)
    if current is not None:
        rows.append(current)
    return rows


def _header_tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _best_table_column(schema_column: str, headers: list[str]) -> str | None:
    """Choose the table header with the largest token overlap with the schema column."""
    target = _header_tokens(schema_column)
    best: str | None = None
    best_score = 0
    for header in headers:
        score = len(target & _header_tokens(header))
        if score > best_score:
            best_score = score
            best = header
    return best


def _parse_table_text(text: str) -> list[dict[str, Any]]:
    """Parse markdown rows ('| h1 | h2 |' + '| v1 | v2 |') and card rows
    ('Columns: h1 | h2 ... Row <label>: h1=v1; h2=v2 ...') into tables."""
    tables: list[dict[str, Any]] = []
    headers: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not any(cells):
            continue
        if all(re.fullmatch(r"[-:]+", cell or "") for cell in cells):
            continue
        if headers is None:
            headers = cells
        else:
            rows.append({"label": cells[0] if cells else "", "cells": cells, "headers": headers})
    if headers is not None:
        tables.append({"headers": headers, "rows": rows})
    card_match = re.search(r"Columns:\s*(.*?)(?=\s*Row\s|\Z)", str(text or ""), re.S)
    if card_match:
        card_headers = [header.strip() for header in card_match.group(1).split("|")]
        card_rows = []
        for row_match in re.finditer(r"Row\s+(.*?):\s*(.*?)(?=\s*Row\s|\Z)", str(text or ""), re.S):
            pairs: dict[str, str] = {}
            for pair in row_match.group(2).split(";"):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    pairs[key.strip()] = value.strip()
            card_rows.append({"label": row_match.group(1).strip(), "pairs": pairs, "headers": card_headers})
        if card_rows:
            tables.append({"headers": card_headers, "rows": card_rows})
    return tables


def _align_table_rows_from_values(
    evidence_values: list[dict[str, Any]],
    table_schema: Any,
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover complete schema rows from bare evidence values.

    Each model-provided value is located inside a bound table row; the row
    label fills the row-key schema column and the best-matching table column
    fills the remaining schema columns.  This fixes missing labels (q_025) and
    wrong-column readings (q_054: value 38.68 locates the row, then the schema
    column picks the AP-nus cell instead of the kit cell the model read).
    """
    schema_columns = [
        str(column) for column in (table_schema or [])
        if isinstance(column, dict) and str(column.get("name") or "")
    ] or [str(column) for column in (table_schema or []) if str(column)]
    if not schema_columns or not evidence_values:
        return []
    values = {str(item.get("value") or "").strip() for item in evidence_values}
    row_key_column = schema_columns[0]
    results: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for table in tables:
        headers = [str(header) for header in (table.get("headers") or [])]
        for row in table.get("rows") or []:
            cells = row.get("cells")
            pairs = row.get("pairs")
            if cells is not None:
                cell_by_header = dict(zip(headers, cells))
                if not values.intersection(str(value) for value in cell_by_header.values()):
                    continue
            elif pairs:
                if not values.intersection(str(value) for value in pairs.values()):
                    continue
            else:
                continue
            out: dict[str, Any] = {}
            for index, column in enumerate(schema_columns):
                if index == 0:
                    out[column] = str(row.get("label") or "")
                    continue
                header = _best_table_column(column, headers)
                if cells is not None and header and header in cell_by_header:
                    out[column] = cell_by_header[header]
                elif pairs and header and header in pairs:
                    out[column] = pairs[header]
            if not out.get(row_key_column):
                continue
            key = tuple(sorted((str(key), str(value)) for key, value in out.items()))
            if key in seen:
                continue
            seen.add(key)
            results.append({"values": out, "support_keys": []})
    return results


def validate_slot_extraction(
    raw: Any,
    slot: dict[str, Any],
    hierarchy: dict[str, Any],
    table_schema: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind a model output to known C-keys and reject unsupported numbers."""
    row = raw if isinstance(raw, dict) else {}
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    card_map = projection.get("_card_support_keys") or {}
    metadata = projection.get("_card_metadata") or {}
    key_index = projection.get("_key_index") or {}
    catalog = {str(record.get("evidence_ref") or ""): record for record in hierarchy.get("l0_catalog") or [] if isinstance(record, dict)}
    keys = []
    for key in _strings(row.get("support_keys"), 4):
        if key in card_map and key not in keys:
            keys.append(key)
    evidence_values, evidence_values_ok = _evidence_values(row.get("evidence_values"), card_map)
    for item in evidence_values:
        for key in item["support_keys"]:
            if key not in keys:
                keys.append(key)
    raw_table_rows = row.get("table_rows")
    table_rows: list[dict[str, Any]] = []
    if isinstance(raw_table_rows, list):
        for item in raw_table_rows[:16]:
            if not isinstance(item, dict):
                continue
            values = item.get("values") if isinstance(item.get("values"), dict) else {}
            row_keys = [key for key in _strings(item.get("support_keys"), 4) if key in card_map]
            clean_values = {str(column): value for column, value in values.items() if str(value or "").strip()}
            if not clean_values or not row_keys:
                continue
            for key in row_keys:
                if key not in keys:
                    keys.append(key)
            table_rows.append({"values": clean_values, "support_keys": row_keys})
    table_mode = isinstance(raw_table_rows, list) or bool(table_schema)
    if table_mode and table_schema:
        if not table_rows and evidence_values:
            table_rows = _rows_from_evidence_values(evidence_values, table_schema)
        if table_schema:
            schema_columns = [
                str(column) for column in (table_schema or [])
                if isinstance(column, dict) and str(column.get("name") or "")
            ] or [str(column) for column in (table_schema or []) if str(column)]
            table_sources = []
            for key in keys:
                proposition = str((metadata.get(key) or {}).get("proposition") or "")
                if proposition:
                    table_sources.append(proposition)
                for record_key in card_map.get(key) or []:
                    ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
                    record = catalog.get(ref) or {}
                    if str(record.get("source_type") or "") == "table":
                        table_sources.append(str(record.get("text") or ""))
            tables = [table for source in table_sources for table in _parse_table_text(source)]
            if not table_rows and evidence_values:
                aligned = _align_table_rows_from_values(evidence_values, table_schema, tables)
                for table_row in aligned:
                    table_row["support_keys"] = list(keys)
                table_rows = aligned
            completed: list[dict[str, Any]] = []
            seen: set[str] = set()
            for table_row in table_rows:
                values = dict(table_row.get("values") or {})
                missing = [column for column in schema_columns if not str(values.get(column) or "").strip()]
                if missing:
                    anchors = [{"value": value} for value in values.values() if str(value or "").strip()]
                    for aligned in _align_table_rows_from_values(anchors, table_schema, tables):
                        merged = {**aligned["values"], **{k: v for k, v in values.items() if str(v or "").strip()}}
                        if all(str(merged.get(column) or "").strip() for column in schema_columns):
                            values = merged
                            break
                key_text = json.dumps(values, ensure_ascii=False, sort_keys=True)
                if key_text not in seen:
                    seen.add(key_text)
                    completed.append({"values": values, "support_keys": table_row.get("support_keys") or list(keys)})
            table_rows = completed
        for table_row in table_rows:
            for key in table_row.get("support_keys") or []:
                if key not in keys:
                    keys.append(key)
    status = str(row.get("status") or "unsupported")
    if status not in _STATUSES:
        status = "unsupported"
    source_types = set(slot.get("required_source_types") or [])
    type_ok = bool(keys) and (not source_types or any(str((metadata.get(key) or {}).get("source_type") or "") in source_types for key in keys))
    source_blocks = [str((metadata.get(key) or {}).get("proposition") or "") for key in keys]
    for key in keys:
        for record_key in card_map.get(key) or []:
            ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
            source_blocks.append(str((catalog.get(ref) or {}).get("text") or ""))
    source_text = "\n".join(source_blocks)
    visual_keys = [
        key
        for key in keys
        if str((metadata.get(key) or {}).get("source_type") or "") in {"figure", "table", "equation_algorithm"}
        and _has_attached_crop(key, card_map, key_index, catalog)
    ]
    visual_supported = bool(visual_keys) and set(visual_keys) == set(keys)
    operation = str(slot.get("operation") or "direct")
    # Direct/count/list answers require one accepted `value` and a C-key. The
    # optional operand list is meaningful only for arithmetic differences;
    # malformed auxiliary operands must not discard an otherwise grounded fact.
    if operation != "difference":
        evidence_values = []
        evidence_values_ok = True
    required_condition_terms = _condition_terms(" ".join(slot.get("required_conditions") or []))
    condition_ok = True
    same_table_operands = False
    if operation == "difference" and required_condition_terms:
        operand_terms: list[set[str]] = []
        for item in evidence_values:
            conditions = item.get("conditions")
            terms = (
                _condition_terms(" ".join(str(value) for value in conditions.values()))
                if isinstance(conditions, dict)
                else set()
            )
            operand_terms.append(terms)
            if not required_condition_terms.issubset(terms):
                condition_ok = False
        # Both operands must co-occur in one bound table record; a value from a
        # different table row or a hallucinated number must not pair with the
        # minuend merely because both numbers are numeric.
        operand_values = [str(item.get("value") or "").strip() for item in evidence_values]
        if all(operand_values):
            for key in keys:
                for record_key in card_map.get(key) or []:
                    ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
                    record = catalog.get(ref) or {}
                    if str(record.get("source_type") or "") != "table":
                        continue
                    text = str(record.get("text") or "")
                    if all(value in text for value in operand_values):
                        same_table_operands = True
                        break
                if same_table_operands:
                    break
        if not same_table_operands:
            condition_ok = False
    deterministic_reported = (
        _table_reported_difference(evidence_values, keys, metadata, card_map, key_index, catalog)
        if operation == "difference" else None
    )
    reported_value = deterministic_reported or row.get("reported_value")
    reported_numbers = _numbers(reported_value)
    source_numbers = set(_numbers(source_text))
    # Values read from an attached crop (figure/table/equation image) are not
    # expected to appear in the serialized caption text.  The crop is the L0
    # proof; require containment in text only for text-sourced values.
    numeric_inputs = list(evidence_values) if operation == "difference" else [{"value": row.get("value")}]
    if reported_numbers:
        numeric_inputs.append({"value": reported_value})
    if table_mode:
        # Table cells are copied verbatim from the serialized table; the
        # post-hoc grounding gate re-validates them through slot-supported
        # values. Number containment per cell is not meaningful here.
        numeric_ok = True
    elif operation == "difference" or operation in ("maximum", "minimum", "count") or _value_is_numeric(row.get("value")):
        numeric_ok = all(
            number in source_numbers or visual_supported
            for item in numeric_inputs
            for number in _numbers(item.get("value"))
        )
    else:
        # List/identifier/prose answers are not required to contain their
        # digits verbatim in the serialized text; type and condition checks
        # plus C-key binding already bound them to evidence.
        numeric_ok = True
    if operation == "count" and not re.fullmatch(r"[0-9]+", str(row.get("value") or "").strip()):
        numeric_ok = False
    value = row.get("value")
    derived = False
    reported_difference = False
    reported_difference_source = ""
    if operation == "difference":
        if reported_numbers and all(number in source_numbers for number in reported_numbers):
            value = str(reported_value)
            reported_difference = True
            reported_difference_source = "deterministic_table_delta" if deterministic_reported else "model_reported_delta"
        else:
            try:
                if len(evidence_values) != 2:
                    raise InvalidOperation
                value = _decimal_text(Decimal(str(evidence_values[0]["value"])) - Decimal(str(evidence_values[1]["value"])))
                derived = True
            except (InvalidOperation, ValueError):
                value = None
    table_ok = (not table_mode) or bool(table_rows)
    accepted = (
        status == "supported"
        and bool(keys)
        and type_ok
        and numeric_ok
        and evidence_values_ok
        and condition_ok
        and table_ok
        and (table_mode or value not in (None, ""))
    )
    output = {
        "slot_id": str(slot.get("id") or ""), "status": "supported" if accepted else "partial",
        "value": value if accepted else None,
        "conditions": row.get("conditions") if isinstance(row.get("conditions"), dict) else {},
        "support_keys": keys, "evidence_values": evidence_values, "missing_conditions": _strings(row.get("missing_conditions")),
        "table_rows": table_rows if table_mode else [],
        "validation": {
            "type_ok": type_ok, "numeric_ok": numeric_ok, "evidence_values_ok": evidence_values_ok,
            "model_status": status, "derived_value": derived, "reported_difference": reported_difference,
            "reported_difference_source": reported_difference_source, "visual_supported": visual_supported,
            "visual_keys": visual_keys, "condition_ok": condition_ok, "same_table_operands": same_table_operands,
        },
    }
    audit = [{"slot_id": output["slot_id"], "status": "slot_supported" if accepted else "slot_needs_backfill", **output["validation"]}]
    return output, audit


def slot_composition_messages(sample: dict[str, Any], candidates: list[dict[str, Any]], contract: dict[str, Any], slots: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Compose only from already validated slot values; no retrieval is possible here."""
    answer_types = [str(value) for value in contract.get("answer_types") or []]
    answer_shape: dict[str, Any] = {}
    if "freeform" in answer_types:
        answer_shape["freeform"] = {"text": "<supported concise answer>"}
    if "multiple_choice" in answer_types:
        keys = [str(row.get("key") or "") for row in (contract.get("multiple_choice") or {}).get("options") or [] if row.get("key")]
        answer_shape["multiple_choice"] = {"gold": "<option key>" if not keys else f"<one of {keys}>"}
    if "table" in answer_types:
        answer_shape["table"] = {"rows": []}
    payload = {
        "question": sample.get("question"), "answer_contract": contract,
        "candidate_papers": [{"paper_id": row.get("paper_id")} for row in candidates],
        "validated_slots": slots,
    }
    system = (
        "Compose the required answer only from validated_slots. Do not add facts, values, papers, or evidence. "
        "Return every answer type required by answer_contract: when both freeform and multiple_choice are required, "
        "output both, with multiple_choice.gold as a single uppercase option key such as \"A\". "
        "For a numeric freeform answer, copy the exact validated numeric value only, without a sentence or unit. Return JSON only."
    )
    target: dict[str, Any] = {
        "gold_papers": [{"paper_id": "<candidate id>"}],
        "claim_to_slot_ids": {"Q01": ["S001"]},
        "answer": answer_shape,
    }
    if "multi" in str(sample.get("task_family") or "").lower():
        target["contributing_papers"] = [{"paper_id": "<candidate id>"}]
    if "table" in answer_types:
        target["table_answer_plan"] = [{"row_slot_ids": ["S001"], "values": {"<exact table schema column>": "<supported value>"}}]
    user = (
        "Match TARGET_JSON_SHAPE exactly. `answer` must be an object, never a scalar. "
        "Omit only optional fields absent from TARGET_JSON_SHAPE; table values must use the exact public schema.\nINPUT:"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\nTARGET_JSON_SHAPE:" + json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def align_composed_values(raw: Any, sample: dict[str, Any], validated_slots: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply only deterministic numeric freeform formatting from accepted slots."""
    result = dict(raw) if isinstance(raw, dict) else {}
    answer = result.get("answer") if isinstance(result.get("answer"), dict) else {}
    freeform = answer.get("freeform") if isinstance(answer.get("freeform"), dict) else None
    question = str(sample.get("question") or "").lower()
    numeric_question = bool(re.search(r"\b(?:how many|how much|by how much|number of|score|accuracy|f1|fid|nrmse|standard deviation|rate|percentage|value)\b", question))
    if not numeric_question or not freeform:
        return result, []
    text = str(freeform.get("text") or "")
    values = list(dict.fromkeys(
        str(slot.get("value"))
        for slot in validated_slots
        if slot.get("status") == "supported" and _numbers(slot.get("value"))
    ))
    matched = [value for value in values if value and value in text]
    if len(matched) != 1:
        return result, []
    answer = dict(answer)
    answer["freeform"] = {"text": matched[0]}
    result["answer"] = answer
    return result, [{"status": "composition_numeric_value_normalized", "value": matched[0]}]


def bind_composition_support(raw: Any, validated_slots: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert composition slot IDs into the C-keys required by keyed grounding."""
    result = dict(raw) if isinstance(raw, dict) else {}
    by_id = {str(row.get("slot_id") or ""): row for row in validated_slots if row.get("status") == "supported"}
    mapping: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    for claim_id, slot_ids in (result.get("claim_to_slot_ids") or {}).items():
        keys: list[str] = []
        for slot_id in _strings(slot_ids, 4):
            for key in by_id.get(slot_id, {}).get("support_keys") or []:
                if key not in keys:
                    keys.append(key)
        if keys:
            mapping[str(claim_id)] = keys[:4]
            audit.append({"claim_id": str(claim_id), "status": "composition_claim_bound", "support_keys": keys[:4]})
        else:
            audit.append({"claim_id": str(claim_id), "status": "composition_claim_unbound"})
    result.pop("claim_to_slot_ids", None)
    result["claim_to_support_keys"] = mapping
    plans = []
    for row in result.get("table_answer_plan") or []:
        if not isinstance(row, dict):
            continue
        keys = []
        for slot_id in _strings(row.get("row_slot_ids"), 4):
            for key in by_id.get(slot_id, {}).get("support_keys") or []:
                if key not in keys:
                    keys.append(key)
        if keys:
            plans.append({"row_support_key": keys[0], "values": row.get("values") if isinstance(row.get("values"), dict) else {}})
        else:
            audit.append({"status": "composition_table_row_unbound"})
    if "table_answer_plan" in result:
        result["table_answer_plan"] = plans
    return result, audit


__all__ = [
    "SLOT_PLAN_VERSION", "align_composed_values", "bind_composition_support", "ensure_slot_cards", "slot_cards", "slot_composition_messages", "slot_extraction_messages", "slot_image_attachments", "slot_image_paths",
    "slot_plan_messages", "validate_slot_extraction", "validate_slot_plan",
]
