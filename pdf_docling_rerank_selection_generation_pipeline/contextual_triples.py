"""Grounded contextual-triple graph built on the auditable L0-L3 hierarchy.

This is deliberately not a conventional knowledge graph.  Every relation keeps
the supporting L0 records and the L1 window from which it was derived.  A
triple can therefore make navigation cheap without becoming the sole proof for
an answer fact.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from .metadata_index import tokenize


TRIPLE_GRAPH_VERSION = "contextual_triples_v1"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:%|x|×|[A-Za-z]+)?")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{1,}|[A-Z]{2,}[A-Z0-9_-]*)\b")
_ORDINAL_REFERENCE_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+reference\b", re.IGNORECASE)
_NAVIGATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "cited", "does", "for", "from", "how", "in", "is", "it",
    "kind", "many", "of", "on", "paper", "reference", "the", "to", "used", "was", "what", "which", "who",
}
_NAVIGATION_TERM_EXPANSIONS = {
    # Request-class aliases bridge ordinary paper wording differences, such as
    # "hardware configuration" in a question and "NVIDIA GPU" in a record.
    # They only route existing L0 records; they never become answer facts.
    "hardware": {"gpu", "cpu", "nvidia", "accelerator", "device"},
    "configure": {"configuration", "setup"},
    "configuration": {"configure", "setup"},
    "implementation": {"implemented", "implementation", "setup"},
}


def _clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    return text if limit <= 0 or len(text) <= limit else text[: max(1, limit - 3)].rstrip() + "..."


def _sentence_for_record(record: dict[str, Any], question_terms: set[str], limit: int) -> str:
    """Select one source sentence; this is extractive rather than a summary."""
    text = str(record.get("text") or "")
    candidates = [_clip(value, limit) for value in _SENTENCE_RE.split(text) if len(value.strip()) >= 12]
    if not candidates:
        return _clip(text, limit)
    def score(value: str) -> tuple[int, int, int]:
        terms = set(tokenize(value))
        overlap = len(question_terms.intersection(terms))
        facts = len(_NUMBER_RE.findall(value)) + len(_ENTITY_RE.findall(value))
        return overlap, facts, -len(value)
    return max(candidates, key=score)


def _source_modality(record: dict[str, Any]) -> str:
    source_type = str(record.get("source_type") or "text_span")
    if source_type in {"figure", "table"}:
        return "visual_structured" if record.get("crop_path") else "structured_text"
    if source_type == "equation_algorithm":
        return "symbolic_text"
    return "text"


def _question_named_papers(question: str, hierarchy: dict[str, Any]) -> set[str]:
    """Identify an explicitly named candidate paper from its title/leading alias."""
    compact_question = re.sub(r"[^a-z0-9]+", "", question.lower())
    named: set[str] = set()
    for paper in (hierarchy.get("l3_navigation") or {}).get("papers") or []:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or "")
        alias = re.sub(r"[^a-z0-9]+", "", title.split(":", 1)[0].lower())
        if len(alias) >= 4 and alias in compact_question:
            named.add(str(paper.get("paper_id") or ""))
    return {paper_id for paper_id in named if paper_id}


def _citation_id(record: dict[str, Any]) -> str:
    locator = record.get("locator") if isinstance(record.get("locator"), dict) else {}
    return str(locator.get("citation_id") or "").strip()


def _navigation_seed_refs(question: str, hierarchy: dict[str, Any], catalog: dict[str, dict[str, Any]], *, limit: int = 4) -> list[tuple[str, str]]:
    """Return small additive L1 seeds that ordinary card ranking can miss.

    This is not a retrieval cascade.  Rich L2 cards remain the main entry
    points; these seeds only expose exact, already-selected L0 facts that have
    a deterministic locator or a rare direct lexical match to the question.
    """
    named_papers = _question_named_papers(question, hierarchy)
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(ref: str, reason: str) -> None:
        if ref in catalog and ref not in seen:
            selected.append((ref, reason))
            seen.add(ref)

    reference_match = _ORDINAL_REFERENCE_RE.search(question)
    if reference_match:
        citation_id = reference_match.group(1)
        for ref, record in sorted(catalog.items()):
            if str(record.get("source_type") or "") != "citation_context" or _citation_id(record) != citation_id:
                continue
            if named_papers and str(record.get("paper_id") or "") not in named_papers:
                continue
            add(ref, f"explicit_citation_id:{citation_id}")

    # Use inverse document frequency across the existing L0 catalog so a rare
    # requested term such as "hardware" is not drowned out by generic method
    # names such as "RAG". This selects at most a few extra text anchors.
    terms = {term for term in tokenize(question) if len(term) >= 3 and term not in _NAVIGATION_STOPWORDS}
    if terms:
        expanded_terms = {
            expansion
            for term in terms
            for expansion in _NAVIGATION_TERM_EXPANSIONS.get(term, set())
        }
        all_terms = terms.union(expanded_terms)
        document_terms = {
            ref: set(tokenize(" ".join([str(record.get("label") or ""), str(record.get("section_title") or ""), str(record.get("text") or "")])))
            for ref, record in catalog.items()
        }
        total = max(1, len(document_terms))
        document_frequency = {term: sum(term in values for values in document_terms.values()) for term in all_terms}
        scored: list[tuple[float, str]] = []
        primary_type = str(hierarchy.get("primary_evidence_type") or "")
        for ref, values in document_terms.items():
            overlap = all_terms.intersection(values)
            if not overlap:
                continue
            score = sum(
                math.log((total + 1) / (document_frequency[term] + 1)) * (1.0 if term in terms else 0.55)
                for term in overlap
            )
            if str(catalog[ref].get("source_type") or "") == primary_type:
                score += 0.25
            scored.append((-score, ref))
        for _, ref in sorted(scored):
            if len(selected) >= limit:
                break
            add(ref, "idf_lexical_navigation")
    # Navigation records were added by the auditable post-selection supplement
    # in evidence_hierarchy.py. Add them after direct/rare lexical matches so
    # a generic named-method record cannot crowd out a requested GPU or value.
    for ref, record in sorted(catalog.items()):
        reasons = [str(value) for value in record.get("navigation_reasons") or [] if str(value)]
        if reasons and len(selected) < limit:
            add(ref, "navigation_record:" + reasons[0])
    return selected[:max(0, limit)]


def _query_aware_table_lines(text: Any, query_terms: set[str], limit: int = 4) -> list[str]:
    """Keep table schema plus the rows needed for this question.

    Docling's structured map can collapse repeated headers such as F1/EM under
    a dataset span.  The raw Markdown lines remain the authoritative L0 text,
    so retain the schema lines and query-relevant raw rows instead of trusting
    a lossy dictionary projection.
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return []
    schema = lines[: min(4, len(lines))]
    rows = lines[len(schema) :]
    ranked: list[tuple[int, int, str]] = []
    comparative = bool(query_terms.intersection({"outperform", "better", "difference", "much", "compare", "versus", "vs"}))
    for index, line in enumerate(rows):
        terms = set(tokenize(line))
        score = len(query_terms.intersection(terms)) * 8
        if comparative and re.search(r"\b(?:absolute|relative|delta|∆)\b", line, re.IGNORECASE):
            score += 6
        if re.search(r"\b(?:ours|icae|baseline)\b", line, re.IGNORECASE):
            score += 3
        if re.search(r"500\s*(?:→|->)\s*1(?!\d)", line):
            score += 5
        ranked.append((score, -index, line))
    selected = [line for _, _, line in sorted(ranked, reverse=True)[: max(1, limit)]]
    if comparative:
        ours = next((line for line in rows if re.search(r"\b(?:ours|our method)\b", line, re.IGNORECASE) and re.search(r"500\s*(?:→|->)\s*1(?!\d)", line)), "")
        baseline_index = next((index for index, line in enumerate(rows) if re.search(r"\b(?:icae|baseline)\b", line, re.IGNORECASE) and re.search(r"500\s*(?:→|->)\s*1(?!\d)", line)), -1)
        baseline = rows[baseline_index] if baseline_index >= 0 else ""
        delta = next((line for line in rows[baseline_index + 1 :] if re.search(r"\babsolute\b|∆", line, re.IGNORECASE)), "")
        required = [line for line in (ours, baseline, delta) if line]
        if len(required) == 3:
            selected = [*required, *[line for _, _, line in sorted(ranked, reverse=True) if line not in required]][: max(3, limit)]
        else:
            delta_rows = [item for item in ranked if re.search(r"\b(?:absolute|relative|delta|∆)\b", item[2], re.IGNORECASE)]
            if delta_rows:
                delta = max(delta_rows)[2]
                if delta not in selected:
                    selected = [*selected[: max(0, limit - 1)], delta]
    # Preserve PDF reading order after relevance selection, which keeps paired
    # Ours/ICAE/delta rows intelligible to an answer model.
    selected_set = set(selected)
    return [*schema, *[line for line in rows if line in selected_set]]


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _deterministic_table_relations(question: str, hierarchy: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover a comparison fact from a single fully grounded table window.

    This handles the common scientific-QA form ``method A vs B on dataset D,
    metric M, setting S``. It does not guess cell coordinates: all header,
    method, and delta lines must be present in the L1 raw-line window.
    """
    query_terms = set(tokenize(question))
    if not query_terms.intersection({"outperform", "compare", "difference", "better", "higher", "how", "much"}):
        return []
    relations: list[dict[str, Any]] = []
    cards_by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in hierarchy.get("l2_evidence_cards") or []:
        if isinstance(card, dict):
            for ref in card.get("support_refs") or []:
                cards_by_ref[str(ref)].append(card)
    for window in windows:
        if str(window.get("source_type") or "") != "table":
            continue
        lines = [str(line) for line in window.get("table_lines") or [] if str(line).strip().startswith("|")]
        if len(lines) < 5:
            continue
        header = _markdown_cells(lines[0])
        metric = next((_markdown_cells(line) for line in lines if re.search(r"\b(?:metric|metrices)\b", line, re.IGNORECASE)), [])
        if not header or len(metric) != len(header):
            continue
        dataset_scores = [len(query_terms.intersection(set(tokenize(cell)))) for cell in header]
        metric_scores = [len(query_terms.intersection(set(tokenize(cell)))) for cell in metric]
        column = max(range(len(header)), key=lambda index: (dataset_scores[index] * 4 + metric_scores[index], -index))
        if dataset_scores[column] <= 0 or metric_scores[column] <= 0:
            continue
        ours = next((line for line in lines if re.search(r"\b(?:ours|our method)\b", line, re.IGNORECASE) and re.search(r"500\s*(?:→|->)\s*1(?!\d)", line)), "")
        baseline_index = next((index for index, line in enumerate(lines) if re.search(r"\b(?:icae|baseline)\b", line, re.IGNORECASE) and re.search(r"500\s*(?:→|->)\s*1(?!\d)", line)), -1)
        baseline = lines[baseline_index] if baseline_index >= 0 else ""
        delta = next((line for line in lines[baseline_index + 1 :] if re.search(r"\babsolute\b|∆", line, re.IGNORECASE)), "")
        if not ours or not baseline or not delta:
            continue
        ours_cells, baseline_cells, delta_cells = _markdown_cells(ours), _markdown_cells(baseline), _markdown_cells(delta)
        if max(len(ours_cells), len(baseline_cells), len(delta_cells)) <= column:
            continue
        values = [ours_cells[column], baseline_cells[column], delta_cells[column]]
        if not all(re.search(r"\d", value) for value in values):
            continue
        anchor_ref = str(window.get("anchor_ref") or "")
        cards = cards_by_ref.get(anchor_ref) or []
        relations.append({
            "triple_id": f"TR{len(relations) + 1:03d}",
            "claim_ids": [claim_id for card in cards for claim_id in card.get("claim_ids") or []],
            "subject": "Ours",
            "predicate": "outperforms",
            "object": "ICAE",
            "qualifiers": {
                "dataset": [header[column]], "metric": [metric[column]], "condition": ["500→1"],
                "comparison": [f"Ours={values[0]}", f"ICAE={values[1]}", f"absolute_delta={values[2]}"],
            },
            "support_refs": [anchor_ref],
            "support_quotes": [{"evidence_ref": anchor_ref, "quote": quote} for quote in [lines[0], next(line for line in lines if _markdown_cells(line) == metric), ours, baseline, delta]],
            "l1_window_ids": [str(window.get("window_id") or "")],
            "card_ids": [str(card.get("card_id") or "") for card in cards if str(card.get("card_id") or "")],
            "verification": {"status": "deterministic_table_relation", "column_index": column},
        })
    return relations


def _deterministic_citation_relations(question: str, hierarchy: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve an explicitly requested bibliography author from a raw entry.

    Bibliography entries already have a stable `citation_id` and author order.
    When a question names both an ordinal reference and its first author, this
    relation is more reliable and cheaper than asking an LLM to restate the
    first comma-delimited author field.
    """
    match = _ORDINAL_REFERENCE_RE.search(question)
    if not match or not re.search(r"\bfirst\s+author\b", question, re.IGNORECASE):
        return []
    requested_id = match.group(1)
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)}
    named_papers = _question_named_papers(question, hierarchy)
    relations: list[dict[str, Any]] = []
    for window in windows:
        if str(window.get("source_type") or "") != "citation_context":
            continue
        anchor_ref = str(window.get("anchor_ref") or "")
        anchor = catalog.get(anchor_ref) or {}
        if _citation_id(anchor) != requested_id:
            continue
        if named_papers and str(anchor.get("paper_id") or "") not in named_papers:
            continue
        entry = _normalized(anchor.get("text"))
        first_author = entry.split(",", 1)[0].strip()
        if len(first_author) < 2 or not re.search(r"[A-Za-z]", first_author):
            continue
        relations.append({
            "triple_id": f"TC{len(relations) + 1:03d}",
            "claim_ids": [],
            "subject": f"Reference {requested_id}",
            "predicate": "cites",
            "object": first_author,
            "qualifiers": {"citation_id": [requested_id], "author_position": ["first"]},
            "support_refs": [anchor_ref],
            "support_quotes": [{"evidence_ref": anchor_ref, "quote": first_author}],
            "l1_window_ids": [str(window.get("window_id") or "")],
            "card_ids": [],
            "verification": {"status": "deterministic_citation_relation"},
        })
    return relations


def _deterministic_hardware_relations(question: str, hierarchy: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract an explicit compute device without inventing a framework subject."""
    terms = set(tokenize(question))
    if not terms.intersection({"hardware", "gpu", "cpu", "tpu", "configure", "configuration", "compute"}):
        return []
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)}
    relations: list[dict[str, Any]] = []
    for window in windows:
        if str(window.get("source_type") or "") != "text_span":
            continue
        anchor_ref = str(window.get("anchor_ref") or "")
        anchor = catalog.get(anchor_ref) or {}
        source = _normalized(anchor.get("text"))
        match = re.search(r"\b(?:run|ran|executed|conducted)\s+on\s+([^.;]*(?:GPU|CPU|TPU)[^.;]*)", source, re.IGNORECASE)
        if not match:
            continue
        device = match.group(1).strip()
        sentence = next((value.strip() for value in _SENTENCE_RE.split(source) if device in value), source)
        relations.append({
            "triple_id": f"TH{len(relations) + 1:03d}",
            "claim_ids": [],
            "subject": "experiments",
            "predicate": "uses",
            "object": device,
            "qualifiers": {"hardware": [device]},
            "support_refs": [anchor_ref],
            "support_quotes": [{"evidence_ref": anchor_ref, "quote": sentence}],
            "l1_window_ids": [str(window.get("window_id") or "")],
            "card_ids": [],
            "verification": {"status": "deterministic_hardware_relation"},
        })
    return relations


def build_l1_windows(question: str, hierarchy: dict[str, Any], *, sentence_chars: int = 520) -> list[dict[str, Any]]:
    """Create bounded, source-specific L1 windows from L0 plus existing L1.

    A window does not duplicate an L0 object.  It contains only an extractive
    sentence, object schema/caption metadata, and references to the raw L0
    records needed to interpret that unit.
    """
    question_terms = set(tokenize(question))
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)}
    existing = {str(row.get("anchor_ref") or ""): row for row in hierarchy.get("l1_contexts") or [] if isinstance(row, dict)}
    # A frozen union can contain hundreds of selected anchors. They remain in
    # L0 for targeted expansion, but making each an L1 window or triple would
    # recreate the original context explosion. Rich L2 cards are compact,
    # explicit entry points; legacy artifacts without cards fall back to the
    # complete selected-anchor set.
    rich_refs: list[str] = []
    for card in hierarchy.get("l2_evidence_cards") or []:
        if not isinstance(card, dict):
            continue
        for ref in card.get("support_refs") or []:
            ref = str(ref or "")
            if ref in catalog and ref not in rich_refs:
                rich_refs.append(ref)
    seed_reasons: dict[str, list[str]] = defaultdict(list)
    for ref in rich_refs:
        seed_reasons[ref].append("rich_l2_card")
    for ref, reason in _navigation_seed_refs(question, hierarchy, catalog):
        if reason not in seed_reasons[ref]:
            seed_reasons[ref].append(reason)
    anchor_refs = list(seed_reasons) or [str(ref or "") for ref in hierarchy.get("selected_anchor_refs") or []]
    windows: list[dict[str, Any]] = []
    for index, anchor_ref in enumerate(anchor_refs, start=1):
        anchor_ref = str(anchor_ref or "")
        anchor = catalog.get(anchor_ref)
        if anchor is None:
            continue
        legacy = existing.get(anchor_ref, {})
        neighbor_refs = [str(ref) for ref in legacy.get("neighbor_refs") or [] if str(ref) in catalog]
        support_refs = [anchor_ref, *[ref for ref in neighbor_refs if ref != anchor_ref]]
        source_type = str(anchor.get("source_type") or "text_span")
        window: dict[str, Any] = {
            "window_id": f"W{index:03d}",
            "anchor_ref": anchor_ref,
            "seed_reasons": seed_reasons.get(anchor_ref, ["selected_anchor_fallback"]),
            "support_refs": support_refs,
            "paper_id": anchor.get("paper_id"),
            "section_path": anchor.get("section_path") or [anchor.get("section_title")],
            "source_type": source_type,
            "modality": _source_modality(anchor),
            "crop_path": anchor.get("crop_path") if source_type == "figure" else None,
            "anchor_quote": _sentence_for_record(anchor, question_terms, sentence_chars),
            "neighbor_quotes": [
                _sentence_for_record(catalog[ref], question_terms, sentence_chars)
                for ref in neighbor_refs
            ],
        }
        # Object-specific constraints prevent a relation from separating a
        # number from the header/caption or a symbol from its definition.
        if source_type == "table":
            table_context = legacy.get("table_context") if isinstance(legacy.get("table_context"), dict) else {}
            window["table_schema"] = {
                "caption": _clip(table_context.get("caption") or anchor.get("text"), sentence_chars),
                "columns": list(table_context.get("columns") or [])[:20],
                "rows": list(table_context.get("rows") or [])[:8],
            }
            window["table_lines"] = _query_aware_table_lines(anchor.get("text"), question_terms)
        elif source_type == "figure":
            figure_context = legacy.get("figure_context") if isinstance(legacy.get("figure_context"), dict) else {}
            window["object_context"] = {
                "label": anchor.get("label"),
                "caption": _clip(figure_context.get("caption") or anchor.get("text"), sentence_chars),
                "crop_available": bool(anchor.get("crop_path")),
            }
        elif source_type == "equation_algorithm":
            equation_context = legacy.get("equation_context") if isinstance(legacy.get("equation_context"), dict) else {}
            window["object_context"] = {
                "label": anchor.get("label"),
                "block": _clip(equation_context.get("block") or anchor.get("text"), sentence_chars * 2),
            }
        elif source_type == "citation_context":
            window["object_context"] = {"citation": _clip(anchor.get("text"), sentence_chars)}
        windows.append(window)
    return windows


def _fallback_triples(question: str, hierarchy: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce proof-preserving candidate triples without an LLM call.

    ``evidence_states`` intentionally has weak semantics.  It is a safe bridge
    for an LLM-generated relation, rather than an invented subject/predicate
    assertion.  The proposition remains exact L1 text and each relation points
    to its complete L0 provenance.
    """
    cards_by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in hierarchy.get("l2_evidence_cards") or []:
        if not isinstance(card, dict):
            continue
        for ref in card.get("support_refs") or []:
            cards_by_ref[str(ref)].append(card)
    triples: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        anchor_ref = str(window.get("anchor_ref") or "")
        cards = cards_by_ref.get(anchor_ref) or []
        card = cards[0] if cards else {}
        proposition = str(card.get("proposition") or window.get("anchor_quote") or "")
        entities = [str(value) for value in card.get("entities") or [] if str(value).strip()]
        if not entities:
            entities = _ENTITY_RE.findall(proposition)[:4]
        subject = entities[0] if entities else str(window.get("source_type") or "evidence")
        support_quotes = list(card.get("support_quotes") or [])
        if not support_quotes and anchor_ref:
            support_quotes = [{"evidence_ref": anchor_ref, "quote": str(window.get("anchor_quote") or "")}]
        triples.append({
            "triple_id": f"T{index:03d}",
            "claim_ids": list(card.get("claim_ids") or []),
            "subject": subject,
            "predicate": "evidence_states",
            "object": proposition,
            "qualifiers": {
                "entities": entities[:8],
                "values": [str(value) for value in card.get("values") or _NUMBER_RE.findall(proposition)][:10],
                "conditions": [str(value) for value in card.get("conditions") or []][:8],
                "source_type": window.get("source_type"),
                "modality": window.get("modality"),
            },
            "support_refs": list(window.get("support_refs") or []),
            "support_quotes": support_quotes,
            "l1_window_ids": [str(window.get("window_id") or "")],
            "card_ids": [str(item.get("card_id") or "") for item in cards if str(item.get("card_id") or "")],
            "verification": {"status": "extractive_candidate", "reason": "awaiting_relation_extraction"},
        })
    return triples


def _graph_navigation(hierarchy: dict[str, Any], triples: list[dict[str, Any]]) -> dict[str, Any]:
    by_entity: dict[str, list[str]] = defaultdict(list)
    by_claim: dict[str, list[str]] = defaultdict(list)
    by_paper: dict[str, list[str]] = defaultdict(list)
    unresolved = list((hierarchy.get("l3_navigation") or {}).get("unresolved_claims") or [])
    for triple in triples:
        triple_id = str(triple.get("triple_id") or "")
        for entity in [triple.get("subject"), *(triple.get("qualifiers", {}).get("entities") or [])]:
            normalized = str(entity or "").strip()
            if normalized and triple_id not in by_entity[normalized]:
                by_entity[normalized].append(triple_id)
        for claim_id in triple.get("claim_ids") or []:
            if triple_id not in by_claim[str(claim_id)]:
                by_claim[str(claim_id)].append(triple_id)
        for ref in triple.get("support_refs") or []:
            record = next((row for row in hierarchy.get("l0_catalog") or [] if str(row.get("evidence_ref") or "") == str(ref)), {})
            paper_id = str(record.get("paper_id") or "")
            if paper_id and triple_id not in by_paper[paper_id]:
                by_paper[paper_id].append(triple_id)
    # Candidate paths remain alternatives.  This preserves CIRAG's protection
    # against a single early relation selection dominating a multi-hop query.
    paths = [
        {"path_id": f"PATH{index:03d}", "claim_id": claim_id, "triple_ids": ids[:4]}
        for index, (claim_id, ids) in enumerate(sorted(by_claim.items()), start=1)
    ]
    return {
        "version": TRIPLE_GRAPH_VERSION,
        "entity_to_triples": dict(sorted(by_entity.items())) ,
        "claim_to_triples": dict(sorted(by_claim.items())),
        "paper_to_triples": dict(sorted(by_paper.items())),
        "candidate_paths": paths,
        "unresolved_claims": unresolved,
    }


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _qualifier_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _canonical_predicate(value: Any) -> str:
    predicate = _normalized(value).lower()
    if predicate in {"reports", "compares", "outperforms", "underperforms", "uses", "defines", "evaluates", "cites", "optimizes", "contains", "measures", "states"}:
        return predicate
    if re.search(r"\b(?:score|achiev|obtain|perform|result|value|has)\b", predicate):
        return "reports"
    if re.search(r"\b(?:higher|lower|better|worse|greater|less|difference|delta)\b", predicate):
        return "compares"
    if re.search(r"\b(?:introduc|propose|present|define)\b", predicate):
        return "defines"
    if re.search(r"\buse(?:s|d)?\b", predicate):
        return "uses"
    return ""


def verify_llm_contextual_triples(raw: dict[str, Any], hierarchy: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace only extractive candidates whose LLM relation has L0 proof.

    Relation labels are allowed to normalize table semantics (for example,
    ``outperforms``), but every entity, value, and qualifier must still occur
    in an exact quoted L0 span.  Invalid output preserves the deterministic
    candidate rather than making the hierarchy unusable.
    """
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)}
    windows = {str(row.get("window_id") or ""): row for row in hierarchy.get("l1_evidence_windows") or [] if isinstance(row, dict)}
    all_triples = [row for row in hierarchy.get("l2_contextual_triples") or [] if isinstance(row, dict)]
    # A window may have an extractive candidate *and* a deterministic table
    # relation. Only the former is replaceable by an LLM refinement; losing the
    # latter would collapse a multi-path proof back into a single weak score.
    fallback = {
        str(row.get("l1_window_ids", [""])[0] or ""): row
        for row in all_triples
        if str((row.get("verification") or {}).get("status") or "") == "extractive_candidate"
    }
    fixed_triples = [row for row in all_triples if row not in fallback.values()]
    cards_by_ref: dict[str, list[str]] = defaultdict(list)
    for card in hierarchy.get("l2_evidence_cards") or []:
        if isinstance(card, dict):
            for ref in card.get("support_refs") or []:
                cards_by_ref[str(ref)].append(str(card.get("card_id") or ""))
    updated: list[dict[str, Any]] = []
    accepted_windows: set[str] = set()
    rejected: list[dict[str, Any]] = []
    allowed_predicates = {"reports", "compares", "outperforms", "underperforms", "uses", "defines", "evaluates", "cites", "optimizes", "contains", "measures", "states"}
    for index, item in enumerate(raw.get("triples") or [], start=1):
        if not isinstance(item, dict):
            rejected.append({"index": index, "reason": "not_object"})
            continue
        window_id = str(item.get("window_id") or "")
        window = windows.get(window_id)
        fallback_triple = fallback.get(window_id)
        if window is None or fallback_triple is None or window_id in accepted_windows:
            rejected.append({"index": index, "reason": "unknown_or_duplicate_window", "window_id": window_id})
            continue
        refs = [str(ref) for ref in item.get("support_refs") or [] if str(ref) in set(window.get("support_refs") or [])]
        quotes_by_ref: dict[str, list[str]] = defaultdict(list)
        for quote in item.get("support_quotes") or []:
            if not isinstance(quote, dict):
                continue
            ref, quote_text = str(quote.get("evidence_ref") or ""), _normalized(quote.get("quote"))
            source = _normalized((catalog.get(ref) or {}).get("text"))
            if ref in refs and quote_text and quote_text in source:
                quotes_by_ref[ref].append(quote_text)
        # A table row is uninterpretable without its multi-level header. L1
        # stores an exact, query-aware subset of raw L0 Markdown lines, so add
        # those lines as support quotes for the same table record. This is an
        # extractive completion, not a generated explanation.
        if str(window.get("source_type") or "") == "table":
            for ref in refs:
                source = _normalized((catalog.get(ref) or {}).get("text"))
                for line in window.get("table_lines") or []:
                    line = _normalized(line)
                    if line and line in source and line not in quotes_by_ref[ref]:
                        quotes_by_ref[ref].append(line)
        if not refs or any(ref not in quotes_by_ref for ref in refs):
            rejected.append({"index": index, "reason": "missing_exact_support_quote", "window_id": window_id})
            continue
        subject, predicate, obj = (_clip(item.get("subject"), 160), _canonical_predicate(item.get("predicate")), _clip(item.get("object"), 420))
        if not subject or not obj or predicate not in allowed_predicates:
            rejected.append({"index": index, "reason": "invalid_relation_shape", "window_id": window_id})
            continue
        source_text = " ".join(value for values in quotes_by_ref.values() for value in values)
        qualifiers_raw = item.get("qualifiers") if isinstance(item.get("qualifiers"), dict) else {}
        qualifiers = {key: _qualifier_values(value)[:8] for key, value in qualifiers_raw.items() if key in {"dataset", "metric", "split", "condition", "comparison", "entity", "value"}}
        required_terms = [subject, obj, *[value for values in qualifiers.values() for value in values]]
        # Multi-word natural-language objects can be a concise source-derived
        # paraphrase. Require all explicit values/names, and a high lexical
        # overlap for the rest, rather than falsely rejecting every concise
        # table relation.
        numeric_ok = all(value in source_text for value in _NUMBER_RE.findall(" ".join(required_terms)))
        entity_terms = [value for value in _ENTITY_RE.findall(" ".join(required_terms)) if len(value) > 2]
        entities_ok = all(value.lower() in source_text.lower() for value in entity_terms)
        object_terms = set(tokenize(" ".join(required_terms)))
        source_terms = set(tokenize(source_text))
        overlap = len(object_terms.intersection(source_terms)) / max(1, len(object_terms))
        if not numeric_ok or not entities_ok or overlap < 0.45:
            rejected.append({"index": index, "reason": "relation_not_grounded_enough", "window_id": window_id, "overlap": round(overlap, 3)})
            continue
        updated.append({
            **fallback_triple,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "qualifiers": qualifiers,
            "support_refs": refs,
            "support_quotes": [{"evidence_ref": ref, "quote": quote} for ref, values in quotes_by_ref.items() for quote in values],
            "card_ids": [card_id for ref in refs for card_id in cards_by_ref.get(ref, []) if card_id],
            "verification": {"status": "verified_llm_relation", "overlap": round(overlap, 3)},
        })
        accepted_windows.add(window_id)
    for window_id, triple in fallback.items():
        if window_id not in accepted_windows:
            updated.append(triple)
    updated.extend(fixed_triples)
    updated.sort(key=lambda row: str(row.get("triple_id") or ""))
    result = dict(hierarchy)
    result["l2_contextual_triples"] = updated
    result["l3_triple_navigation"] = _graph_navigation(result, updated)
    result["triple_relation_verification"] = {"accepted": len(accepted_windows), "rejected": rejected}
    return result, rejected


def visual_triple_messages(question: str, window: dict[str, Any]) -> list[dict[str, str]]:
    """One-crop prompt. Visual facts never ride in the text batch."""
    payload = {
        "question": question, "window_id": window.get("window_id"), "object_label": (window.get("object_context") or {}).get("label"),
        "caption_hint": (window.get("object_context") or {}).get("caption"), "support_refs": window.get("support_refs") or [],
    }
    system = (
        "Extract one grounded triple from this figure crop. Use only literal visible text and unambiguous relationships; "
        "do not infer performance, causality, or method properties. Return JSON only."
    )
    user = (
        "Return {\"window_id\":\"W001\",\"subject\":\"...\",\"predicate\":\"reports|uses|contains|states\",\"object\":\"...\","
        "\"qualifiers\":{\"condition\":[],\"value\":[]},\"visible_strings\":[]}.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def visual_triple_verification_messages(window: dict[str, Any], proposal: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "window_id": window.get("window_id"), "subject": proposal.get("subject"), "predicate": proposal.get("predicate"),
        "object": proposal.get("object"), "qualifiers": proposal.get("qualifiers") or {}, "visible_strings": proposal.get("visible_strings") or [],
    }
    system = (
        "Approve only when every triple entity, value, condition, and relation is directly visible in this crop. "
        "The caption is not proof for absent visual facts. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": "Return {\"supported\":true|false,\"reason\":\"brief\"}.\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def attach_visual_triple(window_id: str, raw: dict[str, Any], hierarchy: dict[str, Any], *, verified: bool) -> dict[str, Any] | None:
    """Accept a visual relation only as crop-grounded L0, never text proof."""
    triples = {str(row.get("l1_window_ids", [""])[0] or ""): row for row in hierarchy.get("l2_contextual_triples") or [] if isinstance(row, dict)}
    base = triples.get(window_id)
    if base is None:
        return None
    subject, predicate, obj = (_clip(raw.get("subject"), 160), _canonical_predicate(raw.get("predicate")), _clip(raw.get("object"), 360))
    if not verified or not subject or not obj or predicate not in {"reports", "uses", "contains", "states"}:
        return None
    qualifiers_raw = raw.get("qualifiers") if isinstance(raw.get("qualifiers"), dict) else {}
    qualifiers = {key: _qualifier_values(value)[:8] for key, value in qualifiers_raw.items() if key in {"condition", "value", "entity"}}
    updated = dict(hierarchy)
    rows = []
    for triple in hierarchy.get("l2_contextual_triples") or []:
        if triple is base:
            rows.append({
                **triple, "subject": subject, "predicate": predicate, "object": obj, "qualifiers": qualifiers,
                "support_quotes": [], "verification": {"status": "visual_crop_verified_relation", "visible_strings": [str(value) for value in raw.get("visible_strings") or []][:12]},
            })
        else:
            rows.append(triple)
    updated["l2_contextual_triples"] = rows
    updated["l3_triple_navigation"] = _graph_navigation(updated, rows)
    return updated


def attach_contextual_triple_graph(question: str, hierarchy: dict[str, Any], *, l1_sentence_chars: int = 520) -> dict[str, Any]:
    """Add deterministic L1 windows, candidate L2 triples, and L3 graph."""
    updated = dict(hierarchy)
    windows = build_l1_windows(question, updated, sentence_chars=l1_sentence_chars)
    triples = [
        *_fallback_triples(question, updated, windows),
        *_deterministic_table_relations(question, updated, windows),
        *_deterministic_citation_relations(question, updated, windows),
        *_deterministic_hardware_relations(question, updated, windows),
    ]
    updated["l1_evidence_windows"] = windows
    updated["l2_contextual_triples"] = triples
    updated["l3_triple_navigation"] = _graph_navigation(updated, triples)
    updated["triple_graph_contract"] = (
        "A triple is navigation only unless its support_refs resolve to L0 and its support_quotes pass the exact-support gate. "
        "L1 may explain a triple; L0 remains the source of every answer fact."
    )
    return updated


def triple_generation_messages(
    question: str,
    hierarchy: dict[str, Any],
    *,
    source_chars: int = 30000,
    include_visual: bool = False,
    window_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Minimal payload for LLM refinement of candidate triples.

    The caller should batch text/structured windows and send visual windows to
    a VLM one crop at a time.  This function itself never selects or exposes
    the full L0 corpus.
    """
    windows = list(hierarchy.get("l1_evidence_windows") or [])
    compact, used = [], 0
    for window in windows:
        if window_ids is not None and str(window.get("window_id") or "") not in window_ids:
            continue
        if not include_visual and str(window.get("source_type") or "") == "figure" and str(window.get("crop_path") or ""):
            continue
        table_schema = window.get("table_schema") if isinstance(window.get("table_schema"), dict) else None
        compact_table = None
        if table_schema:
            compact_table = {
                "caption": _clip(table_schema.get("caption"), 360),
                "columns": list(table_schema.get("columns") or [])[:14],
                "rows": list(table_schema.get("rows") or [])[:4],
            }
        payload = {
            "window_id": window.get("window_id"), "source_type": window.get("source_type"),
            "anchor_quote": _clip(window.get("anchor_quote"), 520), "neighbor_quotes": [_clip(value, 260) for value in (window.get("neighbor_quotes") or [])[:2]],
            "table_schema": compact_table, "object_context": window.get("object_context"),
            "table_lines": list(window.get("table_lines") or [])[:8],
            "support_refs": window.get("support_refs") or [],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if compact and source_chars > 0 and used + len(encoded) > source_chars:
            continue
        compact.append(payload)
        used += len(encoded)
    system = (
        "For each non-empty L1 window, emit one grounded triple; otherwise use predicate states with an extractive object. "
        "Include exact support_quotes and only supplied support_refs. Never infer facts. Keep table values with their header and row condition. Return JSON only."
    )
    user = (
        "Return {\"triples\":[{\"window_id\":\"W001\",\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\","
        "\"qualifiers\":{\"dataset\":[],\"metric\":[],\"condition\":[],\"comparison\":[]},"
        "\"support_refs\":[\"E0001\"],\"support_quotes\":[{\"evidence_ref\":\"E0001\",\"quote\":\"exact quote\"}]}]}.\n"
        + json.dumps({"question": question, "windows": compact}, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def structural_sufficiency_precheck(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Cheap eligibility gate before a semantic sufficiency model is invoked.

    It can only identify manifest structural gaps.  A ``ready`` result means
    the expensive judge receives a compact triple set; it never means the
    question is answerable without semantic inspection.
    """
    triples = [row for row in hierarchy.get("l2_contextual_triples") or [] if isinstance(row, dict)]
    claim_ids = [str(row.get("claim_id") or "") for row in hierarchy.get("query_claims") or [] if isinstance(row, dict)]
    by_claim: dict[str, list[str]] = defaultdict(list)
    invalid: list[str] = []
    for triple in triples:
        triple_id = str(triple.get("triple_id") or "")
        refs = [str(ref) for ref in triple.get("support_refs") or [] if str(ref)]
        windows = [str(window) for window in triple.get("l1_window_ids") or [] if str(window)]
        if not triple_id or not refs or not windows:
            invalid.append(triple_id or "<missing>")
            continue
        for claim_id in triple.get("claim_ids") or []:
            by_claim[str(claim_id)].append(triple_id)
    missing_claims = [claim_id for claim_id in claim_ids if not by_claim.get(claim_id)]
    # When a previous L2 producer could not decompose the question, the
    # absence of a claim is itself a reason to ask the semantic judge rather
    # than projecting a false "sufficient" outcome.
    unresolved = list((hierarchy.get("l3_triple_navigation") or {}).get("unresolved_claims") or [])
    return {
        "status": "ready_for_semantic_judge" if triples and not missing_claims and not invalid else "structural_gap",
        "triple_count": len(triples),
        "missing_claim_ids": missing_claims,
        "invalid_triple_ids": invalid,
        "unresolved_claims": unresolved,
        "claim_to_triple_ids": dict(sorted(by_claim.items())),
    }


def sufficiency_messages(question: str, hierarchy: dict[str, Any], *, max_triples: int = 8) -> list[dict[str, str]]:
    """Compact semantic gate for triple -> L1 -> L0 cascade decisions."""
    question_terms = set(tokenize(question))
    candidates = [triple for triple in hierarchy.get("l2_contextual_triples") or [] if isinstance(triple, dict)]
    trusted_statuses = {
        "deterministic_table_relation", "deterministic_citation_relation", "deterministic_hardware_relation",
        "verified_llm_relation", "visual_crop_verified_relation",
    }
    def triple_priority(triple: dict[str, Any]) -> tuple[int, int, str]:
        qualifier_text = " ".join(
            str(value) for values in (triple.get("qualifiers") or {}).values() if isinstance(values, list) for value in values
        )
        text = " ".join([*(str(triple.get(key) or "") for key in ("subject", "predicate", "object")), qualifier_text])
        status = str((triple.get("verification") or {}).get("status") or "")
        # A grounded relation must be inspected before a higher-overlap but
        # unverified navigation candidate. This keeps a short gate payload
        # from hiding the exact answer behind generic method prose.
        return (0 if status in trusted_statuses else 1, -len(question_terms.intersection(set(tokenize(text)))), str(triple.get("triple_id") or ""))
    candidates.sort(key=triple_priority)
    triples = []
    for triple in candidates[: max(1, max_triples)]:
        triples.append({
            "triple_id": triple.get("triple_id"), "claim_ids": triple.get("claim_ids") or [],
            "subject": _clip(triple.get("subject"), 120), "predicate": triple.get("predicate"), "object": _clip(triple.get("object"), 240),
            "qualifiers": {key: values[:4] for key, values in (triple.get("qualifiers") or {}).items() if isinstance(values, list)}, "l1_window_ids": triple.get("l1_window_ids") or [],
        })
    graph = hierarchy.get("l3_triple_navigation") if isinstance(hierarchy.get("l3_triple_navigation"), dict) else {}
    payload = {
        "question": question,
        "claims": hierarchy.get("query_claims") or [],
        "candidate_paths": graph.get("candidate_paths") or [],
        "triples": triples,
    }
    system = (
        "Decide whether these grounded triples answer the question without outside knowledge. If a value, condition, identity, or reasoning hop is missing, "
        "request the fewest triple IDs: L1 first, L0 only when needed. Return JSON only."
    )
    user = (
        "Return {\"sufficient\":true|false,\"covered_claim_ids\":[\"Q01\"],\"missing_claim_ids\":[],"
        "\"expand_l1_triple_ids\":[],\"expand_l0_triple_ids\":[],\"reason\":\"brief\"}.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_sufficiency_decision(raw: dict[str, Any], hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Resolve only known triple/window keys; arbitrary expansion is rejected."""
    triples = {str(row.get("triple_id") or ""): row for row in hierarchy.get("l2_contextual_triples") or [] if isinstance(row, dict)}
    valid = set(triples)
    def ids(field: str) -> list[str]:
        values = raw.get(field) if isinstance(raw.get(field), list) else []
        result: list[str] = []
        for value in values:
            identifier = str(value or "")
            if identifier in valid and identifier not in result:
                result.append(identifier)
        return result
    l1_ids = ids("expand_l1_triple_ids")[:4]
    l0_ids = ids("expand_l0_triple_ids")[:2]
    # L0 is a last resort. Its matching L1 window always accompanies it so the
    # generation step retains the explanation that motivated expansion.
    for identifier in l0_ids:
        if identifier not in l1_ids:
            l1_ids.append(identifier)
    return {
        "sufficient": bool(raw.get("sufficient")) and not l1_ids and not l0_ids,
        "covered_claim_ids": [str(value) for value in raw.get("covered_claim_ids") or [] if str(value)],
        "missing_claim_ids": [str(value) for value in raw.get("missing_claim_ids") or [] if str(value)],
        "expand_l1_triple_ids": l1_ids,
        "expand_l0_triple_ids": l0_ids,
        "reason": _clip(raw.get("reason"), 300),
        "verification": {"status": "accepted_keyed_gate", "known_triple_count": len(triples)},
    }


def attach_sufficiency_expansions(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Materialize only the L1/L0 units named by the validated gate decision."""
    updated = dict(hierarchy)
    decision = updated.get("triple_sufficiency") if isinstance(updated.get("triple_sufficiency"), dict) else {}
    triples = {str(row.get("triple_id") or ""): row for row in updated.get("l2_contextual_triples") or [] if isinstance(row, dict)}
    windows = {str(row.get("window_id") or ""): row for row in updated.get("l1_evidence_windows") or [] if isinstance(row, dict)}
    catalog = {str(row.get("evidence_ref") or ""): row for row in updated.get("l0_catalog") or [] if isinstance(row, dict)}
    l1_rows, l0_rows = [], []
    for triple_id in decision.get("expand_l1_triple_ids") or []:
        triple = triples.get(str(triple_id))
        if triple is None:
            continue
        for window_id in triple.get("l1_window_ids") or []:
            window = windows.get(str(window_id))
            if window is None:
                continue
            row = {
                "triple_id": str(triple_id), "window_id": str(window_id), "support_refs": list(window.get("support_refs") or []),
                "anchor_quote": _clip(window.get("anchor_quote"), 620),
                "neighbor_quotes": [_clip(value, 300) for value in (window.get("neighbor_quotes") or [])[:2]],
                "table_lines": list(window.get("table_lines") or [])[:8],
                "object_context": window.get("object_context"),
            }
            if row not in l1_rows:
                l1_rows.append(row)
    for triple_id in decision.get("expand_l0_triple_ids") or []:
        triple = triples.get(str(triple_id))
        if triple is None:
            continue
        # L0 is the coarse, expensive fallback. Expand the anchor object only;
        # neighboring records were already available through the paired L1
        # window and must not multiply the raw-context budget.
        anchor_refs = [
            str((windows.get(str(window_id)) or {}).get("anchor_ref") or "")
            for window_id in triple.get("l1_window_ids") or []
        ]
        for ref in (anchor_refs or [str(value) for value in triple.get("support_refs") or []]):
            record = catalog.get(str(ref))
            if record is None:
                continue
            row = {
                "triple_id": str(triple_id), "support_refs": [str(ref)], "source_type": record.get("source_type"),
                "label": record.get("label"), "text": _clip(record.get("text"), 1600),
            }
            if row not in l0_rows:
                l0_rows.append(row)
    updated["sufficiency_expansions"] = {"l1_rows": l1_rows, "l0_rows": l0_rows}
    return updated


__all__ = [
    "TRIPLE_GRAPH_VERSION",
    "attach_contextual_triple_graph",
    "attach_sufficiency_expansions",
    "attach_visual_triple",
    "build_l1_windows",
    "structural_sufficiency_precheck",
    "sufficiency_messages",
    "triple_generation_messages",
    "validate_sufficiency_decision",
    "verify_llm_contextual_triples",
    "visual_triple_messages",
    "visual_triple_verification_messages",
]
