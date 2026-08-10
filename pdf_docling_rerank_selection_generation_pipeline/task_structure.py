"""Query-visible task structure for the 2026-08-05 official input schema.

The official challenge inputs no longer ship ``task_family`` /
``primary_evidence_type`` (those are gold-only fields in
``validation.jsonl``).  This module derives the two control signals the
pipeline needs from the frozen evidence plan and query-visible structure only:

* ``is_multi_paper`` -- routing decision (local selector vs coverage selector,
  budgets, diversification).
* ``preferred_source_types`` -- a query-conditioned modality signal, never a
  fact, never a substitute for evidence.

Division of labour follows the project direction: the LLM performs semantic
parsing (entities, comparisons, evidence slots, inferred paper count), while
deterministic rules below make the routing decision so an unstable LLM label
can never silently change execution mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


_MULTI_PAPER_QUESTION_PATTERNS = (
    r"\bacross (?:all|the) papers\b",
    r"\bwhich .* papers\b",
    r"\bwhat .* each\b",
    r"\blist\b.*\bpapers?\b",
    r"\bcompare[sd]?\b",
    r"\bamong\b.*\bpapers?\b",
    r"\brespectively\b",
)

_EXPLICIT_OBJECT_ID_RE = re.compile(
    r"\b(table|figure|fig\.?|equation|eq\.?|algorithm)\s*\(?\s*(\d{1,4})\s*\)?",
    re.IGNORECASE,
)
_EXPLICIT_REFERENCE_ID_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\s+(?:reference|citation)\b", re.IGNORECASE)

_SOURCE_TYPE_ALIASES = {
    "text": "text_span",
    "paragraph": "text_span",
    "citation": "citation_context",
    "cited": "citation_context",
    "reference": "citation_context",
    "equation": "equation_algorithm",
    "eq": "equation_algorithm",
    "algorithm": "equation_algorithm",
    "fig": "figure",
    "figs": "figure",
    "tab": "table",
    "tables": "table",
    "figures": "figure",
    "equations": "equation_algorithm",
}


@dataclass(frozen=True)
class TaskStructure:
    """Deterministic routing signals derived from query-visible inputs."""

    is_multi_paper: bool
    preferred_source_types: tuple[str, ...]
    inferred_paper_count: int | None
    comparison_targets: tuple[str, ...]
    entities: tuple[str, ...]

    @property
    def primary_source_type(self) -> str:
        """First preferred source type, or empty when the query is neutral."""
        return self.preferred_source_types[0] if self.preferred_source_types else ""

    @property
    def task_family(self) -> str:
        """Runtime family derived from the frozen slot plan, never gold input."""
        return "multi_paper" if self.is_multi_paper else "single_paper"


def _as_string_list(value: Any, limit: int = 12) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _int_in_range(value: Any, low: int = 1, high: int = 50) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def explicit_source_type_mentions(question: Any) -> tuple[str, ...]:
    """Source types the query explicitly locates (``Figure 3``, ...).

    Only query-visible mentions count.  A mention is a retrieval/modality hint,
    never an asserted fact about where the evidence lives.
    """
    text = str(question or "")
    mentioned: list[str] = []
    object_types = {
        "table": "table", "figure": "figure", "fig": "figure",
        "equation": "equation_algorithm", "eq": "equation_algorithm", "algorithm": "equation_algorithm",
    }
    for match in _EXPLICIT_OBJECT_ID_RE.finditer(text):
        source_type = object_types[match.group(1).lower().rstrip(".")]
        if source_type not in mentioned:
            mentioned.append(source_type)
    if _EXPLICIT_REFERENCE_ID_RE.search(text):
        mentioned.append("citation_context")
    return tuple(mentioned)


def fallback_is_multi_paper(sample: dict[str, Any]) -> bool:
    """Deterministic single/multi fallback when no structured plan exists.

    Mirrors the pre-schema routing heuristics so behaviour is continuous while
    the plan is unavailable.  ``task_family`` is accepted only as a
    backward-compatibility fallback for old-format diagnostic data; the new
    official inputs never contain it.
    """
    task_family = str(sample.get("task_family") or "").lower()
    if "multi" in task_family:
        return True
    question = str(sample.get("question") or "").lower()
    return any(re.search(pattern, question) for pattern in _MULTI_PAPER_QUESTION_PATTERNS)


def _preferred_from_plan(plan: dict[str, Any] | None) -> tuple[str, ...]:
    if not plan:
        return ()
    seen: list[str] = []
    for slot in plan.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        for raw_source_type in slot.get("required_source_types") or []:
            source_type = to_official_source_type(source_type=raw_source_type)
            if source_type in OFFICIAL_EVIDENCE_SOURCE_TYPES and source_type not in seen:
                seen.append(source_type)
    return tuple(seen)


def _preferred_from_query(sample: dict[str, Any]) -> tuple[str, ...]:
    seen: list[str] = []
    for source_type in explicit_source_type_mentions(sample.get("question")):
        if source_type not in seen:
            seen.append(source_type)
    return tuple(seen)


def derive_task_structure(
    sample: dict[str, Any],
    plan: dict[str, Any] | None = None,
) -> TaskStructure:
    """Derive routing signals from the frozen plan and query-visible inputs.

    Order of authority for ``preferred_source_types``: plan slots -> answer
    explicit numbered object mention. ``is_multi_paper`` comes from the plan's
    structured query analysis when available, then falls back to question
    heuristics.
    """
    preferred = _preferred_from_plan(plan)
    if not preferred:
        preferred = _preferred_from_query(sample)

    inferred_paper_count: int | None = None
    comparison_targets: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    cross_paper_required: bool | None = None
    if isinstance(plan, dict):
        query_analysis = plan.get("query_analysis")
        if isinstance(query_analysis, dict):
            inferred_paper_count = _int_in_range(query_analysis.get("inferred_paper_count"))
            comparison_targets = _as_string_list(query_analysis.get("comparison_targets"), 8)
            entities = _as_string_list(query_analysis.get("entities"), 16)
            raw_cross_paper = query_analysis.get("cross_paper_synthesis_required")
            if isinstance(raw_cross_paper, bool):
                cross_paper_required = raw_cross_paper
        plan_cross_paper = plan.get("requires_cross_paper_synthesis")
        if plan_cross_paper is True:
            # This is the validated plan-level decision.  A raw LLM analysis
            # may say false even after its multi-entity list was split.
            cross_paper_required = True
        elif cross_paper_required is None and isinstance(plan_cross_paper, bool):
            cross_paper_required = plan_cross_paper

    structured_multi = bool(
        (inferred_paper_count is not None and inferred_paper_count >= 2)
        or len(comparison_targets) >= 2
        or (cross_paper_required and bool(entities))
    )
    is_multi_paper = structured_multi or fallback_is_multi_paper(sample)
    return TaskStructure(
        is_multi_paper=is_multi_paper,
        preferred_source_types=preferred,
        inferred_paper_count=inferred_paper_count,
        comparison_targets=comparison_targets,
        entities=entities,
    )


def as_source_types(value: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalise a single type or an iterable into a deduplicated tuple."""
    if isinstance(value, str):
        values = (value,) if value else ()
    else:
        values = tuple(value or ())
    seen: list[str] = []
    for raw in values:
        normalized = _SOURCE_TYPE_ALIASES.get(str(raw).strip().lower(), str(raw))
        source_type = to_official_source_type(source_type=normalized)
        if source_type in OFFICIAL_EVIDENCE_SOURCE_TYPES and source_type not in seen:
            seen.append(source_type)
    return tuple(seen)
