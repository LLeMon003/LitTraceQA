"""Evidence-package construction and coverage-aware context selection."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

from .metadata_index import BM25Okapi, tokenize
from .object_references import object_reference_paragraphs, same_object_label
from .section_relevance import query_object_targets
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


@dataclass(frozen=True)
class EvidencePackageConfig:
    package_budget: int = 24
    min_package_budget: int = 4
    min_distinct_papers: int = 1
    adaptive_stop: bool = True
    modality_packages_per_paper: int = 2
    supporting_text_packages_per_paper: int = 0
    page_text_anchors_per_page: int = 0
    max_context_chars: int = 80000
    text_neighbors: int = 1
    rrf_k: int = 60
    candidate_pool_per_route: int = 0
    max_packages_per_page: int = 2


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or record.get("record_id") or record.get("id") or "")


def _order(record: dict[str, Any]) -> tuple[int, int, str]:
    try:
        page = int(record.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    try:
        reading = int(record.get("document_order") or record.get("reading_order") or 0)
    except (TypeError, ValueError):
        reading = 0
    return page, reading, record_id(record)


def _source_type(record: dict[str, Any]) -> str | None:
    return to_official_source_type(record.get("record_type"), record.get("source_type"))


def canonical_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate logical records while keeping the most informative projection."""
    chosen: dict[str, dict[str, Any]] = {}
    for raw in records:
        source_type = _source_type(raw)
        identifier = record_id(raw)
        if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES or not identifier:
            continue
        candidate = dict(raw)
        candidate["source_type"] = source_type
        current = chosen.get(identifier)
        if current is None or len(str(candidate.get("text") or "")) > len(str(current.get("text") or "")):
            chosen[identifier] = candidate
    return sorted(chosen.values(), key=lambda record: (str(record.get("paper_id") or ""), *_order(record)))


def _anchor_score_tracks(trace: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)

    def update(identifier: str, name: str, value: Any) -> None:
        if isinstance(value, (int, float)):
            scores[identifier][name] = max(scores[identifier].get(name, float("-inf")), float(value))

    for unit in trace.get("ranked_units") or []:
        contract = unit.get("score_contract") if isinstance(unit.get("score_contract"), dict) else {}
        identifiers = unit.get("anchor_record_ids") or unit.get("record_ids") or []
        for identifier in identifiers:
            key = str(identifier or "")
            if key:
                update(key, "qwen", contract.get("local_relevance"))
    for section in trace.get("sections") or []:
        for unit in section.get("chunks") or []:
            details = unit.get("llmrerank_call") if isinstance(unit.get("llmrerank_call"), dict) else {}
            identifiers = unit.get("anchor_record_ids") or unit.get("record_ids") or []
            for modality_score in details.get("modality_scores") or []:
                if not isinstance(modality_score, dict):
                    continue
                track = "qwen_visual" if modality_score.get("modality") == "image" else "qwen_text"
                for identifier in identifiers:
                    if str(identifier or ""):
                        update(str(identifier), track, modality_score.get("score"))
    return scores


def _citation_number(record: dict[str, Any]) -> str:
    text = " ".join(str(record.get(key) or "") for key in ("label", "text"))
    match = re.search(r"(?:reference|ref\.)\s*\[?(\d{1,3})\]?|\[(\d{1,3})\]", text, re.IGNORECASE)
    return next((item for item in match.groups() if item), "") if match else ""


def _package_records(anchor: dict[str, Any], paper_records: list[dict[str, Any]], text_neighbors: int) -> list[dict[str, Any]]:
    anchor_id = record_id(anchor)
    source_type = str(anchor["source_type"])
    same_section = [record for record in paper_records if record.get("section_id") == anchor.get("section_id")]
    ordered = sorted(same_section or paper_records, key=_order)
    index = next((index for index, record in enumerate(ordered) if record_id(record) == anchor_id), 0)
    if source_type == "text_span":
        return ordered[max(0, index - text_neighbors) : index + text_neighbors + 1]
    nearby = [record for record in ordered[max(0, index - text_neighbors) : index + text_neighbors + 1] if record["source_type"] == "text_span"]
    # ``text_neighbors`` historically meant one prose record on each side of
    # an object. Preserve that bounded size while replacing layout adjacency
    # with explicit object-reference paragraphs whenever they exist.
    references = object_reference_paragraphs(anchor, paper_records, max(1, text_neighbors * 2))
    narration = references or nearby
    companions = [
        record
        for record in paper_records
        if record_id(record) != anchor_id
        and record["source_type"] == source_type
        and same_object_label(record, anchor)
        and str(record.get("record_type") or "") in {"table_caption", "figure_caption", "table", "figure", "equation", "algorithm"}
    ]
    package = [*narration, anchor, *companions]
    if source_type == "citation_context":
        number = _citation_number(anchor)
        if number:
            bibliography = next(
                (
                    record for record in paper_records
                    if record_id(record) != anchor_id
                    and record["source_type"] == "citation_context"
                    and _citation_number(record) == number
                ),
                None,
            )
            if bibliography is not None:
                package.append(bibliography)
    return package


def build_packages(
    records: list[dict[str, Any]],
    trace: dict[str, Any],
    config: EvidencePackageConfig,
    *,
    context_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build scored anchor packages with an optional broader context catalog.

    ``records`` defines the selectable, scored anchors. ``context_records`` is
    evidence-only and may include unscored prose needed to narrate a selected
    object; it never creates an additional package or ranking candidate.
    """
    canonical = canonical_records(records)
    context_canonical = canonical_records(context_records) if context_records is not None else canonical
    score_by_record = _anchor_score_tracks(trace)
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in context_canonical:
        by_paper[str(record.get("paper_id") or "")].append(record)
    packages: list[dict[str, Any]] = []
    for anchor in canonical:
        anchor_id = record_id(anchor)
        contained = _package_records(anchor, by_paper[str(anchor.get("paper_id") or "")], config.text_neighbors)
        unique: dict[str, dict[str, Any]] = {record_id(record): record for record in contained if record_id(record)}
        packages.append(
            {
                "package_id": f"pkg::{anchor_id}",
                "anchor_record_id": anchor_id,
                "paper_id": anchor.get("paper_id"),
                "page": anchor.get("page"),
                "section_id": anchor.get("section_id"),
                "section_title": anchor.get("section_title"),
                "section_type": anchor.get("section_type"),
                "section_path": anchor.get("section_path"),
                "source_type": anchor["source_type"],
                "label": anchor.get("label"),
                "object_id": anchor.get("label") if anchor["source_type"] != "text_span" else None,
                **score_by_record.get(anchor_id, {}),
                "records": sorted(unique.values(), key=_order),
            }
        )
    return packages


def _package_text(package: dict[str, Any]) -> str:
    parts = [str(package.get("source_type") or ""), str(package.get("label") or ""), str(package.get("section_title") or "")]
    parts.extend(str(record.get("text") or "") for record in package.get("records") or [])
    return " ".join(parts)


def _matches_target(package: dict[str, Any], targets: dict[str, Any]) -> bool:
    source_type = str(package.get("source_type") or "")
    target = targets.get(source_type)
    if target is None:
        return False
    label = str(package.get("label") or "")
    return bool(re.search(rf"\b{re.escape(str(target))}\b", label))


def _requested_modalities(query: str, primary_evidence_type: str | None) -> set[str]:
    text = str(query or "").lower()
    modalities: set[str] = set()
    patterns = {
        "table": r"\btable\b|\btab\.",
        "figure": r"\bfigure\b|\bfig\.",
        "equation_algorithm": r"\bequation\b|\beq\.\b|\balgorithm\b|\bobjective\b",
        "citation_context": r"\bcitation\b|\bcited\b|\breference\b|\bref\.\b|\bauthor\b",
    }
    for source_type, pattern in patterns.items():
        if re.search(pattern, text):
            modalities.add(source_type)
    primary = to_official_source_type(source_type=primary_evidence_type)
    if primary:
        modalities.add(primary)
    return modalities or {"text_span"}


def _layout_section_types(query: str) -> set[str]:
    """Return explicit structural intents without turning them into score fusion."""
    text = str(query or "").lower()
    requested: set[str] = set()
    if re.search(r"\b(?:method|methodology|approach|framework|architecture|model)\b", text):
        requested.add("method")
    if re.search(r"\b(?:experiment|result|evaluation|ablation|benchmark)\b", text):
        requested.add("results")
    if re.search(r"\b(?:introduction|background|related work|literature review)\b", text):
        requested.add("introduction")
    return requested


def _section_matches_layout(package: dict[str, Any], layout_types: set[str]) -> bool:
    if not layout_types:
        return False
    section_type = str(package.get("section_type") or "").lower()
    title = str(package.get("section_title") or "").lower()
    if "method" in layout_types and (section_type in {"method", "approach"} or re.search(r"\b(?:method|approach|framework|architecture|model)\b", title)):
        return True
    if "results" in layout_types and (section_type in {"results", "experiments", "evaluation", "analysis"} or re.search(r"\b(?:experiment|result|evaluation|ablation|analysis)\b", title)):
        return True
    if "introduction" in layout_types and (section_type in {"introduction", "background", "related_work"} or re.search(r"\b(?:introduction|background|related work|literature review)\b", title)):
        return True
    return False


def _layout_score(package: dict[str, Any], modalities: set[str], targets: dict[str, Any]) -> float:
    source_type = str(package.get("source_type") or "")
    score = 1.0 if source_type in modalities else 0.0
    if _matches_target(package, targets):
        score += 3.0
    if source_type != "text_span" and package.get("label"):
        score += 0.1
    return score


def _score(package: dict[str, Any], modality: str | None = None) -> float:
    """Use only Qwen's comparable total relevance as the primary rank signal.

    Per-modality scores remain independent RRF routes below.  They must not be
    substituted for total relevance, because an image score and a text score do
    not share a calibrated numerical scale.
    """
    if isinstance(package.get("qwen"), (int, float)):
        return float(package["qwen"])
    return -1.0


def _rrf_pool(rankings: list[list[int]], config: EvidencePackageConfig) -> set[int]:
    selected: set[int] = set()
    for ranking in rankings:
        limit = config.candidate_pool_per_route
        selected.update(ranking if limit <= 0 else ranking[:limit])
    return selected


def select_packages(
    *,
    query: str,
    packages: list[dict[str, Any]],
    primary_evidence_type: str | None,
    is_multi_paper_task: bool,
    route_queries: list[str],
    paper_route_queries: list[tuple[str, str]] | None = None,
    paper_local_route_queries: list[tuple[str, str]] | None = None,
    config: EvidencePackageConfig,
) -> dict[str, Any]:
    if not packages:
        return {"packages": [], "records": [], "trace": {"candidate_package_count": 0}}
    texts = [_package_text(package) for package in packages]
    bm25 = BM25Okapi([tokenize(text) for text in texts])
    rankings: list[list[int]] = []
    claim_rankings: list[list[int]] = []
    paper_claim_rankings: list[list[int]] = []
    paper_local_rankings: list[list[int]] = []
    routing_queries = list(dict.fromkeys([query, *route_queries]))
    for route_index, route_query in enumerate(routing_queries):
        scores = bm25.get_scores(tokenize(route_query))
        ranking = sorted(range(len(packages)), key=lambda index: (-scores[index], index))
        rankings.append(ranking)
        if route_index:
            claim_rankings.append(ranking)
    for paper_id, route_query in paper_route_queries or []:
        scores = bm25.get_scores(tokenize(route_query))
        ranking = [
            index
            for index in sorted(range(len(packages)), key=lambda index: (-scores[index], index))
            if str(packages[index].get("paper_id") or "") == str(paper_id)
        ]
        if ranking:
            rankings.append(ranking)
            paper_claim_rankings.append(ranking)
    for paper_id, route_query in paper_local_route_queries or []:
        scores = bm25.get_scores(tokenize(route_query))
        ranking = [
            index
            for index in sorted(range(len(packages)), key=lambda index: (-scores[index], index))
            if str(packages[index].get("paper_id") or "") == str(paper_id)
        ]
        if ranking:
            rankings.append(ranking)
            paper_local_rankings.append(ranking)
    source_route_count = 0
    for source_type in sorted(OFFICIAL_EVIDENCE_SOURCE_TYPES):
        scores = bm25.get_scores(tokenize(f"{source_type} {query}"))
        ranking = [
            index
            for index in sorted(range(len(packages)), key=lambda index: (-scores[index], index))
            if packages[index]["source_type"] == source_type
        ]
        if ranking:
            rankings.append(ranking)
            source_route_count += 1
    qwen_ranking = sorted(range(len(packages)), key=lambda index: (-_score(packages[index]), index))
    rankings.append(qwen_ranking)
    targets = query_object_targets(query)
    modalities = _requested_modalities(query, primary_evidence_type)
    layout_types = _layout_section_types(query)
    for modality in sorted(modalities):
        matching = [index for index, package in enumerate(packages) if package["source_type"] == modality]
        rankings.append(sorted(matching, key=lambda index: (-_score(packages[index], modality), index)))
    visual_ranking = sorted(
        (index for index, package in enumerate(packages) if isinstance(package.get("qwen_visual"), (int, float))),
        key=lambda index: (-float(packages[index]["qwen_visual"]), index),
    )
    if visual_ranking:
        rankings.append(visual_ranking)
    for package in packages:
        package["layout_score"] = _layout_score(package, modalities, targets)
    rankings.append(sorted(range(len(packages)), key=lambda index: (-float(packages[index]["layout_score"]), index)))
    targeted = [index for index, package in enumerate(packages) if _matches_target(package, targets)]
    if targeted:
        rankings.append(targeted)
    candidate_indexes = _rrf_pool(rankings, config)
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            if index in candidate_indexes:
                rrf_scores[index] += 1.0 / (config.rrf_k + rank)
    ordered = sorted(
        candidate_indexes,
        key=lambda index: (
            -_score(packages[index]),
            -rrf_scores[index],
            index,
        ),
    )
    budget = max(1, config.package_budget)
    min_budget = min(budget, max(1, config.min_package_budget))
    paper_target = min(budget, max(1, config.min_distinct_papers if is_multi_paper_task else 1))
    selected_indexes: list[int] = []
    page_counts: dict[tuple[str, Any], int] = defaultdict(int)

    def add(index: int, *, force: bool = False) -> bool:
        if index in selected_indexes or len(selected_indexes) >= budget:
            return False
        package = packages[index]
        page_key = (str(package.get("paper_id") or ""), package.get("page"))
        if not force and page_counts[page_key] >= max(1, config.max_packages_per_page):
            return False
        selected_indexes.append(index)
        page_counts[page_key] += 1
        return True

    for index in sorted(targeted, key=lambda item: (-_score(packages[item], packages[item]["source_type"]), item)):
        add(index, force=True)
    for modality in sorted(modalities):
        candidates = [index for index in ordered if packages[index]["source_type"] == modality]
        if candidates:
            add(candidates[0])
    for ranking in claim_rankings:
        if ranking:
            add(ranking[0])
    for ranking in paper_claim_rankings:
        if ranking:
            add(ranking[0])
    for ranking in paper_local_rankings:
        if ranking:
            add(ranking[0])
    if is_multi_paper_task:
        best_by_paper: dict[str, int] = {}
        for index in ordered:
            best_by_paper.setdefault(str(packages[index].get("paper_id") or ""), index)
        for index in list(best_by_paper.values())[:paper_target]:
            add(index)
        covered_papers = {str(packages[index].get("paper_id") or "") for index in selected_indexes}
        for paper_id in sorted(covered_papers):
            for modality in sorted(modalities):
                modality_candidates = sorted(
                    (
                        index
                        for index in ordered
                        if str(packages[index].get("paper_id") or "") == paper_id
                        and packages[index]["source_type"] == modality
                    ),
                    key=lambda index: (-_score(packages[index], modality), -rrf_scores[index], index),
                )
                # This is an independent structure route. It is intentionally not
                # added to Qwen's numeric score: explicit "method/framework"
                # queries should retain one matching object per covered paper.
                structural_candidates = [
                    index for index in modality_candidates
                    if _section_matches_layout(packages[index], layout_types)
                ]
                if structural_candidates:
                    add(structural_candidates[0])
                for index in modality_candidates[: max(1, config.modality_packages_per_paper)]:
                    add(index)
            # A visual/object answer frequently depends on a separate sentence
            # that defines a condition, reports a value, or interprets the
            # object. Keep this an explicit anchor route rather than treating
            # nearby package text as equivalent strict evidence.
            text_candidates = [index for index in ordered if str(packages[index].get("paper_id") or "") == paper_id and packages[index]["source_type"] == "text_span"]
            for index in text_candidates[: max(0, config.supporting_text_packages_per_paper)]:
                add(index)
    # This is deliberately an opt-in audit fallback.  It preserves a small,
    # locator-bearing text sample on every page when semantic ranking cannot
    # distinguish malformed or near-empty extracted records.
    if config.page_text_anchors_per_page > 0:
        by_page: dict[tuple[str, Any], list[int]] = defaultdict(list)
        for index in ordered:
            if packages[index]["source_type"] == "text_span":
                by_page[(str(packages[index].get("paper_id") or ""), packages[index].get("page"))].append(index)
        for page_key in sorted(by_page):
            for index in by_page[page_key][:config.page_text_anchors_per_page]:
                add(index, force=True)
    selected_modalities = {str(packages[index]["source_type"]) for index in selected_indexes}
    selected_papers = {str(packages[index].get("paper_id") or "") for index in selected_indexes}
    stop_reason = "max_budget"
    while len(selected_indexes) < budget:
        choices = [
            index
            for index in ordered
            if index not in selected_indexes
            and page_counts[(str(packages[index].get("paper_id") or ""), packages[index].get("page"))] < max(1, config.max_packages_per_page)
        ]
        if not choices:
            stop_reason = "no_page_diverse_candidate"
            break

        def marginal_key(index: int) -> tuple[int, float, float, int]:
            package = packages[index]
            source_type = str(package["source_type"])
            paper_id = str(package.get("paper_id") or "")
            gain = int(source_type in modalities and source_type not in selected_modalities)
            gain += int(is_multi_paper_task and len(selected_papers) < paper_target and paper_id not in selected_papers)
            return gain, _score(package, source_type), rrf_scores[index], -index

        chosen = max(choices, key=marginal_key)
        gain = marginal_key(chosen)[0]
        coverage_complete = modalities <= selected_modalities and (not is_multi_paper_task or len(selected_papers) >= paper_target)
        if config.adaptive_stop and len(selected_indexes) >= min_budget and coverage_complete and gain == 0:
            stop_reason = "coverage_complete_no_marginal_gain"
            break
        if not add(chosen):
            stop_reason = "candidate_rejected"
            break
        selected_modalities.add(str(packages[chosen]["source_type"]))
        selected_papers.add(str(packages[chosen].get("paper_id") or ""))
    selected_packages = [
        dict(packages[index], rank=rank, rrf_score=rrf_scores[index])
        for rank, index in enumerate(selected_indexes, start=1)
    ]
    records: list[dict[str, Any]] = []
    emitted: set[str] = set()
    chars = 0
    context_packages: list[dict[str, Any]] = []
    for package in selected_packages:
        additions = []
        for record in package["records"]:
            identifier = record_id(record)
            if not identifier or identifier in emitted:
                continue
            additions.append(record)
        addition_chars = sum(len(str(record.get("text") or "")) for record in additions)
        # Keep every chosen package semantically whole. A single oversized first
        # package is retained rather than truncating its table/figure/algorithm.
        if config.max_context_chars > 0 and records and chars + addition_chars > config.max_context_chars:
            continue
        context_packages.append(package)
        for record in additions:
            identifier = record_id(record)
            emitted.add(identifier)
            chars += len(str(record.get("text") or ""))
            records.append(record)
    return {
        "packages": context_packages,
        "records": records,
        "trace": {
            "candidate_package_count": len(candidate_indexes),
            "logical_package_count": len(packages),
            "selected_package_count": len(context_packages),
            "selected_record_count": len(records),
            "selected_char_count": chars,
            "package_budget": budget,
            "min_package_budget": min_budget,
            "min_distinct_papers": paper_target,
            "modality_packages_per_paper": config.modality_packages_per_paper if is_multi_paper_task else 0,
            "supporting_text_packages_per_paper": config.supporting_text_packages_per_paper if is_multi_paper_task else 0,
            "page_text_anchors_per_page": config.page_text_anchors_per_page,
            "adaptive_stop": config.adaptive_stop,
            "adaptive_stop_reason": stop_reason,
            "max_context_chars": config.max_context_chars,
            "route_count": len(rankings),
            "source_route_count": source_route_count,
            "hyde_claim_route_count": len(claim_rankings),
            "paper_conditioned_claim_route_count": len(paper_claim_rankings),
            "paper_local_bm25_route_count": len(paper_local_rankings),
            "targeted_package_count": len(targeted),
            "requested_modalities": sorted(modalities),
            "requested_layout_section_types": sorted(layout_types),
            "selected_modalities": sorted({str(package["source_type"]) for package in context_packages}),
            "score_tracks": {
                "qwen_text": sum(isinstance(package.get("qwen_text"), (int, float)) for package in packages),
                "qwen_visual": sum(isinstance(package.get("qwen_visual"), (int, float)) for package in packages),
                "layout": len(packages),
            },
            "candidate_pool_per_route": config.candidate_pool_per_route,
            "max_packages_per_page": config.max_packages_per_page,
        },
    }


__all__ = ["EvidencePackageConfig", "build_packages", "canonical_records", "select_packages"]
