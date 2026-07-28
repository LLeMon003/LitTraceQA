"""Auditable section and record-aware unit relevance retrieval."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llmrerank_client import LLMRerankClient, LLMRerankError
from .metadata_index import BM25Okapi, tokenize
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, grounding_label_from_record, to_official_source_type


REPRESENTATION_TEMPLATE_VERSION = "v2_record_aware_units"
EMBEDDING_CACHE_VERSION = "v1"
CANONICAL_RERANK_PROJECTION_VERSION = "v1_exact_text_image_projection"
LLMRERANK_CACHE_VERSION = "v4_canonical_projections"
LLMRERANK_INSTRUCTIONS = {
    "v1": (
        "Given a research question and a candidate section from a scientific paper, estimate whether the "
        "section contains information relevant to answering the question. Judge relevance only. Do not judge "
        "whether the section alone is sufficient, and do not use external knowledge. Consider textual content, "
        "tables, figures, equations, algorithms, captions, and nearby explanations when provided."
    ),
    "v2": (
        "Given a research question and a candidate retrieval unit from a scientific-paper section, estimate "
        "whether the unit contains information relevant to answering the question. Judge relevance only. "
        "Use the section metadata as location context, not as proof of relevance. Consider the unit text, "
        "object labels, tables, figures, equations, citations, images, and nearby records when provided."
    ),
}


@dataclass(frozen=True)
class SectionRelevanceConfig:
    backend: str = "bm25"
    chunk_max_tokens: int = 448
    chunk_overlap_tokens: int = 0
    text_max_chars: int = 6000
    record_top_k: int = 0
    unit_mode: str = "token_chunks"
    unit_target_tokens: int = 1280
    unit_max_tokens: int = 1536
    unit_overlap_records: int = 1
    object_units_enabled: bool = True
    object_neighbor_records: int = 1
    aggregation_top_k: int = 3
    section_bonus_weight: float = 0.10
    object_section_bonus_weight: float = 0.15
    section_bonus_max: float = 0.10
    retrieval_unit_top_k: int = 0
    hybrid_bm25_weight: float = 0.4
    hybrid_e5_weight: float = 0.6
    e5_model: str = "intfloat/e5-base-v2"
    pooling: str = "log_mean_exp"
    log_mean_exp_lambda: float = 3.0
    query_section_bonus_enabled: bool = True
    query_section_bonus: float = 0.2
    primary_type_prior_enabled: bool = True
    primary_type_prior_max_bonus: float = 0.05
    llmrerank_api_key: str | None = None
    llmrerank_base_url: str = "https://api.siliconflow.cn/v1"
    llmrerank_model: str = "Qwen/Qwen3-VL-Reranker-8B"
    llmrerank_input_mode: str = "text_with_object_images"
    llmrerank_batch_size: int = 8
    llmrerank_request_concurrency: int = 1
    llmrerank_request_timeout_seconds: float = 120.0
    llmrerank_max_retries: int = 3
    llmrerank_failure_fallback: str = "none"
    llmrerank_section_chunk_max_tokens: int = 6144
    llmrerank_section_chunk_overlap_tokens: int = 128
    llmrerank_section_pooling: str = "log_mean_exp"
    llmrerank_log_mean_exp_lambda: float = 5.0
    llmrerank_top_k_mean_chunks: int = 3
    llmrerank_context_top_k_chunks: int = 0
    llmrerank_max_images_per_section: int = 4
    llmrerank_instruction_version: str = "v1"
    llmrerank_unit_prefilter_enabled: bool = True
    llmrerank_unit_prefilter_top_k: int = 64
    llmrerank_unit_prefilter_per_section: int = 3
    llmrerank_unit_prefilter_primary_top_k: int = 32
    llmrerank_prefilter_fallback_weight: float = 0.25
    llmrerank_deterministic_locator_only_enabled: bool = False
    apply_section_type_bonus: bool = False


SECTION_QUERY_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "experimental setup": "experiments",
    "experiments": "experiments",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "limitations": "limitations",
    "references": "references",
    "appendix": "appendix",
}

PRIMARY_TYPE_SECTION_PRIOR = {
    "citation_context": {
        "introduction": 1.0, "related_work": 1.0, "background": 0.8, "literature_review": 1.0,
        "discussion": 0.5, "experiments": 0.3, "method": 0.2, "references": 0.6, "appendix": 0.2,
    },
    "table": {
        "experiments": 1.0, "results": 1.0, "evaluation": 1.0, "ablation": 1.0, "analysis": 0.8,
        "dataset": 0.6, "implementation_details": 0.6, "method": 0.4, "appendix": 0.6,
    },
    "figure": {
        "method": 1.0, "architecture": 1.0, "framework": 1.0, "overview": 0.8,
        "experiments": 0.7, "results": 0.7, "analysis": 0.7, "appendix": 0.5,
    },
    "equation_algorithm": {
        "method": 1.0, "approach": 1.0, "model": 1.0, "objective": 1.0, "optimization": 1.0,
        "preliminaries": 0.7, "background": 0.5, "implementation_details": 0.5, "appendix": 0.7,
    },
    "text_span": {"*": 0.0},
}


def query_section_targets(query: str) -> list[dict[str, str]]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(query or "").lower()).strip()
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for alias, target in SECTION_QUERY_ALIASES.items():
        if re.search(rf"\b(?:{re.escape(alias)}\s+section|section\s+(?:on\s+)?{re.escape(alias)})\b", normalized):
            if target not in seen:
                seen.add(target)
                targets.append({"target": target, "reason": f"explicit_section:{alias}"})
    reference_intent = re.search(
        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|\d+(?:st|nd|rd|th)|how many)\s+(?:\w+\s+){0,3}references?\b",
        normalized,
    )
    if reference_intent and "references" not in seen:
        targets.append({"target": "references", "reason": "reference_index_or_count"})
    return targets


def query_object_targets(query: str) -> dict[str, Any]:
    text = str(query or "")
    lowered = text.lower()

    def object_number(pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    reference_id = None
    for pattern in (r"\b(?:reference|ref\.)\s*\[?(\d{1,3})\]?", r"\b(\d{1,3})(?:st|nd|rd|th)\s+reference\b"):
        if match := re.search(pattern, lowered):
            reference_id = int(match.group(1))
            break
    return {
        "table": object_number(r"\btable\s*(\d+)"),
        "figure": object_number(r"\b(?:figure|fig\.)\s*(\d+)"),
        "equation_algorithm": object_number(r"\b(?:equation|eq\.|algorithm)\s*\(?(\d+)\)?"),
        "citation_context": reference_id,
        "last_reference": bool(re.search(r"\blast\s+reference\b|\bindex\s+of\s+the\s+last\s+reference\b", lowered)),
    }


def explicit_target_unit_indexes(query: str, chunks: list[dict[str, Any]]) -> set[int]:
    targets = query_object_targets(query)
    matched: set[int] = set()
    citations_by_paper: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        unit_type = str(chunk.get("unit_type") or "")
        text = str(chunk.get("text") or "")
        paper_id = str(chunk.get("paper_id") or "")
        for source_type in ("table", "figure", "equation_algorithm", "citation_context"):
            target = targets.get(source_type)
            if target is not None and unit_type == f"object_{source_type}":
                label = "reference" if source_type == "citation_context" else "(?:equation|algorithm)" if source_type == "equation_algorithm" else source_type
                if re.search(rf"\blabel={label}\s*{target}\b", text, re.IGNORECASE):
                    matched.add(index)
        if unit_type == "object_citation_context":
            label_match = re.search(r"\blabel=reference\s*(\d+)\b", text, re.IGNORECASE)
            if label_match:
                citations_by_paper[paper_id].append((int(label_match.group(1)), index))
    if targets["last_reference"]:
        for values in citations_by_paper.values():
            if values:
                matched.add(max(values)[1])
    return matched


def _normalize_identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def deterministic_locator_only_indexes(
    query: str,
    chunks: list[dict[str, Any]],
    candidate_paper_metadata: list[dict[str, Any]] | None,
) -> tuple[set[int] | None, dict[str, Any]]:
    """Return a safe object-only scope only for an explicitly named, unique paper."""
    explicit_indexes = explicit_target_unit_indexes(query, chunks)
    query_identity = _normalize_identity(query)
    named_papers: set[str] = set()
    for candidate in candidate_paper_metadata or []:
        paper_id = str(candidate.get("paper_id") or "")
        title = _normalize_identity(candidate.get("title"))
        paper_identity = _normalize_identity(paper_id)
        title_match = len(title) >= 16 and title in query_identity
        paper_id_match = len(paper_identity) >= 8 and paper_identity in query_identity
        if paper_id and (title_match or paper_id_match):
            named_papers.add(paper_id)
    if not explicit_indexes:
        return None, {"status": "not_applicable", "reason": "no_explicit_object_target"}
    if len(named_papers) != 1:
        return None, {
            "status": "not_applicable",
            "reason": "paper_identity_not_unique",
            "named_paper_ids": sorted(named_papers),
        }
    paper_id = next(iter(named_papers))
    scoped = {index for index in explicit_indexes if str(chunks[index].get("paper_id") or "") == paper_id}
    if not scoped:
        return None, {"status": "not_applicable", "reason": "explicit_object_not_found_in_named_paper", "paper_id": paper_id}
    return scoped, {
        "status": "active",
        "reason": "unique_named_paper_and_explicit_object",
        "paper_id": paper_id,
        "unit_count": len(scoped),
    }


def _section_target_key(section: dict[str, Any]) -> str:
    section_type = re.sub(r"[^a-z0-9]+", "_", str(section.get("section_type") or "").lower()).strip("_")
    title = re.sub(r"[^a-z0-9]+", " ", str(section.get("section_title") or "").lower()).strip()
    if section_type and section_type != "unknown":
        return section_type
    return SECTION_QUERY_ALIASES.get(title, title.replace(" ", "_"))


def structural_section_key(section: dict[str, Any]) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", str(section.get("section_title") or "").lower()).strip()
    specific_terms = (
        ("literature review", "literature_review"), ("implementation detail", "implementation_details"),
        ("related work", "related_work"), ("preliminar", "preliminaries"), ("architecture", "architecture"),
        ("framework", "framework"), ("overview", "overview"), ("evaluation", "evaluation"),
        ("ablation", "ablation"), ("analysis", "analysis"), ("dataset", "dataset"),
        ("objective", "objective"), ("optimization", "optimization"), ("background", "background"),
        ("approach", "approach"), ("model", "model"),
    )
    for term, key in specific_terms:
        if term in title:
            return key
    return _section_target_key(section)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or f"{record.get('paper_id')}::{record.get('record_id')}::{record.get('page')}")


def _record_fingerprint(record: dict[str, Any]) -> str:
    """Fingerprint only evidence-bearing fields; runtime annotations are not provenance."""
    return _json_hash(
        {
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "section_id": record.get("section_id"),
            "record_type": record.get("record_type"),
            "source_type": record.get("source_type"),
            "label": record.get("label"),
            "locator": record.get("locator"),
            "text": record.get("text"),
            "crop_path": _record_image_path(record),
        }
    )


def _deduplicate_candidate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove only byte-for-byte semantic duplicates; conflicting provenance remains visible."""
    unique: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    fingerprints_by_id: dict[str, set[str]] = defaultdict(set)
    repeated_ids: set[str] = set()
    conflicts: set[str] = set()
    for record in records:
        record_id = _record_key(record)
        fingerprint = _record_fingerprint(record)
        key = (record_id, fingerprint)
        if key in seen:
            repeated_ids.add(record_id)
            continue
        if fingerprints_by_id[record_id] and fingerprint not in fingerprints_by_id[record_id]:
            conflicts.add(record_id)
        copied = dict(record)
        copied["_canonical_record_fingerprint"] = fingerprint
        seen[key] = copied
        fingerprints_by_id[record_id].add(fingerprint)
        unique.append(copied)
    return unique, {
        "input_record_count": len(records),
        "canonical_record_count": len(unique),
        "exact_duplicate_record_count": len(records) - len(unique),
        "repeated_global_record_id_count": len(repeated_ids),
        "conflicting_global_record_id_count": len(conflicts),
    }


def _document_order(record: dict[str, Any]) -> tuple[int, int, str]:
    return (int(record.get("page") or 0), int(record.get("document_order") or record.get("reading_order") or 0), _record_key(record))


def context_records_for_section(
    section: dict[str, Any],
    config: SectionRelevanceConfig,
) -> list[dict[str, Any]]:
    records = sorted(section["records"], key=_document_order)
    if "_context_record_ids" in section:
        allowed = set(section.get("_context_record_ids") or [])
        records = [record for record in records if _record_key(record) in allowed]
    if config.record_top_k > 0:
        records = records[: config.record_top_k]
    return records


def rank_and_select_units(
    sections: list[dict[str, Any]],
    config: SectionRelevanceConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks = [chunk for section in sections for chunk in section["chunks"]]
    for section in sections:
        section_score = min(1.0, max(0.0, float(section["assessment"]["relevance"]["score"])))
        section_bonus = min(config.section_bonus_max, config.section_bonus_weight * section_score)
        for chunk in section["chunks"]:
            score = chunk["score_contract"]
            object_bonus = 0.0
            if str(chunk.get("unit_type") or "").startswith("object_"):
                object_total = min(config.section_bonus_max, config.object_section_bonus_weight * section_score)
                object_bonus = max(0.0, object_total - section_bonus)
            score.update({
                "section_relevance": round(float(section["assessment"]["relevance"]["score"]), 8),
                "section_bonus": round(section_bonus, 8),
                "object_bonus": round(object_bonus, 8),
                "final_relevance": round(float(score["local_relevance"]) + section_bonus + object_bonus, 8),
            })

    for rank, chunk in enumerate(
        sorted(chunks, key=lambda item: (-float(item["score_contract"]["local_relevance"]), item["section_index"], item["chunk_index"])),
        start=1,
    ):
        chunk["score_contract"]["local_rank"] = rank
    ranked = sorted(
        chunks,
        key=lambda item: (-float(item["score_contract"]["final_relevance"]), item["section_index"], item["chunk_index"]),
    )
    selected = set(id(chunk) for chunk in (ranked[: config.retrieval_unit_top_k] if config.retrieval_unit_top_k > 0 else ranked))
    for rank, chunk in enumerate(ranked, start=1):
        chunk["score_contract"]["final_rank"] = rank
        chunk["selected_for_retrieval"] = id(chunk) in selected

    unit_budget_active = config.unit_mode == "record_aware" and config.retrieval_unit_top_k > 0
    for section in sections:
        if unit_budget_active:
            context_chunks = [chunk for chunk in section["chunks"] if id(chunk) in selected]
        elif config.backend == "llmrerank" and config.llmrerank_context_top_k_chunks > 0:
            context_chunks = sorted(
                section["chunks"],
                key=lambda chunk: (-float(chunk["score_contract"]["final_relevance"]), int(chunk["chunk_index"])),
            )[: config.llmrerank_context_top_k_chunks]
        else:
            context_chunks = list(section["chunks"])
        section["_context_chunk_indexes"] = [chunk["chunk_index"] for chunk in context_chunks]
        section["_context_record_ids"] = [record_id for chunk in context_chunks for record_id in chunk["record_ids"]]
    selected_sections = [section for section in sections if section["_context_chunk_indexes"]] if unit_budget_active else sections
    return ranked, selected_sections


def _artifact_fingerprint(processed_root: Path, paper_id: str) -> dict[str, Any]:
    matches = sorted(processed_root.glob(f"*/{paper_id}/artifact_status.json"))
    status_path = matches[0] if matches else processed_root / paper_id / "artifact_status.json"
    if not status_path.exists():
        return {"artifact_status": "missing"}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"artifact_status": "unreadable"}
    return {
        "artifact_version": status.get("artifact_version"),
        "standardizer_version": status.get("standardizer_version"),
        "transcription_backend": status.get("transcription_backend"),
        "records_sha256": status.get("records_sha256"),
    }


def _record_image_path(record: dict[str, Any]) -> str | None:
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
    return None


def _section_groups(records: list[dict[str, Any]], processed_root: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
        if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            continue
        paper_id = str(record.get("paper_id") or "")
        section_id = str(record.get("section_id") or "sec_unassigned")
        groups[(paper_id, section_id)].append(record)

    sections: list[dict[str, Any]] = []
    for (paper_id, section_id), section_records in groups.items():
        ordered = sorted(section_records, key=_document_order)
        first = ordered[0]
        labels = []
        for record in ordered:
            if to_official_source_type(record.get("record_type"), record.get("source_type")) in {"table", "figure", "equation_algorithm"}:
                label = str(record.get("label") or record.get("locator", {}).get("object_id") or "").strip()
                if label and label not in labels:
                    labels.append(label)
        content_parts = []
        for record in ordered:
            text = str(record.get("text") or "").strip()
            if text:
                content_parts.append(f"[{_record_key(record)}] {text}")
        content = "\n".join(content_parts)
        pages = [int(record.get("page") or 0) for record in ordered if record.get("page")]
        image_records = [
            {"record_id": _record_key(record), "image_path": image_path}
            for record in ordered
            if (image_path := _record_image_path(record))
        ]
        sections.append(
            {
                "paper_id": paper_id,
                "section_id": section_id,
                "section_title": str(first.get("section_title") or "Front Matter"),
                "section_type": str(first.get("section_type") or "unknown"),
                "section_path": first.get("section_path") or [str(first.get("section_title") or "Front Matter")],
                "page_start": min(pages) if pages else 0,
                "page_end": max(pages) if pages else 0,
                "object_labels": labels,
                "image_records": image_records,
                "records": ordered,
                "record_ids": [_record_key(record) for record in ordered],
                "content": content,
                "artifact_fingerprint": _artifact_fingerprint(processed_root, paper_id),
            }
        )
    return sorted(sections, key=lambda item: (item["paper_id"], item["page_start"], item["section_id"]))


def _prefix(section: dict[str, Any]) -> str:
    labels = ", ".join(section["object_labels"]) or "None"
    section_path = section.get("section_path")
    path_text = " > ".join(str(item) for item in section_path) if isinstance(section_path, list) else str(section_path or section["section_title"])
    return (
        f"Paper: {section['paper_id']}\n"
        f"Section: {path_text}\n"
        f"Section type: {section['section_type']}\n"
        f"Pages: {section['page_start']}-{section['page_end']}\n"
        f"Objects: {labels}\n"
        "Content:\n"
    )


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer  # lazy: BM25 should have no transformer dependency

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # We deliberately tokenize complete sections before splitting them ourselves.
    tokenizer.model_max_length = 10**9
    return tokenizer


def _unit_from_records(
    prefix: str,
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    unit_type: str,
    anchor_record_ids: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    body = "\n".join(
        f"[{_record_key(record)}] "
        + (
            f"source_type={to_official_source_type(record.get('record_type'), record.get('source_type'))}; "
            f"label={record.get('label') or ''}; locator={json.dumps(record.get('locator') or {}, ensure_ascii=False)}; "
            if to_official_source_type(record.get("record_type"), record.get("source_type")) != "text_span"
            else ""
        )
        + str(record.get('text') or '').strip()
        for record in records
        if str(record.get("text") or "").strip()
    ) or "[empty unit]"
    text = prefix + body
    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    return {
        "unit_type": unit_type,
        "text": text,
        "token_count": token_count,
        "content_token_start": 0,
        "content_token_end": max(0, token_count - len(tokenizer.encode(prefix, add_special_tokens=False))),
        "record_ids": [_record_key(record) for record in records],
        "anchor_record_ids": list(anchor_record_ids or []),
        "image_paths": list(image_paths or []),
        "document_order": min((_document_order(record) for record in records), default=(0, 0, "")),
    }


def _oversized_text_units(
    prefix: str,
    record: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
) -> list[dict[str, Any]]:
    marker = f"[{_record_key(record)}] "
    prefix_tokens = len(tokenizer.encode(prefix + marker, add_special_tokens=True))
    capacity = max(1, max_tokens - prefix_tokens)
    text_ids = tokenizer.encode(str(record.get("text") or ""), add_special_tokens=False)
    units = []
    for start in range(0, len(text_ids) or 1, capacity):
        fragment = tokenizer.decode(text_ids[start : start + capacity], skip_special_tokens=True) if text_ids else ""
        projected = dict(record)
        projected["text"] = fragment
        unit = _unit_from_records(prefix, [projected], tokenizer, unit_type="text_fragment")
        unit["oversized_record_split"] = True
        units.append(unit)
    return units


def _record_aware_units(section: dict[str, Any], tokenizer: Any, config: SectionRelevanceConfig) -> list[dict[str, Any]]:
    prefix = _prefix(section)
    ordered = sorted(section["records"], key=_document_order)
    text_records = [
        record for record in ordered
        if to_official_source_type(record.get("record_type"), record.get("source_type")) == "text_span"
    ]
    target_tokens = max(32, int(config.unit_target_tokens))
    max_tokens = max(target_tokens, int(config.unit_max_tokens))
    overlap_count = max(0, int(config.unit_overlap_records))
    units: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    dirty = False

    def flush() -> None:
        nonlocal current, dirty
        if current and dirty:
            units.append(_unit_from_records(prefix, current, tokenizer, unit_type="text_chunk"))
            current = current[-overlap_count:] if overlap_count else []
            dirty = False

    for record in text_records:
        single = _unit_from_records(prefix, [record], tokenizer, unit_type="text_chunk")
        if single["token_count"] > max_tokens:
            flush()
            current = []
            units.extend(_oversized_text_units(prefix, record, tokenizer, max_tokens))
            continue
        proposed = _unit_from_records(prefix, [*current, record], tokenizer, unit_type="text_chunk")
        if current and proposed["token_count"] > max_tokens:
            flush()
            proposed = _unit_from_records(prefix, [*current, record], tokenizer, unit_type="text_chunk")
            if proposed["token_count"] > max_tokens:
                current = []
        current.append(record)
        dirty = True
        if _unit_from_records(prefix, current, tokenizer, unit_type="text_chunk")["token_count"] >= target_tokens:
            flush()
    if current and dirty:
        units.append(_unit_from_records(prefix, current, tokenizer, unit_type="text_chunk"))

    if config.object_units_enabled:
        image_count = 0
        neighbor_count = max(0, int(config.object_neighbor_records))
        for index, record in enumerate(ordered):
            source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
            if source_type not in {"table", "figure", "equation_algorithm", "citation_context"}:
                continue
            before = (
                [item for item in ordered[:index] if to_official_source_type(item.get("record_type"), item.get("source_type")) == "text_span"][-neighbor_count:]
                if neighbor_count else []
            )
            after = (
                [item for item in ordered[index + 1 :] if to_official_source_type(item.get("record_type"), item.get("source_type")) == "text_span"][:neighbor_count]
                if neighbor_count else []
            )
            image_path = _record_image_path(record)
            images = []
            if (
                config.llmrerank_input_mode == "text_with_object_images"
                and image_path
                and image_count < max(0, int(config.llmrerank_max_images_per_section))
            ):
                images = [image_path]
                image_count += 1
            unit = _unit_from_records(
                prefix,
                [*before, record, *after],
                tokenizer,
                unit_type=f"object_{source_type}",
                anchor_record_ids=[_record_key(record)],
                image_paths=images,
            )
            unit["oversized_atomic_object"] = unit["token_count"] > max_tokens
            units.append(unit)

    units.sort(key=lambda unit: (unit["document_order"], 0 if unit["unit_type"] == "text_chunk" else 1))
    for index, unit in enumerate(units):
        unit["chunk_index"] = index
        unit.pop("document_order", None)
    section["full_representation"] = prefix + section["content"]
    section["prefix"] = prefix
    return units


def _chunk_section(section: dict[str, Any], tokenizer: Any, config: SectionRelevanceConfig) -> list[dict[str, Any]]:
    if config.unit_mode == "record_aware":
        return _record_aware_units(section, tokenizer, config)
    if config.backend == "llmrerank":
        max_tokens = min(32768, max(32, int(config.llmrerank_section_chunk_max_tokens)))
        requested_overlap = config.llmrerank_section_chunk_overlap_tokens
    else:
        max_tokens = min(512, max(32, int(config.chunk_max_tokens)))
        requested_overlap = config.chunk_overlap_tokens
    prefix = _prefix(section)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) >= max_tokens:
        # Object labels are auxiliary metadata; preserve the contract fields while shortening only their value.
        prefix_section = dict(section)
        prefix_section["object_labels"] = section["object_labels"][:8]
        prefix = _prefix(prefix_section)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    body_capacity = max(1, max_tokens - len(prefix_ids) - 2)
    body_ids = tokenizer.encode(section["content"], add_special_tokens=False)
    if not body_ids:
        body_ids = tokenizer.encode("[empty section]", add_special_tokens=False)
    overlap = min(max(0, int(requested_overlap)), max(0, body_capacity - 1))
    chunks = []
    image_budget = max(0, int(config.llmrerank_max_images_per_section))
    eligible_images = {
        str(item["record_id"]): str(item["image_path"])
        for item in section.get("image_records") or []
    }
    assigned_images: set[str] = set()
    start = 0
    while start < len(body_ids):
        end = min(len(body_ids), start + body_capacity)
        body = tokenizer.decode(body_ids[start:end], skip_special_tokens=True)
        text = prefix + body
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        while token_count > max_tokens and end > start + 1:
            end -= max(1, token_count - max_tokens)
            body = tokenizer.decode(body_ids[start:end], skip_special_tokens=True)
            text = prefix + body
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
        record_ids = [record_id for record_id in section.get("record_ids", []) if f"[{record_id}]" in text]
        image_paths: list[str] = []
        if config.llmrerank_input_mode == "text_with_object_images":
            for record_id in record_ids:
                image_path = eligible_images.get(record_id)
                if image_path and image_path not in assigned_images and len(assigned_images) < image_budget:
                    image_paths.append(image_path)
                    assigned_images.add(image_path)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": text,
                "token_count": token_count,
                "content_token_start": start,
                "content_token_end": end,
                "record_ids": record_ids,
                "image_paths": image_paths,
            }
        )
        if end >= len(body_ids):
            break
        start = max(start + 1, end - overlap)
    section["full_representation"] = prefix + section["content"]
    section["prefix"] = prefix
    return chunks


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def pool_scores(scores: list[float], pooling: str, log_mean_exp_lambda: float, top_k: int = 2) -> dict[str, float]:
    if not scores:
        return {"mean": 0.0, "max": 0.0, "top_k_mean": 0.0, "log_mean_exp": 0.0, "selected": 0.0}
    mean = sum(scores) / len(scores)
    maximum = max(scores)
    top_count = min(max(1, int(top_k)), len(scores))
    top_k_mean = sum(sorted(scores, reverse=True)[:top_count]) / top_count
    lam = max(1e-6, float(log_mean_exp_lambda))
    anchor = maximum
    lme = anchor + math.log(sum(math.exp(lam * (score - anchor)) for score in scores) / len(scores)) / lam
    selected = {"mean": mean, "max": maximum, "top_k_mean": top_k_mean, "log_mean_exp": lme}.get(pooling, lme)
    return {"mean": mean, "max": maximum, "top_k_mean": top_k_mean, "log_mean_exp": lme, "selected": selected}


def pool_section_units(
    scores: list[float],
    pooling: str,
    log_mean_exp_lambda: float,
    aggregation_top_k: int,
    top_k_mean: int = 3,
) -> dict[str, float]:
    selected_scores = sorted(scores, reverse=True)[:aggregation_top_k] if aggregation_top_k > 0 else list(scores)
    pooled = pool_scores(selected_scores, pooling, log_mean_exp_lambda, top_k_mean)
    pooled["unit_count"] = len(scores)
    pooled["aggregated_unit_count"] = len(selected_scores)
    return pooled


def _embedding_cache_path(cache_root: Path, key: dict[str, Any]) -> Path:
    return cache_root / f"{_json_hash(key)}.json"


def _e5_embeddings(
    chunks: list[dict[str, Any]], config: SectionRelevanceConfig, cache_root: Path, tokenizer: Any, model: Any
) -> tuple[list[list[float]], list[bool], str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    tokenizer_version = str(getattr(tokenizer, "version", "unknown"))
    cached: list[list[float] | None] = []
    keys: list[dict[str, Any]] = []
    cache_hits: list[bool] = []
    missing_indices: list[int] = []
    for index, chunk in enumerate(chunks):
        key = {
            "cache_version": EMBEDDING_CACHE_VERSION,
            "template_version": REPRESENTATION_TEMPLATE_VERSION,
            "model": config.e5_model,
            "tokenizer": tokenizer.__class__.__name__,
            "tokenizer_version": tokenizer_version,
            "chunk_max_tokens": config.chunk_max_tokens,
            "chunk_overlap_tokens": config.chunk_overlap_tokens,
            "text": chunk["text"],
            "artifact": chunk["artifact_fingerprint"],
        }
        keys.append(key)
        path = _embedding_cache_path(cache_root, key)
        try:
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            embedding = value.get("embedding") if isinstance(value, dict) else None
            if isinstance(embedding, list):
                cached.append([float(item) for item in embedding])
                cache_hits.append(True)
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        cached.append(None)
        cache_hits.append(False)
        missing_indices.append(index)

    if missing_indices:
        import torch
        for start in range(0, len(missing_indices), 16):
            batch_indices = missing_indices[start : start + 16]
            batch = tokenizer(["passage: " + chunks[index]["text"] for index in batch_indices], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                output = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).expand(output.size()).float()
                embeddings = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            for index, embedding in zip(batch_indices, embeddings.cpu().tolist()):
                cached[index] = embedding
                _embedding_cache_path(cache_root, keys[index]).write_text(json.dumps({"embedding": embedding}), encoding="utf-8")
    return [item or [] for item in cached], cache_hits, tokenizer_version


def e5_score_queries(
    queries: list[str],
    chunks: list[dict[str, Any]],
    config: SectionRelevanceConfig,
    cache_root: Path,
) -> tuple[list[list[float]], list[bool], str]:
    import torch
    from transformers import AutoModel

    if not queries:
        return [], [], "unknown"
    tokenizer = _load_tokenizer(config.e5_model)
    model = AutoModel.from_pretrained(config.e5_model)
    model.eval()
    embeddings, hits, tokenizer_version = _e5_embeddings(chunks, config, cache_root, tokenizer, model)
    batch = tokenizer(["query: " + query for query in queries], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(output.size()).float()
        query_embeddings = torch.nn.functional.normalize((output * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)
    matrix = torch.tensor(embeddings, dtype=query_embeddings.dtype)
    return torch.matmul(query_embeddings, matrix.T).cpu().tolist(), hits, tokenizer_version


def _e5_scores(query: str, chunks: list[dict[str, Any]], config: SectionRelevanceConfig, cache_root: Path) -> tuple[list[float], list[bool], str]:
    scores, hits, tokenizer_version = e5_score_queries([query], chunks, config, cache_root)
    return scores[0], hits, tokenizer_version


def _image_hashes(paths: list[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        try:
            values.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except OSError:
            values.append({"path": str(path), "sha256": None})
    return values


def _llmrerank_scores(
    query: str,
    chunks: list[dict[str, Any]],
    config: SectionRelevanceConfig,
    cache_root: Path,
    eligible_indexes: set[int] | None = None,
) -> tuple[list[float | None], list[dict[str, Any]], dict[str, Any]]:
    instruction = LLMRERANK_INSTRUCTIONS.get(config.llmrerank_instruction_version)
    if instruction is None:
        raise ValueError(f"Unknown LLMRERANK_INSTRUCTION_VERSION: {config.llmrerank_instruction_version}")
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    query_cache_root = cache_root / query_hash
    query_cache_root.mkdir(parents=True, exist_ok=True)
    scores: list[float | None] = [None] * len(chunks)
    details: list[dict[str, Any]] = []
    projection_by_id: dict[str, dict[str, Any]] = {}
    projection_ids_by_chunk: dict[int, list[str]] = defaultdict(list)
    image_hash_cache: dict[str, dict[str, Any]] = {}
    eligible = set(range(len(chunks))) if eligible_indexes is None else set(eligible_indexes)

    def image_fingerprint(path_value: str) -> dict[str, Any]:
        if path_value not in image_hash_cache:
            image_hash_cache[path_value] = _image_hashes([path_value])[0]
        return image_hash_cache[path_value]

    def add_projection(index: int, modality: str, document: dict[str, Any], value: Any) -> str:
        chunk = chunks[index]
        key = {
            "cache_version": LLMRERANK_CACHE_VERSION,
            "canonical_projection_version": CANONICAL_RERANK_PROJECTION_VERSION,
            "template_version": REPRESENTATION_TEMPLATE_VERSION,
            "model": config.llmrerank_model,
            "instruction_version": config.llmrerank_instruction_version,
            "input_mode": config.llmrerank_input_mode,
            "query": query,
            "modality": modality,
            "value": value,
            "artifact": chunk["artifact_fingerprint"],
        }
        projection_id = _json_hash(key)
        projection = projection_by_id.setdefault(
            projection_id,
            {
                "projection_id": projection_id,
                "modality": modality,
                "document": document,
                "member_indexes": [],
                "cache_path": str(query_cache_root / f"{projection_id}.json"),
                "token_count": int(chunk.get("token_count") or 0) if modality == "text" else 0,
            },
        )
        projection["member_indexes"].append(index)
        projection_ids_by_chunk[index].append(projection_id)
        return projection_id

    for index, chunk in enumerate(chunks):
        detail = {
            "representation_hash": None,
            "canonical_projection_ids": [],
            "cache_path": None,
            "cache_hit": False,
            "status": "pending",
            "error": None,
            "attempts": 0,
            "api_response_id": None,
            "api_meta": None,
        }
        if index not in eligible:
            detail["status"] = "prefiltered"
            details.append(detail)
            continue
        text_id = add_projection(index, "text", {"text": chunk["text"]}, chunk["text"])
        detail["canonical_projection_ids"].append(text_id)
        seen_images: set[str] = set()
        for image_path in chunk.get("image_paths") or []:
            fingerprint = image_fingerprint(str(image_path))
            image_key = str(fingerprint.get("sha256") or fingerprint.get("path") or "")
            if not image_key or image_key in seen_images:
                continue
            seen_images.add(image_key)
            image_id = add_projection(index, "image", {"image_path": str(image_path)}, fingerprint)
            detail["canonical_projection_ids"].append(image_id)
        detail["representation_hash"] = _json_hash(detail["canonical_projection_ids"])
        details.append(detail)

    projection_scores: dict[str, dict[str, Any]] = {}
    missing_projection_ids: list[str] = []
    for projection_id, projection in projection_by_id.items():
        path = Path(projection["cache_path"])
        try:
            cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if isinstance(cached, dict) and isinstance(cached.get("score"), (int, float)):
                projection_scores[projection_id] = {
                    "score": float(cached["score"]),
                    "cache_hit": True,
                    "api_response_id": cached.get("api_response_id"),
                    "api_meta": cached.get("api_meta"),
                    "attempts": int(cached.get("attempts") or 0),
                }
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        missing_projection_ids.append(projection_id)

    client = LLMRerankClient(
        api_key=config.llmrerank_api_key,
        base_url=config.llmrerank_base_url,
        model=config.llmrerank_model,
        timeout_seconds=config.llmrerank_request_timeout_seconds,
        max_retries=config.llmrerank_max_retries,
    )
    batch_size = max(1, int(config.llmrerank_batch_size))
    units = [
        {
            "projection_id": projection_id,
            "modality": projection_by_id[projection_id]["modality"],
            "document": projection_by_id[projection_id]["document"],
        }
        for projection_id in missing_projection_ids
    ]
    batches = [units[start : start + batch_size] for start in range(0, len(units), batch_size)]
    started = time.monotonic()
    api_calls = retries = failures = input_tokens = image_tokens = 0
    modality_scores: dict[int, list[dict[str, Any]]] = defaultdict(list)
    chunk_errors: dict[int, list[str]] = defaultdict(list)

    def score_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, LLMRerankError | None]:
        try:
            result = client.score_documents(
                query=query,
                documents=[unit["document"] for unit in batch],
                instruction=instruction,
            )
            return batch, result, None
        except LLMRerankError as exc:
            return batch, None, exc
        except Exception as exc:
            return batch, None, LLMRerankError(
                f"Unexpected LLM reranker failure: {exc}",
                attempts=1,
            )

    completed: list[tuple[list[dict[str, Any]], dict[str, Any] | None, LLMRerankError | None]] = []
    concurrency = max(1, int(config.llmrerank_request_concurrency))
    if concurrency == 1:
        completed = [score_batch(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(score_batch, batch) for batch in batches]
            completed = [future.result() for future in as_completed(futures)]

    for batch, result, error in completed:
        api_calls += 1
        if error is not None:
            failures += len(batch)
            retries += max(0, error.attempts - 1)
            for unit in batch:
                chunk_errors[str(unit["projection_id"])].append(str(error))
            continue
        assert result is not None
        attempts = int(result.get("attempts") or 1)
        retries += max(0, attempts - 1)
        response = result.get("raw_response") if isinstance(result.get("raw_response"), dict) else {}
        response_id = response.get("id")
        response_meta = response.get("meta")
        tokens = response_meta.get("tokens") if isinstance(response_meta, dict) and isinstance(response_meta.get("tokens"), dict) else {}
        input_tokens += int(tokens.get("input_tokens") or 0)
        image_tokens += int(tokens.get("image_tokens") or 0)
        for unit, score in zip(batch, result["scores"]):
            projection_scores[str(unit["projection_id"])] = {
                "score": float(score),
                "cache_hit": False,
                "api_response_id": response_id,
                "api_meta": response_meta,
                "attempts": attempts,
            }

    for index in sorted(eligible):
        projection_ids = projection_ids_by_chunk[index]
        values = [projection_scores[projection_id] for projection_id in projection_ids if projection_id in projection_scores]
        if not values:
            errors = [error for projection_id in projection_ids for error in chunk_errors.get(projection_id, [])]
            details[index].update({"status": "failed", "error": "; ".join(errors) or "No reranker score returned."})
            continue
        score = max(float(item["score"]) for item in values)
        scores[index] = score
        status = "ok" if len(values) == len(projection_ids) else "partial"
        modality_scores = [
            {
                "modality": projection_by_id[projection_id]["modality"],
                "score": projection_scores[projection_id]["score"],
                "canonical_projection_id": projection_id,
            }
            for projection_id in projection_ids
            if projection_id in projection_scores
        ]
        details[index].update(
            {
                "status": status,
                "error": "; ".join(error for projection_id in projection_ids for error in chunk_errors.get(projection_id, [])) or None,
                "attempts": max(int(item["attempts"]) for item in values),
                "api_response_id": [item["api_response_id"] for item in values],
                "api_meta": [item["api_meta"] for item in values],
                "modality_scores": modality_scores,
                "modality_pooling": "max",
                "cache_hit": all(bool(item["cache_hit"]) for item in values),
                "cache_path": [projection_by_id[projection_id]["cache_path"] for projection_id in projection_ids],
            }
        )
    for projection_id in missing_projection_ids:
        result = projection_scores.get(projection_id)
        if result is None:
            continue
        Path(projection_by_id[projection_id]["cache_path"]).write_text(
            json.dumps(
                {
                    "score": result["score"],
                    "api_response_id": result["api_response_id"],
                    "api_meta": result["api_meta"],
                    "attempts": result["attempts"],
                    "canonical_projection_version": CANONICAL_RERANK_PROJECTION_VERSION,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    logical_image_projection_count = sum(max(0, len(projection_ids_by_chunk[index]) - 1) for index in eligible)
    canonical_text_projection_count = sum(item["modality"] == "text" for item in projection_by_id.values())
    canonical_image_projection_count = len(projection_by_id) - canonical_text_projection_count
    stats = {
        "instruction": instruction,
        "instruction_version": config.llmrerank_instruction_version,
        "api_calls": api_calls,
        "pair_count": len(units),
        "logical_unit_count": len(chunks),
        "scored_logical_unit_count": len(eligible),
        "canonical_projection_version": CANONICAL_RERANK_PROJECTION_VERSION,
        "canonical_projection_count": len(projection_by_id),
        "canonical_text_projection_count": canonical_text_projection_count,
        "canonical_image_projection_count": canonical_image_projection_count,
        "deduplicated_text_projection_count": len(eligible) - canonical_text_projection_count,
        "deduplicated_image_projection_count": logical_image_projection_count - canonical_image_projection_count,
        "prefilter_eligible_count": len(chunks) if eligible_indexes is None else len(eligible_indexes),
        "prefiltered_unit_count": 0 if eligible_indexes is None else len(chunks) - len(eligible_indexes),
        "cache_hits": sum(bool(detail.get("cache_hit")) for detail in details),
        "projection_cache_hits": sum(bool(item.get("cache_hit")) for item in projection_scores.values()),
        "image_count": logical_image_projection_count,
        "text_input_tokens": sum(int(item["token_count"]) for item in projection_by_id.values() if item["modality"] == "text"),
        "provider_input_tokens": input_tokens,
        "provider_image_tokens": image_tokens,
        "retry_count": retries,
        "failed_pair_count": failures,
        "multimodal_strategy": "separate_text_and_image_pairs_max_pool",
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    return scores, details, stats


def _validate_config(config: SectionRelevanceConfig) -> None:
    if config.backend not in {"bm25", "e5_base_v2", "hybrid", "llmrerank"}:
        raise ValueError(f"Unsupported SECTION_RELEVANCE_BACKEND: {config.backend}")
    if config.llmrerank_input_mode not in {"text_only", "text_with_object_images"}:
        raise ValueError(f"Unsupported LLMRERANK_INPUT_MODE: {config.llmrerank_input_mode}")
    if config.llmrerank_failure_fallback not in {"none", "bm25", "e5_base_v2", "hybrid"}:
        raise ValueError(f"Unsupported LLMRERANK_FAILURE_FALLBACK: {config.llmrerank_failure_fallback}")
    if config.llmrerank_section_pooling not in {"max", "log_mean_exp", "top_k_mean"}:
        raise ValueError(f"Unsupported LLMRERANK_SECTION_POOLING: {config.llmrerank_section_pooling}")
    if config.unit_mode not in {"token_chunks", "record_aware"}:
        raise ValueError(f"Unsupported SECTION_RELEVANCE_UNIT_MODE: {config.unit_mode}")
    if config.unit_target_tokens <= 0 or config.unit_max_tokens < config.unit_target_tokens:
        raise ValueError("Record-aware unit token limits must satisfy 0 < target <= max.")
    if config.aggregation_top_k < 0 or config.retrieval_unit_top_k < 0:
        raise ValueError("Unit top-k values must be non-negative.")
    if config.llmrerank_unit_prefilter_top_k < 0 or config.llmrerank_unit_prefilter_per_section < 0:
        raise ValueError("LLM reranker prefilter values must be non-negative.")
    if config.llmrerank_unit_prefilter_primary_top_k < 0 or not 0 <= config.llmrerank_prefilter_fallback_weight <= 1:
        raise ValueError("Primary prefilter top-k must be non-negative and fallback weight must be in [0, 1].")
    if min(config.section_bonus_weight, config.object_section_bonus_weight, config.section_bonus_max) < 0:
        raise ValueError("Section-to-unit bonus values must be non-negative.")


def retrieve_section_relevance(
    query: str,
    candidate_records: list[dict[str, Any]],
    processed_root: str | Path,
    config: SectionRelevanceConfig,
    top_k_sections: int,
    query_id: str | None = None,
    primary_evidence_type: str | None = None,
    candidate_paper_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank complete candidate sections and expand selected sections' records."""
    _validate_config(config)
    root = Path(processed_root)
    canonical_records, record_dedup_stats = _deduplicate_candidate_records(candidate_records)
    sections = _section_groups(canonical_records, root)
    if not sections:
        return {"query_id": query_id, "ranked_sections": [], "selected_sections": [], "expanded_records": [], "trace": {"query_id": query_id, "sections": []}}

    tokenizer_model = config.llmrerank_model if config.backend == "llmrerank" else config.e5_model
    tokenizer = _load_tokenizer(tokenizer_model)
    chunks: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        section_chunks = _chunk_section(section, tokenizer, config)
        for chunk in section_chunks:
            chunk["section_index"] = section_index
            chunk["paper_id"] = section["paper_id"]
            chunk["section_id"] = section["section_id"]
            chunk["artifact_fingerprint"] = section["artifact_fingerprint"]
        section["chunks"] = section_chunks
        section["text_guard_exceeded"] = len(section["content"]) > config.text_max_chars
        chunks.extend(section_chunks)

    bm25_raw = [float(value) for value in BM25Okapi([tokenize(chunk["text"]) for chunk in chunks]).get_scores(tokenize(query))]
    e5_raw: list[float] | None = None
    e5_cache_hits: list[bool] | None = None
    llmrerank_raw: list[float | None] | None = None
    llmrerank_details: list[dict[str, Any]] | None = None
    llmrerank_stats: dict[str, Any] | None = None
    tokenizer_version = str(getattr(tokenizer, "version", "unknown"))
    e5_tokenizer_version: str | None = None
    needs_e5 = config.backend in {"e5_base_v2", "hybrid"} or (
        config.backend == "llmrerank" and config.llmrerank_failure_fallback in {"e5_base_v2", "hybrid"}
    )
    if needs_e5:
        e5_raw, e5_cache_hits, e5_tokenizer_version = _e5_scores(query, chunks, config, root / ".section_relevance_embedding_cache")
        if config.backend != "llmrerank":
            tokenizer_version = e5_tokenizer_version
    if config.backend == "llmrerank":
        eligible_indexes: set[int] | None = None
        explicit_indexes = explicit_target_unit_indexes(query, chunks)
        locator_only_indexes, locator_only_audit = (
            deterministic_locator_only_indexes(query, chunks, candidate_paper_metadata)
            if config.llmrerank_deterministic_locator_only_enabled
            else (None, {"status": "disabled"})
        )
        if locator_only_indexes is not None:
            eligible_indexes = locator_only_indexes
            rerank_scope = "deterministic_locator_only"
        elif config.unit_mode == "record_aware" and config.llmrerank_unit_prefilter_enabled:
            global_count = max(0, int(config.llmrerank_unit_prefilter_top_k))
            per_section_count = max(0, int(config.llmrerank_unit_prefilter_per_section))
            eligible_indexes = set(sorted(range(len(chunks)), key=lambda item: (-bm25_raw[item], item))[:global_count])
            for section_index in range(len(sections)):
                section_indexes = [index for index, chunk in enumerate(chunks) if chunk["section_index"] == section_index]
                eligible_indexes.update(sorted(section_indexes, key=lambda item: (-bm25_raw[item], item))[:per_section_count])
            primary_unit_type = f"object_{primary_evidence_type}"
            primary_indexes = [index for index, chunk in enumerate(chunks) if chunk.get("unit_type") == primary_unit_type]
            eligible_indexes.update(
                sorted(primary_indexes, key=lambda item: (-bm25_raw[item], item))[
                    : max(0, int(config.llmrerank_unit_prefilter_primary_top_k))
                ]
            )
            eligible_indexes.update(explicit_indexes)
            rerank_scope = "experimental_sparse_qwen"
        else:
            rerank_scope = "full_qwen"
        llmrerank_raw, llmrerank_details, llmrerank_stats = _llmrerank_scores(
            query,
            chunks,
            config,
            root / ".section_relevance_llmrerank_cache",
            eligible_indexes,
        )
        if llmrerank_stats is not None:
            llmrerank_stats["explicit_target_unit_count"] = len(explicit_indexes)
            llmrerank_stats["candidate_union_mode"] = rerank_scope
            llmrerank_stats["deterministic_locator_only"] = locator_only_audit

    aggregation_top_k = config.aggregation_top_k if config.unit_mode == "record_aware" else 0
    section_scores: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        chunk_indexes = [chunk_index for chunk_index, chunk in enumerate(chunks) if chunk["section_index"] == index]
        bm25_pool = pool_section_units(
            [bm25_raw[item] for item in chunk_indexes], config.pooling, config.log_mean_exp_lambda, aggregation_top_k
        )
        e5_pool = (
            pool_section_units(
                [e5_raw[item] for item in chunk_indexes], config.pooling, config.log_mean_exp_lambda, aggregation_top_k
            )
            if e5_raw is not None else None
        )
        llm_values = [float(llmrerank_raw[item]) for item in chunk_indexes if llmrerank_raw is not None and llmrerank_raw[item] is not None]
        llm_pool = (
            pool_section_units(
                llm_values,
                config.llmrerank_section_pooling,
                config.llmrerank_log_mean_exp_lambda,
                aggregation_top_k,
                config.llmrerank_top_k_mean_chunks,
            )
            if llmrerank_raw is not None
            else None
        )
        section_scores.append({"bm25": bm25_pool, "e5": e5_pool, "llmrerank": llm_pool, "llm_value_count": len(llm_values), "chunk_indexes": chunk_indexes})
    bm25_norm = _min_max([item["bm25"]["selected"] for item in section_scores])
    e5_norm = _min_max([item["e5"]["selected"] for item in section_scores]) if e5_raw is not None else [None] * len(sections)
    bm25_unit_norm = _min_max(bm25_raw)
    e5_unit_norm = _min_max(e5_raw) if e5_raw is not None else [None] * len(chunks)
    llm_unit_norm: list[float | None] = [None] * len(chunks)
    if llmrerank_raw is not None:
        available_indexes = [index for index, value in enumerate(llmrerank_raw) if value is not None]
        normalized_values = _min_max([float(llmrerank_raw[index]) for index in available_indexes])
        for index, normalized in zip(available_indexes, normalized_values):
            llm_unit_norm[index] = normalized
    section_targets = query_section_targets(query) if config.query_section_bonus_enabled else []
    for index, (section, score_data) in enumerate(zip(sections, section_scores)):
        section["_chunk_indexes"] = score_data["chunk_indexes"]
        relevance_backend = config.backend
        relevance_status = "ok"
        if config.backend == "e5_base_v2":
            final = float(e5_norm[index] or 0.0)
        elif config.backend == "hybrid":
            final = config.hybrid_bm25_weight * bm25_norm[index] + config.hybrid_e5_weight * float(e5_norm[index] or 0.0)
        elif config.backend == "llmrerank":
            available = int(score_data["llm_value_count"])
            expected = len(score_data["chunk_indexes"])
            if available:
                final = float(score_data["llmrerank"]["selected"])
                relevance_status = "ok" if available == expected else "partial"
            elif config.llmrerank_failure_fallback == "bm25":
                final = bm25_norm[index]
                relevance_backend = "bm25_fallback"
                relevance_status = "llmrerank_failed"
            elif config.llmrerank_failure_fallback == "e5_base_v2":
                final = float(e5_norm[index] or 0.0)
                relevance_backend = "e5_base_v2_fallback"
                relevance_status = "llmrerank_failed"
            elif config.llmrerank_failure_fallback == "hybrid":
                final = config.hybrid_bm25_weight * bm25_norm[index] + config.hybrid_e5_weight * float(e5_norm[index] or 0.0)
                relevance_backend = "hybrid_fallback"
                relevance_status = "llmrerank_failed"
            else:
                final = -1.0
                relevance_status = "unavailable"
        else:
            final = bm25_norm[index]
        base_final = final
        matched_targets = [target for target in section_targets if target["target"] == _section_target_key(section)]
        structural_bonuses_enabled = config.backend != "llmrerank" or config.apply_section_type_bonus
        query_section_bonus = config.query_section_bonus if structural_bonuses_enabled and matched_targets else 0.0
        structural_key = structural_section_key(section)
        prior_weight = (
            PRIMARY_TYPE_SECTION_PRIOR.get(str(primary_evidence_type or ""), {}).get(structural_key, 0.0)
            if structural_bonuses_enabled and config.primary_type_prior_enabled
            else 0.0
        )
        primary_type_prior_bonus = config.primary_type_prior_max_bonus * prior_weight
        final += query_section_bonus + primary_type_prior_bonus
        section["assessment"] = {
            "relevance": {
                "score": round(final, 8),
                "backend": relevance_backend,
                "model": config.llmrerank_model if config.backend == "llmrerank" else None,
                "status": relevance_status,
                "rank": 0,
                "components": {
                    "raw_chunk_scores": [
                        llmrerank_raw[item] for item in score_data["chunk_indexes"]
                    ] if llmrerank_raw is not None else None,
                    "pooling": config.llmrerank_section_pooling if config.backend == "llmrerank" else config.pooling,
                    "pooling_lambda": config.llmrerank_log_mean_exp_lambda if config.backend == "llmrerank" else config.log_mean_exp_lambda,
                },
            },
            "supportance": {"score": None, "status": "not_implemented"},
            "sufficiency": {"score": None, "status": "not_implemented"},
        }
        section["score_detail"] = {
            "bm25": {"raw": score_data["bm25"], "normalized": bm25_norm[index]},
            "e5": {"raw": score_data["e5"], "normalized": e5_norm[index]} if score_data["e5"] is not None else None,
            "llmrerank": score_data["llmrerank"],
            "base_final": base_final,
            "query_section_bonus": query_section_bonus,
            "query_section_matches": matched_targets,
            "structural_section_key": structural_key,
            "primary_evidence_type": primary_evidence_type,
            "primary_type_prior_weight": prior_weight,
            "primary_type_prior_bonus": primary_type_prior_bonus,
            "structural_bonuses_enabled": structural_bonuses_enabled,
            "backend_status": relevance_status,
            "final": final,
        }
        bounded_section_score = min(1.0, max(0.0, float(final)))
        for chunk_index in section["_chunk_indexes"]:
            if config.backend == "llmrerank" and llmrerank_raw is not None and llmrerank_raw[chunk_index] is not None:
                local_relevance = float(llm_unit_norm[chunk_index] or 0.0)
                local_backend = "llmrerank"
            elif config.backend == "e5_base_v2":
                local_relevance = float(e5_unit_norm[chunk_index] or 0.0)
                local_backend = "e5_base_v2"
            elif config.backend == "hybrid":
                local_relevance = (
                    config.hybrid_bm25_weight * bm25_unit_norm[chunk_index]
                    + config.hybrid_e5_weight * float(e5_unit_norm[chunk_index] or 0.0)
                )
                local_backend = "hybrid"
            else:
                local_relevance = bm25_unit_norm[chunk_index] * (
                    config.llmrerank_prefilter_fallback_weight if config.backend == "llmrerank" else 1.0
                )
                local_backend = "bm25_fallback" if config.backend == "llmrerank" else "bm25"
            section_bonus = min(config.section_bonus_max, config.section_bonus_weight * bounded_section_score)
            object_bonus = 0.0
            if str(chunks[chunk_index].get("unit_type") or "").startswith("object_"):
                object_total = min(config.section_bonus_max, config.object_section_bonus_weight * bounded_section_score)
                object_bonus = max(0.0, object_total - section_bonus)
            chunks[chunk_index]["score_contract"] = {
                "local_relevance": round(local_relevance, 8),
                "raw_local_relevance": (
                    round(float(llmrerank_raw[chunk_index]), 8)
                    if config.backend == "llmrerank" and llmrerank_raw is not None and llmrerank_raw[chunk_index] is not None
                    else round(float(bm25_raw[chunk_index]), 8)
                ),
                "local_backend": local_backend,
                "section_relevance": round(float(final), 8),
                "section_bonus": round(section_bonus, 8),
                "object_bonus": round(object_bonus, 8),
                "final_relevance": round(local_relevance + section_bonus + object_bonus, 8),
                "local_rank": 0,
                "final_rank": 0,
            }
    sections.sort(key=lambda item: (-float(item["assessment"]["relevance"]["score"]), item["paper_id"], item["section_id"]))
    for rank, section in enumerate(sections, start=1):
        section["assessment"]["relevance"]["rank"] = rank
    ranked_units, unit_selected_sections = rank_and_select_units(sections, config)
    chunk_index_by_identity = {id(chunk): index for index, chunk in enumerate(chunks)}
    ranked_unit_indexes = [chunk_index_by_identity[id(chunk)] for chunk in ranked_units]
    selected_sections = (
        unit_selected_sections
        if config.unit_mode == "record_aware" and config.retrieval_unit_top_k > 0
        else sections[: max(0, int(top_k_sections))] if top_k_sections > 0 else sections
    )
    expanded_records: list[dict[str, Any]] = []
    for section in selected_sections:
        expanded_records.extend(context_records_for_section(section, config))
    trace_sections = []
    for section in sections:
        trace_chunks = []
        for index in section["_chunk_indexes"]:
            chunk = chunks[index]
            trace_chunks.append({
                **{key: chunk.get(key) for key in (
                    "chunk_index", "unit_type", "text", "token_count", "content_token_start", "content_token_end",
                    "record_ids", "anchor_record_ids", "image_paths", "oversized_record_split", "oversized_atomic_object",
                )},
                "selected_for_context": index in section["_context_chunk_indexes"],
                "selected_for_retrieval": bool(chunk.get("selected_for_retrieval")),
                "score_contract": chunk.get("score_contract"),
                "bm25_raw_score": bm25_raw[index],
                "e5_raw_score": e5_raw[index] if e5_raw is not None else None,
                "e5_embedding_cache_hit": e5_cache_hits[index] if e5_cache_hits is not None else None,
                "llmrerank_raw_score": llmrerank_raw[index] if llmrerank_raw is not None else None,
                "llmrerank_call": llmrerank_details[index] if llmrerank_details is not None else None,
            })
        trace_sections.append({
            "paper_id": section["paper_id"], "section_id": section["section_id"], "section_title": section["section_title"], "section_type": section["section_type"],
            "section_path": section["section_path"],
            "page_start": section["page_start"], "page_end": section["page_end"], "record_ids": section["record_ids"], "object_labels": section["object_labels"],
            "representation_hash": hashlib.sha256(section["full_representation"].encode("utf-8")).hexdigest(),
            "full_representation": section["full_representation"], "assessment": section["assessment"], "score_detail": section["score_detail"], "chunks": trace_chunks,
            "content_char_count": len(section["content"]), "text_guard_exceeded": section["text_guard_exceeded"],
            "selected": section in selected_sections,
            "expanded_record_count": len(context_records_for_section(section, config)) if section in selected_sections else 0,
        })
    return {
        "query_id": query_id,
        "ranked_sections": sections,
        "selected_sections": selected_sections,
        "expanded_records": expanded_records,
        "trace": {
            "query_id": query_id, "query": query, "backend": config.backend, "pooling": config.pooling,
            "log_mean_exp_lambda": config.log_mean_exp_lambda, "tokenizer_version": tokenizer_version,
            "tokenizer_model": tokenizer_model,
            "e5_tokenizer_version": e5_tokenizer_version,
            "template_version": REPRESENTATION_TEMPLATE_VERSION, "top_k_sections": top_k_sections,
            "record_top_k": config.record_top_k,
            "context_top_k_chunks": config.llmrerank_context_top_k_chunks,
            "unit_mode": config.unit_mode,
            "unit_target_tokens": config.unit_target_tokens,
            "unit_max_tokens": config.unit_max_tokens,
            "unit_overlap_records": config.unit_overlap_records,
            "record_deduplication": record_dedup_stats,
            "aggregation_top_k": aggregation_top_k,
            "section_bonus_weight": config.section_bonus_weight,
            "object_section_bonus_weight": config.object_section_bonus_weight,
            "section_bonus_max": config.section_bonus_max,
            "retrieval_unit_top_k": config.retrieval_unit_top_k,
            "sections": trace_sections,
            "ranked_units": [
                {
                    "paper_id": chunks[index]["paper_id"],
                    "section_id": chunks[index]["section_id"],
                    "chunk_index": chunks[index]["chunk_index"],
                    "unit_type": chunks[index].get("unit_type", "token_chunk"),
                    "record_ids": chunks[index].get("record_ids", []),
                    "anchor_record_ids": chunks[index].get("anchor_record_ids", []),
                    "score_contract": chunks[index]["score_contract"],
                    "selected_for_retrieval": bool(chunks[index].get("selected_for_retrieval")),
                }
                for index in ranked_unit_indexes
            ],
            "candidate_union": {
                "mode": (llmrerank_stats or {}).get("candidate_union_mode", "not_applicable"),
                "logical_unit_count": len(chunks),
                "qwen_scored_unit_count": sum(value is not None for value in llmrerank_raw or []),
                "qwen_scored_record_ids": sorted(
                    {
                        record_id
                        for index, chunk in enumerate(chunks)
                        if llmrerank_raw is not None and llmrerank_raw[index] is not None
                        for record_id in chunk.get("record_ids", [])
                    }
                ),
                "qwen_unscored_record_ids": sorted(
                    {
                        record_id
                        for index, chunk in enumerate(chunks)
                        if llmrerank_raw is not None and llmrerank_raw[index] is None
                        for record_id in chunk.get("record_ids", [])
                    }
                ),
                "full_qwen_fallback_available": bool(config.backend == "llmrerank" and config.llmrerank_unit_prefilter_enabled),
            },
            "query_section_targets": section_targets, "query_section_bonus": config.query_section_bonus,
            "primary_evidence_type": primary_evidence_type,
            "primary_type_prior_enabled": config.primary_type_prior_enabled,
            "primary_type_prior_max_bonus": config.primary_type_prior_max_bonus,
            "llmrerank": {
                "model": config.llmrerank_model,
                "input_mode": config.llmrerank_input_mode,
                "pooling": config.llmrerank_section_pooling,
                "pooling_lambda": config.llmrerank_log_mean_exp_lambda,
                "top_k_mean_chunks": config.llmrerank_top_k_mean_chunks,
                "failure_fallback": config.llmrerank_failure_fallback,
                "apply_section_type_bonus": config.apply_section_type_bonus,
                **(llmrerank_stats or {}),
            } if config.backend == "llmrerank" else None,
            "selected_section_ids": [f"{item['paper_id']}::{item['section_id']}" for item in selected_sections],
            "expanded_record_ids": [_record_key(item) for item in expanded_records],
            "expanded_record_count": len(expanded_records),
            "expanded_char_count": sum(len(str(item.get("text") or "")) for item in expanded_records),
        },
    }
