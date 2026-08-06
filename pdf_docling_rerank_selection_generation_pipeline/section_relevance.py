"""Full-recall BM25/Qwen scoring for canonical record-aware units.

This module deliberately has no embedding, hybrid-score, sparse-prefilter, or
section-Top-K path.  BM25 is a routing signal and Qwen scores every canonical
unit before evidence-package coverage decides the final context.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .llmrerank_client import LLMRerankClient, LLMRerankError
from .metadata_index import BM25Okapi, tokenize
from .object_references import object_reference_paragraphs
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


REPRESENTATION_TEMPLATE_VERSION = "v5_qwen_compact_pair"
CANONICAL_PROJECTION_VERSION = "v2_text_image_separate"
CACHE_VERSION = "v7_full_qwen_compact_pair"
PAPER_IDENTITY_REPRESENTATION_TEMPLATE_VERSION = "v6_qwen_compact_pair_paper_identity"
PAPER_IDENTITY_CACHE_VERSION = "v8_full_qwen_compact_pair_paper_identity"
INSTRUCTIONS = {
    "v1": (
        "Given a research question and a candidate retrieval unit from a scientific paper, estimate whether the "
        "unit contains information relevant to answering the question. Judge relevance only and do not use "
        "external knowledge."
    ),
    "v2": (
        "Given a research question and a candidate retrieval unit from a scientific-paper section, estimate "
        "whether the unit contains information relevant to answering the question. Judge relevance only. Use "
        "section metadata as location context, not as proof. Consider text, object labels, tables, figures, "
        "equations, citations, images, and nearby records when provided."
    ),
    "v3_complete_support": (
        "Given a research question and a candidate retrieval unit from a scientific paper, estimate whether the "
        "unit is useful evidence for a complete answer. Count both direct answers and necessary supporting facts. "
        "For comparison or multi-paper questions, a unit may be highly relevant when it gives a baseline method's "
        "configuration, result, definition, training detail, table cell, caption, or experimental condition even "
        "when that paper or method is not named verbatim in the question. Do not require lexical overlap; judge "
        "the unit's concrete factual support for the requested topic. Use section metadata only as location context, "
        "not as proof. Consider text, object labels, tables, figures, equations, citations, images, and nearby records."
    ),
    "v4_complete_support_minimal": (
        "Rank each scientific-paper unit by its usefulness as evidence for answering the query. "
        "Include direct answers and necessary supporting facts. For comparisons, relevant evidence may describe "
        "an unnamed baseline. Use only the unit; section metadata is context, not evidence."
    ),
}


@dataclass(frozen=True)
class SectionRelevanceConfig:
    backend: str = "bm25"
    retriever_pool_budget: int = 0
    unit_mode: str = "record_aware"
    unit_target_tokens: int = 1280
    unit_max_tokens: int = 1536
    unit_overlap_records: int = 1
    object_units_enabled: bool = True
    object_neighbor_records: int = 1
    llmrerank_api_key: str | None = None
    llmrerank_base_url: str = "https://api.siliconflow.cn/v1"
    llmrerank_model: str = "Qwen/Qwen3-VL-Reranker-8B"
    llmrerank_input_mode: str = "text_with_object_images"
    llmrerank_batch_size: int = 8
    llmrerank_request_concurrency: int = 1
    llmrerank_request_timeout_seconds: float = 120.0
    llmrerank_max_retries: int = 3
    llmrerank_failure_fallback: str = "none"
    llmrerank_max_images_per_section: int = 4
    llmrerank_instruction_version: str = "v2"
    llmrerank_query_mode: str = "original"
    llmrerank_include_paper_identity: bool = False


def query_object_targets(query: str) -> dict[str, Any]:
    text = str(query or "")
    lowered = text.lower()

    def number(pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    reference_id = None
    for pattern in (r"\b(?:reference|ref\.)\s*\[?(\d{1,3})\]?", r"\b(\d{1,3})(?:st|nd|rd|th)\s+reference\b"):
        match = re.search(pattern, lowered)
        if match:
            reference_id = int(match.group(1))
            break
    return {
        "table": number(r"\btable\s*(\d+)"),
        "figure": number(r"\b(?:figure|fig\.)\s*(\d+)"),
        "equation_algorithm": number(r"\b(?:equation|eq\.|algorithm)\s*\(?(\d+)\)?"),
        "citation_context": reference_id,
        "last_reference": bool(re.search(r"\blast\s+reference\b|\bindex\s+of\s+the\s+last\s+reference\b", lowered)),
    }


def query_for_relevance_mode(query: str, mode: str) -> str:
    """Return a deterministic query projection for a separately audited route."""
    if mode == "original":
        return query
    if mode != "mask_method_aliases":
        raise ValueError("Unsupported relevance query mode.")
    protected = {"cifar", "imagenet", "geneval", "fid", "mcts", "llm", "vlm", "ocr"}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.lower() in protected or re.search(r"\d", token) else "[METHOD]"

    return re.sub(r"\b(?:[A-Z]{2,}(?:-[A-Z]+)?|[a-z]+[A-Z][A-Za-z]*)\b", replace, query)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or f"{record.get('paper_id')}::{record.get('page')}::{record.get('record_id')}")


def _record_order(record: dict[str, Any]) -> tuple[int, int, str]:
    try:
        page = int(record.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    try:
        order = int(record.get("document_order") or record.get("reading_order") or 0)
    except (TypeError, ValueError):
        order = 0
    return page, order, _record_id(record)


def _image_path(record: dict[str, Any]) -> str | None:
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


def _record_fingerprint(record: dict[str, Any]) -> str:
    return _hash({
        "paper_id": record.get("paper_id"), "page": record.get("page"), "section_id": record.get("section_id"),
        "record_type": record.get("record_type"), "source_type": record.get("source_type"), "label": record.get("label"),
        "locator": record.get("locator"), "text": record.get("text"), "image_path": _image_path(record),
    })


def _canonical_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    repeated_ids: set[str] = set()
    fingerprints: dict[str, set[str]] = defaultdict(set)
    conflicts: set[str] = set()
    for record in records:
        if to_official_source_type(record.get("record_type"), record.get("source_type")) not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            continue
        identifier = _record_id(record)
        fingerprint = _record_fingerprint(record)
        key = identifier, fingerprint
        if key in seen:
            repeated_ids.add(identifier)
            continue
        if fingerprints[identifier] and fingerprint not in fingerprints[identifier]:
            conflicts.add(identifier)
        seen.add(key)
        fingerprints[identifier].add(fingerprint)
        copied = dict(record)
        copied["_canonical_record_fingerprint"] = fingerprint
        unique.append(copied)
    return unique, {
        "input_record_count": len(records), "canonical_record_count": len(unique),
        "exact_duplicate_record_count": len(records) - len(unique),
        "repeated_global_record_id_count": len(repeated_ids), "conflicting_global_record_id_count": len(conflicts),
    }


def _artifact_fingerprint(root: Path, paper_id: str) -> dict[str, Any]:
    matches = sorted(root.glob(f"*/{paper_id}/artifact_status.json"))
    path = matches[0] if matches else root / paper_id / "artifact_status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"artifact_status": "missing_or_unreadable"}
    return {key: value.get(key) for key in ("artifact_version", "standardizer_version", "transcription_backend", "records_sha256")}


def _title_acronym(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(title or ""))
    ignored = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    initials = "".join(word[0].upper() for word in words if word.lower() not in ignored)
    return initials if 2 <= len(initials) <= 12 else ""


def _sections(records: list[dict[str, Any]], root: Path, paper_metadata: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("paper_id") or ""), str(record.get("section_id") or "sec_unassigned"))].append(record)
    result = []
    for (paper_id, section_id), values in grouped.items():
        ordered = sorted(values, key=_record_order)
        first = ordered[0]
        pages = [int(record.get("page") or 0) for record in ordered if record.get("page")]
        labels = list(dict.fromkeys(str(record.get("label") or "") for record in ordered if str(record.get("label") or "")))
        metadata = (paper_metadata or {}).get(paper_id) or {}
        title = str(metadata.get("title") or "")
        matched_aliases = metadata.get("matched_aliases") if isinstance(metadata.get("matched_aliases"), list) else []
        aliases = [str(alias) for alias in matched_aliases if str(alias) and not str(alias).startswith("topic:")]
        acronym = _title_acronym(title)
        if acronym:
            aliases.append(acronym)
        result.append({
            "paper_id": paper_id, "section_id": section_id,
            "paper_title": title,
            "paper_aliases": list(dict.fromkeys(aliases)),
            "section_title": str(first.get("section_title") or "Front Matter"),
            "section_type": str(first.get("section_type") or "unknown"),
            "section_path": first.get("section_path") or [str(first.get("section_title") or "Front Matter")],
            "page_start": min(pages) if pages else 0, "page_end": max(pages) if pages else 0,
            "object_labels": labels, "records": ordered, "artifact_fingerprint": _artifact_fingerprint(root, paper_id),
        })
    return sorted(result, key=lambda item: (item["paper_id"], item["page_start"], item["section_id"]))


def _prefix(section: dict[str, Any]) -> str:
    # This prefix is repeated in every Qwen pair. Keep only readable location
    # context that can affect relevance; provenance remains in the raw record
    # and downstream evidence ledger, not in reranker input.
    section_title = str(section.get("section_title") or "Document").strip()
    section_type = str(section.get("section_type") or "unknown").strip()
    identity = f"Paper: {section['paper_title']}\n" if section.get("paper_title") else ""
    aliases = section.get("paper_aliases") or []
    if aliases:
        identity += f"Method aliases: {', '.join(map(str, aliases[:4]))}\n"
    return f"{identity}Section: {section_title}\nSection type: {section_type}\nContent:\n"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _record_line(record: dict[str, Any]) -> str:
    source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
    text = str(record.get("text") or "").strip()
    if source_type == "text_span":
        return text
    label = str(record.get("label") or "").strip()
    header = f"[{source_type}{': ' + label if label else ''}]"
    return f"{header}\n{text}" if text else header


def _unit(prefix: str, records: list[dict[str, Any]], unit_type: str, anchor_ids: list[str] | None = None, image_paths: list[str] | None = None) -> dict[str, Any]:
    text = prefix + ("\n".join(_record_line(record) for record in records if str(record.get("text") or "").strip()) or "[empty unit]")
    return {
        "unit_type": unit_type, "text": text, "token_count": len(_tokens(text)), "content_token_start": 0,
        "content_token_end": max(0, len(_tokens(text)) - len(_tokens(prefix))),
        "record_ids": [_record_id(record) for record in records], "anchor_record_ids": list(anchor_ids or []),
        "image_paths": list(image_paths or []), "document_order": min((_record_order(record) for record in records), default=(0, 0, "")),
    }


def _record_aware_units(section: dict[str, Any], config: SectionRelevanceConfig, paper_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    prefix = _prefix(section)
    ordered = section["records"]
    text_records = [record for record in ordered if to_official_source_type(record.get("record_type"), record.get("source_type")) == "text_span"]
    target, maximum = max(32, config.unit_target_tokens), max(max(32, config.unit_target_tokens), config.unit_max_tokens)
    overlap = max(0, config.unit_overlap_records)
    units: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            units.append(_unit(prefix, current, "text_chunk"))
            current = current[-overlap:] if overlap else []

    for record in text_records:
        proposed = _unit(prefix, [*current, record], "text_chunk")
        if current and proposed["token_count"] > maximum:
            flush()
        standalone = _unit(prefix, [record], "text_chunk")
        if standalone["token_count"] > maximum:
            words = _tokens(str(record.get("text") or ""))
            capacity = max(1, maximum - len(_tokens(prefix)) - 8)
            for start in range(0, len(words), capacity):
                fragment = dict(record, text=" ".join(words[start:start + capacity]))
                split = _unit(prefix, [fragment], "text_fragment")
                split["oversized_record_split"] = True
                units.append(split)
            current = []
            continue
        current.append(record)
        if _unit(prefix, current, "text_chunk")["token_count"] >= target:
            flush()
    flush()

    if config.object_units_enabled:
        image_count = 0
        neighbors = max(0, config.object_neighbor_records)
        for index, record in enumerate(ordered):
            source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
            if source_type not in {"table", "figure", "equation_algorithm", "citation_context"}:
                continue
            before = [item for item in ordered[:index] if to_official_source_type(item.get("record_type"), item.get("source_type")) == "text_span"][-neighbors:]
            after = [item for item in ordered[index + 1:] if to_official_source_type(item.get("record_type"), item.get("source_type")) == "text_span"][:neighbors]
            narration = object_reference_paragraphs(record, paper_records or ordered, max(1, neighbors * 2))
            context = narration or [*before, *after]
            image = _image_path(record)
            images = []
            if config.llmrerank_input_mode == "text_with_object_images" and image and image_count < max(0, config.llmrerank_max_images_per_section):
                images, image_count = [image], image_count + 1
            object_unit = _unit(prefix, [record, *context], f"object_{source_type}", [_record_id(record)], images)
            object_unit["oversized_atomic_object"] = object_unit["token_count"] > maximum
            units.append(object_unit)
    units.sort(key=lambda item: (item["document_order"], 0 if item["unit_type"] == "text_chunk" else 1))
    for index, value in enumerate(units):
        value["chunk_index"] = index
        value.pop("document_order", None)
    return units


def _image_hash(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    try:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except OSError:
        return {"path": str(path), "sha256": None}


def _qwen_scores(query: str, units: list[dict[str, Any]], config: SectionRelevanceConfig, root: Path) -> tuple[list[float | None], list[dict[str, Any]], dict[str, Any]]:
    instruction = INSTRUCTIONS.get(config.llmrerank_instruction_version)
    if not instruction:
        raise ValueError(f"Unsupported LLMRERANK_INSTRUCTION_VERSION: {config.llmrerank_instruction_version}")
    cache_root = root / ".section_relevance_llmrerank_cache" / hashlib.sha256(query.encode("utf-8")).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    template_version = PAPER_IDENTITY_REPRESENTATION_TEMPLATE_VERSION if config.llmrerank_include_paper_identity else REPRESENTATION_TEMPLATE_VERSION
    cache_version = PAPER_IDENTITY_CACHE_VERSION if config.llmrerank_include_paper_identity else CACHE_VERSION
    projections: dict[str, dict[str, Any]] = {}
    unit_projections: list[list[str]] = []
    image_cache: dict[str, dict[str, Any]] = {}
    for unit in units:
        ids = []
        for modality, document, value in [("text", {"text": unit["text"]}, unit["text"])]:
            key = {"cache_version": cache_version, "projection_version": CANONICAL_PROJECTION_VERSION, "template_version": template_version, "model": config.llmrerank_model, "instruction_version": config.llmrerank_instruction_version, "query": query, "modality": modality, "value": value, "artifact": unit["artifact_fingerprint"]}
            identifier = _hash(key)
            projections.setdefault(identifier, {"modality": modality, "document": document, "cache_path": cache_root / f"{identifier}.json", "members": []})
            projections[identifier]["members"].append(len(unit_projections)); ids.append(identifier)
        for image_path in unit.get("image_paths") or []:
            fingerprint = image_cache.setdefault(str(image_path), _image_hash(str(image_path)))
            key = {"cache_version": cache_version, "projection_version": CANONICAL_PROJECTION_VERSION, "template_version": template_version, "model": config.llmrerank_model, "instruction_version": config.llmrerank_instruction_version, "query": query, "modality": "image", "value": fingerprint, "artifact": unit["artifact_fingerprint"]}
            identifier = _hash(key)
            projections.setdefault(identifier, {"modality": "image", "document": {"image_path": str(image_path)}, "cache_path": cache_root / f"{identifier}.json", "members": []})
            projections[identifier]["members"].append(len(unit_projections)); ids.append(identifier)
        unit_projections.append(ids)

    results: dict[str, dict[str, Any]] = {}
    missing = []
    for identifier, projection in projections.items():
        try:
            cached = json.loads(projection["cache_path"].read_text(encoding="utf-8"))
            if isinstance(cached.get("score"), (int, float)):
                results[identifier] = {**cached, "cache_hit": True}
                continue
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            pass
        missing.append(identifier)

    client = LLMRerankClient(api_key=config.llmrerank_api_key, base_url=config.llmrerank_base_url, model=config.llmrerank_model, timeout_seconds=config.llmrerank_request_timeout_seconds, max_retries=config.llmrerank_max_retries)
    work = [{"id": identifier, "document": projections[identifier]["document"]} for identifier in missing]
    batches = [work[start:start + max(1, config.llmrerank_batch_size)] for start in range(0, len(work), max(1, config.llmrerank_batch_size))]
    failures: dict[str, str] = {}
    started, api_calls, retries = time.monotonic(), 0, 0

    def score(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Exception | None]:
        try:
            return batch, client.score_documents(query=query, documents=[item["document"] for item in batch], instruction=instruction), None
        except Exception as exc:  # keep per-projection failures auditable
            return batch, None, exc

    completed = []
    if max(1, config.llmrerank_request_concurrency) == 1:
        completed = [score(batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=max(1, config.llmrerank_request_concurrency)) as pool:
            futures = [pool.submit(score, batch) for batch in batches]
            completed = [future.result() for future in as_completed(futures)]
    for batch, response, error in completed:
        api_calls += 1
        if error is not None:
            for item in batch:
                failures[item["id"]] = str(error)
            continue
        assert response is not None
        attempts = int(response.get("attempts") or 1)
        retries += max(0, attempts - 1)
        raw = response.get("raw_response") if isinstance(response.get("raw_response"), dict) else {}
        for item, value in zip(batch, response["scores"]):
            results[item["id"]] = {"score": float(value), "attempts": attempts, "api_response_id": raw.get("id"), "api_meta": raw.get("meta"), "cache_hit": False}
    for identifier in missing:
        result = results.get(identifier)
        if result is not None:
            projections[identifier]["cache_path"].write_text(json.dumps({key: value for key, value in result.items() if key != "cache_hit"}, ensure_ascii=False), encoding="utf-8")

    scores: list[float | None] = []
    details: list[dict[str, Any]] = []
    for projection_ids in unit_projections:
        values = [(identifier, results[identifier]) for identifier in projection_ids if identifier in results]
        details.append({
            "canonical_projection_ids": projection_ids,
            "status": "ok" if len(values) == len(projection_ids) else "failed" if not values else "partial",
            "error": "; ".join(failures[identifier] for identifier in projection_ids if identifier in failures) or None,
            "cache_hit": bool(values) and all(bool(value.get("cache_hit")) for _, value in values),
            "modality_scores": [{"modality": projections[identifier]["modality"], "score": value["score"], "canonical_projection_id": identifier} for identifier, value in values],
            "modality_pooling": "max",
        })
        scores.append(max((float(value["score"]) for _, value in values), default=None))
    stats = {
        "candidate_union_mode": "full_qwen", "instruction": instruction, "instruction_version": config.llmrerank_instruction_version,
        "api_calls": api_calls, "pair_count": len(work), "logical_unit_count": len(units), "scored_logical_unit_count": sum(value is not None for value in scores),
        "canonical_projection_version": CANONICAL_PROJECTION_VERSION, "canonical_projection_count": len(projections),
        "canonical_text_projection_count": sum(value["modality"] == "text" for value in projections.values()),
        "canonical_image_projection_count": sum(value["modality"] == "image" for value in projections.values()),
        "logical_text_pair_count": len(units),
        "logical_text_pair_tokens": sum(len(_tokens(str(unit.get("text") or ""))) for unit in units),
        "logical_text_pair_chars": sum(len(str(unit.get("text") or "")) for unit in units),
        "average_logical_text_pair_tokens": round(
            sum(len(_tokens(str(unit.get("text") or ""))) for unit in units) / max(1, len(units)), 3
        ),
        "cache_hits": sum(bool(item["cache_hit"]) for item in details), "projection_cache_hits": sum(bool(value.get("cache_hit")) for value in results.values()),
        "failed_pair_count": len(failures), "retry_count": retries, "elapsed_seconds": round(time.monotonic() - started, 6),
        "multimodal_strategy": "separate_text_and_image_pairs_max_pool",
    }
    return scores, details, stats


def _normalize(values: list[float | None]) -> list[float | None]:
    present = [value for value in values if value is not None]
    if not present:
        return [None] * len(values)
    low, high = min(present), max(present)
    if high == low:
        return [0.5 if value is not None else None for value in values]
    return [(float(value) - low) / (high - low) if value is not None else None for value in values]


def retrieve_section_relevance(query: str, candidate_records: list[dict[str, Any]], processed_root: str | Path, config: SectionRelevanceConfig, top_k_sections: int = 0, query_id: str | None = None, primary_evidence_type: str | None = None, candidate_paper_metadata: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if config.backend not in {"bm25", "llmrerank"}:
        raise ValueError("Only SECTION_RELEVANCE_BACKEND=bm25 or llmrerank is supported.")
    if config.unit_mode != "record_aware":
        raise ValueError("Only SECTION_RELEVANCE_UNIT_MODE=record_aware is supported.")
    if config.llmrerank_failure_fallback not in {"none", "bm25"}:
        raise ValueError("Only LLMRERANK_FAILURE_FALLBACK=none or bm25 is supported.")
    if config.llmrerank_input_mode not in {"text_only", "text_with_object_images"}:
        raise ValueError("Unsupported LLMRERANK_INPUT_MODE.")
    if config.llmrerank_query_mode not in {"original", "mask_method_aliases"}:
        raise ValueError("Unsupported LLMRERANK_QUERY_MODE.")
    canonical, dedup = _canonical_records(candidate_records)
    metadata_by_paper = {
        str(item.get("paper_id") or ""): item
        for item in candidate_paper_metadata or []
        if isinstance(item, dict) and str(item.get("paper_id") or "")
    } if config.llmrerank_include_paper_identity else {}
    sections = _sections(canonical, Path(processed_root), metadata_by_paper)
    records_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical:
        records_by_paper[str(record.get("paper_id") or "")].append(record)
    units: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        section_units = _record_aware_units(
            section,
            config,
            records_by_paper.get(str(section.get("paper_id") or ""), []),
        )
        for unit in section_units:
            unit["section_index"] = section_index
            unit["paper_id"] = section["paper_id"]
            unit["section_id"] = section["section_id"]
            unit["artifact_fingerprint"] = section["artifact_fingerprint"]
        section["chunks"] = section_units
        units.extend(section_units)
    retriever_pool: set[int] | None = None
    if config.retriever_pool_budget > 0:
        from .content_retriever import build_retriever_pool, hybrid_retriever_scores

        retriever_pool = build_retriever_pool(
            units,
            hybrid_retriever_scores(query, units, sections),
            config.retriever_pool_budget,
        )
    bm25_raw = [float(value) for value in BM25Okapi([tokenize(unit["text"]) for unit in units]).get_scores(tokenize(query))] if units else []
    qwen_raw: list[float | None] = [None] * len(units)
    calls: list[dict[str, Any]] = [{} for _ in units]
    stats: dict[str, Any] = {}
    scoring_query = query_for_relevance_mode(query, config.llmrerank_query_mode)
    if config.backend == "llmrerank":
        if retriever_pool is not None:
            ordered = sorted(retriever_pool)
            pool_scores, pool_calls, pool_stats = _qwen_scores(
                scoring_query, [units[index] for index in ordered], config, Path(processed_root)
            )
            for local_index, global_index in enumerate(ordered):
                qwen_raw[global_index] = pool_scores[local_index]
                calls[global_index] = pool_calls[local_index]
            stats = dict(pool_stats)
            stats["retriever_pool_budget"] = config.retriever_pool_budget
            stats["retriever_pool_unit_count"] = len(ordered)
            stats["retriever_excluded_unit_count"] = len(units) - len(ordered)
        else:
            qwen_raw, calls, stats = _qwen_scores(scoring_query, units, config, Path(processed_root))
    bm25_norm = _normalize(bm25_raw)
    qwen_norm = _normalize(qwen_raw)
    for index, unit in enumerate(units):
        if retriever_pool is not None and index not in retriever_pool:
            score, raw, backend = -1.0, bm25_raw[index], "retriever_excluded"
        elif config.backend == "llmrerank" and qwen_norm[index] is not None:
            score, raw, backend = float(qwen_norm[index]), float(qwen_raw[index]), "llmrerank"
        elif config.backend == "llmrerank" and config.llmrerank_failure_fallback == "bm25":
            score, raw, backend = float(bm25_norm[index] or 0.0), bm25_raw[index], "bm25_fallback"
        elif config.backend == "llmrerank":
            score, raw, backend = -1.0, bm25_raw[index], "unavailable"
        else:
            score, raw, backend = float(bm25_norm[index] or 0.0), bm25_raw[index], "bm25"
        unit["score_contract"] = {"local_relevance": round(score, 8), "raw_local_relevance": round(raw, 8), "local_backend": backend, "section_relevance": round(score, 8), "section_bonus": 0.0, "object_bonus": 0.0, "final_relevance": round(score, 8)}
    ranked = sorted(units, key=lambda item: (-float(item["score_contract"]["local_relevance"]), item["section_index"], item["chunk_index"]))
    for rank, unit in enumerate(ranked, start=1):
        unit["score_contract"]["local_rank"] = rank
        unit["score_contract"]["final_rank"] = rank
        unit["selected_for_retrieval"] = True
    trace_sections = []
    for section in sections:
        scores = [float(unit["score_contract"]["local_relevance"]) for unit in section["chunks"]]
        section_score = max(scores, default=-1.0)
        trace_sections.append({
            "paper_id": section["paper_id"], "section_id": section["section_id"], "section_title": section["section_title"], "section_type": section["section_type"], "section_path": section["section_path"], "page_start": section["page_start"], "page_end": section["page_end"],
            "record_ids": [_record_id(record) for record in section["records"]], "object_labels": section["object_labels"],
            "assessment": {"relevance": {"score": section_score, "backend": config.backend, "rank": 0}, "supportance": {"score": None, "status": "not_implemented"}, "sufficiency": {"score": None, "status": "not_implemented"}},
            "chunks": [{key: unit.get(key) for key in ("chunk_index", "unit_type", "text", "token_count", "content_token_start", "content_token_end", "record_ids", "anchor_record_ids", "image_paths", "oversized_record_split", "oversized_atomic_object", "score_contract", "selected_for_retrieval")} | {"bm25_raw_score": bm25_raw[units.index(unit)], "llmrerank_raw_score": qwen_raw[units.index(unit)], "llmrerank_call": calls[units.index(unit)]} for unit in section["chunks"]],
        })
    trace_sections.sort(key=lambda item: (-float(item["assessment"]["relevance"]["score"]), item["paper_id"], item["section_id"]))
    for rank, section in enumerate(trace_sections, start=1): section["assessment"]["relevance"]["rank"] = rank
    return {
        "query_id": query_id, "ranked_sections": sections, "selected_sections": sections[:top_k_sections] if top_k_sections > 0 else sections, "expanded_records": canonical,
        "trace": {
            "query_id": query_id, "query": query, "backend": config.backend, "template_version": REPRESENTATION_TEMPLATE_VERSION,
            "tokenizer_version": "regex_whitespace", "tokenizer_model": None, "top_k_sections": top_k_sections, "unit_mode": config.unit_mode,
            "unit_target_tokens": config.unit_target_tokens, "unit_max_tokens": config.unit_max_tokens, "unit_overlap_records": config.unit_overlap_records,
            "record_deduplication": dedup, "sections": trace_sections,
            "ranked_units": [{"paper_id": unit["paper_id"], "section_id": unit["section_id"], "chunk_index": unit["chunk_index"], "unit_type": unit["unit_type"], "record_ids": unit["record_ids"], "anchor_record_ids": unit["anchor_record_ids"], "score_contract": unit["score_contract"], "selected_for_retrieval": True} for unit in ranked],
            "candidate_union": {
                "mode": stats.get("candidate_union_mode", "bm25_all"),
                "logical_unit_count": len(units),
                "qwen_scored_unit_count": sum(value is not None for value in qwen_raw),
                "qwen_unscored_unit_count": sum(value is None for value in qwen_raw),
                "qwen_scored_record_ids": sorted({
                    record_id
                    for index, unit in enumerate(units)
                    if config.backend == "llmrerank" and qwen_raw[index] is not None
                    for record_id in unit["record_ids"]
                }),
                "qwen_unscored_record_ids": sorted({
                    record_id
                    for index, unit in enumerate(units)
                    if config.backend == "llmrerank" and qwen_raw[index] is None
                    for record_id in unit["record_ids"]
                }),
            },
            "llmrerank": {"model": config.llmrerank_model, "input_mode": config.llmrerank_input_mode, "query_mode": config.llmrerank_query_mode, "scoring_query": scoring_query, "failure_fallback": config.llmrerank_failure_fallback, **stats} if config.backend == "llmrerank" else None,
        },
    }


__all__ = ["SectionRelevanceConfig", "query_object_targets", "retrieve_section_relevance"]
