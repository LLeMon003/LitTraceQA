"""Optional, removable HyDE fusion for multi-paper section relevance."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .section_relevance import SectionRelevanceConfig, _load_tokenizer, context_records_for_section, e5_score_queries, rank_and_select_units
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


PROMPT_VERSION = "v1_value_masked_claims"
SCHEMA_VERSION = "v1"
ALLOWED_SOURCE_TYPES = set(OFFICIAL_EVIDENCE_SOURCE_TYPES)
TEXT_SOURCE_TYPES = {"text_span", "citation_context"}
OBJECT_SOURCE_TYPES = {"table", "figure", "equation_algorithm"}
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_ -]{1,40}\]")


@dataclass(frozen=True)
class MultiPaperHyDEConfig:
    enabled: bool = False
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    original_weight: float = 0.7
    claim_weight: float = 0.3
    max_claims: int = 4
    cache_enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    unit_max_tokens: int = 448
    top_hits_per_claim: int = 20


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_claims(raw: str, max_claims: int) -> tuple[list[dict[str, Any]], list[str]]:
    value = _extract_json(raw)
    if not isinstance(value, dict) or not isinstance(value.get("claims"), list):
        raise ValueError("HyDE output must be an object containing a claims array.")
    warnings: list[str] = []
    raw_claims = value["claims"]
    if len(raw_claims) > max_claims:
        warnings.append(f"excessive_claim_count:{len(raw_claims)}>{max_claims}")
    claims: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for item in raw_claims[: max(0, max_claims)]:
        if not isinstance(item, dict):
            warnings.append("non_object_claim_dropped")
            continue
        text = re.sub(r"\s+", " ", str(item.get("hypothetical_evidence") or "")).strip()
        if not text:
            warnings.append("empty_claim_dropped")
            continue
        if not PLACEHOLDER_RE.search(text):
            warnings.append("claim_without_value_placeholder_dropped")
            continue
        dedupe_key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if dedupe_key in seen_text:
            warnings.append("duplicate_claim_dropped")
            continue
        seen_text.add(dedupe_key)
        requested = item.get("expected_source_types")
        requested = requested if isinstance(requested, list) else []
        unsupported = sorted({str(source) for source in requested} - ALLOWED_SOURCE_TYPES)
        if unsupported:
            warnings.append("unsupported_source_types:" + ",".join(unsupported))
        routed = [str(source) for source in requested if str(source) in ALLOWED_SOURCE_TYPES]
        if not routed:
            routed = sorted(ALLOWED_SOURCE_TYPES)
            warnings.append("source_type_route_rescued_to_all")
        claims.append({
            "claim_id": f"claim_{len(claims) + 1}",
            "hypothetical_evidence": text,
            "expected_source_types": list(dict.fromkeys(routed)),
        })
    if not claims:
        raise ValueError("HyDE output contained no usable claims.")
    return claims, warnings


def _prompt(query: str, primary_evidence_type: str | None, max_claims: int) -> list[dict[str, str]]:
    schema = {
        "claims": [
            {
                "claim_id": "claim_1",
                "hypothetical_evidence": "The reported result for the requested condition is [VALUE].",
                "expected_source_types": ["text_span", "table"],
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate answer-agnostic hypothetical scientific evidence claims for retrieval. Return JSON only. "
                "Preserve every condition, comparison, and paper scope in the question. Mask every unknown answer "
                "with an uppercase placeholder such as [VALUE], [METHOD], [DATASET], [METRIC], or [PAPER]. "
                "Do not guess answers, paper names, section names, object labels, locators, table numbers, figure "
                "numbers, equation numbers, or facts not stated in the question. expected_source_types must contain "
                "one or more of: text_span, citation_context, table, figure, equation_algorithm."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\nPrimary evidence hint: {primary_evidence_type or 'unknown'}\n"
                f"Generate at most {max_claims} distinct claims using this schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            ),
        },
    ]


def _generation_cache_path(
    root: Path,
    *,
    query_id: str | None,
    query: str,
    candidate_papers: list[str],
    config: MultiPaperHyDEConfig,
) -> Path:
    key = {
        "query_id": query_id,
        "query": query,
        "candidate_papers": candidate_papers,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "max_claims": config.max_claims,
    }
    return root / "generations" / f"{_stable_hash(key)}.json"


def generate_claims(
    *,
    query_id: str | None,
    query: str,
    primary_evidence_type: str | None,
    candidate_papers: list[str],
    config: MultiPaperHyDEConfig,
    client: Any,
    cache_root: Path,
) -> tuple[str, list[dict[str, Any]], list[str], bool]:
    cache_path = _generation_cache_path(
        cache_root,
        query_id=query_id,
        query=query,
        candidate_papers=candidate_papers,
        config=config,
    )
    warnings: list[str] = []
    if config.cache_enabled and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw = str(cached["raw_generation"])
            claims, validation_warnings = validate_claims(raw, config.max_claims)
            return raw, claims, validation_warnings, True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"generation_cache_invalid:{type(exc).__name__}")
    response = client.generate_prediction(
        _prompt(query, primary_evidence_type, config.max_claims),
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
    )
    raw = str(response.get("content") or "")
    claims, validation_warnings = validate_claims(raw, config.max_claims)
    warnings.extend(validation_warnings)
    if config.cache_enabled:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "schema_version": SCHEMA_VERSION,
                        "model": config.model,
                        "raw_generation": raw,
                        "parsed_claims": claims,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.append(f"generation_cache_write_failed:{type(exc).__name__}")
    return raw, claims, warnings, False


def _record_id(record: dict[str, Any]) -> str:
    return str(
        record.get("global_record_id")
        or record.get("record_id")
        or record.get("id")
        or record.get("symbolic_id")
        or ""
    )


def _truncate_projection(text: str, tokenizer: Any, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def build_retrieval_units(
    sections: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    max_tokens = min(512, max(32, int(max_tokens)))
    for section in sections:
        paper_id, section_id = section["paper_id"], section["section_id"]
        prefix = f"Paper: {paper_id}\nSection: {section['section_title']}\n"
        text_parts: list[tuple[str, str, int]] = []

        def flush_text() -> None:
            if not text_parts:
                return
            record_ids = [item[0] for item in text_parts]
            source_types = sorted({item[1] for item in text_parts})
            text = prefix + "\n".join(item[2] for item in text_parts)
            units.append(
                {
                    "unit_id": f"{paper_id}::{section_id}::text_{len(units)}",
                    "paper_id": paper_id,
                    "section_id": section_id,
                    "source_types": source_types,
                    "record_ids": record_ids,
                    "page": min((int(record.get("page") or 0) for record in section["records"] if _record_id(record) in record_ids), default=0),
                    "text": text,
                    "artifact_fingerprint": section["artifact_fingerprint"],
                }
            )
            text_parts.clear()

        for record in section["records"]:
            source_type = to_official_source_type(record.get("record_type"), record.get("source_type"))
            text = re.sub(r"\s+", " ", str(record.get("text") or "")).strip()
            if source_type not in ALLOWED_SOURCE_TYPES or not text:
                continue
            record_id = _record_id(record)
            if source_type in OBJECT_SOURCE_TYPES:
                flush_text()
                label = str(record.get("label") or "").strip()
                projection = _truncate_projection(
                    f"{prefix}Source type: {source_type}\nObject: {label or 'unlabeled'}\nContent: {text}",
                    tokenizer,
                    max_tokens,
                )
                units.append(
                    {
                        "unit_id": f"{paper_id}::{section_id}::object_{record_id}",
                        "paper_id": paper_id,
                        "section_id": section_id,
                        "source_types": [source_type],
                        "record_ids": [record_id],
                        "page": int(record.get("page") or 0),
                        "label": label,
                        "text": projection,
                        "artifact_fingerprint": section["artifact_fingerprint"],
                    }
                )
                continue
            projection = f"[{record_id}] {text}"
            candidate = prefix + "\n".join([item[2] for item in text_parts] + [projection])
            if text_parts and len(tokenizer.encode(candidate, add_special_tokens=False)) > max_tokens:
                flush_text()
            text_parts.append((record_id, source_type, _truncate_projection(projection, tokenizer, max_tokens - 8)))
        flush_text()
    return units


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError("Cannot normalize non-finite scores.")
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def fuse_scores(
    original_scores: list[float],
    hyde_scores: list[float],
    original_weight: float,
    claim_weight: float,
) -> tuple[list[float], list[float], list[float]]:
    if len(original_scores) != len(hyde_scores):
        raise ValueError("Original and HyDE score lengths differ.")
    if original_weight < 0 or claim_weight < 0 or original_weight + claim_weight <= 0:
        raise ValueError("Fusion weights must be non-negative and not both zero.")
    original_norm = minmax_normalize(original_scores)
    hyde_norm = minmax_normalize(hyde_scores)
    total = original_weight + claim_weight
    fused = [
        (original_weight * original + claim_weight * hyde) / total
        for original, hyde in zip(original_norm, hyde_norm)
    ]
    return original_norm, hyde_norm, fused


def _fallback_audit(
    *,
    query_id: str | None,
    config: MultiPaperHyDEConfig,
    selection_budget: int,
    stage: str,
    reason: str,
    exc: Exception | None = None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "task_family": "multi_paper",
        "hyde_enabled": True,
        "hyde_model": config.model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "raw_generation": None,
        "parsed_claims": [],
        "validation_warnings": [],
        "original_qwen_section_scores": [],
        "per_claim_retrieval_hits": [],
        "hyde_section_scores": [],
        "normalization_trace": {},
        "fusion_weights": {"original": config.original_weight, "hyde_claim": config.claim_weight},
        "fused_section_scores": [],
        "baseline_rank_vs_fused_rank": [],
        "selected_sections": [],
        "selection_budget": selection_budget,
        "fallback_used": True,
        "fallback_stage": stage,
        "fallback_reason": reason,
        "exception_type": type(exc).__name__ if exc else None,
    }


def apply_multi_paper_hyde(
    *,
    relevance: dict[str, Any],
    query: str,
    query_id: str | None,
    primary_evidence_type: str | None,
    processed_root: str | Path,
    section_config: SectionRelevanceConfig,
    hyde_config: MultiPaperHyDEConfig,
    client: Any,
    selection_budget: int,
) -> dict[str, Any]:
    trace = relevance["trace"]
    if not hyde_config.enabled:
        return relevance
    if section_config.backend != "llmrerank":
        trace["hyde"] = _fallback_audit(
            query_id=query_id,
            config=hyde_config,
            selection_budget=selection_budget,
            stage="precondition",
            reason="HyDE fusion requires SECTION_RELEVANCE_BACKEND=llmrerank.",
        )
        return relevance
    baseline_sections = list(relevance["ranked_sections"])
    baseline_selected_ids = {
        f"{section['paper_id']}::{section['section_id']}" for section in relevance["selected_sections"]
    }
    cache_root = Path(processed_root) / ".multi_paper_hyde_cache"
    try:
        candidate_papers = sorted({str(section["paper_id"]) for section in baseline_sections})
        raw, claims, warnings, generation_cache_hit = generate_claims(
            query_id=query_id,
            query=query,
            primary_evidence_type=primary_evidence_type,
            candidate_papers=candidate_papers,
            config=hyde_config,
            client=client,
            cache_root=cache_root,
        )
    except Exception as exc:
        trace["hyde"] = _fallback_audit(
            query_id=query_id,
            config=hyde_config,
            selection_budget=selection_budget,
            stage="generation_or_validation",
            reason=str(exc),
            exc=exc,
        )
        return relevance
    try:
        tokenizer = _load_tokenizer(section_config.e5_model)
        units = build_retrieval_units(baseline_sections, tokenizer, hyde_config.unit_max_tokens)
        if not units:
            raise ValueError("No HyDE retrieval units were constructed.")
        scoring_chunks = [
            {"text": unit["text"], "artifact_fingerprint": unit["artifact_fingerprint"]}
            for unit in units
        ]
        claim_matrix, embedding_cache_hits, tokenizer_version = e5_score_queries(
            [claim["hypothetical_evidence"] for claim in claims],
            scoring_chunks,
            SectionRelevanceConfig(
                e5_model=section_config.e5_model,
                chunk_max_tokens=hyde_config.unit_max_tokens,
            ),
            cache_root / "e5_embeddings",
        )
        section_claim_scores: dict[str, list[float]] = {
            f"{section['paper_id']}::{section['section_id']}": [] for section in baseline_sections
        }
        best_hyde_by_section: dict[str, dict[str, Any]] = {}
        per_claim_hits: list[dict[str, Any]] = []
        routing_rescues = 0
        claim_unit_pairs = 0
        for claim, raw_scores in zip(claims, claim_matrix):
            routed = [
                index for index, unit in enumerate(units)
                if set(unit["source_types"]).intersection(claim["expected_source_types"])
            ]
            if not routed:
                routed = list(range(len(units)))
                routing_rescues += 1
                warnings.append(f"{claim['claim_id']}:no_routed_candidates_rescued_to_all")
            claim_unit_pairs += len(routed)
            routed_norm = minmax_normalize([float(raw_scores[index]) for index in routed])
            best_by_section: dict[str, tuple[float, int, float]] = {}
            for unit_index, normalized in zip(routed, routed_norm):
                unit = units[unit_index]
                section_key = f"{unit['paper_id']}::{unit['section_id']}"
                current = best_by_section.get(section_key)
                if current is None or normalized > current[0]:
                    best_by_section[section_key] = (normalized, unit_index, float(raw_scores[unit_index]))
            for section_key in section_claim_scores:
                section_claim_scores[section_key].append(best_by_section.get(section_key, (0.0, -1, 0.0))[0])
            for section_key, value in best_by_section.items():
                unit = units[value[1]]
                previous = best_hyde_by_section.get(section_key)
                if previous is None or value[0] > previous["normalized_claim_score"]:
                    best_hyde_by_section[section_key] = {
                        "best_claim_id": claim["claim_id"],
                        "best_source_types": unit["source_types"],
                        "best_retrieval_unit_id": unit["unit_id"],
                        "best_record_ids": unit["record_ids"],
                        "raw_unit_score": value[2],
                        "normalized_claim_score": value[0],
                    }
            ranked_hits = sorted(best_by_section.items(), key=lambda item: (-item[1][0], item[0]))
            per_claim_hits.append(
                {
                    "claim_id": claim["claim_id"],
                    "expected_source_types": claim["expected_source_types"],
                    "routed_unit_count": len(routed),
                    "hits": [
                        {
                            "section_id": section_key,
                            "unit_id": units[value[1]]["unit_id"],
                            "source_types": units[value[1]]["source_types"],
                            "record_ids": units[value[1]]["record_ids"],
                            "raw_unit_score": value[2],
                            "normalized_claim_score": value[0],
                        }
                        for section_key, value in ranked_hits[: hyde_config.top_hits_per_claim]
                    ],
                }
            )
        original_scores = [float(section["assessment"]["relevance"]["score"]) for section in baseline_sections]
        hyde_scores = [
            max(section_claim_scores[f"{section['paper_id']}::{section['section_id']}"], default=0.0)
            for section in baseline_sections
        ]
        original_norm, hyde_norm, fused = fuse_scores(
            original_scores,
            hyde_scores,
            hyde_config.original_weight,
            hyde_config.claim_weight,
        )
        score_rows: list[dict[str, Any]] = []
        for section, original_score, original_normalized, hyde_score, hyde_normalized, fused_score in zip(
            baseline_sections, original_scores, original_norm, hyde_scores, hyde_norm, fused
        ):
            relevance_assessment = section["assessment"]["relevance"]
            original_rank = int(relevance_assessment["rank"])
            relevance_assessment.update(
                {
                    "original_qwen_score": original_score,
                    "original_qwen_normalized_score": original_normalized,
                    "original_qwen_rank": original_rank,
                    "hyde_score": hyde_score,
                    "hyde_normalized_score": hyde_normalized,
                    "fused_score": fused_score,
                    "score": fused_score,
                    "backend": "llmrerank+hyde_e5",
                }
            )
            score_rows.append(
                {
                    "section_id": f"{section['paper_id']}::{section['section_id']}",
                    "original_qwen_score": original_score,
                    "original_qwen_normalized_score": original_normalized,
                    "original_qwen_rank": original_rank,
                    "hyde_score": hyde_score,
                    "hyde_normalized_score": hyde_normalized,
                    "fused_score": fused_score,
                }
            )
        baseline_sections.sort(
            key=lambda section: (
                -float(section["assessment"]["relevance"]["fused_score"]),
                int(section["assessment"]["relevance"]["original_qwen_rank"]),
            )
        )
        for rank, section in enumerate(baseline_sections, start=1):
            section["assessment"]["relevance"]["rank"] = rank
        unit_budget_active = section_config.unit_mode == "record_aware" and section_config.retrieval_unit_top_k > 0
        ranked_units, unit_selected_sections = (
            rank_and_select_units(baseline_sections, section_config) if unit_budget_active else ([], baseline_sections)
        )
        selected_sections = (
            unit_selected_sections
            if unit_budget_active
            else baseline_sections[: max(0, selection_budget)] if selection_budget > 0 else baseline_sections
        )
        selected_ids = {f"{section['paper_id']}::{section['section_id']}" for section in selected_sections}
        expanded_records: list[dict[str, Any]] = []
        for section in selected_sections:
            expanded_records.extend(context_records_for_section(section, section_config))
        rank_by_id = {
            f"{section['paper_id']}::{section['section_id']}": int(section["assessment"]["relevance"]["rank"])
            for section in baseline_sections
        }
        score_by_id = {row["section_id"]: row for row in score_rows}
        trace_sections = {f"{section['paper_id']}::{section['section_id']}": section for section in trace["sections"]}
        expanded_count_by_id = {
            f"{section['paper_id']}::{section['section_id']}": len(
                context_records_for_section(section, section_config)
            )
            for section in selected_sections
        }
        if set(score_by_id) != set(trace_sections):
            raise ValueError("HyDE fusion section scope differs from baseline trace scope.")
        for section_id, trace_section in trace_sections.items():
            row = score_by_id[section_id]
            if unit_budget_active:
                section = next(item for item in baseline_sections if f"{item['paper_id']}::{item['section_id']}" == section_id)
                current_chunks = {int(chunk["chunk_index"]): chunk for chunk in section["chunks"]}
                for trace_chunk in trace_section["chunks"]:
                    current = current_chunks[int(trace_chunk["chunk_index"])]
                    trace_chunk["score_contract"] = current["score_contract"]
                    trace_chunk["selected_for_context"] = int(trace_chunk["chunk_index"]) in section["_context_chunk_indexes"]
                    trace_chunk["selected_for_retrieval"] = bool(current.get("selected_for_retrieval"))
            trace_section["baseline_selected"] = section_id in baseline_selected_ids
            trace_section["selected"] = section_id in selected_ids
            trace_section["expanded_record_count"] = expanded_count_by_id.get(section_id, 0)
            trace_section["assessment"]["relevance"].update(
                {
                    **row,
                    "rank": rank_by_id[section_id],
                    "score": row["fused_score"],
                    "backend": "llmrerank+hyde_e5",
                }
            )
        if unit_budget_active:
            trace["ranked_units"] = [
                {
                    "paper_id": chunk["paper_id"],
                    "section_id": chunk["section_id"],
                    "chunk_index": chunk["chunk_index"],
                    "unit_type": chunk.get("unit_type", "token_chunk"),
                    "record_ids": chunk.get("record_ids", []),
                    "anchor_record_ids": chunk.get("anchor_record_ids", []),
                    "score_contract": chunk["score_contract"],
                    "selected_for_retrieval": bool(chunk.get("selected_for_retrieval")),
                }
                for chunk in ranked_units
            ]
        audit = {
            "query_id": query_id,
            "task_family": "multi_paper",
            "hyde_enabled": True,
            "hyde_model": hyde_config.model,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "raw_generation": raw,
            "parsed_claims": claims,
            "validation_warnings": warnings,
            "original_qwen_section_scores": [
                {key: row[key] for key in ("section_id", "original_qwen_score", "original_qwen_rank")}
                for row in score_rows
            ],
            "per_claim_retrieval_hits": per_claim_hits,
            "hyde_section_scores": [
                {
                    **{key: row[key] for key in ("section_id", "hyde_score", "hyde_normalized_score")},
                    **best_hyde_by_section.get(row["section_id"], {}),
                }
                for row in score_rows
            ],
            "normalization_trace": {"original": "minmax_all_sections", "claim_units": "minmax_per_claim_routed_units", "hyde_sections": "minmax_all_sections"},
            "fusion_weights": {"original": hyde_config.original_weight, "hyde_claim": hyde_config.claim_weight},
            "fused_section_scores": score_rows,
            "baseline_rank_vs_fused_rank": [
                {
                    "section_id": row["section_id"],
                    "baseline_rank": row["original_qwen_rank"],
                    "fused_rank": rank_by_id[row["section_id"]],
                    "rank_delta": row["original_qwen_rank"] - rank_by_id[row["section_id"]],
                }
                for row in score_rows
            ],
            "selected_sections": [
                {
                    "section_id": section_id,
                    "fused_rank": rank_by_id[section_id],
                    "selection_reason": "top_units_with_fused_section_bonus" if section_config.unit_mode == "record_aware" else "top_k_fused_relevance",
                    **{key: score_by_id[section_id][key] for key in ("original_qwen_normalized_score", "hyde_normalized_score", "fused_score")},
                }
                for section_id in sorted(selected_ids, key=rank_by_id.get)
            ],
            "selection_budget": selection_budget,
            "retrieval_unit_count": len(units),
            "claim_unit_pair_count": claim_unit_pairs,
            "document_embedding_cache_hits": sum(embedding_cache_hits),
            "e5_tokenizer_version": tokenizer_version,
            "generation_cache_hit": generation_cache_hit,
            "routing_rescue_count": routing_rescues,
            "newly_selected_vs_baseline": sorted(selected_ids - baseline_selected_ids),
            "displaced_baseline_sections": sorted(baseline_selected_ids - selected_ids),
            "fallback_used": False,
            "fallback_stage": None,
            "fallback_reason": None,
            "exception_type": None,
        }
        relevance.update(
            {
                "ranked_sections": baseline_sections,
                "selected_sections": selected_sections,
                "expanded_records": expanded_records,
            }
        )
        trace["hyde"] = audit
        trace["selected_section_ids"] = sorted(selected_ids, key=rank_by_id.get)
        trace["expanded_record_ids"] = [_record_id(record) for record in expanded_records]
        trace["expanded_record_count"] = len(expanded_records)
        trace["expanded_char_count"] = sum(len(str(record.get("text") or "")) for record in expanded_records)
        return relevance
    except Exception as exc:
        trace["hyde"] = _fallback_audit(
            query_id=query_id,
            config=hyde_config,
            selection_budget=selection_budget,
            stage="retrieval_or_fusion",
            reason=str(exc),
            exc=exc,
        )
        return relevance


__all__ = [
    "MultiPaperHyDEConfig",
    "apply_multi_paper_hyde",
    "build_retrieval_units",
    "fuse_scores",
    "minmax_normalize",
    "validate_claims",
]
