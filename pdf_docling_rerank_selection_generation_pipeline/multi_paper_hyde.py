"""Multi-paper HyDE claims used only as additional lexical routing queries."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES


PROMPT_VERSION = "v1_value_masked_claims"
SCHEMA_VERSION = "v1"
ALLOWED_SOURCE_TYPES = set(OFFICIAL_EVIDENCE_SOURCE_TYPES)
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_ -]{1,40}\]")


@dataclass(frozen=True)
class MultiPaperHyDEConfig:
    enabled: bool = False
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    max_claims: int = 4
    cache_enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 120.0


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value["claims"][: max(0, max_claims)]:
        if not isinstance(item, dict):
            warnings.append("non_object_claim_dropped")
            continue
        text = re.sub(r"\s+", " ", str(item.get("hypothetical_evidence") or "")).strip()
        if not text or not PLACEHOLDER_RE.search(text):
            warnings.append("invalid_claim_dropped")
            continue
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key in seen:
            warnings.append("duplicate_claim_dropped")
            continue
        seen.add(key)
        requested = item.get("expected_source_types") if isinstance(item.get("expected_source_types"), list) else []
        routed = [str(source) for source in requested if str(source) in ALLOWED_SOURCE_TYPES]
        if len(routed) != len(requested):
            warnings.append("unsupported_source_types_dropped")
        claims.append({
            "claim_id": f"claim_{len(claims) + 1}",
            "hypothetical_evidence": text,
            "expected_source_types": list(dict.fromkeys(routed or sorted(ALLOWED_SOURCE_TYPES))),
        })
    if not claims:
        raise ValueError("HyDE output contained no usable claims.")
    return claims, warnings


def _prompt(query: str, primary_evidence_type: str | None, max_claims: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "Generate answer-agnostic hypothetical scientific evidence claims for lexical retrieval routing. Return JSON only. "
            "Preserve every condition, comparison, and paper scope. Mask every unknown answer with [VALUE], [METHOD], "
            "[DATASET], [METRIC], or [PAPER]. Do not guess facts, paper names, section names, or object labels. "
            "expected_source_types may contain text_span, citation_context, table, figure, equation_algorithm."
        )},
        {"role": "user", "content": f"Question: {query}\nPrimary evidence hint: {primary_evidence_type or 'unknown'}\n"
         f"Generate at most {max_claims} claims as {{\"claims\":[{{\"hypothetical_evidence\":\"... [VALUE].\",\"expected_source_types\":[\"text_span\"]}}]}}."},
    ]


def _generation_cache_path(root: Path, *, query_id: str | None, query: str, candidate_papers: list[str], config: MultiPaperHyDEConfig) -> Path:
    return root / "generations" / f"{_stable_hash({'query_id': query_id, 'query': query, 'candidate_papers': candidate_papers, 'model': config.model, 'prompt_version': PROMPT_VERSION, 'schema_version': SCHEMA_VERSION, 'temperature': config.temperature, 'max_tokens': config.max_tokens, 'max_claims': config.max_claims})}.json"


def generate_claims(*, query_id: str | None, query: str, primary_evidence_type: str | None, candidate_papers: list[str], config: MultiPaperHyDEConfig, client: Any, cache_root: Path) -> tuple[str, list[dict[str, Any]], list[str], bool]:
    cache_path = _generation_cache_path(cache_root, query_id=query_id, query=query, candidate_papers=candidate_papers, config=config)
    warnings: list[str] = []
    if config.cache_enabled and cache_path.exists():
        try:
            raw = str(json.loads(cache_path.read_text(encoding="utf-8"))["raw_generation"])
            claims, cached_warnings = validate_claims(raw, config.max_claims)
            return raw, claims, cached_warnings, True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"generation_cache_invalid:{type(exc).__name__}")
    response = client.generate_prediction(_prompt(query, primary_evidence_type, config.max_claims), model=config.model, temperature=config.temperature, max_tokens=config.max_tokens, timeout_seconds=config.timeout_seconds)
    raw = str(response.get("content") or "")
    claims, validation_warnings = validate_claims(raw, config.max_claims)
    warnings.extend(validation_warnings)
    if config.cache_enabled:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION, "model": config.model, "raw_generation": raw, "parsed_claims": claims}, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.append(f"generation_cache_write_failed:{type(exc).__name__}")
    return raw, claims, warnings, False


__all__ = ["MultiPaperHyDEConfig", "generate_claims", "validate_claims"]
