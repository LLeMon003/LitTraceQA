"""Candidate-metadata evidence planning for multi-paper package routing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES


PROMPT_VERSION = "v3_compact_evidence_plan"
ALLOWED_SOURCE_TYPES = set(OFFICIAL_EVIDENCE_SOURCE_TYPES)


@dataclass(frozen=True)
class PaperConditionedClaimsConfig:
    enabled: bool = False
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    max_papers: int = 12
    cache_enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 1800
    timeout_seconds: float = 120.0


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _extract_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("paper-conditioned claim response must be an object")
    return value


def _prompt(query: str, primary_evidence_type: str | None, papers: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = [
        {
            "paper_id": str(paper.get("paper_id") or ""),
            "title": str(paper.get("title") or ""),
            "abstract": str(paper.get("abstract") or "")[:1200],
        }
        for paper in papers
    ]
    return [
        {"role": "system", "content": (
            "Plan evidence retrieval, not an answer. From the candidate metadata, identify only papers likely needed and "
            "the evidence type to look for. Preserve named methods, datasets, metrics, settings, and comparisons. "
            "Do not invent values or facts. Return JSON only."
        )},
        {"role": "user", "content": (
            f"Question: {query}\nPrimary evidence type: {primary_evidence_type or 'unknown'}\n"
            f"Candidate papers: {json.dumps(compact, ensure_ascii=False)}\n"
            "Return {\"cross_paper\":true|false,\"plans\":[{\"paper_id\":\"...\","
            "\"source_types\":[\"table\"],\"retrieval_query\":\"concise evidence terms\"}]}."
        )},
    ]


def generate_evidence_plan(
    *, query_id: str | None, query: str, primary_evidence_type: str | None, candidate_papers: list[dict[str, Any]],
    config: PaperConditionedClaimsConfig, client: Any, cache_root: Path,
) -> tuple[dict[str, Any], list[str], bool]:
    papers = [dict(paper) for paper in candidate_papers[: max(0, config.max_papers)] if paper.get("paper_id")]
    key = {
        "prompt_version": PROMPT_VERSION, "query_id": query_id, "query": query,
        "primary_evidence_type": primary_evidence_type, "papers": [
            {field: paper.get(field) for field in ("paper_id", "title", "abstract")} for paper in papers
        ], "model": config.model, "temperature": config.temperature, "max_tokens": config.max_tokens,
    }
    path = cache_root / "evidence_planning" / f"{_hash(key)}.json"
    warnings: list[str] = []
    raw = ""
    if config.cache_enabled and path.exists():
        try:
            raw = str(json.loads(path.read_text(encoding="utf-8"))["raw_generation"])
            cached = True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            cached = False
    else:
        cached = False
    if not raw:
        response = client.generate_prediction(
            _prompt(query, primary_evidence_type, papers), model=config.model, temperature=config.temperature,
            max_tokens=config.max_tokens, timeout_seconds=config.timeout_seconds,
        )
        raw = str(response.get("content") or "")
    value = _extract_json(raw)
    allowed = {str(paper.get("paper_id") or "") for paper in papers}
    plans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value.get("plans") or []:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        text = re.sub(r"\s+", " ", str(item.get("retrieval_query") or "")).strip()
        source_types = [str(source) for source in item.get("source_types") or [] if str(source) in ALLOWED_SOURCE_TYPES]
        if paper_id not in allowed or not text or paper_id in seen:
            continue
        seen.add(paper_id)
        plans.append({"paper_id": paper_id, "source_types": list(dict.fromkeys(source_types)), "retrieval_query": text})
    missing = allowed - seen
    for paper_id in sorted(missing):
        warnings.append(f"missing_plan:{paper_id}")
    if not plans:
        raise ValueError("evidence plan contained no usable routes")
    plan = {"cross_paper": bool(value.get("cross_paper")), "plans": plans}
    if config.cache_enabled and not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "raw_generation": raw, "plan": plan}, ensure_ascii=False), encoding="utf-8")
    return plan, warnings, cached


def generate_paper_conditioned_claims(**kwargs: Any) -> tuple[list[dict[str, str]], list[str], bool]:
    """Compatibility wrapper for callers that only need paper-local routes."""
    plan, warnings, cached = generate_evidence_plan(**kwargs)
    claims = [
        {"paper_id": item["paper_id"], "hypothetical_evidence": item["retrieval_query"]}
        for item in plan["plans"]
    ]
    return claims, warnings, cached
