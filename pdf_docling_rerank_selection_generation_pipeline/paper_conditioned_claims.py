"""Candidate-paper-conditioned lexical claims for multi-paper routing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROMPT_VERSION = "v1_candidate_local_evidence"


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
            "You create lexical retrieval queries for local evidence inside each candidate scientific paper. "
            "A question can name one method while correct evidence comes from comparison baselines or related papers. "
            "For every provided paper, write one concise paper-local claim describing the fact to retrieve from that "
            "paper to help answer the question. Do not guess facts. Preserve requested dataset, metric, setting, and "
            "property, but replace unknown values with [VALUE]. Return JSON only: "
            '{"claims":[{"paper_id":"...","hypothetical_evidence":"... [VALUE]"}]}.'
        )},
        {"role": "user", "content": (
            f"Question: {query}\nPrimary evidence type: {primary_evidence_type or 'unknown'}\n"
            f"Candidate papers: {json.dumps(compact, ensure_ascii=False)}"
        )},
    ]


def generate_paper_conditioned_claims(
    *, query_id: str | None, query: str, primary_evidence_type: str | None, candidate_papers: list[dict[str, Any]],
    config: PaperConditionedClaimsConfig, client: Any, cache_root: Path,
) -> tuple[list[dict[str, str]], list[str], bool]:
    papers = [dict(paper) for paper in candidate_papers[: max(0, config.max_papers)] if paper.get("paper_id")]
    key = {
        "prompt_version": PROMPT_VERSION, "query_id": query_id, "query": query,
        "primary_evidence_type": primary_evidence_type, "papers": [
            {field: paper.get(field) for field in ("paper_id", "title", "abstract")} for paper in papers
        ], "model": config.model, "temperature": config.temperature, "max_tokens": config.max_tokens,
    }
    path = cache_root / "paper_conditioned" / f"{_hash(key)}.json"
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
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value.get("claims") or []:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        text = re.sub(r"\s+", " ", str(item.get("hypothetical_evidence") or "")).strip()
        if paper_id not in allowed or not text or paper_id in seen:
            continue
        seen.add(paper_id)
        claims.append({"paper_id": paper_id, "hypothetical_evidence": text})
    missing = allowed - seen
    for paper_id in sorted(missing):
        warnings.append(f"missing_claim:{paper_id}")
    if not claims:
        raise ValueError("paper-conditioned claims contained no usable routes")
    if config.cache_enabled and not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "raw_generation": raw, "claims": claims}, ensure_ascii=False), encoding="utf-8")
    return claims, warnings, cached
