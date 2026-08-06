"""Add verified, query-aware visual L2 cards to a frozen evidence hierarchy.

The input hierarchy is never reranked or rewritten at L0/L1.  Each accepted
card retains the figure's L0 support_ref and crop hash, while only the compact
proposition is exposed to the final keyed generation prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .data_io import find_official_file, read_jsonl, write_jsonl
from .parser import extract_json_object
from .vlm_answer_client import VLMAnswerClient


_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:%|x|×|[A-Za-z]+)?")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]{1,}|[A-Z]{2,}[A-Z0-9_-]*)\b")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create crop-verified visual L2 cards from a frozen hierarchy.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--only-query-ids", default="")
    parser.add_argument("--max-per-query", type=int, default=None)
    parser.add_argument("--max-per-paper", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _crop_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    return text if len(text) <= limit else text[: max(1, limit - 3)].rstrip() + "..."


def select_visual_candidates(hierarchy: dict[str, Any], max_per_query: int, max_per_paper: int) -> list[dict[str, Any]]:
    """Choose selected figure anchors only, deduplicated by crop content."""
    refs = {str(value) for value in hierarchy.get("selected_anchor_refs") or []}
    rows = [
        row for row in hierarchy.get("l0_catalog") or []
        if isinstance(row, dict)
        and str(row.get("evidence_ref") or "") in refs
        and str(row.get("source_type") or "") == "figure"
        and str(row.get("crop_path") or "").strip()
    ]
    rows.sort(key=lambda row: (int(row.get("selection_rank") or 10**8), int(row.get("page") or 10**8), str(row.get("evidence_ref") or "")))
    selected: list[dict[str, Any]] = []
    by_paper: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        path = Path(str(row.get("crop_path") or ""))
        if not path.is_file() or by_paper[paper_id] >= max(0, max_per_paper):
            continue
        try:
            digest = _crop_hash(path)
        except OSError:
            continue
        if digest in seen_hashes:
            continue
        selected.append({**row, "_crop_hash": digest})
        by_paper[paper_id] += 1
        seen_hashes.add(digest)
        if len(selected) >= max(0, max_per_query):
            break
    return selected


def _observation_messages(question: str, record: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "object_label": record.get("label"),
        "caption_hint": _clip(record.get("text"), 420),
    }
    system = (
        "Create one compact literal observation from this figure crop. Use visible text and unambiguous relationships only; "
        "do not infer purpose, performance, comparison, causality, or properties. The caption only disambiguates visible text. Return JSON only."
    )
    user = (
        "Return {\"proposition\":\"one <=320-character literal visual observation\",\"visible_strings\":[\"exact text visibly present\"],"
        "\"entities\":[\"visible entity\"],\"values\":[\"visible value\"],\"conditions\":[\"visible qualifier\"]}.\nINPUT:"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _verification_messages(question: str, record: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "proposition": _clip(observation.get("proposition"), 360),
        "visible_strings": observation.get("visible_strings") or [],
        "entities": observation.get("entities") or [],
        "values": observation.get("values") or [],
        "conditions": observation.get("conditions") or [],
    }
    system = (
        "Approve only when every stated entity, number, relation, and condition is directly visible in this crop. "
        "Do not use outside knowledge or infer missing text. Return JSON only."
    )
    user = "Return {\"supported\":true|false,\"reason\":\"brief\"}.\nINPUT:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _visual_card(record: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any] | None:
    proposition = _clip(observation.get("proposition"), 360)
    if len(proposition) < 4:
        return None
    entities = [str(value) for value in observation.get("entities") or [] if str(value).strip()][:8]
    values = [str(value) for value in observation.get("values") or [] if str(value).strip()][:10]
    conditions = [str(value) for value in observation.get("conditions") or [] if str(value).strip()][:6]
    # Do not give the next generation stage untraceable metadata: every visual
    # fact stays bound to exactly one L0 crop and its immutable content hash.
    return {
        "card_id": "V_PENDING",
        "claim_ids": ["Q01"],
        "proposition": proposition,
        "entities": entities or _ENTITY_RE.findall(proposition)[:8],
        "values": values or _NUMBER_RE.findall(proposition)[:10],
        "conditions": conditions,
        "paper_id": record.get("paper_id"),
        "source_type": "figure",
        "locator": record.get("locator"),
        "support_refs": [record.get("evidence_ref")],
        "support_quotes": [],
        "l1_refs": [],
        "verification": {
            "status": "visual_verified",
            "crop_sha256": record.get("_crop_hash"),
            "verifier": "independent_vlm_crop_check",
        },
    }


def _json_from_crop_call(client: VLMAnswerClient, messages: list[dict[str, str]], path: Path, max_tokens: int) -> dict[str, Any]:
    """Retry malformed JSON once without changing the image or evidence scope."""
    current = messages
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            content = str(client.generate_prediction(current, [path], max_tokens=max_tokens)["content"])
            return extract_json_object(content)
        except Exception as exc:
            last_error = exc
            if attempt:
                break
            current = [dict(message) for message in messages]
            current[-1]["content"] = str(current[-1].get("content") or "") + "\nFORMAT RETRY: Return exactly one complete JSON object and no prose or Markdown."
    raise RuntimeError(f"visual_crop_json_invalid: {last_error}")


def enrich_hierarchy(question: str, hierarchy: dict[str, Any], client: VLMAnswerClient | None, *, max_per_query: int, max_per_paper: int, observation_tokens: int, verify_tokens: int, dry_run: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    updated = dict(hierarchy)
    cards = list(updated.get("l2_evidence_cards") or [])
    report: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in select_visual_candidates(updated, max_per_query, max_per_paper):
        ref = str(record.get("evidence_ref") or "")
        path = Path(str(record.get("crop_path") or ""))
        entry = {"evidence_ref": ref, "paper_id": record.get("paper_id"), "crop_sha256": record.get("_crop_hash")}
        if dry_run:
            report.append({**entry, "status": "planned"})
            continue
        if client is None or not client.supports_image_input():
            errors.append({**entry, "type": "visual_cards_image_model_unavailable"})
            continue
        try:
            observed = _json_from_crop_call(client, _observation_messages(question, record), path, observation_tokens)
            proposed = _visual_card(record, observed)
            if proposed is None:
                report.append({**entry, "status": "rejected_empty_observation", "observation": observed})
                continue
            verified = _json_from_crop_call(client, _verification_messages(question, record, proposed), path, verify_tokens)
            if not bool(verified.get("supported")):
                report.append({**entry, "status": "rejected_by_visual_verifier", "proposition": proposed["proposition"], "observation": observed, "verification": verified, "reason": _clip(verified.get("reason"), 180)})
                continue
            proposed["card_id"] = f"V{len([card for card in cards if str(card.get('card_id') or '').startswith('V')]) + 1:03d}"
            cards.append(proposed)
            report.append({**entry, "status": "accepted", "card_id": proposed["card_id"], "proposition": proposed["proposition"], "observation": observed, "verification": verified})
        except Exception as exc:
            # A malformed crop response simply leaves this optional L2 card
            # absent. It must not turn an otherwise usable query hierarchy
            # into a generation failure; network/API failures remain visible
            # as errors for operational monitoring.
            if str(exc).startswith("visual_crop_json_invalid:"):
                report.append({**entry, "status": "rejected_invalid_json", "reason": _clip(exc, 180)})
            else:
                errors.append({**entry, "type": "visual_card_generation_failure", "error": str(exc)})
    updated["l2_evidence_cards"] = cards
    updated["visual_card_enrichment"] = {"version": "visual_l2_v1", "report": report, "error_count": len(errors)}
    return updated, report, errors


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    only = {part.strip() for part in args.only_query_ids.split(",") if part.strip()}
    questions = {str(row.get("query_id") or ""): str(row.get("question") or "") for row in read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))}
    max_per_query = config.evidence_hierarchy_visual_cards_max_per_query if args.max_per_query is None else args.max_per_query
    max_per_paper = config.evidence_hierarchy_visual_cards_max_per_paper if args.max_per_paper is None else args.max_per_paper
    client = None if args.dry_run else VLMAnswerClient(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in read_jsonl(args.hierarchy_input):
        query_id = str(row.get("query_id") or "")
        if only and query_id not in only:
            continue
        hierarchy, report, row_errors = enrich_hierarchy(
            questions.get(query_id, str(row.get("question") or "")), dict(row.get("hierarchy") or {}), client,
            max_per_query=max_per_query, max_per_paper=max_per_paper,
            observation_tokens=config.evidence_hierarchy_visual_cards_max_tokens,
            verify_tokens=config.evidence_hierarchy_visual_verify_max_tokens,
            dry_run=args.dry_run,
        )
        rows.append({**row, "hierarchy": hierarchy})
        report_rows.extend({"query_id": query_id, **item} for item in report)
        errors.extend({"query_id": query_id, **item} for item in row_errors)
        write_jsonl(output / "evidence_hierarchy.jsonl", rows)
        write_jsonl(output / "visual_card_report.jsonl", report_rows)
        write_jsonl(output / "errors.jsonl", errors)
    summary = {"queries": len(rows), "accepted": sum(row.get("status") == "accepted" for row in report_rows), "planned": sum(row.get("status") == "planned" for row in report_rows), "errors": len(errors), "dry_run": args.dry_run}
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
