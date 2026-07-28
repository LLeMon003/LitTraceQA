"""Build contextual triples and a sufficiency decision from a frozen hierarchy.

This post-selection transform is intentionally cache-safe: it reads only the
public validation question and the supplied hierarchy, and does not rerun Qwen,
retrieval, extraction, or answer generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .contextual_triples import (
    attach_contextual_triple_graph,
    attach_sufficiency_expansions,
    attach_visual_triple,
    structural_sufficiency_precheck,
    sufficiency_messages,
    triple_generation_messages,
    validate_sufficiency_decision,
    verify_llm_contextual_triples,
    visual_triple_messages,
    visual_triple_verification_messages,
)
from .data_io import find_official_file, read_jsonl, write_jsonl
from .parser import extract_json_object
from .vlm_answer_client import VLMAnswerClient
from .metadata_index import tokenize


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grounded contextual triples and a compact sufficiency decision without reranking.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reuse-raw-output",
        default="",
        help="Prior enrich output whose raw triple batches may be reused only when the rebuilt L1 windows are byte-identical.",
    )
    parser.add_argument("--only-query-ids", default="")
    parser.add_argument("--l1-sentence-chars", type=int, default=520)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--mode", choices=("extractive", "verified_llm"), default="", help="Default comes from EVIDENCE_TRIPLE_MODE.")
    parser.add_argument("--sufficiency", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Build artifacts and prompt previews but make no model calls.")
    return parser.parse_args()


def _visual_windows(hierarchy: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [
        row for row in hierarchy.get("l1_evidence_windows") or []
        if isinstance(row, dict) and str(row.get("source_type") or "") == "figure" and Path(str(row.get("crop_path") or "")).is_file()
    ]
    return rows[:max(0, limit)]


def _json_call(client: VLMAnswerClient, messages: list[dict[str, Any]], *, model: str, max_tokens: int, timeout_seconds: float, image_path: str | None = None) -> dict[str, Any]:
    result = client.generate_prediction(messages, [image_path] if image_path else None, model=model, max_tokens=max_tokens, timeout_seconds=timeout_seconds)
    return extract_json_object(str(result.get("content") or ""))


def _cached_json_call(
    client: VLMAnswerClient,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    cache_root: Path,
    cache_enabled: bool,
    phase: str,
    image_path: str | None = None,
) -> tuple[dict[str, Any], bool]:
    image_signature = ""
    if image_path:
        path = Path(image_path)
        if path.is_file():
            image_signature = hashlib.sha256(path.read_bytes()).hexdigest()
    key_payload = {"phase": phase, "model": model, "max_tokens": max_tokens, "messages": messages, "image_sha256": image_signature}
    key = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = cache_root / f"{key}.json"
    if cache_enabled and path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw, True
        except (OSError, json.JSONDecodeError):
            pass
    raw = _json_call(client, messages, model=model, max_tokens=max_tokens, timeout_seconds=timeout_seconds, image_path=image_path)
    if cache_enabled:
        cache_root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return raw, False


def _text_window_batches(hierarchy: dict[str, Any], max_chars: int) -> list[set[str]]:
    """Stable L1 batches; each query can checkpoint successful relation calls."""
    batches: list[set[str]] = []
    current: set[str] = set()
    used = 0
    for window in hierarchy.get("l1_evidence_windows") or []:
        if not isinstance(window, dict) or (str(window.get("source_type") or "") == "figure" and str(window.get("crop_path") or "")):
            continue
        estimate = len(json.dumps(window, ensure_ascii=False, separators=(",", ":")))
        if current and used + estimate > max(800, max_chars):
            batches.append(current)
            current, used = set(), 0
        current.add(str(window.get("window_id") or ""))
        used += estimate
    if current:
        batches.append(current)
    return batches


def _top_text_windows(question: str, hierarchy: dict[str, Any], limit: int) -> set[str]:
    query_terms = set(tokenize(question))
    rows = []
    forced: list[str] = []
    for window in hierarchy.get("l1_evidence_windows") or []:
        if not isinstance(window, dict) or (str(window.get("source_type") or "") == "figure" and str(window.get("crop_path") or "")):
            continue
        window_id = str(window.get("window_id") or "")
        seed_reasons = {str(value) for value in window.get("seed_reasons") or []}
        # A navigation seed is only useful if it reaches the relation model.
        # Existing rich-card windows already have the normal ranking path, so
        # force only additive seeds into the fixed per-query model budget.
        if window_id and "rich_l2_card" not in seed_reasons and any(
            reason.startswith(("explicit_citation_id:", "navigation_record:", "idf_lexical_navigation"))
            for reason in seed_reasons
        ):
            forced.append(window_id)
        text = " ".join([str(window.get("anchor_quote") or ""), *[str(value) for value in window.get("table_lines") or []]])
        overlap = len(query_terms.intersection(set(tokenize(text))))
        source_bonus = 2 if str(window.get("source_type") or "") in {"table", "equation_algorithm"} else 0
        rows.append((-(overlap * 10 + source_bonus), window_id))
    selected = list(dict.fromkeys(forced))[: max(1, limit)]
    for _, window_id in sorted(rows):
        if len(selected) >= max(1, limit):
            break
        if window_id and window_id not in selected:
            selected.append(window_id)
    return set(selected)


def _l1_fingerprint(hierarchy: dict[str, Any]) -> str:
    payload = hierarchy.get("l1_evidence_windows") or []
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    mode = args.mode or config.evidence_triple_mode
    sufficiency_enabled = config.evidence_triple_sufficiency_enabled if args.sufficiency == "auto" else args.sufficiency == "true"
    only = {value.strip() for value in args.only_query_ids.split(",") if value.strip()}
    questions = {
        str(row.get("query_id") or ""): str(row.get("question") or row.get("query") or "")
        for row in read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    }
    prior_hierarchies: dict[str, dict[str, Any]] = {}
    if args.reuse_raw_output:
        prior_path = Path(args.reuse_raw_output) / "evidence_hierarchy.jsonl"
        if not prior_path.is_file():
            raise FileNotFoundError(f"Missing prior hierarchy artifact: {prior_path}")
        prior_hierarchies = {
            str(row.get("query_id") or ""): dict(row.get("hierarchy") or {})
            for row in read_jsonl(prior_path)
            if isinstance(row.get("hierarchy"), dict)
        }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "triple_llm_cache"
    client = None if args.dry_run or mode != "verified_llm" else VLMAnswerClient(config, retries=0)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prompt_previews: list[dict[str, Any]] = []
    for row in read_jsonl(args.hierarchy_input):
        query_id = str(row.get("query_id") or "")
        if only and query_id not in only:
            continue
        question = questions.get(query_id, str(row.get("question") or ""))
        hierarchy = attach_contextual_triple_graph(
            question,
            dict(row.get("hierarchy") or {}),
            l1_sentence_chars=max(80, args.l1_sentence_chars),
        )
        prior = prior_hierarchies.get(query_id)
        can_reuse_raw = bool(
            prior
            and _l1_fingerprint(prior) == _l1_fingerprint(hierarchy)
            and isinstance(prior.get("triple_relation_raw_batches"), list)
        )
        if mode == "verified_llm":
            if can_reuse_raw:
                raw_batches = list(prior.get("triple_relation_raw_batches") or [])
                for batch_index, raw in enumerate(raw_batches, start=1):
                    hierarchy, rejected = verify_llm_contextual_triples(raw, hierarchy)
                    prompt_previews.append({"query_id": query_id, "phase": "text_triples_reused", "batch_index": batch_index, "source_output": str(args.reuse_raw_output)})
                    if rejected:
                        errors.extend({"query_id": query_id, "type": "triple_relation_rejected", "batch_index": batch_index, **entry} for entry in rejected)
                hierarchy["triple_relation_raw_batches"] = raw_batches
            elif client is None:
                errors.append({"query_id": query_id, "type": "triple_text_model_unavailable"})
            elif not args.dry_run:
                raw_batches = []
                allowed_windows = _top_text_windows(question, hierarchy, config.evidence_triple_text_max_windows)
                for batch_index, window_ids in enumerate(_text_window_batches({**hierarchy, "l1_evidence_windows": [row for row in hierarchy.get("l1_evidence_windows") or [] if str(row.get("window_id") or "") in allowed_windows]}, config.evidence_triple_batch_source_chars), start=1):
                    messages = triple_generation_messages(question, hierarchy, source_chars=config.evidence_triple_batch_source_chars, window_ids=window_ids)
                    prompt_previews.append({"query_id": query_id, "phase": "text_triples", "batch_index": batch_index, "window_ids": sorted(window_ids), "messages": messages})
                    try:
                        raw, cache_hit = _cached_json_call(client, messages, model=config.evidence_triple_text_model, max_tokens=config.evidence_triple_text_max_tokens, timeout_seconds=config.evidence_triple_timeout_seconds, cache_root=cache_root, cache_enabled=config.evidence_triple_cache_enabled, phase="text_triple",)
                        prompt_previews[-1]["cache_hit"] = cache_hit
                        hierarchy, rejected = verify_llm_contextual_triples(raw, hierarchy)
                        raw_batches.append(raw)
                        if rejected:
                            errors.extend({"query_id": query_id, "type": "triple_relation_rejected", "batch_index": batch_index, **entry} for entry in rejected)
                    except Exception as exc:
                        errors.append({"query_id": query_id, "type": "triple_text_generation_failure", "batch_index": batch_index, "error": str(exc)})
                hierarchy["triple_relation_raw_batches"] = raw_batches
            # Images are never included in the text batch. Each accepted crop
            # receives extraction and independent visual verification.
            if not can_reuse_raw and client is not None and client.supports_image_input(config.evidence_triple_visual_model) and not args.dry_run:
                for window in _visual_windows(hierarchy, config.evidence_triple_visual_max_per_query):
                    crop_path = str(window.get("crop_path") or "")
                    try:
                        visual_messages = visual_triple_messages(question, window)
                        prompt_previews.append({"query_id": query_id, "phase": "visual_triple", "window_id": window.get("window_id"), "messages": visual_messages})
                        proposal, _ = _cached_json_call(client, visual_messages, model=config.evidence_triple_visual_model, max_tokens=config.evidence_triple_visual_max_tokens, timeout_seconds=config.evidence_triple_timeout_seconds, cache_root=cache_root, cache_enabled=config.evidence_triple_cache_enabled, phase="visual_triple", image_path=crop_path)
                        verification, _ = _cached_json_call(client, visual_triple_verification_messages(window, proposal), model=config.evidence_triple_visual_model, max_tokens=120, timeout_seconds=config.evidence_triple_timeout_seconds, cache_root=cache_root, cache_enabled=config.evidence_triple_cache_enabled, phase="visual_verify", image_path=crop_path)
                        verified = attach_visual_triple(str(window.get("window_id") or ""), proposal, hierarchy, verified=bool(verification.get("supported")))
                        if verified is None:
                            errors.append({"query_id": query_id, "type": "visual_triple_rejected", "window_id": window.get("window_id"), "verification": verification})
                        else:
                            hierarchy = verified
                    except Exception as exc:
                        errors.append({"query_id": query_id, "type": "visual_triple_failure", "window_id": window.get("window_id"), "error": str(exc)})
        precheck = structural_sufficiency_precheck(hierarchy)
        if sufficiency_enabled and precheck["status"] == "ready_for_semantic_judge" and client is not None and not args.dry_run:
            messages = sufficiency_messages(question, hierarchy, max_triples=config.evidence_triple_text_max_windows)
            prompt_previews.append({"query_id": query_id, "phase": "sufficiency", "messages": messages})
            try:
                raw, cache_hit = _cached_json_call(client, messages, model=config.evidence_triple_sufficiency_model, max_tokens=config.evidence_triple_sufficiency_max_tokens, timeout_seconds=config.evidence_triple_timeout_seconds, cache_root=cache_root, cache_enabled=config.evidence_triple_cache_enabled, phase="sufficiency")
                prompt_previews[-1]["cache_hit"] = cache_hit
                hierarchy["triple_sufficiency"] = {**validate_sufficiency_decision(raw, hierarchy), "precheck": precheck, "raw": raw}
            except Exception as exc:
                hierarchy["triple_sufficiency"] = {"sufficient": False, "precheck": precheck, "verification": {"status": "model_failure"}}
                errors.append({"query_id": query_id, "type": "triple_sufficiency_failure", "error": str(exc)})
        else:
            hierarchy["triple_sufficiency"] = {
                "sufficient": False,
                "precheck": precheck,
                "verification": {"status": "not_called" if not sufficiency_enabled or args.dry_run else "structural_gap"},
            }
        hierarchy = attach_sufficiency_expansions(hierarchy)
        if bool((hierarchy.get("triple_sufficiency") or {}).get("sufficient")):
            # A positive semantic gate permits a compact triple-first prompt.
            # If the gate is negative or unavailable, generation retains the
            # broad keyed-card fallback rather than silently losing coverage.
            hierarchy["triple_prompt_policy"] = "accepted_triples_first"
        rows.append({**row, "hierarchy": hierarchy})
    write_jsonl(output / "evidence_hierarchy.jsonl", rows)
    write_jsonl(output / "errors.jsonl", errors)
    write_jsonl(output / "prompt_previews.jsonl", prompt_previews)
    summary = {
        "queries": len(rows),
        "l1_windows": sum(len((row.get("hierarchy") or {}).get("l1_evidence_windows") or []) for row in rows),
        "l2_candidate_triples": sum(len((row.get("hierarchy") or {}).get("l2_contextual_triples") or []) for row in rows),
        "verified_llm_triples": sum(sum(item.get("verification", {}).get("status") == "verified_llm_relation" for item in (row.get("hierarchy") or {}).get("l2_contextual_triples") or []) for row in rows),
        "sufficiency_called": sum((row.get("hierarchy") or {}).get("triple_sufficiency", {}).get("verification", {}).get("status") == "accepted_keyed_gate" for row in rows),
        "errors": len(errors),
        "mode": mode,
        "dry_run": args.dry_run,
        "source_hierarchy": str(args.hierarchy_input),
        "reused_raw_output": str(args.reuse_raw_output),
    }
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
