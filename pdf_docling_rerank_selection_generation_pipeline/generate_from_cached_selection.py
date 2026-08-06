"""Generate predictions from a frozen offline selection artifact, without reranking."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .data_io import extract_answer_contract, find_official_file, read_jsonl, write_jsonl
from .evidence_hierarchy import (
    hierarchy_prompt_projection,
    keyed_hierarchy_prompt_projection,
    resolve_claim_support_keys,
)
from .metadata_index import tokenize
from .parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding
from .symbolic_context_selector import _compact_package_packets, grounding_label_from_record, project_context_for_vlm2


GENERATION_PROVENANCE_VERSION = "v1_inference_inputs_only"


class JSONDraftError(RuntimeError):
    """A model response was received but could not be parsed as the contract JSON."""

    def __init__(self, message: str, raw_attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.raw_attempts = raw_attempts
from .symbolic_schema import to_official_source_type
from .vlm_answer_client import VLMAnswerClient
from .vlm_answer_prompt_builder import build_symbolic_answer_prompt
from .slot_generation import (
    align_composed_values,
    bind_composition_support,
    deterministic_count_extraction,
    ensure_slot_cards,
    slot_composition_messages,
    slot_extraction_messages,
    slot_image_attachments,
    slot_plan_messages,
    validate_slot_extraction,
    validate_slot_plan,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer generation from frozen selected records.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--selected-contexts-input", required=True)
    parser.add_argument("--candidate-papers-input", required=True)
    parser.add_argument(
        "--answer-contracts-input",
        default="",
        help="Optional sanitized answer-shape contracts (options/schema only; never gold answers).",
    )
    parser.add_argument("--hierarchy-input", default="", help="L0-L3 artifact produced by build_evidence_hierarchy.")
    parser.add_argument(
        "--hierarchy-prompt-mode",
        choices=("legacy", "keyed"),
        default="keyed",
        help="Use L2 cards plus navigation keys only, or the legacy hierarchy projection.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--max-context-chars", type=int, default=30000)
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=90000,
        help="Hard limit on the fully serialized system and user prompt. Set <=0 to disable.",
    )
    parser.add_argument(
        "--packing-mode",
        choices=("coverage", "ranked_prefix"),
        default="coverage",
        help="Use budget-aware query coverage packing or preserve the legacy record-order prefix.",
    )
    parser.add_argument(
        "--record-text-budget", type=int, default=72000,
        help="Estimated serialized evidence budget before prompt framing. Set <=0 to use --max-context-chars.",
    )
    parser.add_argument(
        "--keyed-micro-text-chars",
        type=int,
        default=None,
        help="Experimental L2 proposition width override; default comes from the hierarchy artifact.",
    )
    parser.add_argument(
        "--keyed-micro-order",
        choices=("selection", "query_aware", "stability_query_aware"),
        default=None,
        help="Experimental order for L2 micro propositions; never changes frozen selection membership.",
    )
    parser.add_argument(
        "--keyed-micro-index-chars",
        type=int,
        default=None,
        help="Maximum serialized keyed L2 micro index before prompt fitting; default comes from the hierarchy artifact.",
    )
    parser.add_argument(
        "--keyed-card-limit",
        type=int,
        default=None,
        help="Experimental maximum number of rich L2 cards; micro propositions retain the remaining budget.",
    )
    parser.add_argument(
        "--keyed-table-view-limit",
        type=int,
        default=None,
        help="Maximum structured table views exposed in keyed L2 mode; default comes from the hierarchy.",
    )
    parser.add_argument(
        "--keyed-table-view-rows",
        type=int,
        default=None,
        help="Maximum rows retained per keyed structured table view; default comes from the hierarchy.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum attached figure crops; default uses EVIDENCE_HIERARCHY_MAX_IMAGES.",
    )
    parser.add_argument(
        "--posthoc-refinement",
        choices=("auto", "true", "false"),
        default="auto",
        help="Run a fail-soft raw-L0 verifier after keyed grounding; auto uses the environment setting.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("direct", "slots"),
        default="direct",
        help="Use the existing one-pass answer prompt or plan/extract/compose slots from keyed hierarchy cards.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the exact packed contexts without calling the answer model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed predictions in --output-dir and continue only missing query IDs.",
    )
    parser.add_argument(
        "--retry-query-ids",
        default="",
        help="Comma-separated completed query IDs to replace, for example after a fallback prediction.",
    )
    parser.add_argument("--only-query-ids", default="")
    return parser.parse_args()


def _candidate_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            result.setdefault(query_id, []).append(row)
    for candidates in result.values():
        candidates.sort(key=lambda row: int(row.get("rank") or 10**9))
    return result


def _prompt_records_ranked_prefix(records: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for raw in records:
        source_type = to_official_source_type(raw.get("record_type"), raw.get("source_type"))
        if not source_type:
            continue
        text = str(raw.get("text") or "")
        cost = len(text) + 200
        if selected and max_chars > 0 and used + cost > max_chars:
            continue
        record = dict(raw)
        record["source_type"] = source_type
        record["grounding_label"] = grounding_label_from_record(source_type, record.get("label"))
        record["evidence_ref"] = f"E{len(selected) + 1:04d}"
        selected.append(record)
        used += cost
    return selected


def _query_targets(query: str) -> dict[str, str]:
    """Extract deterministic object names mentioned directly in the question."""
    targets: dict[str, str] = {}
    patterns = {
        "table": r"\btable\s+([A-Za-z0-9.\-]+)",
        "figure": r"\b(?:figure|fig\.)\s+([A-Za-z0-9.\-]+)",
        "equation_algorithm": r"\b(?:equation|eq\.|algorithm)\s*\(?([A-Za-z0-9.\-]+)\)?",
        "citation_context": r"\b(?:reference|ref\.)\s*\[?(\d{1,3})\]?|\b(\d{1,3})(?:st|nd|rd|th)\s+reference\b",
    }
    for source_type, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            targets[source_type] = next((group for group in match.groups() if group), "")
    return targets


def _compact_table_text(text: str, query_terms: set[str], limit: int = 3000) -> str:
    """Retain table headers and query-relevant rows without changing its locator."""
    if len(text) <= limit:
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:limit]
    header = lines[:3]
    candidates = lines[3:]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            -len(query_terms.intersection(tokenize(item[1]))),
            item[0],
        ),
    )
    kept = list(header)
    # Keep both lexical matches and a small ordered sample for questions whose
    # requested cell uses an abbreviation absent from the question.
    for _, line in ranked[:10]:
        if line not in kept:
            kept.append(line)
    for line in candidates[:3] + candidates[-2:]:
        if line not in kept:
            kept.append(line)
    compact = "\n".join(kept)
    return compact if len(compact) <= limit else compact[:limit - 3].rstrip() + "..."


def _compact_record_text(record: dict[str, Any], query_terms: set[str], targets: dict[str, str]) -> str:
    text = str(record.get("text") or "").strip()
    source_type = str(record.get("source_type") or "")
    target = str(targets.get(source_type) or "")
    direct_target = bool(target and re.search(rf"\b{re.escape(target)}\b", f"{record.get('label') or ''}\n{text}", re.IGNORECASE))
    limits = {
        "table": 3000 if direct_target else 1500,
        "figure": 1500 if direct_target else 800,
        "equation_algorithm": 1800 if direct_target else 900,
        "citation_context": 1200 if direct_target else 500,
        # text spans are strict-locator anchors. Their surrounding package was
        # already selected upstream, so a concise head/tail projection lets the
        # answer model inspect many independent claims under one context budget.
        "text_span": 350,
    }
    limit = limits.get(source_type, 1100)
    if source_type == "table":
        return _compact_table_text(text, query_terms, limit)
    if len(text) <= limit:
        return text
    # Preserve a tail too: it commonly contains the final cited author, table
    # value, or qualifying condition that a pure prefix would discard.
    head = max(1, int(limit * 0.78))
    tail = max(1, limit - head - 5)
    return text[:head].rstrip() + " ... " + text[-tail:].lstrip()


def _record_priority(
    record: dict[str, Any],
    query_terms: set[str],
    targets: dict[str, str],
    primary_evidence_type: str,
    rank: int,
) -> tuple[float, bool]:
    source_type = str(record.get("source_type") or "")
    label = str(record.get("label") or "")
    text = str(record.get("text") or "")
    terms = set(tokenize(" ".join((label, text))))
    overlap = len(query_terms.intersection(terms)) / max(1, len(query_terms))
    # The original offline order is frozen-Qwen package order. It remains the
    # dominant signal; lexical/object signals only repair budget allocation.
    score = 4.0 / (1.0 + rank / 30.0) + 3.0 * overlap
    if source_type == primary_evidence_type:
        score += 0.9
    target = targets.get(source_type)
    direct_target = bool(target and re.search(rf"\b{re.escape(target)}\b", f"{label}\n{text}", re.IGNORECASE))
    if direct_target:
        score += 12.0
    if source_type != "text_span" and label:
        score += 0.3
    if not text:
        score -= 2.0
    return score, direct_target


def _prompt_records_coverage(
    records: list[dict[str, Any]],
    query: str,
    primary_evidence_type: str | None,
    task_family: str | None,
    budget: int,
) -> list[dict[str, Any]]:
    """Pack frozen-Qwen records under a real prompt budget without gold access.

    The previous implementation flattened selected packages and then consumed a
    prefix.  It systematically dropped later papers and sparse text anchors.
    This allocator uses only query-visible structure and the frozen package
    order, retains exact records/locators, and favors marginal paper/modality
    coverage per serialized character.
    """
    query_terms = set(tokenize(query))
    targets = _query_targets(query)
    primary = to_official_source_type(source_type=primary_evidence_type) or str(primary_evidence_type or "")
    is_multi = "multi" in str(task_family or "").lower()
    unique: dict[str, tuple[int, dict[str, Any]]] = {}
    for rank, raw in enumerate(records, start=1):
        source_type = to_official_source_type(raw.get("record_type"), raw.get("source_type"))
        identifier = str(raw.get("global_record_id") or raw.get("record_id") or "")
        if not source_type or not identifier or identifier in unique:
            continue
        record = dict(raw)
        record["source_type"] = source_type
        record["text"] = _compact_record_text(record, query_terms, targets)
        unique[identifier] = (rank, record)

    prepared: list[dict[str, Any]] = []
    for rank, record in unique.values():
        priority, direct_target = _record_priority(record, query_terms, targets, primary, rank)
        prepared.append({
            "record": record,
            "priority": priority,
            "direct_target": direct_target,
            "cost": len(str(record.get("text") or "")) + 220,
            "paper_id": str(record.get("paper_id") or ""),
            "source_type": str(record.get("source_type") or ""),
            "page": record.get("page"),
        })

    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        by_paper[item["paper_id"]].append(item)
        by_type[item["source_type"]].append(item)
    for items in [*by_paper.values(), *by_type.values()]:
        items.sort(key=lambda item: (-item["priority"], item["cost"]))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used = 0
    selected_papers: set[str] = set()
    selected_types: set[str] = set()

    def add(item: dict[str, Any], force: bool = False) -> bool:
        nonlocal used
        record = item["record"]
        identifier = str(record.get("global_record_id") or record.get("record_id") or "")
        if not identifier or identifier in selected_ids:
            return False
        if selected and budget > 0 and used + item["cost"] > budget:
            return False
        selected_ids.add(identifier)
        selected.append(record)
        selected_papers.add(item["paper_id"])
        selected_types.add(item["source_type"])
        used += item["cost"]
        return True

    # Exact query locators are provable constraints, so reserve them first.
    for item in sorted((item for item in prepared if item["direct_target"]), key=lambda item: (-item["priority"], item["cost"])):
        add(item, force=True)
    # Multi-paper questions need evidence from multiple papers before a single
    # paper's dense table/paragraph projections consume the whole budget.
    if is_multi:
        paper_order = sorted(by_paper, key=lambda paper_id: (-by_paper[paper_id][0]["priority"], paper_id))
        for paper_id in paper_order:
            add(by_paper[paper_id][0])
    # Preserve the explicitly requested source type when it exists.
    if primary in by_type:
        add(by_type[primary][0])

    while True:
        remaining = [item for item in prepared if str(item["record"].get("global_record_id") or item["record"].get("record_id") or "") not in selected_ids]
        if not remaining:
            break

        def utility(item: dict[str, Any]) -> tuple[float, float, float]:
            gain = 0.0
            if item["source_type"] == primary and item["source_type"] not in selected_types:
                gain += 1.8
            if is_multi and item["paper_id"] not in selected_papers:
                gain += 1.2
            if item["source_type"] not in selected_types:
                gain += 0.35
            # A weak page diversity term avoids selecting many duplicate OCR
            # fragments from the same page solely because their prefix rank is
            # adjacent in the frozen trace.
            return (item["priority"] + gain) / max(1, item["cost"]), item["priority"], -item["cost"]

        candidate = max(remaining, key=utility)
        if not add(candidate):
            # Oversized records have already been compacted; discard only this
            # candidate and let a smaller competing record use the remaining
            # budget on the next pass.
            prepared.remove(candidate)
            continue

    for index, record in enumerate(selected, start=1):
        record["grounding_label"] = grounding_label_from_record(record["source_type"], record.get("label"))
        record["evidence_ref"] = f"E{index:04d}"
    return selected


def _selected_context(
    evidence: list[dict[str, Any]],
    *,
    context_mode: str = "text_only",
    max_images: int = 0,
    evidence_hierarchy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages = [
        {
            "package_id": f"offline::{record['evidence_ref']}",
            "anchor_record_id": record.get("global_record_id"),
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "section_id": record.get("section_id"),
            "section_title": record.get("section_title"),
            "section_type": record.get("section_type"),
            "source_type": record.get("source_type"),
            "label": record.get("label"),
            "records": [record],
        }
        for record in evidence
    ]
    attachments: list[dict[str, Any]] = []
    attachment_by_path: dict[str, dict[str, Any]] = {}
    if context_mode == "cropped_image":
        for record in evidence:
            if str(record.get("source_type") or "") not in {"figure", "table", "equation_algorithm"}:
                continue
            path = Path(str(record.get("crop_path") or record.get("image_path") or ""))
            if not path.is_file():
                continue
            attachment = attachment_by_path.get(str(path))
            if attachment is None:
                if max_images > 0 and len(attachments) >= max_images:
                    continue
                attachment = {"image_ref": f"IMG{len(attachments) + 1:03d}", "path": str(path), "evidence_refs": []}
                attachments.append(attachment)
                attachment_by_path[str(path)] = attachment
            record["image_ref"] = attachment["image_ref"]
            attachment["evidence_refs"].append(record.get("evidence_ref"))
    hierarchy = copy.deepcopy(evidence_hierarchy) if evidence_hierarchy else None
    if hierarchy is not None:
        hierarchy["image_map"] = [
            {"image_ref": item["image_ref"], "evidence_refs": item["evidence_refs"]}
            for item in attachments
        ]
    return {
        "selected_evidence": [project_context_for_vlm2(record, context_mode) for record in evidence],
        "compact_chunk_packets": _compact_package_packets(packages, evidence),
        # A flat ledger avoids repeatedly serializing package/section defaults.
        # The exact records above remain the sole source of truth for locator
        # restoration after the model echoes an evidence_ref.
        "evidence_ledger": _evidence_ledger(evidence),
        "attached_image_refs": [{"image_ref": item["image_ref"], "evidence_refs": item["evidence_refs"]} for item in attachments],
        "attached_image_paths": [item["path"] for item in attachments],
        "evidence_hierarchy": hierarchy,
    }


def _fit_hierarchy_to_prompt(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    hierarchy: dict[str, Any],
    config: Any,
    contract: dict[str, Any],
    max_prompt_chars: int,
    max_images: int,
    keyed_mode: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Fit an L2 prompt while retaining the complete off-prompt L0 catalog."""
    cards = list(hierarchy.get("l2_evidence_cards") or [])
    best_context: dict[str, Any] | None = None
    best_messages: list[dict[str, Any]] = []
    configured_micro_chars = hierarchy.get("keyed_micro_index_chars")
    # Zero means "let the final prompt ceiling decide". The binary search
    # still needs a finite high bound, well above any current selected L2
    # index, instead of silently restoring the legacy 48k cap.
    max_micro_chars = 250000 if configured_micro_chars is None or int(configured_micro_chars) == 0 else int(configured_micro_chars)

    # Prime query-aware micro priorities once on the base hierarchy. Subsequent
    # binary-search projections deep-copy this tiny cache instead of tokenizing
    # every L0 record repeatedly.
    if keyed_mode and str(hierarchy.get("keyed_micro_order") or "") in {"query_aware", "stability_query_aware"}:
        keyed_hierarchy_prompt_projection(hierarchy)

    def build_current(current: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        projection = keyed_hierarchy_prompt_projection(current)
        card_map = projection["_card_support_keys"]
        visible_card_keys = [
            str(card.get("key") or "") for card in projection["l2_cards"]
            if isinstance(card, dict)
        ]
        visible_refs = {
            str(projection["_key_index"][record_key]["evidence_ref"])
            for row in projection.get("l2_micro_rows") or []
            if isinstance(row, list) and row and str(row[0]) in card_map
            for record_key in card_map[str(row[0])]
            if str(record_key) in projection["_key_index"]
        }
        visible_refs = {
            str(projection["_key_index"][record_key]["evidence_ref"])
            for card_key in visible_card_keys
            for record_key in card_map.get(card_key, [])
            if str(record_key) in projection["_key_index"]
        } | visible_refs
        prompt_evidence = [record for record in evidence if str(record.get("evidence_ref") or "") in visible_refs]
        # Attach scarce visual slots to the query's requested modality first.
        # This changes only attachment order, never L2 key order or selection.
        primary_type = str(sample.get("primary_evidence_type") or "")
        prompt_evidence.sort(key=lambda record: str(record.get("source_type") or "") != primary_type)
        context = _selected_context(
            [dict(record) for record in prompt_evidence],
            # Keyed L2 cards remain the sole provenance keys, but the linked
            # Figure crops must reach the answer model in cropped_image mode.
            # image_map exposes only IMG->opaque evidence refs at runtime; the
            # keyed prompt maps each visible C-card to its IMG identifier.
            context_mode=config.vlm2_context_mode,
            max_images=max_images,
            evidence_hierarchy=current,
        )
        messages = build_symbolic_answer_prompt(
            sample, candidates, context, False, "docling", config.answer_model, contract,
        )
        return prompt_evidence, context, messages

    for count in range(len(cards), -1, -1):
        current = copy.deepcopy(hierarchy)
        current["l2_evidence_cards"] = cards[:count]
        if keyed_mode:
            current["prompt_mode"] = "keyed_l2_only"
            # The configured micro index is a preference, not permission to
            # violate the model ceiling. Hold the rich cards fixed and binary
            # search the largest micro-proposition budget that still fits.
            low, high = 0, max_micro_chars
            candidate: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]] | None = None
            while low <= high:
                micro_chars = (low + high) // 2
                current["keyed_micro_index_chars"] = micro_chars
                prompt_evidence, context, messages = build_current(current)
                chars = sum(len(str(message.get("content") or "")) for message in messages)
                if max_prompt_chars <= 0 or chars <= max_prompt_chars:
                    # _selected_context appends the runtime IMG->L0 map. Keep
                    # that map in the returned hierarchy so debug artifacts
                    # and post-hoc audits describe the actual prompt, not an
                    # image-less pre-projection copy.
                    candidate = (copy.deepcopy(context.get("evidence_hierarchy") or current), prompt_evidence, context, messages)
                    low = micro_chars + 1
                else:
                    high = micro_chars - 1
            if candidate is not None:
                return candidate
            # Too many rich cards can exceed the ceiling even with no micro
            # index. Drop the lowest-priority card and retry.
            continue
        else:
            visible_refs = {
                str(ref)
                for card in current["l2_evidence_cards"]
                for ref in card.get("support_refs") or []
            }
            visible_refs.update(
                str(row.get("support_ref") or "")
                for row in hierarchy_prompt_projection(current).get("l2_micro_evidence") or []
                if str(row.get("support_ref") or "")
            )
        prompt_evidence = [record for record in evidence if str(record.get("evidence_ref") or "") in visible_refs]
        context = _selected_context(
            [dict(record) for record in prompt_evidence],
            context_mode=config.vlm2_context_mode,
            max_images=max_images,
            evidence_hierarchy=current,
        )
        messages = build_symbolic_answer_prompt(
            sample, candidates, context, bool(context.get("attached_image_paths")), "docling", config.answer_model, contract,
        )
        chars = sum(len(str(message.get("content") or "")) for message in messages)
        if max_prompt_chars <= 0 or chars <= max_prompt_chars:
            return copy.deepcopy(context.get("evidence_hierarchy") or current), prompt_evidence, context, messages
        best_context, best_messages = context, messages
    if keyed_mode:
        empty = copy.deepcopy(hierarchy)
        empty["l2_evidence_cards"] = []
        empty["keyed_micro_index_chars"] = 0
        _, context, messages = build_current(empty)
        return empty, [], context, messages
    return {**hierarchy, "l2_evidence_cards": []}, [], best_context or _selected_context([]), best_messages


def _evidence_ledger(evidence: list[dict[str, Any]]) -> str:
    papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in evidence:
        papers[str(record.get("paper_id") or "unknown")].append(record)
    lines: list[str] = []
    for paper_id, rows in papers.items():
        lines.append(f"PAPER {paper_id}")
        current_section: tuple[str, str] | None = None
        for record in rows:
            section = (str(record.get("section_id") or ""), str(record.get("section_title") or ""))
            if section != current_section:
                current_section = section
                lines.append(f"SECTION {section[1] or section[0] or 'unsectioned'}")
            header = "\t".join(
                value for value in (
                    str(record.get("evidence_ref") or ""),
                    f"p{record.get('page')}",
                    str(record.get("source_type") or "text_span"),
                    str(record.get("label") or ""),
                )
                if value
            )
            text = str(record.get("text") or "").strip().replace("\x00", "")
            lines.append(f"{header}\t{text}")
    return "\n".join(lines)


def _restrict_prediction_to_visible_evidence(prediction: dict[str, Any], evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Enforce the L2-to-L0 proof path after parser table-plan expansion."""
    allowed = {
        (
            str(record.get("paper_id") or ""),
            str(record.get("source_type") or ""),
            json.dumps(record.get("locator") or {}, ensure_ascii=False, sort_keys=True),
        )
        for record in evidence
    }
    kept = []
    removed = False
    for item in prediction.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("paper_id") or ""),
            str(item.get("source_type") or ""),
            json.dumps(item.get("locator") or {}, ensure_ascii=False, sort_keys=True),
        )
        if key in allowed:
            kept.append(item)
        else:
            removed = True
    if removed:
        prediction = dict(prediction)
        prediction["evidence"] = kept
    return prediction, removed


def _posthoc_ground_keyed_prediction(
    internal: dict[str, Any],
    hierarchy: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace model-visible card keys with exact L0 refs before parsing.

    The model never receives a raw ref, page, or locator in keyed mode.  This
    makes the only possible proof path C-key -> R-key -> frozen L0 record.
    """
    grounded = dict(internal)
    refs, audit = resolve_claim_support_keys(grounded.get("claim_to_support_keys"), hierarchy)
    # Do not allow a model to bypass the keyed contract with an incidental raw
    # evidence field in malformed JSON.
    grounded.pop("evidence", None)

    projection = keyed_hierarchy_prompt_projection(hierarchy)
    key_index = projection["_key_index"]
    card_map = projection["_card_support_keys"]
    by_ref = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    card_metadata = projection.get("_card_metadata") or {}
    claim_text = " ".join(str(item.get("claim") or "") for item in hierarchy.get("query_claims") or [] if isinstance(item, dict))
    primary_source_type = str(hierarchy.get("primary_evidence_type") or "")
    explicit_object_claim = bool(re.search(r"\b(?:explicitly\s+(?:mention|reference)|explicit\s+(?:mention|reference)|labeled?|named)\b", claim_text, re.IGNORECASE))
    required_anchor_terms = {term.lower() for term in re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", claim_text) if term.upper() not in {"PDF", "QA"}}
    venue_match = re.search(r"\b(ACL|NAACL|EMNLP|ICLR|ICML|NEURIPS)\s*(20\d{2})\b", claim_text, re.IGNORECASE)
    venue_prefix = f"{venue_match.group(1).lower()}{venue_match.group(2)}" if venue_match else ""
    catalog_papers = {str(row.get("paper_id") or "").lower() for row in by_ref.values()}
    if venue_prefix and not any(paper.startswith(venue_prefix) for paper in catalog_papers):
        venue_prefix = ""

    # Deterministic constraints are applied only when the query itself names a
    # venue/year or requires an explicit textual/object mention. They remove
    # impossible C -> L0 proof paths; they do not rank or prefilter the corpus.
    raw_claim_mapping = grounded.get("claim_to_support_keys") if isinstance(grounded.get("claim_to_support_keys"), dict) else {}
    filtered_claim_mapping: dict[str, list[str]] = {}
    constraint_audit: list[dict[str, Any]] = []
    for row in audit:
        claim_id = str(row.get("claim_id") or "")
        allowed: list[str] = []
        for card_key in row.get("valid_card_keys") or []:
            metadata = card_metadata.get(str(card_key)) or {}
            record_keys = card_map.get(str(card_key)) or []
            paper_id = str((key_index.get(str(record_keys[0])) or {}).get("paper_id") or "").lower() if record_keys else ""
            source_type = str(metadata.get("source_type") or "")
            proposition = str(metadata.get("proposition") or "").lower()
            # A venue string inside the question (for example "NAACL 2025"
            # inside a citation or slot condition) is not proof of the paper's
            # venue and routinely removed the correct ACL evidence. Paper
            # identity is already constrained to evidence-bound C-keys, so the
            # venue heuristic is disabled for keyed grounding.
            venue_ok = True
            anchor_ok = not (explicit_object_claim and required_anchor_terms and source_type == primary_source_type) or any(term in proposition for term in required_anchor_terms)
            if venue_ok and anchor_ok:
                allowed.append(str(card_key))
            else:
                constraint_audit.append({"claim_id": claim_id, "card_key": str(card_key), "status": "claim_key_constraint_removed", "venue_ok": venue_ok, "anchor_ok": anchor_ok})
        if allowed:
            filtered_claim_mapping[claim_id] = allowed
    if raw_claim_mapping != filtered_claim_mapping:
        grounded["claim_to_support_keys"] = filtered_claim_mapping
        refs, audit = resolve_claim_support_keys(filtered_claim_mapping, hierarchy)
        audit.extend(constraint_audit)
    grounded["evidence_refs"] = refs
    cleaned_plan: list[dict[str, Any]] = []
    visual_card_keys = [
        key
        for row in audit for key in row.get("valid_card_keys") or []
        if str((projection.get("_card_metadata", {}).get(str(key)) or {}).get("verification_status") or "") == "visual_verified"
    ]
    linked_card_keys = {str(key) for row in audit for key in row.get("valid_card_keys") or []}
    # Uppercase abbreviations are usually method/model/dataset anchors (for
    # example MCTS, RAG, BERT). A Paper Title row can only be identity-mapped
    # when its C-card states those explicit query anchors, preventing a nearby
    # but irrelevant figure from claiming a paper merely by shared layout.
    for raw in grounded.get("table_answer_plan") or []:
        if not isinstance(raw, dict):
            continue
        card_key = str(raw.get("row_support_key") or "")
        if card_key not in linked_card_keys:
            audit.append({"table_support_key": card_key, "status": "table_plan_unlinked_to_claim_removed"})
            continue
        record_keys = card_map.get(card_key) or []
        # `answer.table` is an output shape, not proof that the supporting
        # evidence is a PDF table. A row may be grounded by a figure caption or
        # prose sentence; the direct fact gate below still verifies every cell
        # against the resolved L0 source.
        support_ref = next(
            (str(key_index[key]["evidence_ref"]) for key in record_keys if key in key_index),
            "",
        )
        record = by_ref.get(support_ref)
        if not record:
            audit.append({"table_support_key": card_key, "status": "table_plan_ungrounded"})
            continue
        row_source = {
            "paper_id": record.get("paper_id"),
            "page": record.get("page"),
            "label": record.get("label"),
            "source_type": record.get("source_type"),
        }
        values = raw.get("values") if isinstance(raw.get("values"), dict) else {}
        if any(str(column).strip().lower() == "paper title" for column in values):
            metadata = card_metadata.get(card_key) or {}
            proposition = str(metadata.get("proposition") or "").lower()
            primary_ok = not primary_source_type or str(metadata.get("source_type") or "") == primary_source_type
            anchor_ok = not required_anchor_terms or any(term in proposition for term in required_anchor_terms)
            if not primary_ok or not anchor_ok:
                audit.append({
                    "table_support_key": card_key,
                    "status": "table_plan_identity_support_removed",
                    "primary_ok": primary_ok,
                    "anchor_ok": anchor_ok,
                })
                continue
        cleaned_plan.append({"row_evidence_ref": support_ref, "row_source": row_source, "values": values})
        if str((projection.get("_card_metadata", {}).get(card_key) or {}).get("verification_status") or "") == "visual_verified":
            visual_card_keys.append(card_key)
        if support_ref not in grounded["evidence_refs"]:
            grounded["evidence_refs"].append(support_ref)
        audit.append({"table_support_key": card_key, "evidence_ref": support_ref, "status": "table_plan_grounded"})
    if "table_answer_plan" in grounded:
        grounded["table_answer_plan"] = cleaned_plan
    grounded["visual_support_card_keys"] = sorted(set(visual_card_keys))
    grounded["posthoc_grounding_audit"] = audit
    _apply_direct_fact_gate(grounded, hierarchy, audit, visual_card_keys=grounded["visual_support_card_keys"])
    # Once table rows have passed their value gate, retain only proof refs that
    # still belong to a generated claim or an accepted row. This prevents a
    # rejected table draft from leaving an unrelated official locator behind.
    claim_refs, _ = resolve_claim_support_keys(grounded.get("claim_to_support_keys"), hierarchy)
    accepted_plan_refs = [str(row.get("row_evidence_ref") or "") for row in grounded.get("table_answer_plan") or []]
    grounded["evidence_refs"] = list(dict.fromkeys([*claim_refs, *[ref for ref in accepted_plan_refs if ref]]))
    supported_papers = {str((by_ref.get(ref) or {}).get("paper_id") or "") for ref in grounded["evidence_refs"]}
    for field in ("gold_papers", "contributing_papers"):
        original = grounded.get(field)
        if not isinstance(original, list):
            continue
        filtered = []
        for item in original:
            paper_id = str((item.get("paper_id") or "") if isinstance(item, dict) else (item or ""))
            if paper_id and paper_id in supported_papers:
                filtered.append(item)
            elif paper_id:
                audit.append({"field": field, "paper_id": paper_id, "status": "unsupported_paper_removed"})
        grounded[field] = filtered
    # A model-supplied answer.table is only a presentation draft.  In keyed
    # mode it must be reconstructed from the post-hoc verified plan; otherwise
    # a rejected row can survive normalization merely because it had the right
    # output shape.  This also makes the plan the single proof path for every
    # table cell, regardless of whether the underlying L0 source is a table,
    # figure, equation, or text span.
    answer = grounded.get("answer") if isinstance(grounded.get("answer"), dict) else {}
    if "table" in answer or "table_answer_plan" in grounded:
        answer = dict(answer)
        answer["table"] = {"rows": [dict(row.get("values") or {}) for row in grounded.get("table_answer_plan") or []]}
        grounded["answer"] = answer
    return grounded, audit


def _canonical_fact_text(value: Any) -> str:
    text = str(value or "").lower().replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _fact_is_directly_supported(value: Any, raw_text: str) -> bool:
    """Conservative lexical gate for entities, values, and table cells.

    It intentionally does not attempt semantic verification of an MC option;
    those labels often do not occur verbatim in the evidence.  Freeform and
    table fields, however, must survive a literal normalized check.
    """
    claim = _canonical_fact_text(value)
    source = _canonical_fact_text(raw_text)
    if not claim or not source:
        return False
    if claim in source:
        return True
    claim_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", claim)
    if claim_numbers:
        source_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", source))
        return all(number in source_numbers for number in claim_numbers)
    # Multi-word names are the reliable entity case.  Avoid treating short
    # function words or a single letter as a fact merely because it appears.
    tokens = [token for token in tokenize(claim) if len(token) >= 3]
    return bool(tokens) and " ".join(tokens) in " ".join(tokenize(source))


def _apply_direct_fact_gate(
    internal: dict[str, Any],
    hierarchy: dict[str, Any],
    audit: list[dict[str, Any]],
    *,
    visual_card_keys: list[str] | None = None,
) -> None:
    """Drop generated fact fields that cannot be found in their resolved L0."""
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    refs = [str(ref) for ref in internal.get("evidence_refs") or []]
    raw_text = "\n".join(str((catalog.get(ref) or {}).get("text") or "") for ref in refs)
    # Visual L2 propositions are allowed only when an independent crop check
    # accepted them and the model explicitly selected the corresponding C-key.
    # They remain bound to the same L0 figure through card_support_keys.
    projection = keyed_hierarchy_prompt_projection(hierarchy)
    key_index = projection.get("_key_index") or {}
    card_map = projection.get("_card_support_keys") or {}
    card_metadata = projection.get("_card_metadata") or {}
    visual_text_by_ref: dict[str, list[str]] = defaultdict(list)
    for card_key in visual_card_keys if visual_card_keys is not None else internal.get("visual_support_card_keys") or []:
        metadata = card_metadata.get(str(card_key)) or {}
        if str(metadata.get("verification_status") or "") != "visual_verified":
            continue
        proposition = str(metadata.get("proposition") or "")
        for record_key in card_map.get(str(card_key)) or []:
            ref = str((key_index.get(str(record_key)) or {}).get("evidence_ref") or "")
            if ref and proposition:
                visual_text_by_ref[ref].append(proposition)
    visual_text = "\n".join(value for ref in refs for value in visual_text_by_ref.get(ref, []))
    selected_card_keys = {
        str(card_key)
        for keys in (internal.get("claim_to_support_keys") or {}).values()
        if isinstance(keys, list)
        for card_key in keys
    }
    derived_values = {
        _canonical_fact_text(slot.get("value"))
        for slot in internal.get("_validated_slots") or []
        if isinstance(slot, dict)
        and bool((slot.get("validation") or {}).get("derived_value"))
        and set(str(key) for key in slot.get("support_keys") or []).intersection(selected_card_keys)
        and _canonical_fact_text(slot.get("value"))
    }
    # Values read from an attached crop (figure/table/equation image) are L0
    # proof even when the serialized caption does not contain the number.  The
    # slot extractor already bound them to a crop-backed C-key and validated
    # them; the grounding gate must not erase them again.
    visual_slot_values = {
        _canonical_fact_text(slot.get("value"))
        for slot in internal.get("_validated_slots") or []
        if isinstance(slot, dict)
        and bool((slot.get("validation") or {}).get("visual_supported"))
        and set(str(key) for key in slot.get("support_keys") or []).intersection(selected_card_keys)
        and _canonical_fact_text(slot.get("value"))
    }
    # Any slot the extractor accepted (status supported) is already bound to
    # visible C-keys and passed deterministic type/numeric/condition checks.
    # Its value is a verified extraction, so the freeform/table gate must not
    # erase it merely because the literal span is absent from serialized text
    # (list answers, synthesized values, or values read from a crop).
    slot_supported_values = {
        _canonical_fact_text(slot.get("value"))
        for slot in internal.get("_validated_slots") or []
        if isinstance(slot, dict)
        and slot.get("status") == "supported"
        and set(str(key) for key in slot.get("support_keys") or []).intersection(selected_card_keys)
        and _canonical_fact_text(slot.get("value"))
    }
    for slot in internal.get("_validated_slots") or []:
        if not isinstance(slot, dict) or slot.get("status") != "supported":
            continue
        for table_row in slot.get("table_rows") or []:
            if not isinstance(table_row, dict):
                continue
            for cell in (table_row.get("values") or {}).values():
                canonical_cell = _canonical_fact_text(cell)
                if canonical_cell:
                    slot_supported_values.add(canonical_cell)
    answer = internal.get("answer") if isinstance(internal.get("answer"), dict) else {}
    freeform = answer.get("freeform") if isinstance(answer.get("freeform"), dict) else None
    if freeform and str(freeform.get("text") or "").strip():
        value = freeform.get("text")
        canonical = _canonical_fact_text(value)
        if (
            not _fact_is_directly_supported(value, "\n".join((raw_text, visual_text)))
            and canonical not in derived_values
            and canonical not in visual_slot_values
            and canonical not in slot_supported_values
        ):
            freeform["text"] = ""
            audit.append({"field": "answer.freeform.text", "value": str(value), "status": "unsupported_value_removed"})
        else:
            audit.append({
                "field": "answer.freeform.text",
                "status": "visual_slot_grounded"
                if canonical in visual_slot_values
                else "derived_value_grounded"
                if canonical in derived_values
                else "slot_supported_grounded"
                if canonical in slot_supported_values
                else "directly_grounded",
            })

    cleaned_plan: list[dict[str, Any]] = []
    for item in internal.get("table_answer_plan") or []:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("row_evidence_ref") or "")
        table_text = "\n".join((str((catalog.get(ref) or {}).get("text") or ""), *visual_text_by_ref.get(ref, [])))
        values = item.get("values") if isinstance(item.get("values"), dict) else {}
        nonempty = [value for value in values.values() if str(value or "").strip()]
        paper_id = str((catalog.get(ref) or {}).get("paper_id") or "")
        def supported_table_value(column: Any, value: Any) -> bool:
            # Paper title is a deterministic navigation identity, not a claim
            # copied from L0 text. It is allowed only when the model echoes the
            # exact source paper_id attached to its C-key; parser normalization
            # maps it to the trusted candidate title after this gate.
            if str(column).strip().lower() == "paper title" and str(value).strip() == paper_id:
                return True
            return (
                _fact_is_directly_supported(value, table_text)
                or _canonical_fact_text(value) in derived_values
                or _canonical_fact_text(value) in visual_slot_values
                or _canonical_fact_text(value) in slot_supported_values
            )
        if nonempty and all(supported_table_value(column, value) for column, value in values.items() if str(value or "").strip()):
            cleaned_plan.append(item)
        else:
            audit.append({"field": "table_answer_plan", "evidence_ref": ref, "status": "unsupported_table_row_removed"})
    if "table_answer_plan" in internal:
        internal["table_answer_plan"] = cleaned_plan


def _keyed_refinement_messages(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    draft: dict[str, Any],
    resolved_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Use only key-selected L0/L1 evidence to verify a keyed L2 draft."""
    rows = []
    for record in resolved_records:
        rows.append({
            "evidence_ref": record.get("evidence_ref"), "paper_id": record.get("paper_id"),
            "source_type": record.get("source_type"), "page": record.get("page"), "label": record.get("label"),
            "text": record.get("text"),
        })
    payload = {
        "query_id": sample.get("query_id"), "question": sample.get("question"),
        "answer_contract": contract, "candidate_papers": [{"paper_id": row.get("paper_id"), "title": row.get("title")} for row in candidates],
        "draft": {key: draft.get(key) for key in ("gold_papers", "contributing_papers", "answer", "table_answer_plan")},
        "resolved_evidence": rows,
    }
    user = (
        "You are a post-hoc factual verifier. The draft was produced from compressed evidence keys. "
        "Revise it only when the resolved raw evidence below directly contradicts or completes it. Do not add facts, papers, values, rows, or evidence beyond resolved_evidence. "
        "Return JSON only with query_id, gold_papers, evidence_refs, answer, and table_answer_plan when table is required. "
        "evidence_refs must contain only provided E#### strings. For a table answer, output at most 16 plan rows; each has row_evidence_ref copied from resolved_evidence, row_source with its provided paper/page/label, and values using exactly table_schema columns.\n"
        f"INPUT:{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": user}]


def _refine_keyed_draft(
    client: VLMAnswerClient,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    draft: dict[str, Any],
    hierarchy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    refs = [str(ref) for ref in draft.get("evidence_refs") or []]
    catalog = {str(row.get("evidence_ref") or ""): row for row in hierarchy.get("l0_catalog") or []}
    records = [catalog[ref] for ref in refs if ref in catalog]
    if not records:
        return draft, None
    result = client.generate_prediction(_keyed_refinement_messages(sample, candidates, contract, draft, records))
    revised, repaired = _extract_json_object_with_suffix_repair(str(result["content"]))
    draft_answer = draft.get("answer") if isinstance(draft.get("answer"), dict) else {}
    revised_answer = revised.get("answer") if isinstance(revised.get("answer"), dict) else {}
    # Refinement is corrective, never destructive: retain a grounded draft
    # field when the verifier omits a contract-required answer type.
    for answer_type in contract.get("answer_types") or []:
        if answer_type not in revised_answer and answer_type in draft_answer:
            revised_answer[answer_type] = draft_answer[answer_type]
    revised["answer"] = revised_answer
    if not revised.get("gold_papers"):
        revised["gold_papers"] = draft.get("gold_papers") or []
    if not revised.get("contributing_papers") and draft.get("contributing_papers"):
        revised["contributing_papers"] = draft.get("contributing_papers")
    allowed = {str(record.get("evidence_ref") or "") for record in records}
    revised["evidence_refs"] = [str(ref) for ref in revised.get("evidence_refs") or [] if str(ref) in allowed]
    if not revised["evidence_refs"]:
        revised["evidence_refs"] = refs
    audit: list[dict[str, Any]] = []
    revised["visual_support_card_keys"] = list(draft.get("visual_support_card_keys") or [])
    _apply_direct_fact_gate(revised, hierarchy, audit)
    revised["posthoc_grounding_audit"] = [*(draft.get("posthoc_grounding_audit") or []), *audit]
    return revised, {"content": result["content"], "raw_response": result["raw_response"], "json_suffix_repaired": repaired}


def _bounded_messages(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    config: Any,
    contract: dict[str, Any],
    max_prompt_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the highest-ranked prefix that fits the actual serialized prompt."""

    def build(prefix: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return build_symbolic_answer_prompt(
            sample,
            candidates,
            _selected_context(prefix),
            False,
            "docling",
            config.answer_model,
            contract,
        )

    if max_prompt_chars <= 0:
        return evidence, build(evidence)

    low, high, best_count = 0, len(evidence), 0
    best_messages = build([])
    while low <= high:
        midpoint = (low + high) // 2
        messages = build(evidence[:midpoint])
        if sum(len(str(message.get("content") or "")) for message in messages) <= max_prompt_chars:
            low = midpoint + 1
            best_count = midpoint
            best_messages = messages
        else:
            high = midpoint - 1
    return evidence[:best_count], best_messages


def _fit_coverage_to_prompt(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    config: Any,
    contract: dict[str, Any],
    max_prompt_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit records in coverage order against the serialized prompt, not a proxy.

    The package payload has nontrivial JSON metadata overhead. Rebuilding the
    prompt after each accepted record is cheap at the already budgeted scale and
    prevents a final prefix truncation from undoing the allocator's diversity.
    """
    if max_prompt_chars <= 0:
        return evidence, _bounded_messages(sample, candidates, evidence, config, contract, max_prompt_chars)[1]
    accepted: list[dict[str, Any]] = []
    messages = _bounded_messages(sample, candidates, [], config, contract, 0)[1]
    for record in evidence:
        proposed = [*accepted, record]
        # max_prompt_chars=0 requests one exact serialization without the
        # binary-prefix fitting used by the legacy helper.
        _, proposed_messages = _bounded_messages(sample, candidates, proposed, config, contract, 0)
        serialized = sum(len(str(message.get("content") or "")) for message in proposed_messages)
        if serialized <= max_prompt_chars:
            accepted = proposed
            messages = proposed_messages
    return accepted, messages


def _generate_json_draft(
    client: VLMAnswerClient,
    messages: list[dict[str, Any]],
    image_paths: list[str] | None,
    parse_retries: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Request a JSON draft, retrying only malformed model responses.

    Transport failures are already retried by ``VLMAnswerClient``.  A model can
    still produce prose or a truncated object after a successful HTTP response;
    retaining each attempt in the raw audit makes a replacement run traceable.
    """
    request_messages = messages
    raw_attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(max(0, parse_retries) + 1):
        result = client.generate_prediction(request_messages, image_paths=image_paths)
        raw_attempts.append({"parse_attempt": attempt, "content": result["content"], "raw_response": result["raw_response"]})
        try:
            parsed, repaired = _extract_json_object_with_suffix_repair(str(result["content"]))
            if repaired:
                raw_attempts[-1]["json_suffix_repaired"] = True
            return parsed, result, raw_attempts
        except Exception as exc:
            last_error = exc
            if attempt >= max(0, parse_retries):
                break
            request_messages = [
                *messages,
                {
                    "role": "user",
                    "content": "The previous response was not a complete JSON object. Return the requested JSON object only, with no markdown or explanation.",
                },
            ]
    raise JSONDraftError(f"Unable to parse answer JSON: {last_error}", raw_attempts)


def _extract_json_object_with_suffix_repair(content: str) -> tuple[dict[str, Any], bool]:
    """Parse model JSON, allowing only a tiny deterministic closing-delimiter repair.

    Some vision responses consistently omit the final ``}`` while otherwise
    returning a complete object.  The repair adds no values or keys: it is
    allowed only for an object prefix with at most two unclosed structural
    delimiters and no unterminated string or mismatched close delimiter.
    """
    try:
        return extract_json_object(content), False
    except Exception as original_error:
        start = content.find("{")
        candidate = content[start:].strip() if start >= 0 else ""
        if not candidate:
            raise original_error
        stack: list[str] = []
        in_string = False
        escaped = False
        for char in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or {"}": "{", "]": "["}[char] != stack[-1]:
                    raise original_error
                stack.pop()
        if in_string or not stack or len(stack) > 2:
            raise original_error
        suffix = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        try:
            parsed = json.loads(candidate + suffix)
        except json.JSONDecodeError:
            raise original_error
        if not isinstance(parsed, dict):
            raise original_error
        return parsed, True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_generation_provenance(
    *,
    validation_inputs_path: Path,
    selected_contexts_path: Path,
    candidate_papers_path: Path,
    hierarchy_path: Path | None,
    answer_contracts_path: Path | None,
    env_path: str,
    generation_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint every inference input and explicitly exclude gold labels."""
    source_paths = {
        "validation_inputs": validation_inputs_path,
        "selected_contexts": selected_contexts_path,
        "candidate_papers": candidate_papers_path,
    }
    if hierarchy_path:
        source_paths["evidence_hierarchy"] = hierarchy_path
    if answer_contracts_path:
        source_paths["answer_contracts"] = answer_contracts_path
    sources = {
        name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
        for name, path in source_paths.items()
    }
    if any(Path(item["path"]).name == "validation.jsonl" for item in sources.values()):
        raise ValueError("Generation provenance may not include validation.jsonl")
    return {
        "version": GENERATION_PROVENANCE_VERSION,
        "inference_inputs_only": True,
        "forbidden_inference_input": "validation.jsonl",
        "sources": sources,
        "env_path": str(env_path),
        "generation_parameters": generation_parameters,
    }


def _write_or_validate_provenance(output: Path, provenance: dict[str, Any], *, resume: bool) -> None:
    path = output / "generation_provenance.json"
    if resume and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError(
                "Refusing --resume with different generation inputs; use a new output directory or remove the stale cache explicitly."
            )
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    client = VLMAnswerClient(config, retries=config.generation_request_retries)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    only = {part.strip() for part in args.only_query_ids.split(",") if part.strip()}
    retry_query_ids = {part.strip() for part in args.retry_query_ids.split(",") if part.strip()}
    validation_inputs_path = find_official_file(args.official_dir, "validation_inputs.jsonl")
    selected_contexts_path = Path(args.selected_contexts_input)
    candidate_papers_path = Path(args.candidate_papers_input)
    hierarchy_path = Path(args.hierarchy_input) if args.hierarchy_input else None
    answer_contracts_path = Path(args.answer_contracts_input) if args.answer_contracts_input else None
    provenance = build_generation_provenance(
        validation_inputs_path=validation_inputs_path,
        selected_contexts_path=selected_contexts_path,
        candidate_papers_path=candidate_papers_path,
        hierarchy_path=hierarchy_path,
        answer_contracts_path=answer_contracts_path,
        env_path=args.env_path,
        generation_parameters={
            "hierarchy_prompt_mode": args.hierarchy_prompt_mode,
            "max_context_chars": args.max_context_chars,
            "max_prompt_chars": args.max_prompt_chars,
            "packing_mode": args.packing_mode,
            "record_text_budget": args.record_text_budget,
            "keyed_micro_text_chars": args.keyed_micro_text_chars,
            "keyed_micro_order": args.keyed_micro_order,
            "keyed_micro_index_chars": args.keyed_micro_index_chars,
            "keyed_card_limit": args.keyed_card_limit,
            "max_images": args.max_images,
            "posthoc_refinement": args.posthoc_refinement,
            "generation_mode": args.generation_mode,
        },
    )
    _write_or_validate_provenance(output, provenance, resume=args.resume)
    inputs = {str(row.get("query_id") or ""): row for row in read_jsonl(validation_inputs_path)}
    # Generation must be inference-only.  The public validation input provides
    # the question and output shape; validation.jsonl contains gold answers and
    # is intentionally reserved for the evaluator that runs after prediction.
    selections = {str(row.get("query_id") or ""): row for row in read_jsonl(selected_contexts_path)}
    hierarchies = {
        str(row.get("query_id") or ""): row.get("hierarchy")
        for row in read_jsonl(hierarchy_path)
        if isinstance(row.get("hierarchy"), dict)
    } if args.hierarchy_input else {}
    candidates_by_query = _candidate_map(candidate_papers_path)
    answer_contracts = {
        str(row.get("query_id") or ""): row
        for row in read_jsonl(answer_contracts_path)
    } if args.answer_contracts_input else {}
    def previous_rows(name: str) -> list[dict[str, Any]]:
        path = output / name
        return read_jsonl(path) if args.resume and path.exists() else []

    predictions = previous_rows("predictions.jsonl")
    internal_rows = previous_rows("internal_predictions.jsonl")
    raw_rows = previous_rows("raw_vlm_answer_responses.jsonl")
    recoveries = previous_rows("generation_recoveries.jsonl")
    previews = previous_rows("prompt_previews.jsonl")
    errors = previous_rows("errors.jsonl")
    generation_contexts = previous_rows("generation_selected_contexts.debug.jsonl")
    slot_plans = previous_rows("slot_plans.jsonl")
    slot_extractions = previous_rows("slot_extractions.jsonl")
    slot_validations = previous_rows("slot_validations.jsonl")
    if retry_query_ids:
        def without_retried(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [row for row in rows if str(row.get("query_id") or "") not in retry_query_ids]

        predictions = without_retried(predictions)
        internal_rows = without_retried(internal_rows)
        raw_rows = without_retried(raw_rows)
        recoveries = without_retried(recoveries)
        previews = without_retried(previews)
        errors = without_retried(errors)
        generation_contexts = without_retried(generation_contexts)
        slot_plans = without_retried(slot_plans)
        slot_extractions = without_retried(slot_extractions)
        slot_validations = without_retried(slot_validations)
    completed_query_ids = {str(row.get("query_id") or "") for row in predictions if row.get("query_id")}

    def checkpoint() -> None:
        """Persist completed queries so a transient API failure cannot lose a run."""
        write_jsonl(output / "predictions.jsonl", predictions)
        write_jsonl(output / "internal_predictions.jsonl", internal_rows)
        write_jsonl(output / "raw_vlm_answer_responses.jsonl", raw_rows)
        write_jsonl(output / "generation_recoveries.jsonl", recoveries)
        write_jsonl(output / "prompt_previews.jsonl", previews)
        write_jsonl(output / "generation_selected_contexts.debug.jsonl", generation_contexts)
        write_jsonl(output / "slot_plans.jsonl", slot_plans)
        write_jsonl(output / "slot_extractions.jsonl", slot_extractions)
        write_jsonl(output / "slot_validations.jsonl", slot_validations)
        write_jsonl(output / "errors.jsonl", errors)

    for query_id, sample in inputs.items():
        if only and query_id not in only:
            continue
        if query_id in completed_query_ids:
            continue
        candidates = candidates_by_query.get(query_id, [])
        hierarchy = hierarchies.get(query_id)
        records = (selections.get(query_id) or {}).get("selected_records") or []
        raw_records = [row for row in records if isinstance(row, dict)]
        if hierarchy:
            raw_records = [dict(row) for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)]
            evidence = raw_records
        elif args.packing_mode == "coverage":
            evidence = _prompt_records_coverage(
                raw_records,
                str(sample.get("question") or sample.get("query") or ""),
                sample.get("primary_evidence_type"),
                sample.get("task_family"),
                args.record_text_budget if args.record_text_budget > 0 else args.max_context_chars,
            )
        else:
            evidence = _prompt_records_ranked_prefix(raw_records, args.max_context_chars)
        contract = extract_answer_contract(sample, answer_contracts.get(query_id))
        max_images = config.evidence_hierarchy_max_images if args.max_images is None else max(0, args.max_images)
        refinement_enabled = (
            config.evidence_hierarchy_posthoc_refinement_enabled
            if args.posthoc_refinement == "auto"
            else args.posthoc_refinement == "true"
        )
        pre_prompt_count = len(evidence)
        slot_hierarchy: dict[str, Any] | None = None
        if hierarchy:
            hierarchy = copy.deepcopy(hierarchy)
            # Table-derived facts commonly answer freeform/MC questions. Keep
            # a bounded header + relevant-row view whenever table is the
            # primary source, rather than exposing only a caption.
            hierarchy["keyed_table_structure_enabled"] = (
                "table" in (contract.get("answer_types") or [])
                or str(sample.get("primary_evidence_type") or "") == "table"
            )
            if args.keyed_micro_text_chars is not None:
                hierarchy["keyed_micro_text_chars"] = max(40, args.keyed_micro_text_chars)
            if args.keyed_micro_order is not None:
                hierarchy["keyed_micro_order"] = args.keyed_micro_order
            if args.keyed_micro_index_chars is not None:
                hierarchy["keyed_micro_index_chars"] = max(0, args.keyed_micro_index_chars)
            if args.keyed_card_limit is not None:
                cards = list(hierarchy.get("l2_evidence_cards") or [])
                primary = str(sample.get("primary_evidence_type") or "")
                # A verified visual card is the only L2 source allowed to
                # carry crop-only facts. Prefer it within the query's primary
                # modality, while preserving tables as the primary choice for
                # normal table questions and never changing frozen selection.
                cards.sort(key=lambda card: (
                    str(card.get("source_type") or "") != primary,
                    str((card.get("verification") or {}).get("status") or "") != "visual_verified",
                ))
                if "multi" in str(sample.get("task_family") or "").lower():
                    preferred = [card for card in cards if str(card.get("source_type") or "") == primary]
                    fallback = [card for card in cards if card not in preferred]
                    # A multi-paper answer needs independent evidence from
                    # several papers. Preserve one primary-type card per paper
                    # before allowing repeated cards from a dominant paper.
                    diversified: list[dict[str, Any]] = []
                    seen_papers: set[str] = set()
                    for card in preferred:
                        paper_id = str(card.get("paper_id") or "")
                        if paper_id and paper_id not in seen_papers:
                            diversified.append(card)
                            seen_papers.add(paper_id)
                    diversified.extend(card for card in preferred if card not in diversified)
                    diversified.extend(fallback)
                    cards = diversified
                hierarchy["l2_evidence_cards"] = cards[: max(0, args.keyed_card_limit)]
            if args.keyed_table_view_limit is not None:
                hierarchy["keyed_table_view_limit"] = max(0, args.keyed_table_view_limit)
            if args.keyed_table_view_rows is not None:
                hierarchy["keyed_table_view_rows"] = max(1, args.keyed_table_view_rows)
            if args.generation_mode == "slots" and args.hierarchy_prompt_mode == "keyed":
                slot_hierarchy = ensure_slot_cards(copy.deepcopy(hierarchy))
            hierarchy, evidence, selected_context, messages = _fit_hierarchy_to_prompt(
                sample, candidates, evidence, hierarchy, config, contract, args.max_prompt_chars,
                max_images=max_images, keyed_mode=args.hierarchy_prompt_mode == "keyed",
            )
            if slot_hierarchy is not None:
                # Prompt fitting controls the legacy preview only. Slot calls
                # read from the full frozen L0 reservoir through C-keys.
                hierarchy = slot_hierarchy
                evidence = [dict(row) for row in hierarchy.get("l0_catalog") or [] if isinstance(row, dict)]
        else:
            evidence, messages = _fit_coverage_to_prompt(
                sample, candidates, evidence, config, contract, args.max_prompt_chars,
            )
            selected_context = _selected_context(evidence)
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        previews.append({
            "query_id": query_id,
            "pre_prompt_evidence_count": pre_prompt_count,
            "selected_evidence_count": len(evidence),
                "hierarchy_card_count": len((hierarchy or {}).get("l2_evidence_cards") or []),
            "prompt_chars": prompt_chars,
            "messages": messages,
        })
        generation_contexts.append({"query_id": query_id, "selected_records": evidence, "prompt_chars": prompt_chars, "evidence_hierarchy": hierarchy})
        if args.dry_run:
            continue
        try:
            image_paths = selected_context.get("attached_image_paths") if config.vlm2_context_mode == "cropped_image" else None
            if args.generation_mode == "slots":
                if not hierarchy or args.hierarchy_prompt_mode != "keyed":
                    raise ValueError("--generation-mode slots requires --hierarchy-input and --hierarchy-prompt-mode keyed")
                raw_plan, _result, raw_attempts = _generate_json_draft(
                    client, slot_plan_messages(sample, contract), None,
                    config.generation_parse_max_retries,
                )
                raw_rows.extend({"query_id": query_id, "phase": "slot_plan", **attempt} for attempt in raw_attempts)
                plan, plan_audit = validate_slot_plan(raw_plan, sample, candidates)
                slot_plans.append({"query_id": query_id, "plan": plan, "audit": plan_audit})
                extracted_slots: list[dict[str, Any]] = []
                for slot in plan["slots"]:
                    if slot.get("operation") in ("count", "direct"):
                        deterministic = deterministic_count_extraction(
                            slot, hierarchy, str(sample.get("question") or ""), candidates
                        )
                        if deterministic is not None:
                            extracted = deterministic
                            extraction_audit = [{
                                "slot_id": slot["id"],
                                "status": "slot_supported_deterministic",
                                "value": extracted.get("value"),
                            }]
                            raw_rows.append({
                                "query_id": query_id, "phase": f"slot_extract:{slot['id']}",
                                "parse_attempt": 0, "content": "deterministic_count", "raw_response": {},
                            })
                            extracted_slots.append(extracted)
                            slot_extractions.append({"query_id": query_id, "slot": slot, "extraction": extracted})
                            slot_validations.append({"query_id": query_id, "slot_id": slot["id"], "audit": extraction_audit})
                            continue
                    slot_attachments = (
                        slot_image_attachments(sample, slot, hierarchy, candidates, max_images=max_images)
                        if config.vlm2_context_mode == "cropped_image" else None
                    )
                    slot_images = [str(row["path"]) for row in slot_attachments or []]
                    table_schema = (contract.get("table") or {}).get("table_schema")
                    extracted = None
                    extraction_audit: list[dict[str, Any]] = []
                    for card_limit in (24, 48):
                        raw_slot, _result, raw_attempts = _generate_json_draft(
                            client,
                            slot_extraction_messages(
                                sample, slot, hierarchy, candidates, slot_attachments,
                                table_schema=table_schema, card_limit=card_limit,
                            ),
                            slot_images,
                            config.generation_parse_max_retries,
                        )
                        raw_rows.extend({"query_id": query_id, "phase": f"slot_extract:{slot['id']}", **attempt} for attempt in raw_attempts)
                        extracted, extraction_audit = validate_slot_extraction(
                            raw_slot, slot, hierarchy, table_schema=table_schema
                        )
                        if extracted["status"] == "supported" or card_limit == 48:
                            break
                        errors.append({
                            "query_id": query_id,
                            "type": "slot_extraction_retry_expanded_packet",
                            "slot_id": slot["id"],
                            "first_status": extracted["status"],
                            "card_limit": card_limit,
                        })
                    extracted_slots.append(extracted)
                    slot_extractions.append({"query_id": query_id, "slot": slot, "extraction": extracted})
                    slot_validations.append({"query_id": query_id, "slot_id": slot["id"], "audit": extraction_audit})
                internal, _result, raw_attempts = _generate_json_draft(
                    client, slot_composition_messages(sample, candidates, contract, extracted_slots), None,
                    config.generation_parse_max_retries,
                )
                raw_rows.extend({"query_id": query_id, "phase": "slot_compose", **attempt} for attempt in raw_attempts)
                internal, composition_audit = bind_composition_support(internal, extracted_slots)
                internal, format_audit = align_composed_values(internal, sample, extracted_slots)
                deterministic_table_plan: list[dict[str, Any]] = []
                for validated in extracted_slots:
                    if validated.get("status") != "supported":
                        continue
                    for table_row in validated.get("table_rows") or []:
                        row_keys = [str(key) for key in table_row.get("support_keys") or []]
                        if row_keys:
                            deterministic_table_plan.append({
                                "row_support_key": row_keys[0],
                                "values": dict(table_row.get("values") or {}),
                                "slot_id": validated.get("slot_id"),
                            })
                if deterministic_table_plan:
                    internal["table_answer_plan"] = deterministic_table_plan
                    format_audit.append({"status": "table_rows_filled_from_validated_slots", "row_count": len(deterministic_table_plan)})
                internal["_validated_slots"] = extracted_slots
                slot_validations.append({"query_id": query_id, "slot_id": "composition", "audit": [*composition_audit, *format_audit]})
            else:
                try:
                    internal, _result, raw_attempts = _generate_json_draft(
                        client,
                        messages,
                        image_paths=image_paths,
                        parse_retries=config.generation_parse_max_retries,
                    )
                    raw_rows.extend({"query_id": query_id, "phase": "primary", **attempt} for attempt in raw_attempts)
                except Exception as image_error:
                    # A vision endpoint may return a non-JSON response or reject a
                    # multimodal request even though the text-only endpoint is
                    # healthy. Preserve the failed response where available and
                    # recover with the identical keyed L2 prompt minus attachments.
                    # This never changes selection membership or exposes L0 data.
                    if not image_paths:
                        raise
                    if isinstance(image_error, JSONDraftError):
                        raw_rows.extend({"query_id": query_id, "phase": "cropped_image_parse_failure", **attempt} for attempt in image_error.raw_attempts)
                    recoveries.append({"query_id": query_id, "type": "cropped_image_text_only_recovery", "error": str(image_error)})
                    internal, _result, raw_attempts = _generate_json_draft(
                        client,
                        messages,
                        image_paths=None,
                        parse_retries=config.generation_parse_max_retries,
                    )
                    raw_rows.extend({"query_id": query_id, "phase": "text_only_recovery", **attempt} for attempt in raw_attempts)
            if hierarchy and args.hierarchy_prompt_mode == "keyed":
                internal, grounding_audit = _posthoc_ground_keyed_prediction(internal, hierarchy)
                if refinement_enabled and args.generation_mode == "direct":
                    try:
                        internal, refinement = _refine_keyed_draft(client, sample, candidates, contract, internal, hierarchy)
                        if refinement is not None:
                            raw_rows.append({"query_id": query_id, "phase": "posthoc_refinement", **refinement})
                    except Exception as refinement_error:
                        # The first-pass prediction is already grounded. A
                        # verifier outage must not replace it with a fallback.
                        recoveries.append({"query_id": query_id, "type": "posthoc_refinement_preserved_draft", "error": str(refinement_error)})
            else:
                grounding_audit = []
            prediction, row_errors = normalize_prediction(
                internal, sample, [str(row.get("paper_id") or "") for row in candidates],
                answer_contract=contract, selected_evidence=evidence,
                symbolic_evidence_standardization=config.symbolic_evidence_standardization,
                candidate_records=candidates,
            )
            if hierarchy:
                prediction, outside_removed = _restrict_prediction_to_visible_evidence(prediction, evidence)
                if outside_removed:
                    row_errors.append("evidence_outside_hierarchy_support_removed")
            if hierarchy and args.hierarchy_prompt_mode == "keyed" and not any(
                row.get("status") == "grounded" for row in grounding_audit
            ):
                row_errors.append("keyed_posthoc_grounding_empty")
            predictions.append(strip_internal_grounding(prediction))
            internal_rows.append(internal)
            for error in row_errors:
                if isinstance(error, dict):
                    errors.append({"query_id": query_id, **error})
                else:
                    errors.append({"query_id": query_id, "type": str(error)})
        except Exception as exc:
            predictions.append(make_fallback_prediction(sample, candidates[0] if candidates else None))
            errors.append({"query_id": query_id, "type": "generation_failure", "error": str(exc)})
        checkpoint()
        print(
            json.dumps(
                {"query_id": query_id, "completed": len(predictions), "errors": len(errors)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    checkpoint()
    print(json.dumps({"predictions": len(predictions), "contexts": len(generation_contexts), "errors": len(errors), "recoveries": len(recoveries), "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
