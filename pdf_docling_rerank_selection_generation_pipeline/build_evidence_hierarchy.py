"""Build L0-L3 artifacts from frozen selected contexts without reranking."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .contextual_triples import attach_contextual_triple_graph
from .data_io import find_official_file, read_jsonl, write_jsonl
from .evidence_hierarchy import attach_cards, build_l0_l1_l3, card_generation_messages, hierarchy_metrics, load_processed_records
from .vlm_answer_client import VLMAnswerClient


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build auditable L0-L3 evidence hierarchy from frozen selection.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--selected-contexts-input", required=True)
    parser.add_argument("--candidate-papers-input", required=True)
    parser.add_argument("--processed-output-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--mode", choices=("extractive", "verified_llm"), default="")
    parser.add_argument(
        "--contextual-triples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add deterministic L1 evidence windows plus L2 candidate triples and L3 relation navigation. This never reranks selection.",
    )
    parser.add_argument("--only-query-ids", default="")
    parser.add_argument("--dry-run", action="store_true", help="Build deterministic L0/L1/L3 and extractive L2 cards without an API call.")
    return parser.parse_args()


def _candidate_map(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            rows[query_id].append(row)
    for values in rows.values():
        values.sort(key=lambda row: int(row.get("rank") or 10**9))
    return rows


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    mode = args.mode or config.evidence_hierarchy_card_mode
    only = {part.strip() for part in args.only_query_ids.split(",") if part.strip()}
    inputs = {str(row.get("query_id") or ""): row for row in read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))}
    selections = {str(row.get("query_id") or ""): row for row in read_jsonl(args.selected_contexts_input)}
    candidates_by_query = _candidate_map(args.candidate_papers_input)
    paper_ids = {str(record.get("paper_id") or "") for row in selections.values() for record in row.get("selected_records") or [] if isinstance(record, dict)}
    processed_records = load_processed_records(args.processed_output_dir, paper_ids)
    client = VLMAnswerClient(config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, prompts = [], [], []
    aggregate: dict[str, float] = defaultdict(float)
    for query_id, sample in inputs.items():
        if only and query_id not in only:
            continue
        selected = (selections.get(query_id) or {}).get("selected_records") or []
        candidates = candidates_by_query.get(query_id, [])
        hierarchy = build_l0_l1_l3(
            [record for record in selected if isinstance(record, dict)], candidates, processed_records,
            question=str(sample.get("question") or ""),
            l1_max_chars=config.evidence_hierarchy_l1_max_chars,
            l3_paper_chars=config.evidence_hierarchy_l3_paper_chars,
        )
        hierarchy["task_family"] = sample.get("task_family")
        hierarchy["primary_evidence_type"] = sample.get("primary_evidence_type")
        hierarchy["prompt_micro_index_chars"] = config.evidence_hierarchy_micro_index_chars
        hierarchy["prompt_micro_text_chars"] = config.evidence_hierarchy_micro_text_chars
        hierarchy["keyed_micro_index_chars"] = config.evidence_hierarchy_keyed_micro_index_chars
        hierarchy["keyed_micro_text_chars"] = config.evidence_hierarchy_keyed_micro_text_chars
        hierarchy["keyed_micro_order"] = config.evidence_hierarchy_keyed_micro_order
        raw_card_response: dict[str, Any] | None = None
        if mode == "verified_llm" and not args.dry_run:
            messages = card_generation_messages(
                str(sample.get("question") or ""), hierarchy, config.evidence_hierarchy_max_claims,
                config.evidence_hierarchy_max_cards, config.evidence_hierarchy_card_source_chars,
            )
            prompts.append({"query_id": query_id, "messages": messages})
            try:
                result = client.generate_prediction(messages, max_tokens=config.evidence_hierarchy_card_max_tokens)
                from .parser import extract_json_object
                raw_card_response = extract_json_object(str(result["content"]))
                hierarchy["raw_card_response"] = raw_card_response
            except Exception as exc:
                errors.append({"query_id": query_id, "type": "hierarchy_card_generation_failure", "error": str(exc)})
        hierarchy = attach_cards(
            str(sample.get("question") or ""), hierarchy, mode=mode,
            max_claims=config.evidence_hierarchy_max_claims, max_cards=config.evidence_hierarchy_max_cards,
            primary_evidence_type=str(sample.get("primary_evidence_type") or ""),
            llm_result=raw_card_response,
        )
        if args.contextual_triples:
            hierarchy = attach_contextual_triple_graph(
                str(sample.get("question") or ""), hierarchy,
                l1_sentence_chars=config.evidence_hierarchy_l1_max_chars,
            )
        metrics = hierarchy_metrics(hierarchy)
        aggregate["queries"] += 1
        for key in ("l0_record_count", "selected_anchor_count", "l1_context_count", "l2_card_count", "l0_raw_chars", "l2_l3_serialized_chars"):
            aggregate[key] += float(metrics[key])
        rows.append({"query_id": query_id, "question": sample.get("question"), "task_family": sample.get("task_family"), "primary_evidence_type": sample.get("primary_evidence_type"), "hierarchy": hierarchy, "metrics": metrics})
        write_jsonl(output / "evidence_hierarchy.jsonl", rows)
        write_jsonl(output / "errors.jsonl", errors)
        write_jsonl(output / "card_prompt_previews.jsonl", prompts)
    count = max(1.0, aggregate["queries"])
    summary = {"queries": int(aggregate["queries"]), "mode": mode, **{f"avg_{key}": round(value / count, 3) for key, value in aggregate.items() if key != "queries"}, "error_count": len(errors)}
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
