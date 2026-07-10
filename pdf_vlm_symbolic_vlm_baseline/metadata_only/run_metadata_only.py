from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from ..config import load_pipeline_config
from ..data_io import append_jsonl, find_official_file, read_jsonl
from ..metadata_index import HYBRID_SCORE_WEIGHTS, build_metadata_records, retrieve_candidates
from ..metadata_selection import (
    build_metadata_selection_messages,
    empty_answer_for_sample,
    empty_metadata_prediction,
    normalize_metadata_selection,
    selection_candidates_for_metadata_vlm,
)
from ..parser import extract_json_object
from ..topic_expansion import expand_candidates_with_topic_profiles
from ..vlm_answer_client import VLMAnswerClient

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata-only LitTraceQA retrieval run.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--top-k-papers", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--paper-output-mode",
        choices=["top1", "top-k", "task-family"],
        default="task-family",
        help="Diagnostic direct mode only: how many retrieved papers to submit as predicted gold_papers.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["vlm2", "direct"],
        default="vlm2",
        help="Use VLM-2 over metadata candidates. The direct mode is only for retrieval diagnostics.",
    )
    return parser.parse_args()


def _task_family_bucket(task_family: str) -> str:
    normalized = task_family.strip().lower().replace("-", "_")
    return "multi_paper" if "multi" in normalized else "single_paper"


def _query_top_k(sample: dict[str, Any], config: Any, fallback_top_k: int) -> int:
    if not config.task_family_budget_enabled:
        return fallback_top_k
    if _task_family_bucket(str(sample.get("task_family") or "")) == "multi_paper":
        return int(config.multi_paper_top_k_papers)
    return int(config.single_paper_top_k_papers)


def _empty_answer(sample: dict[str, Any]) -> dict[str, Any]:
    return empty_answer_for_sample(sample)


def _predicted_papers(sample: dict[str, Any], candidates: list[dict[str, Any]], mode: str) -> list[dict[str, str]]:
    if mode == "top1":
        selected = candidates[:1]
    elif mode == "top-k":
        selected = candidates
    else:
        if _task_family_bucket(str(sample.get("task_family") or "")) == "multi_paper":
            topic_candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("retrieval_method") or "") == "hybrid_alias_topic_optin"
            ]
            selected = topic_candidates or candidates
        else:
            selected = candidates[:1]
    seen: set[str] = set()
    papers: list[dict[str, str]] = []
    for candidate in selected:
        paper_id = str(candidate.get("paper_id") or "")
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            papers.append({"paper_id": paper_id})
    return papers


def _prediction(sample: dict[str, Any], candidates: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "query_id": sample.get("query_id"),
        "gold_papers": _predicted_papers(sample, candidates, mode),
        "evidence": [],
        "answer": _empty_answer(sample),
    }


def _build_metadata_selection_messages(sample: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    return build_metadata_selection_messages(sample, candidates)


def _normalize_selected_papers(
    obj: dict[str, Any],
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return normalize_metadata_selection(obj, sample, candidates)


def _gold_ids(row: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in row.get("gold_papers") or row.get("papers") or []:
        if isinstance(item, dict):
            value = item.get("paper_id")
        else:
            value = item
        if value:
            ids.add(str(value))
    return ids


def _paper_prf(pred_ids: set[str], gold_ids: set[str]) -> tuple[float, float, float]:
    if not pred_ids and not gold_ids:
        return 1.0, 1.0, 1.0
    correct = len(pred_ids & gold_ids)
    precision = correct / len(pred_ids) if pred_ids else 0.0
    recall = correct / len(gold_ids) if gold_ids else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Metadata-Only Retrieval Report",
        "",
        f"- created_at: {stats['created_at']}",
        f"- retrieval_method: `{stats['retrieval_method']}`",
        f"- query_decomposition_enabled: {stats['query_decomposition_enabled']}",
        f"- subquery_top_k: {stats['subquery_top_k']}",
        f"- topic_expansion_enabled: {stats['topic_expansion_enabled']}",
        f"- paper_output_mode: `{stats['paper_output_mode']}`",
        f"- selection_mode: `{stats['selection_mode']}`",
        f"- direct_mode_is_diagnostic_only: {stats['selection_mode'] == 'direct'}",
        f"- answer_model: `{stats['answer_model']}`",
        f"- metadata_only_retrieval_eval_model: `{stats['metadata_only_retrieval_eval_model']}`",
        f"- vlm2_selection_calls: {stats['vlm2_selection_calls']}",
        f"- vlm2_selection_failures: {stats['vlm2_selection_failures']}",
        f"- processed_queries: {stats['processed_queries']}",
        f"- macro_paper_precision: {stats['macro_paper_precision']}",
        f"- macro_paper_recall: {stats['macro_paper_recall']}",
        f"- macro_paper_f1: {stats['macro_paper_f1']}",
        f"- top_k_full_gold_coverage: {stats['top_k_full_gold_coverage']}",
        f"- top1_gold_hit_rate: {stats['top1_gold_hit_rate']}",
        f"- predictions_path: `{stats['predictions_path']}`",
        f"- candidate_papers_path: `{stats['candidate_papers_path']}`",
        "",
        "This command runs metadata retrieval plus paper selection only. It does not perform page routing, PDF rendering, VLM-1 parsing, or symbolic selection. In the default VLM-2 selection mode, retrieval top-k papers are only metadata candidates; VLM-2 sees their title/abstract metadata and selects prediction.gold_papers. Evidence and non-paper answer fields are intentionally empty. Direct mode is a retrieval diagnostic and is not the metadata-only baseline contract.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    candidates_path = output_dir / "candidate_papers.jsonl"
    raw_vlm_path = output_dir / "raw_vlm_metadata_selection.jsonl"
    prompt_path = output_dir / "metadata_selection_prompts.jsonl"
    errors_path = output_dir / "errors.jsonl"
    report_path = output_dir / "metadata_only_report.md"
    for path in (predictions_path, candidates_path, raw_vlm_path, prompt_path, errors_path):
        path.write_text("", encoding="utf-8")

    config = load_pipeline_config(args.env_path)
    metadata_selection_config = replace(config, answer_model=config.metadata_only_retrieval_eval_model)
    answer_client = VLMAnswerClient(metadata_selection_config)
    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    if args.max_queries is not None:
        inputs = inputs[: args.max_queries]
    gold_by_id = {str(row.get("query_id") or ""): row for row in read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))}
    metadata = build_metadata_records(read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl")))

    predictions: list[dict[str, Any]] = []
    paper_precisions: list[float] = []
    paper_recalls: list[float] = []
    paper_f1s: list[float] = []
    full_coverage = 0
    top1_hits = 0

    iterator: Any = inputs
    if args.show_progress and tqdm is not None:
        iterator = tqdm(inputs, desc="metadata-only", unit="query")

    for sample in iterator:
        query_id = str(sample.get("query_id") or "")
        effective_top_k = _query_top_k(sample, config, args.top_k_papers)
        candidates = retrieve_candidates(
            str(sample.get("question") or ""),
            metadata,
            effective_top_k,
            method=config.retrieval_method,
            enable_query_decomposition=(
                config.retrieval_enable_query_decomposition
                and _task_family_bucket(str(sample.get("task_family") or "")) == "multi_paper"
            ),
            subquery_top_k=config.retrieval_subquery_top_k,
        )
        topic_info = None
        if config.retrieval_enable_topic_expansion:
            candidates, topic_info = expand_candidates_with_topic_profiles(sample, candidates, metadata, effective_top_k)
        selection_candidates, selection_policy = selection_candidates_for_metadata_vlm(candidates)
        if args.selection_mode == "vlm2":
            messages = _build_metadata_selection_messages(sample, selection_candidates)
            append_jsonl(prompt_path, {"query_id": query_id, "selection_policy": selection_policy, "messages": messages})
            try:
                result = answer_client.generate_prediction(messages, image_paths=None)
                append_jsonl(raw_vlm_path, {"query_id": query_id, "content": result["content"], "raw_response": result["raw_response"]})
                selected = extract_json_object(str(result["content"]))
                prediction, errors = _normalize_selected_papers(selected, sample, selection_candidates)
                for error in errors:
                    append_jsonl(errors_path, error)
            except Exception as exc:
                append_jsonl(errors_path, {"query_id": query_id, "type": "metadata_vlm_selection_failure", "error": str(exc)})
                prediction = empty_metadata_prediction(sample)
        else:
            prediction = _prediction(sample, candidates, args.paper_output_mode)
        append_jsonl(
            candidates_path,
            {
                "query_id": query_id,
                "effective_top_k_papers": effective_top_k,
                "topic_expansion": topic_info,
                "selection_policy": selection_policy,
                "selection_candidates": selection_candidates,
                "candidates": candidates,
            },
        )
        append_jsonl(predictions_path, prediction)
        predictions.append(prediction)

        gold_ids = _gold_ids(gold_by_id.get(query_id, {}))
        candidate_ids = {str(candidate.get("paper_id") or "") for candidate in candidates if candidate.get("paper_id")}
        pred_ids = {paper["paper_id"] for paper in prediction["gold_papers"]}
        precision, recall, f1 = _paper_prf(pred_ids, gold_ids)
        paper_precisions.append(precision)
        paper_recalls.append(recall)
        paper_f1s.append(f1)
        if gold_ids and gold_ids <= candidate_ids:
            full_coverage += 1
        if candidates and str(candidates[0].get("paper_id") or "") in gold_ids:
            top1_hits += 1

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "retrieval_method": config.retrieval_method,
        "query_decomposition_enabled": config.retrieval_enable_query_decomposition,
        "subquery_top_k": config.retrieval_subquery_top_k,
        "topic_expansion_enabled": config.retrieval_enable_topic_expansion,
        "paper_output_mode": args.paper_output_mode,
        "selection_mode": args.selection_mode,
        "answer_model": config.answer_model,
        "metadata_only_retrieval_eval_model": metadata_selection_config.answer_model,
        "vlm2_selection_calls": len(predictions) if args.selection_mode == "vlm2" else 0,
        "vlm2_selection_failures": sum(
            1
            for line in errors_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("type") == "metadata_vlm_selection_failure"
        ),
        "processed_queries": len(predictions),
        "macro_paper_precision": round(mean(paper_precisions), 6) if paper_precisions else 0.0,
        "macro_paper_recall": round(mean(paper_recalls), 6) if paper_recalls else 0.0,
        "macro_paper_f1": round(mean(paper_f1s), 6) if paper_f1s else 0.0,
        "top_k_full_gold_coverage": round(full_coverage / max(1, len(predictions)), 6),
        "top1_gold_hit_rate": round(top1_hits / max(1, len(predictions)), 6),
        "predictions_path": str(predictions_path),
        "candidate_papers_path": str(candidates_path),
    }
    _write_report(report_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
