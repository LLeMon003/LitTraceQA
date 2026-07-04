from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from .answer_client import AnswerClient, resolve_answer_generation_model
from .config import load_pipeline_config
from .context_index import retrieve_contexts_for_query
from .data_io import append_jsonl, find_official_file, read_jsonl
from .deepseek_ocr_converter import load_or_convert_pdf
from .metadata_index import build_metadata_records, retrieve_candidates
from .ocr_context_prompt_builder import build_ocr_context_answer_prompt
from .parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding
from .pdf_cache import ensure_candidate_pdfs, write_pdf_availability


BASELINE_TYPE = "pdf_ocr_context_vlm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF OCR context VLM baseline.")
    p.add_argument("--official-dir", default="official_dev")
    p.add_argument("--output-dir", default="outputs/pdf_ocr_context_vlm_baseline")
    p.add_argument("--pdf-output-dir", default="raw_pdfs")
    p.add_argument("--processed-output-dir", default="processed_pdfs/deepseek_ocr")
    p.add_argument("--index-output-dir", default="indexes/ocr_chunk_lexical")
    p.add_argument("--top-k-papers", type=int, default=4)
    p.add_argument("--top-n-text-contexts", type=int, default=12)
    p.add_argument("--top-n-visual-contexts", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fresh-run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--skip-context-selection", action="store_true")
    p.add_argument("--skip-embedding", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--ocr-overwrite", action="store_true")
    p.add_argument("--embedding-overwrite", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--pdf-sleep-seconds", type=float, default=2.0)
    p.add_argument("--pdf-timeout-seconds", type=float, default=60.0)
    p.add_argument("--pdf-max-retries", type=int, default=2)
    p.add_argument("--pdf-overwrite", action="store_true")
    p.add_argument("--env-path", default=".env")
    return p.parse_args()


def _reset_outputs(output_dir: Path, dry_run: bool, skip_generation: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_papers": output_dir / "candidate_papers.jsonl",
        "pdf_availability": output_dir / "pdf_availability.jsonl",
        "ocr_artifacts": output_dir / "ocr_artifacts.jsonl",
        "selected_contexts": output_dir / "selected_contexts.jsonl",
        "context_selection_report": output_dir / "context_selection_report.jsonl",
        "prompt_previews": output_dir / "prompt_previews.jsonl",
        "errors": output_dir / "errors.jsonl",
        "raw": output_dir / "raw_llm_responses.jsonl",
        "internal_predictions": output_dir / "internal_predictions.jsonl",
        "predictions": output_dir / "predictions.jsonl",
        "report": output_dir / "run_report.md",
    }
    for key, path in paths.items():
        if key == "predictions" and (dry_run or skip_generation):
            if path.exists():
                path.unlink()
            continue
        path.write_text("", encoding="utf-8")
    return paths


def _prepare_outputs(output_dir: Path, dry_run: bool, skip_generation: bool, resume: bool) -> dict[str, Path]:
    if not resume:
        return _reset_outputs(output_dir, dry_run, skip_generation)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_papers": output_dir / "candidate_papers.jsonl",
        "pdf_availability": output_dir / "pdf_availability.jsonl",
        "ocr_artifacts": output_dir / "ocr_artifacts.jsonl",
        "selected_contexts": output_dir / "selected_contexts.jsonl",
        "context_selection_report": output_dir / "context_selection_report.jsonl",
        "prompt_previews": output_dir / "prompt_previews.jsonl",
        "errors": output_dir / "errors.jsonl",
        "raw": output_dir / "raw_llm_responses.jsonl",
        "internal_predictions": output_dir / "internal_predictions.jsonl",
        "predictions": output_dir / "predictions.jsonl",
        "report": output_dir / "run_report.md",
    }
    for path in paths.values():
        if path.name == "predictions.jsonl" and (dry_run or skip_generation):
            continue
        path.touch(exist_ok=True)
    return paths


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# PDF OCR Context VLM Baseline Report",
        "",
        f"- baseline type: `{BASELINE_TYPE}`",
        f"- run_id: `{stats.get('run_id')}`",
        f"- processed query 数: {stats.get('processed_queries', 0)}",
        f"- resume skipped existing predictions 数: {stats.get('skipped_existing_predictions', 0)}",
        f"- top_k_papers: {stats.get('top_k_papers')}",
        f"- top_n_text_contexts: {stats.get('top_n_text_contexts')}",
        f"- top_n_visual_contexts: {stats.get('top_n_visual_contexts')}",
        f"- PDF cache 路径: `{stats.get('pdf_output_dir')}`",
        f"- existing PDFs 数: {stats.get('existing_pdfs', 0)}",
        f"- newly downloaded PDFs 数: {stats.get('newly_downloaded_pdfs', 0)}",
        f"- failed PDF downloads 数: {stats.get('failed_pdf_downloads', 0)}",
        f"- OCR model: `deepseek-ai/DeepSeek-OCR`",
        f"- OCR 成功 paper 数: {stats.get('ocr_success', 0)}",
        f"- OCR 失败 paper 数: {stats.get('ocr_failed', 0)}",
        f"- context selection method: `{stats.get('context_selection_method')}`",
        f"- OCR prompt version: `{stats.get('ocr_prompt_version')}`",
        f"- OCR provider: `{stats.get('ocr_provider')}`",
        f"- selected text context 总数: {stats.get('selected_text_contexts', 0)}",
        f"- selected visual context 总数: {stats.get('selected_visual_contexts', 0)}",
        f"- answer generation model: `{stats.get('answer_model')}`",
        f"- answer model source: `{stats.get('answer_model_source')}`",
        f"- VLM generation probe can_generate: {stats.get('vlm_generation_probe_can_generate')}",
        f"- VLM generation probe reason: {stats.get('vlm_generation_probe_reason')}",
        f"- answer model 是否支持 image input: {stats.get('answer_supports_images')}",
        f"- attached answer images 数: {stats.get('attached_answer_images', 0)}",
        f"- successful API calls: {stats.get('successful_api_calls', 0)}",
        f"- failed API calls: {stats.get('failed_api_calls', 0)}",
        f"- parse failures: {stats.get('parse_failures', 0)}",
        f"- fallback predictions: {stats.get('fallback_predictions', 0)}",
        f"- predictions generated: {stats.get('predictions_generated', False)}",
        f"- predictions 路径: `{stats.get('predictions_path')}`",
        f"- selected_contexts 路径: `{stats.get('selected_contexts_path')}`",
        "",
        "## Evaluator",
        "",
        (
            "```bash\n"
            f"python -m metadata_only_baseline.evaluate_local --official-dir official_dev --pred {stats.get('predictions_path')}\n"
            "```"
            if stats.get("predictions_generated")
            else "当前 run 没有生成可评估的 `predictions.jsonl`；不运行官方 evaluator。"
        ),
        "",
        "## 关键限制",
        "",
        "- 当前 baseline 是 `pdf_ocr_context_vlm`，不是 `pdf_native_llm`。",
        "- 当前不依赖 LLM API 原生读取 PDF。",
        "- 当前不做 bbox / region-level grounding，只做 context-level grounding。",
        "- 当前不做 table cell extraction、figure crop extraction、equation structure parsing、citation graph construction。",
        "- DeepSeek-OCR 是 input conversion 模块。",
        "- DeepSeek-OCR 输出 pages/chunks/visual_contexts，保留 page、bbox、reading_order、table_id、figure_id 等结构化字段。",
        "- 当前不需要独立 embedding 模型；context selection 从 OCR chunks / visual captions 做轻量 lexical selection。",
        "- 多模态 answer generation model 负责基于候选 metadata 与 OCR structured contexts 生成最终 JSON。",
        "- 当前 schema 为 bbox、reading_order、table_id、figure_id、equation_id、visual_id、parser_confidence 预留字段。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_image_paths(selected_contexts: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for visual in selected_contexts.get("selected_visual_contexts", []):
        if not isinstance(visual, dict):
            continue
        image_path = str(visual.get("image_path", "") or "")
        if image_path and image_path not in seen and Path(image_path).exists():
            seen.add(image_path)
            paths.append(image_path)
    return paths


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.fresh_run:
        args.resume = False
        args.ocr_overwrite = True
    paths = _prepare_outputs(output_dir, args.dry_run, args.skip_generation, args.resume)
    config = load_pipeline_config(args.env_path)
    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    metadata = build_metadata_records(read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl")))
    metadata_by_id = {str(record.get("paper_id", "")): record for record in metadata}
    if args.max_queries is not None:
        inputs = inputs[: args.max_queries]
    answer_resolution = resolve_answer_generation_model(config)
    answer_client = AnswerClient(config, model_override=str(answer_resolution["answer_model"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    completed_query_ids: set[str] = set()
    if args.resume and paths["predictions"].exists():
        completed_query_ids = {str(row.get("query_id", "")) for row in read_jsonl(paths["predictions"]) if row.get("query_id")}
    stats: dict[str, Any] = {
        "run_id": run_id,
        "processed_queries": 0,
        "skipped_existing_predictions": 0,
        "top_k_papers": args.top_k_papers,
        "top_n_text_contexts": args.top_n_text_contexts,
        "top_n_visual_contexts": args.top_n_visual_contexts,
        "pdf_output_dir": args.pdf_output_dir,
        "existing_pdfs": 0,
        "newly_downloaded_pdfs": 0,
        "failed_pdf_downloads": 0,
        "ocr_success": 0,
        "ocr_failed": 0,
        "context_selection_method": "ocr_chunk_lexical_without_embedding",
        "ocr_prompt_version": "deepseek_ocr_page_json_v2",
        "ocr_provider": config.ocr_provider,
        "selected_text_contexts": 0,
        "selected_visual_contexts": 0,
        "answer_model": str(answer_resolution["answer_model"]),
        "answer_model_source": answer_resolution["source"],
        "vlm_generation_probe_can_generate": answer_resolution["vlm_generation_probe"].get("can_generate"),
        "vlm_generation_probe_reason": answer_resolution["vlm_generation_probe"].get("reason"),
        "answer_supports_images": answer_client.supports_image_input(),
        "attached_answer_images": 0,
        "successful_api_calls": 0,
        "failed_api_calls": 0,
        "parse_failures": 0,
        "fallback_predictions": 0,
        "predictions_generated": False,
        "predictions_path": str(paths["predictions"]),
        "selected_contexts_path": str(paths["selected_contexts"]),
    }
    for sample in inputs:
        query_id = str(sample.get("query_id", ""))
        if query_id in completed_query_ids:
            stats["skipped_existing_predictions"] += 1
            continue
        candidates = retrieve_candidates(str(sample.get("question", "")), metadata, top_k=args.top_k_papers)
        append_jsonl(paths["candidate_papers"], {"query_id": query_id, "candidates": candidates})
        availability = ensure_candidate_pdfs(
            candidates,
            args.pdf_output_dir,
            overwrite=args.pdf_overwrite,
            metadata_by_id=metadata_by_id,
            sleep_seconds=args.pdf_sleep_seconds,
            timeout_seconds=args.pdf_timeout_seconds,
            max_retries=args.pdf_max_retries,
        )
        write_pdf_availability(paths["pdf_availability"], availability, query_id)
        stats["existing_pdfs"] += availability["existing_count"]
        stats["newly_downloaded_pdfs"] += availability["newly_downloaded_count"]
        stats["failed_pdf_downloads"] += availability["failed_count"]
        ocr_records: list[dict[str, Any]] = []
        if not args.skip_ocr:
            for pdf_row in availability["rows"]:
                if not pdf_row["available"]:
                    continue
                record = load_or_convert_pdf(
                    pdf_row["paper_id"],
                    pdf_row["local_path"],
                    args.processed_output_dir,
                    max_pages=config.ocr_max_pages_per_paper or None,
                    overwrite=args.ocr_overwrite,
                    api_key=config.ocr_api_key,
                    provider=config.ocr_provider,
                    base_url=config.ocr_base_url,
                    model=config.ocr_model,
                    timeout_seconds=config.answer_timeout_seconds,
                )
                ocr_records.append(record)
                append_jsonl(paths["ocr_artifacts"], {"query_id": query_id, **record})
                if record.get("status") == "ok":
                    stats["ocr_success"] += 1
                else:
                    stats["ocr_failed"] += 1
        selected = {"selected_text_contexts": [], "selected_visual_contexts": []}
        if not args.skip_context_selection and not args.skip_embedding:
            selected = retrieve_contexts_for_query(
                str(sample.get("question", "")),
                candidates,
                args.processed_output_dir,
                args.index_output_dir,
                None,
                top_n_text=args.top_n_text_contexts,
                top_n_visual=args.top_n_visual_contexts,
            )
            append_jsonl(
                paths["context_selection_report"],
                {
                    "query_id": query_id,
                    "method": stats["context_selection_method"],
                    "selected_text_count": len(selected["selected_text_contexts"]),
                    "selected_visual_count": len(selected["selected_visual_contexts"]),
                },
            )
        stats["selected_text_contexts"] += len(selected["selected_text_contexts"])
        stats["selected_visual_contexts"] += len(selected["selected_visual_contexts"])
        append_jsonl(paths["selected_contexts"], {"query_id": query_id, **selected})
        messages = build_ocr_context_answer_prompt(sample, candidates, selected, answer_client.supports_image_input())
        append_jsonl(paths["prompt_previews"], {"query_id": query_id, "messages": messages})
        stats["processed_queries"] += 1
        if args.dry_run or args.skip_generation:
            continue
        if not selected["selected_text_contexts"] and not selected["selected_visual_contexts"]:
            append_jsonl(
                paths["errors"],
                {
                    "query_id": query_id,
                    "type": "context_retrieval_unavailable",
                    "error": "No selected OCR text or visual contexts are available; generation skipped to avoid faking pdf_ocr_context_vlm predictions.",
                },
            )
            continue
        try:
            image_paths = _selected_image_paths(selected) if answer_client.supports_image_input() else []
            stats["attached_answer_images"] += len(image_paths)
            result = answer_client.generate_json(messages, image_paths=image_paths)
            stats["successful_api_calls"] += 1
            append_jsonl(paths["raw"], {"query_id": query_id, **result})
            parsed = extract_json_object(result["content"])
            internal, errors = normalize_prediction(parsed, sample, [str(c.get("paper_id", "")) for c in candidates])
            for err in errors:
                append_jsonl(paths["errors"], err)
            append_jsonl(paths["internal_predictions"], internal)
            append_jsonl(paths["predictions"], strip_internal_grounding(internal))
        except Exception as exc:
            stats["failed_api_calls"] += 1
            stats["fallback_predictions"] += 1
            append_jsonl(paths["errors"], {"query_id": query_id, "type": "generation_failure", "error": str(exc)})
            fallback = make_fallback_prediction(sample, candidates[0] if candidates else None)
            append_jsonl(paths["predictions"], fallback)
    if stats["successful_api_calls"] == 0 and paths["predictions"].exists() and paths["predictions"].stat().st_size == 0:
        paths["predictions"].unlink()
    stats["predictions_generated"] = paths["predictions"].exists() and paths["predictions"].stat().st_size > 0
    _write_report(paths["report"], stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
