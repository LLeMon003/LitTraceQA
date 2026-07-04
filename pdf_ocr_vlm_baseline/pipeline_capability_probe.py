from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer_client import AnswerClient, resolve_answer_generation_model
from .config import is_api_key_configured, load_pipeline_config, mask_api_key
from .data_io import append_jsonl, find_official_file, read_jsonl
from .metadata_index import build_metadata_records, retrieve_candidates
from .pdf_cache import ensure_candidate_pdfs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capability probe for PDF OCR context VLM baseline.")
    p.add_argument("--official-dir", default="official_dev")
    p.add_argument("--pdf-output-dir", default="raw_pdfs")
    p.add_argument("--processed-output-dir", default="processed_pdfs/deepseek_ocr")
    p.add_argument("--index-output-dir", default="indexes/ocr_chunk_lexical")
    p.add_argument("--output-dir", default="outputs/pdf_ocr_context_vlm_baseline")
    p.add_argument("--max-papers", type=int, default=1)
    p.add_argument("--env-path", default=".env")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "pipeline_capability_report.md"
    raw_path = out / "capability_raw.jsonl"
    errors_path = out / "errors.jsonl"
    raw_path.write_text("", encoding="utf-8")
    errors_path.touch(exist_ok=True)
    config = load_pipeline_config(args.env_path)
    result: dict[str, object] = {
        "metadata_readable": False,
        "pdf_available": False,
        "ocr_model": config.ocr_model,
        "ocr_available": False,
        "answer_model": config.answer_model,
        "answer_generation_available": False,
        "answer_supports_image_input": False,
    }
    try:
        metadata = build_metadata_records(read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl")))
        result["metadata_readable"] = True
        probe_candidates = metadata[: args.max_papers]
        availability = ensure_candidate_pdfs(probe_candidates, args.pdf_output_dir)
        result["pdf_available"] = availability["existing_count"] > 0
        result["pdf_availability"] = availability
    except Exception as exc:
        append_jsonl(errors_path, {"type": "metadata_or_pdf_probe_failure", "error": str(exc)})
    result["ocr_available"] = bool(config.ocr_model and config.ocr_provider != "local" and is_api_key_configured(config.ocr_api_key))
    result["ocr_reason"] = (
        "OCR API configuration is present; provider-specific OCR request must be validated by an OCR smoke call."
        if result["ocr_available"]
        else "OCR API config is incomplete or provider is local."
    )
    answer_resolution = resolve_answer_generation_model(config)
    answer = AnswerClient(config, model_override=str(answer_resolution["answer_model"]))
    result["answer_generation_available"] = answer.supports_text_generation()
    result["answer_supports_image_input"] = answer.supports_image_input()
    result["vlm_generation_probe"] = {
        "model": answer_resolution["vlm_generation_probe"].get("model"),
        "can_generate": answer_resolution["vlm_generation_probe"].get("can_generate"),
        "reason": answer_resolution["vlm_generation_probe"].get("reason"),
        "content": answer_resolution["vlm_generation_probe"].get("content", ""),
    }
    result["resolved_answer_model"] = answer_resolution["answer_model"]
    result["answer_model_source"] = answer_resolution["source"]
    append_jsonl(raw_path, result)
    lines = [
        "# Pipeline Capability Report",
        "",
        f"- metadata 可读取: {result['metadata_readable']}",
        f"- 至少 1 个 PDF 本地可用: {result['pdf_available']}",
        f"- OCR model: `{config.ocr_model}`",
        f"- DeepSeek-OCR 可用: {result['ocr_available']}",
        f"- OCR 不可用原因: {result.get('ocr_reason', '')}",
        "- 独立 embedding 模型: 不需要",
        "- context selection: OCR chunks / visual captions lexical selection",
        f"- VLM model generation capability: {result['vlm_generation_probe']}",
        f"- resolved answer model: `{answer_resolution['answer_model']}`",
        f"- answer model source: `{answer_resolution['source']}`",
        f"- answer API key: {mask_api_key(config.answer_api_key or '')}",
        f"- answer generation 可用: {answer.supports_text_generation()}",
        f"- answer model 支持 image input: {answer.supports_image_input()}",
        "",
        "## 结论",
        "",
        "当前 baseline 按 API provider 设计，不需要本地 torch/transformers。",
        "生成模型选择逻辑：先测试 ANSWER_MODEL 是否能 chat generation；只有不能生成时，才 fallback 到 BASE_GENERATION_MODEL / SILICONFLOW_MODEL。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "raw": str(raw_path), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
