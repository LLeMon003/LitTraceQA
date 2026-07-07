from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_pipeline_config, model_slug
from .data_io import append_jsonl, extract_answer_contract, find_official_file, read_jsonl
from .metadata_index import build_metadata_records, retrieve_candidates
from .parser import extract_json_object
from .pdf_cache import ensure_candidate_pdfs
from .pdf_page_renderer import render_pdf_pages
from .symbolic_context_selector import select_symbolic_contexts
from .symbolic_validator import normalize_page_records, validate_page_structure
from .vlm_answer_client import VLMAnswerClient, probe_text_json
from .vlm_answer_prompt_builder import build_symbolic_answer_prompt
from .vlm_parser_client import VLMParserClient
from .vlm_parser_prompt_builder import build_page_parser_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe PDF VLM symbolic VLM baseline capabilities.")
    p.add_argument("--official-dir", default="official_dev")
    p.add_argument("--pdf-output-dir", default="raw_pdfs")
    p.add_argument("--processed-output-dir", default="processed_pdfs/vlm_symbolic")
    p.add_argument("--output-dir", default="outputs/pdf_vlm_symbolic_vlm_baseline")
    p.add_argument("--max-papers", type=int, default=1)
    p.add_argument("--env-path", default=".env")
    return p.parse_args()


def _report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# PDF VLM Symbolic VLM Capability Probe", ""]
    for row in rows:
        status = "通过" if row.get("ok") else "失败"
        lines.append(f"- {row.get('step')}: {status}。{row.get('detail', '')}")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本 probe 不伪造 parser artifacts、bbox、answer 或 evidence。",
            "- 如果 parser VLM 不支持 image input 或不能返回结构化 JSON，则 baseline 没有真正完成 VLM-1 document-to-symbol 步骤。",
            "- 如果 answer VLM 不能生成 JSON，则不会生成正式 prediction。",
            "- 当前 baseline 不使用 OCR，也不让 answer model 原生读取 PDF。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "capability_raw.jsonl"
    errors_path = out / "errors.jsonl"
    raw_path.write_text("", encoding="utf-8")
    report_path = out / "pipeline_capability_report.md"
    rows: list[dict[str, Any]] = []
    config = load_pipeline_config(args.env_path)
    try:
        metadata_rows = read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl"))
        metadata = build_metadata_records(metadata_rows)
        rows.append({"step": "读取 paper_metadata.jsonl", "ok": True, "detail": f"{len(metadata)} 条 metadata"})
    except Exception as exc:
        rows.append({"step": "读取 paper_metadata.jsonl", "ok": False, "detail": str(exc)})
        _report(report_path, rows)
        return 2
    try:
        inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
        rows.append({"step": "读取 validation_inputs.jsonl", "ok": True, "detail": f"{len(inputs)} 条 query"})
    except Exception as exc:
        rows.append({"step": "读取 validation_inputs.jsonl", "ok": False, "detail": str(exc)})
        _report(report_path, rows)
        return 2
    sample = inputs[0]
    candidates = retrieve_candidates(str(sample.get("question", "")), metadata, top_k=max(1, args.max_papers))
    metadata_by_id = {str(r.get("paper_id")): r for r in metadata}
    availability = ensure_candidate_pdfs(candidates[: args.max_papers], args.pdf_output_dir, metadata_by_id=metadata_by_id)
    append_jsonl(raw_path, {"step": "pdf_availability", "data": availability})
    pdf_rows = [r for r in availability["rows"] if r.get("available")]
    rows.append({"step": "找到或下载 1 个 PDF", "ok": bool(pdf_rows), "detail": f"available={len(pdf_rows)}"})
    if not pdf_rows:
        append_jsonl(errors_path, {"type": "pdf_unavailable", "availability": availability})
        _report(report_path, rows)
        return 2
    paper_id = pdf_rows[0]["paper_id"]
    paper_dir = Path(args.processed_output_dir) / model_slug(config.parser_model) / paper_id
    try:
        manifest = render_pdf_pages(paper_id, pdf_rows[0]["local_path"], paper_dir, dpi=config.render_dpi, image_format=config.render_format, max_pages=1)
        page = manifest["pages"][0]
        rows.append({"step": "渲染至少 1 页 PDF image", "ok": True, "detail": page["image_path"]})
    except Exception as exc:
        rows.append({"step": "渲染至少 1 页 PDF image", "ok": False, "detail": str(exc)})
        _report(report_path, rows)
        return 2
    parser = VLMParserClient(config)
    parser_supports = parser.supports_image_input()
    rows.append({"step": "parser VLM 支持 image input", "ok": parser_supports, "detail": config.parser_model})
    parser_records: list[dict[str, Any]] = []
    if parser_supports:
        try:
            messages = build_page_parser_prompt(
                paper_id,
                int(page["page"]),
                metadata_by_id.get(paper_id, {}),
                int(page["width_px"]),
                int(page["height_px"]),
                max_records_per_call=config.parser_max_records_per_call,
            )
            result = parser.generate_page_structure(messages, page["image_path"])
            append_jsonl(raw_path, {"step": "parser_vlm", "content": result["content"], "raw_response": result["raw_response"]})
            raw_obj = extract_json_object(str(result["content"]))
            repaired = validate_page_structure(raw_obj, paper_id, int(page["page"]), config.parser_extraction_mode)
            parser_records = normalize_page_records(
                repaired,
                config.parser_model,
                int(page["width_px"]),
                int(page["height_px"]),
                artifact_version=config.symbolic_artifact_version,
                parser_mode=config.parser_extraction_mode,
            )
            valid_count = sum(1 for r in parser_records if r.get("validation_status") != "rejected")
            rows.append({"step": "parser VLM 输出 parseable structured JSON", "ok": True, "detail": f"records={len(parser_records)}"})
            rows.append({"step": "symbolic validator 生成 valid symbolic record", "ok": valid_count > 0, "detail": f"valid={valid_count}"})
        except Exception as exc:
            append_jsonl(errors_path, {"type": "parser_probe_failure", "error": str(exc)})
            rows.append({"step": "parser VLM 输出 parseable structured JSON", "ok": False, "detail": str(exc)})
    answer_probe = probe_text_json(config.answer_model, config.answer_api_key, config.answer_base_url, min(config.answer_timeout_seconds, 30))
    append_jsonl(raw_path, {"step": "answer_text_json_probe", "data": answer_probe})
    rows.append({"step": "answer VLM 能生成 parseable JSON", "ok": bool(answer_probe.get("can_generate")), "detail": str(answer_probe.get("reason", ""))})
    answer_client = VLMAnswerClient(config)
    rows.append({"step": "answer VLM 支持 image input", "ok": answer_client.supports_image_input(), "detail": config.answer_model})
    if parser_records and answer_probe.get("can_generate"):
        selected = select_symbolic_contexts(str(sample.get("question", "")), parser_records, args.processed_output_dir, model_slug(config.parser_model), query_id=str(sample.get("query_id")))
        prompt = build_symbolic_answer_prompt(
            sample,
            candidates,
            selected,
            answer_client.supports_image_input(),
            config.parser_model,
            config.answer_model,
            extract_answer_contract(sample),
        )
        rows.append({"step": "完成一次 mini flow", "ok": bool(prompt and selected.get("selected_records")), "detail": f"selected_records={len(selected.get('selected_records', []))}"})
    else:
        rows.append({"step": "完成一次 mini flow", "ok": False, "detail": "parser 或 answer probe 不可用，未伪造 mini flow"})
    _report(report_path, rows)
    print(json.dumps({"report": str(report_path), "rows": rows}, ensure_ascii=False, indent=2))
    return 0 if all(row.get("ok") for row in rows if row.get("step") not in {"answer VLM 支持 image input"}) else 2


if __name__ == "__main__":
    raise SystemExit(main())
