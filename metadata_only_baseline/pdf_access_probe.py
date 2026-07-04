from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import is_api_key_configured, load_config, mask_api_key
from .data_io import append_jsonl, find_official_file, read_jsonl
from .link_utils import extract_identifiers, extract_online_links
from .llm_client import SiliconFlowClient
from .parser import extract_json_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny probe for online PDF/URL access capability.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", default="outputs/pdf_access_probe")
    parser.add_argument("--max-papers", type=int, default=2)
    parser.add_argument("--env-path", default=".env")
    return parser.parse_args()


def _build_probe_prompt(paper: dict[str, Any], links: list[str]) -> list[dict[str, str]]:
    payload = {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "links": links[:5],
    }
    system = (
        "You are testing whether the current API/model can access online URLs or PDFs. "
        "Be honest. If you cannot open external links, say so clearly. Output valid JSON only."
    )
    user = (
        "Try to determine whether you can access the provided online paper links. "
        "If you can access a URL/PDF, attempt to open one link and answer a simple question that "
        "should be verifiable near the paper title or abstract: what is the paper title? "
        "If you cannot access URLs/PDFs, do not pretend that you accessed them.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return this JSON shape only:\n"
        "{\n"
        '  "can_access_online_pdf": true,\n'
        '  "accessed_links": [],\n'
        '  "evidence_of_access": "",\n'
        '  "answer": "",\n'
        '  "limitations": ""\n'
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _select_papers(metadata_rows: list[dict[str, Any]], max_papers: int) -> list[tuple[dict[str, Any], list[str], dict[str, list[str]]]]:
    selected: list[tuple[dict[str, Any], list[str], dict[str, list[str]]]] = []
    for row in metadata_rows:
        links = extract_online_links(row)
        if not links:
            continue
        selected.append((row, links, extract_identifiers(row)))
        if len(selected) >= max_papers:
            break
    return selected


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "pdf_access_probe_raw_responses.jsonl"
    errors_path = output_dir / "pdf_access_probe_errors.jsonl"
    report_path = output_dir / "pdf_access_probe_report.md"
    raw_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")

    config = load_config(args.env_path)
    if not config.env_exists:
        print(f"未检测到 .env 文件：{config.env_path}。停止 PDF-access probe。")
        report_path.write_text("# PDF-Access Probe Report\n\n未检测到 `.env`，未进行 API probe。\n", encoding="utf-8")
        return 2
    if not is_api_key_configured(config.api_key):
        print("SILICONFLOW_API_KEY 未补齐或仍是占位符。停止 PDF-access probe。")
        report_path.write_text("# PDF-Access Probe Report\n\nAPI key 未补齐，未进行 API probe。\n", encoding="utf-8")
        return 2

    metadata_path = find_official_file(args.official_dir, "paper_metadata.jsonl")
    selected = _select_papers(read_jsonl(metadata_path), args.max_papers)
    client = SiliconFlowClient(config)
    parsed_results: list[dict[str, Any]] = []
    for paper, links, identifiers in selected:
        messages = _build_probe_prompt(paper, links)
        try:
            content, raw_response = client.chat_completions(messages)
            parsed: dict[str, Any]
            try:
                parsed = extract_json_object(content)
            except Exception as exc:
                parsed = {"parse_error": str(exc), "content": content}
            record = {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "links": links[:5],
                "identifiers": identifiers,
                "content": content,
                "parsed": parsed,
                "raw_response": raw_response,
            }
            append_jsonl(raw_path, record)
            parsed_results.append(record)
        except Exception as exc:
            append_jsonl(errors_path, {"paper_id": paper.get("paper_id"), "error": str(exc), "links": links[:5]})

    claims_access = [
        item
        for item in parsed_results
        if isinstance(item.get("parsed"), dict) and item["parsed"].get("can_access_online_pdf") is True
    ]
    enough_evidence = [
        item
        for item in claims_access
        if isinstance(item.get("parsed"), dict) and str(item["parsed"].get("evidence_of_access", "")).strip()
    ]
    lines = [
        "# PDF-Access Probe Report",
        "",
        f"- 是否完成 probe: {bool(parsed_results)}",
        f"- 使用模型: `{config.model}`",
        f"- base url: `{config.base_url}`",
        f"- API key: {mask_api_key(config.api_key or '')}",
        f"- 测试论文数量: {len(selected)}",
        f"- 模型声称可访问 PDF/URL 的次数: {len(claims_access)}",
        f"- 有访问证据描述的次数: {len(enough_evidence)}",
        "",
        "## 测试链接",
    ]
    for item in parsed_results:
        lines.append(f"- `{item.get('paper_id')}`: {', '.join(item.get('links', []))}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "该 probe 只用于极微量能力验证，不参与 metadata-only baseline，也不修改 predictions。"
            "如果模型只是声称能访问但没有给出可核查证据，不应据此开发正式 PDF-access baseline。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tested": len(selected), "raw": str(raw_path), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

