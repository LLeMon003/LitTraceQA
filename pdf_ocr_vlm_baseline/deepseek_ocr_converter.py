from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import urllib.request

import fitz
from PIL import Image

from .parser import extract_json_object
from .config import is_api_key_configured
from .data_io import write_jsonl


OCR_SYSTEM_PROMPT = (
    "You are an OCR and document-layout parser for academic papers. "
    "Extract the page into structured JSON only. Preserve reading order, section headings, tables, figures, equations, "
    "captions, and approximate bounding boxes when visible. Do not summarize."
)
OCR_PROMPT_VERSION = "deepseek_ocr_page_json_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(paper_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("document.json", "pages.jsonl", "chunks.jsonl", "visual_contexts.jsonl", "raw_ocr_responses.jsonl", "document.md"):
        path = paper_dir / name
        if path.exists():
            hashes[name] = _sha256_file(path)
    return hashes


def _check_runtime(api_key: str | None = None, provider: str = "siliconflow", base_url: str = "", model: str = "") -> tuple[bool, str]:
    if provider.lower() == "local":
        return False, "Local DeepSeek-OCR is disabled for this baseline; OCR is expected to run via API."
    if not is_api_key_configured(api_key):
        return False, "OCR API key is missing or placeholder."
    if not base_url:
        return False, "OCR base URL is missing."
    if not model:
        return False, "OCR model is missing."
    return True, "OCR API configuration is present."


def _render_pdf_pages(pdf_path: str | Path, output_dir: Path, max_pages: int | None = None, dpi: int = 144) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))
    try:
        limit = min(len(doc), max_pages) if max_pages else len(doc)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in range(limit):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{idx + 1:03d}.jpg"
            pix.save(str(image_path), jpg_quality=85)
            rendered.append(
                {
                    "page": idx + 1,
                    "image_path": str(image_path),
                    "width": pix.width,
                    "height": pix.height,
                }
            )
    finally:
        doc.close()
    return rendered


def _image_to_data_url(image_path: str | Path, max_side: int = 1800, quality: int = 85) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _call_ocr_page(
    *,
    image_path: str | Path,
    page_number: int,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    prompt = (
        f"Extract OCR from page {page_number} of this academic paper image.\n"
        "Return one JSON object only, with these keys: page, text, chunks, figures, tables, equations.\n"
        "Rules:\n"
        "- text: transcribe the actual visible page text in reading order.\n"
        "- chunks: array of real visible text blocks. Each item must include type, text, bbox, reading_order.\n"
        "- type must be one of paragraph, heading, table, figure_caption, equation, reference, other.\n"
        "- figures: array of visible figures with figure_id, caption, bbox. Empty array if none.\n"
        "- tables: array of visible tables with table_id, caption, bbox, markdown. Empty array if none.\n"
        "- equations: array of visible equations with equation_id, text, bbox. Empty array if none.\n"
        "- bbox uses normalized [x1,y1,x2,y2] coordinates in 0..1 when possible; use null if uncertain.\n"
        "- Do not copy these instructions. Do not use placeholder text. Do not summarize.\n"
        "- Return JSON only, without markdown fences."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path), "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    content = str(data.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    try:
        parsed = extract_json_object(content)
    except Exception:
        parsed = {
            "page": page_number,
            "text": content,
            "chunks": [{"type": "page_text", "text": content, "bbox": None, "reading_order": 1}],
            "figures": [],
            "tables": [],
            "equations": [],
            "parse_warning": "OCR model did not return JSON; stored raw content as a page-level OCR chunk.",
        }
    parsed["raw_content"] = content
    return parsed


def _looks_like_placeholder_ocr(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "full page text in reading order",
        "chunk text",
        "visible caption text",
        "markdown table if readable",
        "extract structured ocr from this academic-paper page",
        "do not use markdown",
        "return json only",
        "do not copy these instructions",
    ]
    if any(marker in lowered for marker in markers):
        return True
    words = lowered.split()
    if len(words) > 80 and len(set(words)) / max(len(words), 1) < 0.12:
        return True
    return False


def _safe_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if any(x < -0.05 or x > 1.05 for x in bbox):
        return None
    return [max(0.0, min(1.0, x)) for x in bbox]


def _chunk_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_pdf_with_deepseek_ocr(
    paper_id: str,
    pdf_path: str | Path,
    output_root: str | Path,
    max_pages: int | None = None,
    overwrite: bool = False,
    api_key: str | None = None,
    provider: str = "siliconflow",
    base_url: str = "",
    model: str = "deepseek-ai/DeepSeek-OCR",
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    ok, reason = _check_runtime(api_key=api_key, provider=provider, base_url=base_url, model=model)
    paper_dir = Path(output_root) / paper_id
    if overwrite and paper_dir.exists():
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    if not ok:
        document = {
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "status": "ocr_unavailable",
            "parser": model,
            "provider": provider,
            "prompt_version": OCR_PROMPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "max_pages": max_pages,
        }
        (paper_dir / "document.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_jsonl(paper_dir / "pages.jsonl", [])
        write_jsonl(paper_dir / "chunks.jsonl", [])
        write_jsonl(paper_dir / "visual_contexts.jsonl", [])
        write_jsonl(paper_dir / "raw_ocr_responses.jsonl", [])
        (paper_dir / "document.md").write_text("", encoding="utf-8")
        (paper_dir / "page_images").mkdir(exist_ok=True)
        document["artifact_hashes"] = _artifact_hashes(paper_dir)
        (paper_dir / "document.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {**document, "output_dir": str(paper_dir), "chunks_path": str(paper_dir / "chunks.jsonl"), "visual_contexts_path": str(paper_dir / "visual_contexts.jsonl")}
    page_image_dir = paper_dir / "page_images"
    rendered_pages = _render_pdf_pages(pdf_path, page_image_dir, max_pages=max_pages)
    pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    raw_ocr_responses: list[dict[str, Any]] = []
    markdown_parts: list[str] = []
    failures: list[dict[str, Any]] = []
    for page in rendered_pages:
        try:
            parsed = _call_ocr_page(
                image_path=page["image_path"],
                page_number=int(page["page"]),
                model=model,
                api_key=str(api_key),
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            failures.append({"page": page["page"], "error": str(exc)})
            raw_ocr_responses.append(
                {
                    "paper_id": paper_id,
                    "page": page["page"],
                    "image_path": page["image_path"],
                    "status": "request_or_parse_failed",
                    "error": str(exc),
                    "raw_content": "",
                }
            )
            continue
        raw_ocr_responses.append(
            {
                "paper_id": paper_id,
                "page": page["page"],
                "image_path": page["image_path"],
                "status": "received",
                "parse_warning": parsed.get("parse_warning", ""),
                "raw_content": parsed.get("raw_content", ""),
            }
        )
        page_text = _chunk_text(parsed.get("text"))
        if _looks_like_placeholder_ocr(page_text) or _looks_like_placeholder_ocr(str(parsed.get("raw_content", ""))):
            failures.append({"page": page["page"], "error": "OCR response looked like a prompt/schema echo, not page transcription."})
            continue
        page_record = {
            "paper_id": paper_id,
            "page": page["page"],
            "image_path": page["image_path"],
            "width": page["width"],
            "height": page["height"],
            "text": page_text,
            "raw_content": parsed.get("raw_content", ""),
        }
        pages.append(page_record)
        markdown_parts.append(f"\n\n<!-- page {page['page']} -->\n\n{page_text}")
        for idx, item in enumerate(parsed.get("chunks", []) if isinstance(parsed.get("chunks"), list) else []):
            text = _chunk_text(item.get("text") if isinstance(item, dict) else item)
            if not text:
                continue
            chunks.append(
                {
                    "paper_id": paper_id,
                    "page": page["page"],
                    "chunk_id": f"p{int(page['page']):03d}_c{idx + 1:03d}",
                    "text": text,
                    "type": item.get("type", "other") if isinstance(item, dict) else "other",
                    "bbox": _safe_bbox(item.get("bbox")) if isinstance(item, dict) else None,
                    "reading_order": item.get("reading_order", idx + 1) if isinstance(item, dict) else idx + 1,
                    "parser_confidence": item.get("confidence") if isinstance(item, dict) else None,
                }
            )
        if page_text and not any(c["page"] == page["page"] for c in chunks):
            chunks.append(
                {
                    "paper_id": paper_id,
                    "page": page["page"],
                    "chunk_id": f"p{int(page['page']):03d}_c001",
                    "text": page_text,
                    "type": "page_text",
                    "bbox": None,
                    "reading_order": 1,
                    "parser_confidence": None,
                }
            )
        for idx, fig in enumerate(parsed.get("figures", []) if isinstance(parsed.get("figures"), list) else []):
            if not isinstance(fig, dict):
                continue
            visuals.append(
                {
                    "paper_id": paper_id,
                    "page": page["page"],
                    "visual_id": fig.get("figure_id") or f"fig_p{int(page['page']):03d}_{idx + 1:03d}",
                    "source_type": "figure",
                    "caption": _chunk_text(fig.get("caption")),
                    "bbox": _safe_bbox(fig.get("bbox")),
                    "image_path": page["image_path"],
                }
            )
        for idx, table in enumerate(parsed.get("tables", []) if isinstance(parsed.get("tables"), list) else []):
            if not isinstance(table, dict):
                continue
            table_text = _chunk_text(table.get("markdown") or table.get("caption"))
            table_id = table.get("table_id") or f"tab_p{int(page['page']):03d}_{idx + 1:03d}"
            visuals.append(
                {
                    "paper_id": paper_id,
                    "page": page["page"],
                    "visual_id": table_id,
                    "source_type": "table",
                    "caption": _chunk_text(table.get("caption")),
                    "bbox": _safe_bbox(table.get("bbox")),
                    "image_path": page["image_path"],
                }
            )
            if table_text:
                chunks.append(
                    {
                        "paper_id": paper_id,
                        "page": page["page"],
                        "chunk_id": f"{table_id}_text",
                        "text": table_text,
                        "type": "table",
                        "bbox": _safe_bbox(table.get("bbox")),
                        "reading_order": 900 + idx,
                        "table_id": table_id,
                        "parser_confidence": table.get("confidence"),
                    }
                )
        for idx, eq in enumerate(parsed.get("equations", []) if isinstance(parsed.get("equations"), list) else []):
            if not isinstance(eq, dict):
                continue
            equation_text = _chunk_text(eq.get("text"))
            if equation_text:
                chunks.append(
                    {
                        "paper_id": paper_id,
                        "page": page["page"],
                        "chunk_id": eq.get("equation_id") or f"eq_p{int(page['page']):03d}_{idx + 1:03d}",
                        "text": equation_text,
                        "type": "equation",
                        "bbox": _safe_bbox(eq.get("bbox")),
                        "reading_order": 950 + idx,
                        "parser_confidence": eq.get("confidence"),
                    }
                )
    status = "ok" if pages and chunks else "ocr_failed"
    document = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "status": status,
        "parser": model,
        "provider": provider,
        "prompt_version": OCR_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason if status == "ok" else "No OCR chunks were produced.",
        "max_pages": max_pages,
        "page_count_processed": len(rendered_pages),
        "page_count_succeeded": len(pages),
        "page_failures": failures,
    }
    write_jsonl(paper_dir / "pages.jsonl", pages)
    write_jsonl(paper_dir / "chunks.jsonl", chunks)
    write_jsonl(paper_dir / "visual_contexts.jsonl", visuals)
    write_jsonl(paper_dir / "raw_ocr_responses.jsonl", raw_ocr_responses)
    (paper_dir / "document.md").write_text("\n".join(markdown_parts).strip() + "\n", encoding="utf-8")
    (paper_dir / "document.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document["artifact_hashes"] = _artifact_hashes(paper_dir)
    (paper_dir / "document.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**document, "output_dir": str(paper_dir), "chunks_path": str(paper_dir / "chunks.jsonl"), "visual_contexts_path": str(paper_dir / "visual_contexts.jsonl")}


def load_or_convert_pdf(
    paper_id: str,
    pdf_path: str | Path,
    output_root: str | Path,
    max_pages: int | None = None,
    overwrite: bool = False,
    api_key: str | None = None,
    provider: str = "siliconflow",
    base_url: str = "",
    model: str = "deepseek-ai/DeepSeek-OCR",
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    paper_dir = Path(output_root) / paper_id
    document_path = paper_dir / "document.json"
    if document_path.exists() and not overwrite:
        data = json.loads(document_path.read_text(encoding="utf-8"))
        return {**data, "output_dir": str(paper_dir), "chunks_path": str(paper_dir / "chunks.jsonl"), "visual_contexts_path": str(paper_dir / "visual_contexts.jsonl")}
    return convert_pdf_with_deepseek_ocr(
        paper_id,
        pdf_path,
        output_root,
        max_pages=max_pages,
        overwrite=overwrite,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
