from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_io import write_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_fitz() -> Any:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for native PDF text extraction. Install PyMuPDF first.") from exc
    return fitz


def _normalize_native_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_native_page_text(
    paper_id: str,
    pdf_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    extraction_method: str = "pymupdf_text",
    min_chars_per_page: int = 40,
) -> dict[str, Any]:
    """Extract native PDF text per page with PyMuPDF only; no OCR and no model calls."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    page_text_path = output / "page_text.jsonl"
    status_path = output / "page_text_status.json"
    if page_text_path.exists() and status_path.exists() and not overwrite:
        rows = []
        with page_text_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return {"rows": rows, "status": status, "page_text_path": str(page_text_path), "status_path": str(status_path), "cache_status": "hit"}

    if extraction_method != "pymupdf_text":
        raise ValueError(f"Unsupported native text extraction method: {extraction_method}")
    fitz = _load_fitz()
    rows: list[dict[str, Any]] = []
    pdf_file = Path(pdf_path)
    with fitz.open(pdf_file) as doc:
        page_count = len(doc)
        for index in range(page_count):
            page_no = index + 1
            raw_text = doc.load_page(index).get_text("text")
            text = _normalize_native_text(raw_text)
            lines = [line for line in text.splitlines() if line.strip()]
            rows.append(
                {
                    "paper_id": paper_id,
                    "page": page_no,
                    "text": text,
                    "char_count": len(text),
                    "line_count": len(lines),
                    "has_native_text": len(text) >= min_chars_per_page,
                    "extraction_method": extraction_method,
                }
            )
    non_empty_pages = sum(1 for row in rows if row["has_native_text"])
    status = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_file),
        "page_count": len(rows),
        "non_empty_pages": non_empty_pages,
        "empty_pages": len(rows) - non_empty_pages,
        "native_text_available": non_empty_pages > 0,
        "extraction_method": extraction_method,
        "min_chars_per_page": min_chars_per_page,
        "created_at": _now_iso(),
    }
    write_jsonl(page_text_path, rows)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": rows, "status": status, "page_text_path": str(page_text_path), "status_path": str(status_path), "cache_status": "miss"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract native PDF text page-by-page for page routing.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-chars-per-page", type=int, default=40)
    args = parser.parse_args()
    result = extract_native_page_text(
        args.paper_id,
        args.pdf,
        args.output_dir,
        overwrite=args.overwrite,
        min_chars_per_page=args.min_chars_per_page,
    )
    print(json.dumps({"status": result["status"], "page_text_path": result["page_text_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
