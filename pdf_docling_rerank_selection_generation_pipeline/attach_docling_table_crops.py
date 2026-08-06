"""Attach deterministic Docling table crops to an existing L0-L3 hierarchy."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl


_TABLE_NUMBER_RE = re.compile(r"\btable\s+(\d+)\b", re.IGNORECASE)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Docling table bboxes and attach their crop paths to hierarchy L0 records.")
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--processed-output-dir", required=True)
    parser.add_argument(
        "--crop-output-dir",
        default="",
        help="Optional separate root that receives <paper_id>/raw_docling_output/table_crops/*.png. "
        "Defaults to --processed-output-dir so existing invocations keep rendering into the docling root.",
    )
    parser.add_argument("--pdf-output-dir", default="raw_pdfs/pdf")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--only-query-ids", default="", help="Optional comma-separated query IDs.")
    return parser.parse_args()


def _table_number(value: Any) -> int | None:
    match = _TABLE_NUMBER_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def _docling_table_index(path: Path) -> dict[int, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
    return {index: table for index, table in enumerate(tables, start=1) if isinstance(table, dict)}


def _render_crop(pdf_path: Path, table: dict[str, Any], output_path: Path, dpi: int) -> bool:
    provenance = table.get("prov") if isinstance(table.get("prov"), list) else []
    if not provenance or not isinstance(provenance[0], dict):
        return False
    item = provenance[0]
    bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
    try:
        page_no = int(item.get("page_no"))
        left, right = float(bbox["l"]), float(bbox["r"])
        top, bottom = float(bbox["t"]), float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return False
    if page_no < 1 or not pdf_path.is_file():
        return False
    try:
        import fitz  # type: ignore
        document = fitz.open(pdf_path)
        page = document[page_no - 1]
        height = float(page.rect.height)
        # Docling provenance is bottom-left; PyMuPDF clip rectangles are top-left.
        rect = fitz.Rect(left - 6, height - top - 6, right + 6, height - bottom + 6) & page.rect
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            document.close()
            return False
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(output_path))
        document.close()
        return True
    except Exception:
        return False


def main() -> int:
    args = _args()
    processed_root = Path(args.processed_output_dir)
    crop_root = Path(args.crop_output_dir) if args.crop_output_dir else processed_root
    pdf_root = Path(args.pdf_output_dir)
    indices: dict[str, dict[int, dict[str, Any]]] = {}
    attached = rendered = unresolved = 0
    only = {value.strip() for value in args.only_query_ids.split(",") if value.strip()}
    rows = [row for row in read_jsonl(args.hierarchy_input) if not only or str(row.get("query_id") or "") in only]
    for row in rows:
        hierarchy = row.get("hierarchy") if isinstance(row.get("hierarchy"), dict) else {}
        for record in hierarchy.get("l0_catalog") or []:
            if not isinstance(record, dict) or str(record.get("source_type") or "") != "table":
                continue
            paper_id = str(record.get("paper_id") or "")
            number = _table_number((record.get("locator") or {}).get("table_id") if isinstance(record.get("locator"), dict) else record.get("label"))
            if not paper_id or number is None:
                unresolved += 1
                continue
            table_index = indices.get(paper_id)
            if table_index is None:
                raw = processed_root / paper_id / "raw_docling_output" / f"{paper_id}.docling.json"
                table_index = _docling_table_index(raw)
                indices[paper_id] = table_index
            table = table_index.get(number)
            crop_path = crop_root / paper_id / "raw_docling_output" / "table_crops" / f"table_{number:03d}.png"
            if table and not crop_path.is_file():
                _render_crop(pdf_root / f"{paper_id}.pdf", table, crop_path, args.dpi)
            # Saving is best effort (some malformed PDFs acknowledge a render
            # but leave no artifact). Attach only a file that actually exists.
            if table and crop_path.is_file():
                record["crop_path"] = str(crop_path)
                attached += 1
                rendered += int(crop_path.stat().st_size > 0)
            else:
                unresolved += 1
    write_jsonl(args.output, rows)
    print(json.dumps({"queries": len(rows), "attached_table_records": attached, "rendered_or_reused": rendered, "unresolved": unresolved, "output": args.output}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
