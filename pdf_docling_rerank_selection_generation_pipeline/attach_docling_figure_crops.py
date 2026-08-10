"""Attach deterministic high-resolution Docling figure crops to an L0-L3 hierarchy.

Docling's native picture exports are the embedded bitmap resolution, which is
often too small for a VLM to read panel labels or count subfigures (for example
the DynaPipe Figure 4 crop is only 394x135 px while the paper's rendered figure
contains 8 readable panels).  This utility renders each figure's page bbox from
the original PDF at a configurable DPI and points the hierarchy L0 record at
the readable crop, following the same pattern as attach_docling_table_crops.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Docling figure bboxes from PDFs at high DPI and attach their crop paths to hierarchy L0 records."
    )
    parser.add_argument("--hierarchy-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--processed-output-dir",
        default="processed_pdfs/hrcrops_v1",
        help="Root that receives <paper_id>/raw_docling_output/figure_crops/*.png.",
    )
    parser.add_argument(
        "--docling-output-dir",
        default="processed_pdfs/standard_pipeline_v3.0/docling",
        help="Docling artifact root that contains <paper_id>/symbolic_records.debug.jsonl.",
    )
    parser.add_argument("--pdf-output-dir", default="raw_pdfs/pdf")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI for figure crops.")
    parser.add_argument("--only-query-ids", default="", help="Optional comma-separated query IDs.")
    return parser.parse_args()


def _debug_bbox_index(debug_path: Path) -> dict[str, dict[str, Any]]:
    """Map global_record_id -> {'page_no': int, 'bbox': [l, t, r, b] bottom-left}."""
    index: dict[str, dict[str, Any]] = {}
    if not debug_path.is_file():
        return index
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        record_id = str(row.get("global_record_id") or row.get("record_id") or "")
        bbox = row.get("bbox")
        if not record_id or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            page_no = int(row.get("page") or 0)
            values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if page_no >= 1:
            index[record_id] = {"page_no": page_no, "bbox": values}
    return index


def _pictures_on_page(docling_path: Path, page: int) -> list[tuple[int, list[float]]]:
    """Docling pictures on a page as (index, [l, t, r, b]) bottom-left bboxes."""
    try:
        payload = json.loads(docling_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    pictures: list[tuple[int, list[float]]] = []
    for index, picture in enumerate(payload.get("pictures") or [], start=1):
        prov = (picture.get("prov") or [{}])[0]
        bbox = prov.get("bbox")
        if prov.get("page_no") != page or not isinstance(bbox, dict):
            continue
        try:
            pictures.append((index, [float(bbox["l"]), float(bbox["t"]), float(bbox["r"]), float(bbox["b"])]))
        except (KeyError, TypeError, ValueError):
            continue
    return pictures


class _PdfDocumentCache:
    """Keep one PyMuPDF document open per paper during a run."""

    def __init__(self, pdf_root: Path, dpi: int) -> None:
        self.pdf_root = pdf_root
        self.dpi = dpi
        self._documents: dict[str, Any] = {}

    def render(self, paper_id: str, page_no: int, bbox: list[float], output_path: Path) -> bool:
        if page_no < 1:
            return False
        document = self._documents.get(paper_id)
        if document is None:
            pdf_path = self.pdf_root / f"{paper_id}.pdf"
            if not pdf_path.is_file():
                return False
            try:
                import fitz  # type: ignore

                document = fitz.open(pdf_path)
            except Exception:
                return False
            self._documents[paper_id] = document
        try:
            left, top_bottom, right, bottom = bbox
            page = document[page_no - 1]
            height = float(page.rect.height)
            rect = fitz.Rect(left - 6, height - top_bottom - 6, right + 6, height - bottom + 6) & page.rect
            if rect.is_empty or rect.width < 4 or rect.height < 4:
                return False
            pixmap = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72), clip=rect, alpha=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap.save(str(output_path))
            return True
        except Exception:
            return False

    def close(self) -> None:
        for document in self._documents.values():
            try:
                document.close()
            except Exception:
                pass
        self._documents.clear()


def _crop_name(record: dict[str, Any], index: int) -> str:
    """Stable figure file name: prefer the figure label, else the old picture index."""
    label = str((record.get("locator") or {}).get("figure_id") or record.get("label") or "")
    match = re.search(r"(\d{1,4})", label)
    if match:
        return f"figure_{int(match.group(1)):04d}.png"
    old = str(record.get("crop_path") or "")
    old_match = re.search(r"(\d{4})_pictures_(\d+)\.png$", old)
    if old_match:
        return f"figure_{int(old_match.group(1)):04d}.png"
    return f"figure_{index:04d}.png"


def main() -> int:
    args = _args()
    processed_root = Path(args.processed_output_dir)
    pdf_root = Path(args.pdf_output_dir)
    indices: dict[str, dict[str, dict[str, Any]]] = {}
    renderer = _PdfDocumentCache(pdf_root, args.dpi)
    attached = rendered = unresolved = 0
    only = {value.strip() for value in args.only_query_ids.split(",") if value.strip()}
    rows = [row for row in read_jsonl(args.hierarchy_input) if not only or str(row.get("query_id") or "") in only]
    for row in rows:
        hierarchy = row.get("hierarchy") if isinstance(row.get("hierarchy"), dict) else {}
        for record in hierarchy.get("l0_catalog") or []:
            if not isinstance(record, dict) or str(record.get("source_type") or "") != "figure":
                continue
            paper_id = str(record.get("paper_id") or "")
            record_id = str(record.get("global_record_id") or record.get("record_id") or "")
            if not paper_id or not record_id:
                unresolved += 1
                continue
            bbox_index = indices.get(paper_id)
            if bbox_index is None:
                debug_path = Path(args.docling_output_dir) / paper_id / "symbolic_records.debug.jsonl"
                bbox_index = _debug_bbox_index(debug_path)
                indices[paper_id] = bbox_index
            info = bbox_index.get(record_id)
            page = record.get("page")
            if not info and page is not None:
                docling_path = Path(args.docling_output_dir) / paper_id / "raw_docling_output" / f"{paper_id}.docling.json"
                pictures = _pictures_on_page(docling_path, int(page))
                if len(pictures) == 1:
                    picture_index, bbox = pictures[0]
                    info = {"page_no": int(page), "bbox": bbox, "fallback": f"page_single_picture_{picture_index}"}
            if not info:
                unresolved += 1
                continue
            crop_dir = processed_root / paper_id / "raw_docling_output" / "figure_crops"
            crop_path = crop_dir / _crop_name(record, len(bbox_index))
            if not crop_path.is_file():
                rendered += int(renderer.render(paper_id, int(info["page_no"]), info["bbox"], crop_path))
            if crop_path.is_file():
                record["crop_path"] = str(crop_path)
                record["figure_crop_path"] = str(crop_path)
                attached += 1
            else:
                unresolved += 1
    renderer.close()
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "queries": len(rows),
                "attached_figure_records": attached,
                "rendered_or_reused": rendered,
                "unresolved": unresolved,
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
