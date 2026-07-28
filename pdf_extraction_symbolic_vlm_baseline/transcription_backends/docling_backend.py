from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import TranscribedDocument, TranscribedElement


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return str(obj)


def _first_prov(payload: dict[str, Any]) -> dict[str, Any]:
    prov = payload.get("prov")
    if isinstance(prov, list) and prov and isinstance(prov[0], dict):
        return prov[0]
    return {}


def _page(payload: dict[str, Any]) -> int | None:
    value = _first_prov(payload).get("page_no")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bbox(payload: dict[str, Any]) -> list[float] | None:
    raw = _first_prov(payload).get("bbox")
    if not isinstance(raw, dict):
        return None
    try:
        return [float(raw["l"]), float(raw["t"]), float(raw["r"]), float(raw["b"])]
    except Exception:
        return None


def _ref_id(ref: Any) -> str:
    if isinstance(ref, dict):
        return str(ref.get("$ref") or ref.get("cref") or "")
    return str(ref or "")


def _self_ref_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("self_ref") or ""): item for item in items if item.get("self_ref")}


def _caption_text(payload: dict[str, Any], text_by_ref: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for ref in payload.get("captions") or []:
        target = text_by_ref.get(_ref_id(ref))
        if target and target.get("text"):
            parts.append(re.sub(r"\s+", " ", str(target.get("text") or "")).strip())
    return "\n".join(part for part in parts if part)


def _caption_label(text: str, source_type: str) -> str | None:
    if source_type == "table":
        match = re.search(r"\b(?:Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", text, re.IGNORECASE)
        return f"Table {match.group(1).rstrip('.,;:')}" if match else None
    if source_type == "figure":
        match = re.search(r"\b(?:Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*(?:\([a-z]\))?)", text, re.IGNORECASE)
        return f"Figure {match.group(1).rstrip('.,;:')}" if match else None
    return None


def _formula_label(text: str) -> str | None:
    """Return a displayed equation number only when it appears at the formula end."""
    match = re.search(r"\(\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)\s*[.,;:]?\s*$", text)
    return f"Equation {match.group(1)}" if match else None


def _bbox_distance(left: list[float] | None, right: list[float] | None) -> float:
    """Distance for Docling page bboxes, independent of its coordinate origin."""
    if not left or not right:
        return float("inf")
    l1, t1, r1, b1 = left
    l2, t2, r2, b2 = right
    x1_lo, x1_hi = sorted((l1, r1))
    x2_lo, x2_hi = sorted((l2, r2))
    y1_lo, y1_hi = sorted((t1, b1))
    y2_lo, y2_hi = sorted((t2, b2))
    x_gap = max(x1_lo - x2_hi, x2_lo - x1_hi, 0.0)
    y_gap = max(y1_lo - y2_hi, y2_lo - y1_hi, 0.0)
    return y_gap + 0.25 * x_gap


def _caption_candidates(texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in texts:
        if str(item.get("label") or "").lower() != "caption":
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        if not text:
            continue
        for source_type in ("table", "figure"):
            label = _caption_label(text, source_type)
            if label:
                candidates.append(
                    {
                        "ref": str(item.get("self_ref") or ""),
                        "page": _page(item),
                        "bbox": _bbox(item),
                        "text": text,
                        "source_type": source_type,
                        "label": label,
                    }
                )
    return candidates


def _resolve_object_caption(
    payload: dict[str, Any],
    source_type: str,
    text_by_ref: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    claimed_caption_refs: set[str],
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Use Docling refs first, then a conservative same-page spatial association."""
    direct = _caption_text(payload, text_by_ref)
    direct_label = _caption_label(direct, source_type)
    if direct:
        for ref in payload.get("captions") or []:
            claimed_caption_refs.add(_ref_id(ref))
        return direct, direct_label, {"method": "docling_caption_ref"}

    page = _page(payload)
    bbox = _bbox(payload)
    choices = [
        candidate
        for candidate in candidates
        if candidate["source_type"] == source_type
        and candidate["page"] == page
        and candidate["ref"] not in claimed_caption_refs
    ]
    if not choices:
        return "", None, None
    best = min(choices, key=lambda candidate: _bbox_distance(bbox, candidate["bbox"]))
    distance = _bbox_distance(bbox, best["bbox"])
    # Captions normally touch the object. A 96 pt page-space limit avoids linking
    # ordinary prose that happens to mention a figure or table on the same page.
    if distance > 96:
        return "", None, None
    claimed_caption_refs.add(str(best["ref"]))
    return str(best["text"]), str(best["label"]), {
        "method": "spatial_caption_fallback",
        "caption_ref": best["ref"],
        "distance": round(distance, 3),
    }


def _table_to_markdown(table: dict[str, Any]) -> str:
    data = table.get("data") if isinstance(table.get("data"), dict) else {}
    cells = data.get("table_cells") if isinstance(data.get("table_cells"), list) else []
    try:
        rows = int(data.get("num_rows") or 0)
        cols = int(data.get("num_cols") or 0)
    except Exception:
        rows = cols = 0
    if rows <= 0 or cols <= 0 or not cells:
        return ""
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = re.sub(r"\s+", " ", str(cell.get("text") or "")).strip()
        r0 = int(cell.get("start_row_offset_idx") or 0)
        r1 = int(cell.get("end_row_offset_idx") or r0 + 1)
        c0 = int(cell.get("start_col_offset_idx") or 0)
        c1 = int(cell.get("end_col_offset_idx") or c0 + 1)
        for row in range(max(0, r0), min(rows, r1)):
            for col in range(max(0, c0), min(cols, c1)):
                grid[row][col] = text if not grid[row][col] else grid[row][col]
    if not any(any(cell for cell in row) for row in grid):
        return ""
    lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join("---" for _ in range(cols)) + " |"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _safe_artifact_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return name or fallback


def _save_picture_artifacts(document: Any, data: dict[str, Any], artifact_dir: Path) -> dict[str, str]:
    pictures = getattr(document, "pictures", None) or []
    if not pictures:
        return {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_by_ref: dict[str, str] = {}
    for index, picture in enumerate(pictures, start=1):
        self_ref = str(getattr(picture, "self_ref", "") or f"picture_{index}")
        try:
            image = picture.get_image(document)
        except Exception:
            image = None
        if image is None:
            continue
        path = artifact_dir / f"{index:04d}_{_safe_artifact_name(self_ref, f'picture_{index}')}.png"
        image.save(path)
        saved_by_ref[self_ref] = str(path)

    json_pictures = data.get("pictures") if isinstance(data.get("pictures"), list) else []
    for item in json_pictures:
        if not isinstance(item, dict):
            continue
        path = saved_by_ref.get(str(item.get("self_ref") or ""))
        if not path:
            continue
        image = item.get("image") if isinstance(item.get("image"), dict) else {}
        image = dict(image)
        image["uri"] = path
        image.setdefault("mimetype", "image/png")
        item["image"] = image
        item["_littraceqa_image_path"] = path
    return saved_by_ref


class DoclingTranscriptionBackend:
    name = "docling"

    def transcribe_pdf(
        self,
        paper_id: str,
        pdf_path: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> TranscribedDocument:
        try:
            import docling  # type: ignore
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Docling backend requested but docling is not installed. "
                "Please install docling in the littraceqa conda environment."
            ) from exc

        raw_dir = output_dir / "raw_docling_output"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_json_path = raw_dir / f"{paper_id}.docling.json"
        raw_md_path = raw_dir / f"{paper_id}.md"
        raw_text_path = raw_dir / f"{paper_id}.txt"
        figure_artifact_dir = raw_dir / "figures"

        data: dict[str, Any] | None = None
        if raw_json_path.exists() and not overwrite:
            data = json.loads(raw_json_path.read_text(encoding="utf-8"))
        else:
            from docling.datamodel.base_models import InputFormat  # type: ignore
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
            from docling.document_converter import PdfFormatOption  # type: ignore

            pipeline_options = PdfPipelineOptions()
            docling_do_ocr = bool(kwargs.get("docling_do_ocr", True))
            pipeline_options.do_ocr = docling_do_ocr
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_page_images = False
            pipeline_options.generate_table_images = False
            converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
            result = converter.convert(pdf_path, max_num_pages=int(kwargs.get("max_pages") or 9223372036854775807))
            document = result.document
            if hasattr(document, "export_to_dict"):
                data = document.export_to_dict()
            elif hasattr(document, "model_dump"):
                data = document.model_dump(mode="json")
            else:
                data = json.loads(document.export_to_json())
            _save_picture_artifacts(document, data, figure_artifact_dir)
            raw_json_path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if hasattr(document, "export_to_markdown"):
                raw_md_path.write_text(str(document.export_to_markdown()), encoding="utf-8")
            if hasattr(document, "export_to_text"):
                raw_text_path.write_text(str(document.export_to_text()), encoding="utf-8")

        data = data or {}
        pages = data.get("pages") if isinstance(data.get("pages"), dict) else {}
        page_count = len(pages) if pages else None
        texts = data.get("texts") if isinstance(data.get("texts"), list) else []
        tables = data.get("tables") if isinstance(data.get("tables"), list) else []
        pictures = data.get("pictures") if isinstance(data.get("pictures"), list) else []
        text_by_ref = _self_ref_index([item for item in texts if isinstance(item, dict)])
        caption_candidates = _caption_candidates([item for item in texts if isinstance(item, dict)])
        claimed_caption_refs: set[str] = set()
        elements: list[TranscribedElement] = []
        order = 0

        for item in texts:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("label") or "text")
            # Docling stores formula content in ``orig`` while leaving ``text`` empty.
            text = re.sub(r"\s+", " ", str(item.get("text") or item.get("orig") or "")).strip()
            if not text:
                continue
            order += 1
            page = _page(item)
            warnings = [] if page is not None else ["missing_page"]
            elements.append(
                TranscribedElement(
                    paper_id=paper_id,
                    page=page,
                    element_id=str(item.get("self_ref") or f"text_{order}"),
                    element_type=raw_type,
                    text=text,
                    label=_formula_label(text) if raw_type.lower() in {"formula", "equation"} else None,
                    bbox=_bbox(item),
                    reading_order=order,
                    raw_backend_type=raw_type,
                    raw_backend_payload=item,
                    backend=self.name,
                    warnings=warnings,
                )
            )

        for item in tables:
            if not isinstance(item, dict):
                continue
            order += 1
            caption, label, association = _resolve_object_caption(
                item, "table", text_by_ref, caption_candidates, claimed_caption_refs
            )
            markdown = _table_to_markdown(item)
            text = "\n".join(part for part in [caption, markdown] if part).strip()
            if not text:
                text = "Extracted table text may be incomplete."
            page = _page(item)
            warnings = [] if page is not None else ["missing_page"]
            elements.append(
                TranscribedElement(
                    paper_id=paper_id,
                    page=page,
                    element_id=str(item.get("self_ref") or f"table_{order}"),
                    element_type="table",
                    text=text,
                    label=label,
                    bbox=_bbox(item),
                    reading_order=order,
                    raw_backend_type="table",
                    raw_backend_payload={
                        **item,
                        "_littraceqa_markdown": markdown,
                        "_littraceqa_caption": caption,
                        "_littraceqa_caption_association": association,
                    },
                    backend=self.name,
                    warnings=warnings,
                )
            )

        for item in pictures:
            if not isinstance(item, dict):
                continue
            order += 1
            caption, label, association = _resolve_object_caption(
                item, "figure", text_by_ref, caption_candidates, claimed_caption_refs
            )
            text = caption or "Figure evidence extracted by Docling; no caption text available."
            page = _page(item)
            warnings = [] if page is not None else ["missing_page"]
            elements.append(
                TranscribedElement(
                    paper_id=paper_id,
                    page=page,
                    element_id=str(item.get("self_ref") or f"picture_{order}"),
                    element_type="picture",
                    text=text,
                    label=label,
                    bbox=_bbox(item),
                    reading_order=order,
                    raw_backend_type="picture",
                    raw_backend_payload={**item, "_littraceqa_caption_association": association},
                    backend=self.name,
                    warnings=warnings,
                )
            )

        elements.sort(key=lambda item: ((item.page if item.page is not None else 10**9), item.reading_order or 0, item.element_id))
        return TranscribedDocument(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            backend=self.name,
            page_count=page_count,
            elements=elements,
            raw_output_path=str(raw_json_path),
            warnings=[],
            backend_version=str(getattr(docling, "__version__", "")),
            backend_config={
                "table_mode": "accurate",
                "source": "docling.DocumentConverter",
                "generate_picture_images": True,
                "generate_page_images": False,
                "generate_table_images": False,
                "do_ocr": bool(kwargs.get("docling_do_ocr", True)),
                "figure_artifact_dir": str(figure_artifact_dir),
            },
        )
