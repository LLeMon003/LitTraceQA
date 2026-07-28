from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .pdf_layout_blocks import BBox, TextBlock, bbox_center, bbox_list, save_page_crop, union_bbox


TABLE_RE = re.compile(r"\b(?:Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)


def normalize_table_label(text: str) -> str | None:
    match = TABLE_RE.search(text or "")
    if not match:
        return None
    return f"Table {match.group(1).rstrip('.,;:')}"


def _table_to_rows(table: Any) -> list[list[str]]:
    try:
        rows = table.extract()
    except Exception:
        rows = []
    if not isinstance(rows, list):
        return []
    cleaned: list[list[str]] = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
            if any(cells):
                cleaned.append(cells)
    return cleaned


def _rows_to_markdown(rows: list[list[str]], max_chars: int = 2200) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows[:40]]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _crop_name(label: str | None, fallback: str) -> str:
    value = label or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "table"


def _caption_blocks(blocks: list[TextBlock]) -> list[tuple[str, TextBlock]]:
    rows: list[tuple[str, TextBlock]] = []
    for block in blocks:
        label = normalize_table_label(block.text)
        if not label:
            continue
        text = re.sub(r"\s+", " ", block.text).strip()
        # Exclude prose mentions such as "Table 1 demonstrates..." from caption matching.
        if not re.match(r"^\s*(?:Table|Tab\.)\s*[A-Za-z]?\d+[A-Za-z0-9.\-]*\s*[:.]", text, re.IGNORECASE):
            continue
        rows.append((label, block))
    return rows


def _mention_blocks(blocks: list[TextBlock]) -> list[tuple[str, TextBlock]]:
    rows: list[tuple[str, TextBlock]] = []
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue
        if re.match(r"^\s*(?:Table|Tab\.)\s*[A-Za-z]?\d+[A-Za-z0-9.\-]*\s*[:.]", text, re.IGNORECASE):
            continue
        labels: list[str] = []
        for match in TABLE_RE.finditer(text):
            label = f"Table {match.group(1).rstrip('.,;:')}"
            if label not in labels:
                labels.append(label)
        for label in labels:
            rows.append((label, block))
    return rows


def _caption_neighbor_body_bbox(caption: TextBlock, blocks: list[TextBlock], page: Any | None = None) -> BBox:
    if page is not None:
        page_rect = getattr(page, "rect", None)
        page_bottom = float(getattr(page_rect, "y1", caption.bbox[3] + 620.0) or caption.bbox[3] + 620.0)
        word_region = (
            max(float(getattr(page_rect, "x0", 0.0) or 0.0), caption.bbox[0] - 12),
            caption.bbox[3] + 0.5,
            min(float(getattr(page_rect, "x1", caption.bbox[2]) or caption.bbox[2]), caption.bbox[2] + 12),
            min(page_bottom, caption.bbox[3] + 620.0),
        )
        word_lines = _group_lines(_words_in_bbox(page, word_region))
        line_boxes: list[BBox] = []
        last_bottom = caption.bbox[3]
        for line in word_lines:
            line_box = _line_bbox(line)
            if line_box is None:
                continue
            if line_boxes and line_box[1] - last_bottom > 24:
                break
            line_text = _line_text(line)
            if line_boxes and re.match(r"^\s*(?:Figure|Fig\.|Table|Tab\.)\s*\d+", line_text, re.IGNORECASE):
                break
            line_boxes.append(line_box)
            last_bottom = line_box[3]
        if len(line_boxes) >= 2:
            return union_bbox([caption.bbox, *line_boxes]) or caption.bbox

    body_boxes: list[BBox] = []
    last_y = caption.bbox[3]
    for block in sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0])):
        if block is caption or block.bbox[1] <= caption.bbox[3]:
            continue
        if block.bbox[1] - caption.bbox[3] > 190:
            break
        if body_boxes and block.bbox[1] - last_y > 18:
            break
        if normalize_table_label(block.text) or re.match(r"^\s*(?:Figure|Fig\.)\s*\d+", block.text, re.IGNORECASE):
            if body_boxes:
                break
            continue
        caption_cx, _ = bbox_center(caption.bbox)
        block_cx, _ = bbox_center(block.bbox)
        if abs(block_cx - caption_cx) > 150 and block.bbox[2] < caption.bbox[0]:
            continue
        body_boxes.append(block.bbox)
        last_y = block.bbox[3]
        if len(body_boxes) >= 80:
            break
    if body_boxes:
        return union_bbox([caption.bbox, *body_boxes]) or caption.bbox

    upper_boxes: list[BBox] = []
    last_y = caption.bbox[1]
    for block in sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]), reverse=True):
        if block is caption or block.bbox[3] >= caption.bbox[1]:
            continue
        if caption.bbox[1] - block.bbox[3] > 160:
            break
        if upper_boxes and last_y - block.bbox[3] > 18:
            break
        if normalize_table_label(block.text) or re.match(r"^\s*(?:Figure|Fig\.)\s*\d+", block.text, re.IGNORECASE):
            break
        upper_boxes.append(block.bbox)
        last_y = block.bbox[1]
        if len(upper_boxes) >= 80:
            break
    return union_bbox([*upper_boxes, caption.bbox]) or caption.bbox


def _nearby_caption(blocks: list[TextBlock], bbox: BBox) -> tuple[str | None, str, BBox | None]:
    candidates: list[tuple[float, TextBlock]] = []
    cx, _ = bbox_center(bbox)
    for label, block in _caption_blocks(blocks):
        bx, _ = bbox_center(block.bbox)
        vertical_gap = min(abs(block.bbox[3] - bbox[1]), abs(block.bbox[1] - bbox[3]))
        if vertical_gap <= 90:
            candidates.append((vertical_gap + abs(cx - bx) * 0.08, block))
    if not candidates:
        return None, "", None
    _, block = sorted(candidates, key=lambda item: item[0])[0]
    return normalize_table_label(block.text), re.sub(r"\s+", " ", block.text).strip(), block.bbox


def _words_in_bbox(page: Any, bbox: BBox) -> list[dict[str, Any]]:
    try:
        words = page.get_text("words", sort=True)
    except TypeError:
        words = page.get_text("words")
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for word in words or []:
        if not isinstance(word, (list, tuple)) or len(word) < 5:
            continue
        x0, y0, x1, y1 = (float(word[0]), float(word[1]), float(word[2]), float(word[3]))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            rows.append({"bbox": (x0, y0, x1, y1), "x": cx, "y": cy, "text": str(word[4] or "")})
    return rows


def _group_lines(words: list[dict[str, Any]], y_tol: float = 4.0) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["y"], item["x"])):
        if not lines:
            lines.append([word])
            continue
        last_y = sum(float(item["y"]) for item in lines[-1]) / len(lines[-1])
        if abs(float(word["y"]) - last_y) <= y_tol:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda item: item["x"]) for line in lines]


def _line_text(line: list[dict[str, Any]]) -> str:
    return " ".join(str(word["text"]) for word in sorted(line, key=lambda item: item["x"])).strip()


def _line_bbox(line: list[dict[str, Any]]) -> BBox | None:
    boxes = [word["bbox"] for word in line if isinstance(word.get("bbox"), tuple)]
    return union_bbox(boxes) if boxes else None


def _numeric_like(value: str) -> bool:
    return bool(re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?%?|0", value.strip()))


def _tableish_line(line: list[dict[str, Any]]) -> bool:
    if len(line) < 3:
        return False
    text = _line_text(line)
    if re.search(r"\b(?:Table|Figure|Section)\b", text):
        return False
    numeric_count = sum(1 for word in line if _numeric_like(str(word["text"])))
    return numeric_count >= 2 or len(line) >= 4


def _candidate_regions_from_alignment(page: Any, blocks: list[TextBlock]) -> list[dict[str, Any]]:
    page_rect = getattr(page, "rect", None)
    page_bbox: BBox = (
        float(getattr(page_rect, "x0", 0.0) or 0.0),
        float(getattr(page_rect, "y0", 0.0) or 0.0),
        float(getattr(page_rect, "x1", 595.0) or 595.0),
        float(getattr(page_rect, "y1", 842.0) or 842.0),
    )
    captions = _caption_blocks(blocks)
    search_regions: list[tuple[str | None, TextBlock | None, BBox]] = []
    for label, caption in captions:
        # Most paper tables have the caption below or above the body. Search a compact band around it.
        search_regions.append(
            (
                label,
                caption,
                (
                    max(page_bbox[0], caption.bbox[0] - 80),
                    max(page_bbox[1], caption.bbox[1] - 150),
                    min(page_bbox[2], caption.bbox[2] + 80),
                    min(page_bbox[3], caption.bbox[3] + 560),
                ),
            )
        )
    if not search_regions:
        search_regions.append((None, None, page_bbox))

    candidates: list[dict[str, Any]] = []
    for label, caption, region in search_regions:
        words = _words_in_bbox(page, region)
        lines = _group_lines(words)
        runs: list[list[list[dict[str, Any]]]] = []
        current: list[list[dict[str, Any]]] = []
        for line in lines:
            if _tableish_line(line):
                current.append(line)
                continue
            if len(current) >= 3:
                runs.append(current)
            current = []
        if len(current) >= 3:
            runs.append(current)
        for run in runs:
            boxes = [_line_bbox(line) for line in run]
            bbox = union_bbox([box for box in boxes if box is not None])
            if bbox is None:
                continue
            if caption and bbox[1] <= caption.bbox[3] and bbox[3] >= caption.bbox[1]:
                continue
            data_lines = [line for line in run if sum(1 for word in line if _numeric_like(str(word["text"]))) >= 2]
            if len(data_lines) < 2:
                continue
            score = len(data_lines) * 10 + sum(len(line) for line in run)
            if caption:
                vertical_gap = min(abs(caption.bbox[1] - bbox[3]), abs(caption.bbox[3] - bbox[1]))
                score -= vertical_gap * 0.05
            candidates.append({"label": label, "caption_block": caption, "bbox": bbox, "lines": run, "score": score})
    return sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True)


def _column_centers_from_data(data_lines: list[list[dict[str, Any]]]) -> list[float]:
    centers_by_index: dict[int, list[float]] = {}
    for line in data_lines:
        numeric_words = [word for word in sorted(line, key=lambda item: item["x"]) if _numeric_like(str(word["text"]))]
        for index, word in enumerate(numeric_words):
            centers_by_index.setdefault(index, []).append(float(word["x"]))
    centers = [sum(values) / len(values) for _, values in sorted(centers_by_index.items()) if values]
    return centers


def _assign_line_to_columns(line: list[dict[str, Any]], centers: list[float], *, drop_far_words: bool = False) -> list[str]:
    cells = ["" for _ in centers]
    for word in sorted(line, key=lambda item: item["x"]):
        if not centers:
            continue
        distances = [abs(float(word["x"]) - center) for center in centers]
        nearest = min(range(len(centers)), key=lambda idx: distances[idx])
        if drop_far_words and distances[nearest] > 24.0:
            continue
        cells[nearest] = f"{cells[nearest]} {word['text']}".strip()
    return cells


def _infer_header(lines: list[list[dict[str, Any]]], data_start: int, centers: list[float]) -> list[str]:
    header_lines = lines[:data_start]
    if not header_lines:
        return [f"Column {idx + 1}" for idx in range(len(centers))]
    top = _assign_line_to_columns(header_lines[0], centers)
    lower_parts = [_assign_line_to_columns(line, centers) for line in header_lines[1:]]
    headers: list[str] = []
    for idx in range(len(centers)):
        parts: list[str] = []
        if top[idx]:
            parts.append(top[idx])
        for lower in lower_parts:
            if idx < len(lower) and lower[idx]:
                parts.append(lower[idx])
        headers.append(" ".join(parts).strip() or f"Column {idx + 1}")
    if len(headers) >= 5:
        if headers[0].lower() in {"output length", "output"} or headers[0].lower().startswith("output"):
            headers[0] = "Output Length"
        for idx in (1, 2):
            if not headers[idx].lower().startswith("calculations"):
                headers[idx] = f"Calculations {headers[idx]}".strip()
        for idx in (3, 4):
            if not headers[idx].lower().startswith("memory"):
                headers[idx] = f"Memory {headers[idx]}".strip()
    return headers


def _aligned_candidate_to_rows(candidate: dict[str, Any]) -> tuple[list[list[str]], dict[str, Any]]:
    lines = candidate.get("lines") if isinstance(candidate.get("lines"), list) else []
    data_start = 0
    for index, line in enumerate(lines):
        if sum(1 for word in line if _numeric_like(str(word["text"]))) >= 2:
            data_start = index
            break
    data_lines = [line for line in lines[data_start:] if sum(1 for word in line if _numeric_like(str(word["text"]))) >= 2]
    centers = _column_centers_from_data(data_lines)
    if len(centers) < 3:
        return [], {"column_centers": centers}
    header = _infer_header(lines, data_start, centers)
    rows = [header]
    for line in data_lines:
        cells = _assign_line_to_columns(line, centers, drop_far_words=True)
        if any(cells):
            rows.append(cells)
    return rows, {"column_centers": [round(value, 3) for value in centers], "data_row_count": len(data_lines), "header_line_count": data_start}


def extract_tables(
    page: Any,
    blocks: list[TextBlock],
    page_no: int,
    output_dir: Path,
    *,
    enable_crops: bool,
    crop_dpi: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[BBox]]:
    runtime_like: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    used_caption_boxes: list[BBox] = []
    tables: list[Any] = []
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", []) or [])
    except Exception:
        tables = []
    seen_find_table_bboxes: set[tuple[int, int, int, int]] = set()
    seen_find_table_labels: set[str] = set()
    for idx, table in enumerate(tables, start=1):
        raw_bbox = getattr(table, "bbox", None)
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        bbox = tuple(float(v) for v in raw_bbox)  # type: ignore[assignment]
        bbox_key = tuple(round(value) for value in bbox)
        if bbox_key in seen_find_table_bboxes:
            continue
        rows = _table_to_rows(table)
        if len(rows) < 2 or (bbox[3] - bbox[1]) < 20:
            continue
        label, caption, caption_bbox = _nearby_caption(blocks, bbox)
        if label and label in seen_find_table_labels:
            continue
        seen_find_table_bboxes.add(bbox_key)
        if label:
            seen_find_table_labels.add(label)
        if caption_bbox:
            used_caption_boxes.append(caption_bbox)
        crop_path = save_page_crop(page, bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(label, f'table_{idx}')}.png", crop_dpi, padding=8.0) if enable_crops else None
        markdown = ""
        try:
            markdown = str(table.to_markdown() or "").strip()
        except Exception:
            markdown = ""
        if not markdown:
            markdown = _rows_to_markdown(rows)
        text_parts = []
        if caption:
            text_parts.append(caption)
        if markdown:
            text_parts.append(markdown)
        text = "\n".join(text_parts).strip() or f"Table candidate on page {page_no}"
        runtime_like.append(
            {
                "record_type": "table",
                "source_type": "table",
                "label": label,
                "locator": {"page": page_no, **({"table_id": label} if label else {})},
                "text": text,
            }
        )
        debug.append(
            {
                "source_type": "table",
                "record_type": "table",
                "label": label,
                "table_bbox": bbox_list(bbox),
                "caption_bbox": bbox_list(caption_bbox),
                "row_count": len(rows),
                "col_count": max((len(row) for row in rows), default=0),
                "cells": rows,
                "crop_path": crop_path,
                "table_crop_path": crop_path,
                "extraction_method": "pymupdf_find_tables",
            }
        )
    used_labels: set[str] = {str(record.get("label") or "") for record in runtime_like if record.get("label")}
    aligned_candidates = _candidate_regions_from_alignment(page, blocks)
    for idx, candidate in enumerate(aligned_candidates, start=1):
        rows, meta = _aligned_candidate_to_rows(candidate)
        if len(rows) < 3:
            continue
        label = candidate.get("label")
        if label and str(label) in used_labels:
            continue
        if label:
            used_labels.add(str(label))
        caption_block = candidate.get("caption_block")
        caption = re.sub(r"\s+", " ", caption_block.text).strip() if isinstance(caption_block, TextBlock) else ""
        bbox = candidate.get("bbox")
        caption_bbox = caption_block.bbox if isinstance(caption_block, TextBlock) else None
        crop_bbox = union_bbox([box for box in [bbox, caption_bbox] if isinstance(box, tuple)]) or bbox
        crop_path = (
            save_page_crop(page, crop_bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(str(label or ''), f'aligned_table_{idx}')}.png", crop_dpi, padding=8.0)
            if enable_crops and isinstance(crop_bbox, tuple)
            else None
        )
        markdown = _rows_to_markdown(rows, max_chars=3200)
        text = "\n".join(part for part in [caption, markdown] if part).strip() or f"Table candidate on page {page_no}"
        runtime_like.append(
            {
                "record_type": "table",
                "source_type": "table",
                "label": label,
                "locator": {"page": page_no, **({"table_id": label} if label else {})},
                "text": text,
            }
        )
        debug.append(
            {
                "source_type": "table",
                "record_type": "table",
                "label": label,
                "table_bbox": bbox_list(bbox if isinstance(bbox, tuple) else None),
                "caption_bbox": bbox_list(caption_bbox),
                "row_count": max(0, len(rows) - 1),
                "col_count": max((len(row) for row in rows), default=0),
                "cells": rows,
                "column_centers": meta.get("column_centers", []),
                "crop_path": crop_path,
                "table_crop_path": crop_path,
                "extraction_method": "aligned_words_fallback",
            }
        )
        if caption_bbox:
            used_caption_boxes.append(caption_bbox)
        if isinstance(bbox, tuple):
            used_caption_boxes.append(bbox)

    for label, block in _caption_blocks(blocks):
        if label in used_labels:
            continue
        used_labels.add(label)
        text = re.sub(r"\s+", " ", block.text).strip()
        crop_bbox = _caption_neighbor_body_bbox(block, blocks, page)
        crop_path = save_page_crop(page, crop_bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(label, 'caption_table')}.png", crop_dpi, padding=8.0) if enable_crops else None
        runtime_like.append({"record_type": "table", "source_type": "table", "label": label, "locator": {"page": page_no, "table_id": label}, "text": text})
        debug.append(
            {
                "source_type": "table",
                "record_type": "table",
                "label": label,
                "table_bbox": bbox_list(crop_bbox),
                "caption_bbox": bbox_list(block.bbox),
                "row_count": 0,
                "col_count": 0,
                "cells": [],
                "crop_path": crop_path,
                "table_crop_path": crop_path,
                "extraction_method": "caption_only_fallback",
                "warnings": ["table matrix unavailable", "crop_bbox_inferred_from_neighbor_text_blocks"],
            }
        )
        used_caption_boxes.append(crop_bbox)

    for label, block in _mention_blocks(blocks):
        if label in used_labels:
            continue
        used_labels.add(label)
        text = re.sub(r"\s+", " ", block.text).strip()
        crop_path = save_page_crop(page, block.bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(label, 'table_mention')}.png", crop_dpi, padding=8.0) if enable_crops else None
        runtime_like.append({"record_type": "table", "source_type": "table", "label": label, "locator": {"page": page_no, "table_id": label}, "text": text})
        debug.append(
            {
                "source_type": "table",
                "record_type": "table",
                "label": label,
                "table_bbox": bbox_list(block.bbox),
                "caption_bbox": None,
                "row_count": 0,
                "col_count": 0,
                "cells": [],
                "crop_path": crop_path,
                "table_crop_path": crop_path,
                "extraction_method": "table_mention_fallback",
                "warnings": ["table body unavailable", "record created from in-page table mention"],
            }
        )
        used_caption_boxes.append(block.bbox)
    return runtime_like, debug, used_caption_boxes
