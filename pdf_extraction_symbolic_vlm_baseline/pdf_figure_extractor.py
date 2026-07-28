from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .pdf_layout_blocks import BBox, TextBlock, bbox_center, bbox_list, save_page_crop, union_bbox


FIGURE_RE = re.compile(r"\b(?:Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*(?:\([a-z]\))?)", re.IGNORECASE)
FIGURE_CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.)\s*[A-Za-z]?\d+[A-Za-z0-9.\-]*\s*[:.]", re.IGNORECASE)


def normalize_figure_label(text: str) -> str | None:
    match = FIGURE_RE.search(text or "")
    if not match:
        return None
    return f"Figure {match.group(1).rstrip('.,;:')}"


def _base_figure_label(label: str) -> str:
    match = re.fullmatch(r"Figure\s+([A-Za-z]?)(\d+)[a-z](?:\([a-z]\))?", label, re.IGNORECASE)
    if not match:
        return label
    return f"Figure {match.group(1)}{match.group(2)}"


def _caption_label(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not FIGURE_CAPTION_RE.match(normalized):
        # Multi-panel figures often have leading panel text before the shared caption,
        # e.g. "(f) Time-varying CartPole Figure 2: ...".
        if not re.search(r"(?:^|\s)(?:Figure|Fig\.)\s*[A-Za-z]?\d+[A-Za-z0-9.\-]*\s*[:.]", normalized, re.IGNORECASE):
            return None
        before = re.split(r"(?:Figure|Fig\.)\s*[A-Za-z]?\d+[A-Za-z0-9.\-]*\s*[:.]", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
        if len(before) > 160:
            return None
    if re.search(r"\b(?:see|shown in|as in|from|of)\s+(?:Figure|Fig\.)\s*\d+", normalized[:80], re.IGNORECASE):
        return None
    label = normalize_figure_label(normalized)
    if label and re.search(r"\([a-z]\)$", label, re.IGNORECASE):
        return None
    return label


def _mention_labels(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    if _caption_label(normalized):
        return []
    labels: list[str] = []
    for match in FIGURE_RE.finditer(normalized):
        label = f"Figure {match.group(1).rstrip('.,;:')}"
        base_label = _base_figure_label(label)
        for candidate in (label, base_label):
            if candidate not in labels:
                labels.append(candidate)
    return labels


def _nearest_visual(caption_box: BBox, visual_boxes: list[BBox], page_height: float) -> BBox | None:
    if not visual_boxes:
        return None
    cx, cy = bbox_center(caption_box)
    candidates: list[tuple[float, BBox]] = []
    for box in visual_boxes:
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area < 400:
            continue
        bx, by = bbox_center(box)
        vertical = abs(by - cy)
        if vertical > page_height * 0.45:
            continue
        candidates.append((vertical + abs(bx - cx) * 0.18, box))
    if not candidates:
        return None
    _, anchor = sorted(candidates, key=lambda item: item[0])[0]
    _, anchor_y = bbox_center(anchor)
    nearby: list[BBox] = []
    for _, box in candidates:
        _, by = bbox_center(box)
        vertical_gap = max(0.0, max(anchor[1], box[1]) - min(anchor[3], box[3]))
        if abs(by - anchor_y) <= page_height * 0.18 or vertical_gap <= 45:
            nearby.append(box)
    if not nearby:
        nearby = [anchor]
    return union_bbox(nearby)


def _figure_crop_bbox(body_bbox: BBox | None, caption_bbox: BBox) -> BBox | None:
    if body_bbox is None:
        return None
    x0 = min(body_bbox[0], caption_bbox[0])
    x1 = max(body_bbox[2], caption_bbox[2])
    body_mid_y = (body_bbox[1] + body_bbox[3]) / 2.0
    caption_mid_y = (caption_bbox[1] + caption_bbox[3]) / 2.0
    if caption_mid_y >= body_mid_y:
        return (x0, body_bbox[1], x1, caption_bbox[3])
    return (x0, caption_bbox[1], x1, body_bbox[3])


def _trim_surrounding_text(crop_bbox: BBox | None, caption_bbox: BBox, blocks: list[TextBlock]) -> BBox | None:
    if crop_bbox is None:
        return None
    x0, y0, x1, y1 = crop_bbox
    caption_mid_y = (caption_bbox[1] + caption_bbox[3]) / 2.0
    crop_mid_y = (y0 + y1) / 2.0
    if caption_mid_y >= crop_mid_y:
        for block in blocks:
            if block.bbox == caption_bbox:
                continue
            if block.bbox[3] <= y0 or block.bbox[1] >= y0 + 36:
                continue
            overlap_x = max(0.0, min(x1, block.bbox[2]) - max(x0, block.bbox[0]))
            if overlap_x > 20:
                y0 = max(y0, block.bbox[3] + 10.0)
    return (x0, y0, x1, y1) if y1 > y0 else crop_bbox


def extract_figures(
    page: Any,
    blocks: list[TextBlock],
    visual_boxes: list[BBox],
    page_no: int,
    output_dir: Path,
    *,
    enable_crops: bool,
    crop_dpi: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[BBox]]:
    _, page_height = (float(getattr(getattr(page, "rect", None), "width", 0.0) or 0.0), float(getattr(getattr(page, "rect", None), "height", 0.0) or 0.0))
    runtime_like: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    used_caption_boxes: list[BBox] = []
    seen_labels: set[str] = set()
    for idx, block in enumerate(blocks, start=1):
        label = _caption_label(re.sub(r"\s+", " ", block.text).strip())
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        body_bbox = _nearest_visual(block.bbox, visual_boxes, page_height)
        crop_bbox = _trim_surrounding_text(_figure_crop_bbox(body_bbox, block.bbox), block.bbox, blocks)
        crop_path = None
        if enable_crops and crop_bbox is not None:
            crop_path = save_page_crop(page, crop_bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{label.replace(' ', '_')}.png", crop_dpi)
        caption = re.sub(r"\s+", " ", block.text).strip()
        runtime_like.append(
            {
                "record_type": "figure",
                "source_type": "figure",
                "label": label,
                "locator": {"page": page_no, "figure_id": label},
                "text": caption or f"Figure candidate on page {page_no}",
            }
        )
        debug_record = {
            "source_type": "figure",
            "record_type": "figure",
            "label": label,
            "caption_bbox": bbox_list(block.bbox),
            "figure_bbox": bbox_list(body_bbox),
            "crop_bbox": bbox_list(crop_bbox),
            "visual_component_bboxes": [bbox_list(box) for box in visual_boxes],
            "crop_path": crop_path,
            "figure_crop_path": crop_path,
            "extraction_method": "caption_nearest_visual",
        }
        debug.append(debug_record)
        used_caption_boxes.append(block.bbox)
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        for label in _mention_labels(text):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            crop_path = None
            if enable_crops:
                crop_path = save_page_crop(page, block.bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{label.replace(' ', '_')}_mention.png", crop_dpi, padding=8.0)
            runtime_like.append(
                {
                    "record_type": "figure",
                    "source_type": "figure",
                    "label": label,
                    "locator": {"page": page_no, "figure_id": label},
                    "text": text or f"Figure mention on page {page_no}",
                }
            )
            debug.append(
                {
                    "source_type": "figure",
                    "record_type": "figure",
                    "label": label,
                    "caption_bbox": None,
                    "figure_bbox": None,
                    "crop_bbox": bbox_list(block.bbox),
                    "visual_component_bboxes": [bbox_list(box) for box in visual_boxes],
                    "crop_path": crop_path,
                    "figure_crop_path": crop_path,
                    "extraction_method": "figure_mention_fallback",
                    "warnings": ["figure body unavailable", "record created from in-page figure mention"],
                }
            )
            used_caption_boxes.append(block.bbox)
    return runtime_like, debug, used_caption_boxes
