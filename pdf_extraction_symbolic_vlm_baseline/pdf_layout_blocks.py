from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextBlock:
    bbox: BBox
    text: str
    block_no: int
    line_count: int = 0


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def bbox_list(box: BBox | None) -> list[float] | None:
    if box is None:
        return None
    return [round(v, 3) for v in box]


def union_bbox(boxes: list[BBox]) -> BBox | None:
    valid = [box for box in boxes if box and box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def bbox_intersects(a: BBox, b: BBox, *, pad: float = 0.0) -> bool:
    return not (a[2] < b[0] - pad or b[2] < a[0] - pad or a[3] < b[1] - pad or b[3] < a[1] - pad)


def bbox_center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def extract_text_blocks(page: Any) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    try:
        data = page.get_text("dict", sort=True)
    except TypeError:
        data = page.get_text("dict")
    except Exception:
        data = {}
    for index, block in enumerate(data.get("blocks", []) if isinstance(data, dict) else []):
        if not isinstance(block, dict) or block.get("type", 0) != 0:
            continue
        box = _bbox(block.get("bbox"))
        if box is None:
            continue
        lines: list[str] = []
        for line in block.get("lines", []) if isinstance(block.get("lines"), list) else []:
            spans = line.get("spans", []) if isinstance(line, dict) else []
            line_text = "".join(str(span.get("text") or "") for span in spans if isinstance(span, dict)).strip()
            if line_text:
                lines.append(line_text)
        text = "\n".join(lines).strip()
        if text:
            blocks.append(TextBlock(bbox=box, text=text, block_no=index, line_count=len(lines)))
    return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1), b.block_no))


def extract_words(page: Any) -> list[dict[str, Any]]:
    try:
        words = page.get_text("words", sort=True)
    except TypeError:
        words = page.get_text("words")
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in words or []:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        box = _bbox(item[:4])
        if box is None:
            continue
        rows.append({"bbox": box, "text": str(item[4] or ""), "block": item[5] if len(item) > 5 else None})
    return rows


def extract_image_bboxes(page: Any) -> list[BBox]:
    boxes: list[BBox] = []
    try:
        infos = page.get_image_info(xrefs=True)
    except TypeError:
        infos = page.get_image_info()
    except Exception:
        infos = []
    for info in infos or []:
        if isinstance(info, dict):
            box = _bbox(info.get("bbox"))
            if box is not None:
                boxes.append(box)
    if boxes:
        return boxes
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    for image in images or []:
        try:
            for rect in page.get_image_rects(image[0]):
                box = _bbox([rect.x0, rect.y0, rect.x1, rect.y1])
                if box is not None:
                    boxes.append(box)
        except Exception:
            continue
    return boxes


def extract_drawing_bboxes(page: Any) -> list[BBox]:
    boxes: list[BBox] = []
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for item in clusters or []:
        box = _bbox([getattr(item, "x0", None), getattr(item, "y0", None), getattr(item, "x1", None), getattr(item, "y1", None)])
        if box is not None:
            boxes.append(box)
    if boxes:
        return boxes
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings or []:
        rect = drawing.get("rect") if isinstance(drawing, dict) else None
        box = _bbox([getattr(rect, "x0", None), getattr(rect, "y0", None), getattr(rect, "x1", None), getattr(rect, "y1", None)])
        if box is not None:
            boxes.append(box)
    return boxes


def page_size(page: Any) -> tuple[float, float]:
    rect = getattr(page, "rect", None)
    return (float(getattr(rect, "width", 0.0) or 0.0), float(getattr(rect, "height", 0.0) or 0.0))


def save_page_crop(page: Any, bbox: BBox, crop_path: Path, dpi: int, padding: float = 8.0) -> str | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    rect = getattr(page, "rect", None)
    x0 = max(float(getattr(rect, "x0", 0.0)), bbox[0] - padding)
    y0 = max(float(getattr(rect, "y0", 0.0)), bbox[1] - padding)
    x1 = min(float(getattr(rect, "x1", bbox[2])), bbox[2] + padding)
    y1 = min(float(getattr(rect, "y1", bbox[3])), bbox[3] + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), clip=fitz.Rect(x0, y0, x1, y1), alpha=False)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(crop_path))
        return str(crop_path)
    except Exception:
        return None
