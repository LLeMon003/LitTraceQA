from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .pdf_layout_blocks import BBox, TextBlock, bbox_list, save_page_crop


ALGORITHM_RE = re.compile(r"^\s*Algorithm\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\b", re.IGNORECASE)
EQ_NUMBER_RE = re.compile(r"\(\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)\s*$")
EQ_MENTION_RE = re.compile(r"\b(?:Equation|Eq\.)\s*\(?\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)?", re.IGNORECASE)
MATH_RE = re.compile(r"(?:=|≤|≥|≈|∑|∏|∇|∂|∫|\barg\b|\bmin\b|\bmax\b|\blog\b|\bexp\b|\\[a-zA-Z]+|[α-ωΑ-Ω])")


def _algorithm_label(text: str) -> str | None:
    match = ALGORITHM_RE.search(text or "")
    if not match:
        return None
    return f"Algorithm {match.group(1).rstrip('.,;:')}"


def _equation_label(text: str) -> str | None:
    match = EQ_NUMBER_RE.search(text or "")
    if not match:
        return None
    if not MATH_RE.search(text or ""):
        return None
    return f"Equation {match.group(1).rstrip('.,;:')}"


def _equation_mention_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in EQ_MENTION_RE.finditer(text or ""):
        label = f"Equation {match.group(1).rstrip('.,;:')}"
        if label not in labels:
            labels.append(label)
    return labels


def _crop_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "equation_algorithm"


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
    if not boxes:
        return None
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))


def _line_math_heavy(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) < 8 or len(compact) > 420:
        return False
    if not MATH_RE.search(compact):
        return False
    math_marks = len(re.findall(r"(?:=|≤|≥|≈|∑|∏|∇|∂|∫|\\[a-zA-Z]+|[α-ωΑ-Ω]|\^|_|[{}])", compact))
    prose_words = len(re.findall(r"\b[a-zA-Z]{4,}\b", compact))
    return math_marks >= 2 or (math_marks >= 1 and prose_words <= 8)


def _all_page_lines(page: Any) -> list[list[dict[str, Any]]]:
    rect = getattr(page, "rect", None)
    bbox: BBox = (
        float(getattr(rect, "x0", 0.0) or 0.0),
        float(getattr(rect, "y0", 0.0) or 0.0),
        float(getattr(rect, "x1", 595.0) or 595.0),
        float(getattr(rect, "y1", 842.0) or 842.0),
    )
    return _group_lines(_words_in_bbox(page, bbox))


def _equation_crop_bbox(page: Any, block: TextBlock) -> BBox:
    line_boxes: list[BBox] = []
    lines = _group_lines(_words_in_bbox(page, block.bbox))
    for line in lines:
        text = _line_text(line)
        if _equation_label(text):
            box = _line_bbox(line)
            if box is not None:
                line_boxes.append(box)
    if line_boxes:
        return (
            min(box[0] for box in line_boxes),
            min(box[1] for box in line_boxes),
            max(box[2] for box in line_boxes),
            max(box[3] for box in line_boxes),
        )
    return block.bbox


def extract_equations_algorithms(
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
    used_boxes: list[BBox] = []
    seen_labels: set[str] = set()
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue
        algorithm_label = _algorithm_label(text)
        if algorithm_label and (
            re.search(r"\b(?:Input|Output|Require|Ensure|for|while|if|return|repeat)\b", text, re.IGNORECASE)
            or len(text) < 500
        ):
            crop_path = None
            if enable_crops:
                crop_path = save_page_crop(page, block.bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(algorithm_label)}.png", crop_dpi, padding=10.0)
            runtime_like.append(
                {
                    "record_type": "algorithm",
                    "source_type": "equation_algorithm",
                    "label": algorithm_label,
                    "locator": {"page": page_no, "algorithm_id": algorithm_label},
                    "text": f"{algorithm_label}. Algorithm evidence is stored as a page crop image on page {page_no}.",
                }
            )
            seen_labels.add(algorithm_label)
            debug.append(
                {
                    "record_type": "algorithm",
                    "source_type": "equation_algorithm",
                    "label": algorithm_label,
                    "bbox": bbox_list(block.bbox),
                    "crop_path": crop_path,
                    "equation_algorithm_crop_path": crop_path,
                    "detected_text_preview": text[:500],
                    "detection_rules": ["algorithm_heading", "pseudocode_tokens_or_short_heading"],
                }
            )
            used_boxes.append(block.bbox)
            continue
        equation_label = _equation_label(text)
        if equation_label:
            crop_bbox = _equation_crop_bbox(page, block)
            crop_path = None
            if enable_crops:
                crop_path = save_page_crop(page, crop_bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(equation_label)}.png", crop_dpi, padding=2.0)
            runtime_like.append(
                {
                    "record_type": "equation",
                    "source_type": "equation_algorithm",
                    "label": equation_label,
                    "locator": {"page": page_no, "equation_id": equation_label},
                    "text": f"{equation_label}. Equation evidence is stored as a page crop image on page {page_no}.",
                }
            )
            seen_labels.add(equation_label)
            debug.append(
                {
                    "record_type": "equation",
                    "source_type": "equation_algorithm",
                    "label": equation_label,
                    "bbox": bbox_list(crop_bbox),
                    "block_bbox": bbox_list(block.bbox),
                    "crop_path": crop_path,
                    "equation_algorithm_crop_path": crop_path,
                    "detected_text_preview": text[:500],
                    "detection_rules": ["right_numbered_math_heavy_block"],
                }
            )
            used_boxes.append(block.bbox)
    equation_zero_added = "Equation 0" in seen_labels
    for line in _all_page_lines(page):
        text = _line_text(line)
        if not _line_math_heavy(text):
            continue
        label = _equation_label(text)
        if not label:
            label = "Equation 0"
            if equation_zero_added:
                continue
            equation_zero_added = True
        if label in seen_labels:
            continue
        bbox = _line_bbox(line)
        if bbox is None:
            continue
        crop_path = None
        if enable_crops:
            crop_path = save_page_crop(page, bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(label)}.png", crop_dpi, padding=2.0)
        runtime_like.append(
            {
                "record_type": "equation",
                "source_type": "equation_algorithm",
                "label": label,
                "locator": {"page": page_no, "equation_id": label},
                "text": f"{label}. Equation evidence is stored as a page crop image on page {page_no}.",
            }
        )
        debug.append(
            {
                "record_type": "equation",
                "source_type": "equation_algorithm",
                "label": label,
                "bbox": bbox_list(bbox),
                "block_bbox": bbox_list(bbox),
                "crop_path": crop_path,
                "equation_algorithm_crop_path": crop_path,
                "detected_text_preview": text[:500],
                "detection_rules": ["line_level_math_fallback"],
            }
        )
        used_boxes.append(bbox)
        seen_labels.add(label)
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue
        for label in _equation_mention_labels(text):
            if label in seen_labels:
                continue
            crop_path = None
            if enable_crops:
                crop_path = save_page_crop(page, block.bbox, output_dir / "object_crops" / f"page_{page_no:03d}_{_crop_name(label)}_mention.png", crop_dpi, padding=8.0)
            runtime_like.append(
                {
                    "record_type": "equation",
                    "source_type": "equation_algorithm",
                    "label": label,
                    "locator": {"page": page_no, "equation_id": label},
                    "text": f"{label}. Equation mention: {text[:500]}",
                }
            )
            debug.append(
                {
                    "record_type": "equation",
                    "source_type": "equation_algorithm",
                    "label": label,
                    "bbox": bbox_list(block.bbox),
                    "block_bbox": bbox_list(block.bbox),
                    "crop_path": crop_path,
                    "equation_algorithm_crop_path": crop_path,
                    "detected_text_preview": text[:500],
                    "detection_rules": ["equation_mention_fallback"],
                    "warnings": ["equation body unavailable", "record created from in-page equation mention"],
                }
            )
            used_boxes.append(block.bbox)
            seen_labels.add(label)
    return runtime_like, debug, used_boxes
