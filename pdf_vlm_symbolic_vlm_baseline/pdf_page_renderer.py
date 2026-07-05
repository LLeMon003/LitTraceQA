from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_pdf_pages(
    paper_id: str,
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 160,
    image_format: str = "jpg",
    max_pages: int | None = None,
    overwrite: bool = False,
    selected_pages: set[int] | None = None,
) -> dict[str, Any]:
    pdf_file = Path(pdf_path)
    paper_dir = Path(output_dir)
    page_dir = paper_dir / "page_images"
    page_dir.mkdir(parents=True, exist_ok=True)
    fmt = "jpg" if image_format.lower() in {"jpg", "jpeg"} else image_format.lower()
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_file) as doc:
        page_count = len(doc)
        limit = page_count if not max_pages or max_pages <= 0 else min(page_count, max_pages)
        if selected_pages is None:
            page_numbers = list(range(1, limit + 1))
        else:
            page_numbers = sorted(page for page in selected_pages if 1 <= int(page) <= limit)
        for page_no in page_numbers:
            index = page_no - 1
            image_path = page_dir / f"page_{page_no:03d}.{fmt}"
            if image_path.exists() and not overwrite:
                with Image.open(image_path) as image:
                    width, height = image.size
                status = "existing"
            else:
                page = doc.load_page(index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                if fmt in {"jpg", "jpeg"}:
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    image.save(image_path, format="JPEG", quality=90, optimize=True)
                else:
                    pix.save(str(image_path))
                width, height = pix.width, pix.height
                status = "rendered"
            pages.append(
                {
                    "paper_id": paper_id,
                    "page": page_no,
                    "image_path": str(image_path),
                    "width_px": width,
                    "height_px": height,
                    "dpi": dpi,
                    "render_status": status,
                }
            )
    manifest = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_file),
        "page_count": page_count,
        "rendered_pages": len(pages),
        "selected_pages": page_numbers,
        "dpi": dpi,
        "image_format": fmt,
        "created_at": _now_iso(),
        "pages": pages,
    }
    (paper_dir / "document_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
