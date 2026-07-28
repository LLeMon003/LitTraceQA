from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TranscribedDocument


class PyMuPDFTranscriptionBackend:
    name = "pymupdf"

    def transcribe_pdf(
        self,
        paper_id: str,
        pdf_path: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> TranscribedDocument:
        from ..pdf_extraction_parser import extract_pdf_symbolic_records

        status = extract_pdf_symbolic_records(
            paper_id,
            pdf_path,
            output_dir,
            overwrite=overwrite,
            extract_all_pages=bool(kwargs.get("extract_all_pages", True)),
            selected_pages=kwargs.get("selected_pages"),
            max_pages=kwargs.get("max_pages"),
            enable_figure_crops=bool(kwargs.get("enable_figure_crops", True)),
            enable_table_crops=bool(kwargs.get("enable_table_crops", False)),
            enable_equation_crops=bool(kwargs.get("enable_equation_crops", False)),
            crop_dpi=int(kwargs.get("crop_dpi", 160)),
            min_text_block_chars=int(kwargs.get("min_text_block_chars", 20)),
            include_debug_bbox=bool(kwargs.get("include_debug_bbox", False)),
            cache_policy=str(kwargs.get("cache_policy", "reuse_complete_only")),
        )
        return TranscribedDocument(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            backend=self.name,
            page_count=int(status.get("page_count") or 0) or None,
            elements=[],
            raw_output_path=None,
            warnings=[] if status.get("status") != "failed" else [str(status.get("error") or "pymupdf extraction failed")],
            backend_version=str(status.get("parser_version") or ""),
            backend_config={"delegated_to_legacy_parser": True},
        )
