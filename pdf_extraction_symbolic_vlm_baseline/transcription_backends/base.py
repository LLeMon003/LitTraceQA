from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TranscribedElement:
    paper_id: str
    page: int | None
    element_id: str
    element_type: str
    text: str | None = None
    label: str | None = None
    bbox: list[float] | None = None
    reading_order: int | None = None
    raw_backend_type: str | None = None
    raw_backend_payload: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class TranscribedDocument:
    paper_id: str
    pdf_path: str
    backend: str
    page_count: int | None
    elements: list[TranscribedElement]
    raw_output_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    backend_version: str | None = None
    backend_config: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(Protocol):
    name: str

    def transcribe_pdf(
        self,
        paper_id: str,
        pdf_path: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> TranscribedDocument:
        ...
