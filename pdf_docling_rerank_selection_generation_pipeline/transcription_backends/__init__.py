from __future__ import annotations

from typing import Literal

from .base import TranscribedDocument, TranscribedElement, TranscriptionBackend

BACKEND_CHOICES = ("pymupdf", "docling")
BackendName = Literal["pymupdf", "docling"]


def normalize_backend_name(value: str | None, *, default: str = "docling") -> str:
    name = str(value or default).strip().lower()
    if name not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported transcription backend {value!r}; expected one of {', '.join(BACKEND_CHOICES)}")
    return name


def get_transcription_backend(name: str) -> TranscriptionBackend:
    normalized = normalize_backend_name(name)
    if normalized == "pymupdf":
        from .pymupdf_backend import PyMuPDFTranscriptionBackend

        return PyMuPDFTranscriptionBackend()
    if normalized == "docling":
        from .docling_backend import DoclingTranscriptionBackend

        return DoclingTranscriptionBackend()
    raise AssertionError(f"unreachable backend: {normalized}")


__all__ = [
    "BACKEND_CHOICES",
    "BackendName",
    "TranscribedDocument",
    "TranscribedElement",
    "TranscriptionBackend",
    "get_transcription_backend",
    "normalize_backend_name",
]
