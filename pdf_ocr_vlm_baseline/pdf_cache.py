from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_io import append_jsonl
from metadata_only_baseline.pdf_download_core import DEFAULT_USER_AGENT, now_iso, sha256_file, try_download_pdf


def _try_download_pdf(
    metadata: dict[str, Any],
    output_path: Path,
    *,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[bool, dict[str, Any]]:
    return try_download_pdf(
        metadata,
        output_path,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
        user_agent=user_agent,
    )


def ensure_candidate_pdfs(
    candidate_records: list[dict[str, Any]],
    pdf_output_dir: str | Path,
    *,
    overwrite: bool = False,
    metadata_by_id: dict[str, dict[str, Any]] | None = None,
    sleep_seconds: float = 2.0,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    pdf_root = Path(pdf_output_dir)
    pdf_dir = pdf_root / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    existing = 0
    downloaded = 0
    failed = 0
    for candidate in candidate_records:
        paper_id = str(candidate.get("paper_id", ""))
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        available = pdf_path.exists() and not overwrite
        if available:
            existing += 1
            status = "existing"
            note = "Local cache hit."
            result: dict[str, Any] = {}
        else:
            metadata = (metadata_by_id or {}).get(paper_id, candidate)
            ok, result = _try_download_pdf(
                metadata,
                pdf_path,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            available = ok
            if ok:
                downloaded += 1
                status = "downloaded"
                note = "Downloaded on demand before PDF page rendering."
            else:
                failed += 1
                status = str(result.get("status", "missing_not_downloaded"))
                note = "No local PDF available after on-demand download attempts."
        rows.append(
            {
                "paper_id": paper_id,
                "available": available,
                "local_path": str(pdf_path),
                "status": status,
                "note": note,
                "file_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "sha256": sha256_file(pdf_path) if pdf_path.exists() else "",
                "checked_at": now_iso(),
                "source_type": result.get("source_type", ""),
                "source_url": result.get("source_url", ""),
                "resolved_url": result.get("resolved_url", ""),
                "http_status": result.get("http_status"),
                "download_attempts": result.get("attempts", []),
            }
        )
    return {
        "rows": rows,
        "existing_count": existing,
        "newly_downloaded_count": downloaded,
        "failed_count": failed,
    }


def write_pdf_availability(path: str | Path, availability: dict[str, Any], query_id: str) -> None:
    append_jsonl(
        path,
        {
            "query_id": query_id,
            "existing_count": availability.get("existing_count", 0),
            "newly_downloaded_count": availability.get("newly_downloaded_count", 0),
            "failed_count": availability.get("failed_count", 0),
            "papers": availability.get("rows", []),
        },
    )
