from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_io import append_jsonl
from .link_utils import extract_pdf_candidate_urls


PDF_MAGIC_BYTES = b"%PDF"
DEFAULT_USER_AGENT = "LitTraceQA-PDF-OCR-Baseline/1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_pdf(content_type: str, body: bytes) -> bool:
    return "application/pdf" in content_type.lower() or body.startswith(PDF_MAGIC_BYTES)


def _try_download_pdf(
    metadata: dict[str, Any],
    output_path: Path,
    *,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[bool, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    candidates = extract_pdf_candidate_urls(metadata)
    for candidate in candidates:
        for attempt_no in range(1, max(1, max_retries + 1) + 1):
            error = ""
            http_status: int | None = None
            content_type = ""
            resolved_url = ""
            body = b""
            try:
                request = urllib.request.Request(
                    candidate["url"],
                    headers={"User-Agent": user_agent, "Accept": "application/pdf,*/*;q=0.8"},
                )
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    body = response.read()
                    http_status = int(getattr(response, "status", response.getcode()))
                    content_type = response.headers.get("Content-Type", "")
                    resolved_url = response.geturl()
                if http_status == 200 and _looks_like_pdf(content_type, body):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(body)
                    return True, {
                        "status": "downloaded",
                        "source_type": candidate.get("source_type", ""),
                        "source_url": candidate.get("url", ""),
                        "resolved_url": resolved_url,
                        "http_status": http_status,
                        "content_type": content_type,
                        "attempt_count": attempt_no,
                        "attempts": attempts,
                    }
                error = "not a pdf" if http_status == 200 else f"http status {http_status}"
            except urllib.error.HTTPError as exc:
                http_status = exc.code
                content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
                resolved_url = exc.geturl()
                error = str(exc)
            except Exception as exc:
                error = str(exc)
            attempts.append(
                {
                    "source_type": candidate.get("source_type", ""),
                    "url": candidate.get("url", ""),
                    "resolved_url": resolved_url,
                    "http_status": http_status,
                    "content_type": content_type,
                    "attempt_no": attempt_no,
                    "error": error,
                }
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if attempt_no < max(1, max_retries + 1):
                time.sleep(min(30.0, 2.0 ** (attempt_no - 1)))
    return False, {
        "status": "download_failed" if candidates else "missing_no_pdf_candidates",
        "attempts": attempts,
        "candidate_count": len(candidates),
    }


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
                note = "Downloaded on demand before OCR."
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
                "sha256": _sha256_file(pdf_path) if pdf_path.exists() else "",
                "checked_at": _now_iso(),
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
