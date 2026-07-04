from __future__ import annotations

import argparse
import hashlib
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_io import append_jsonl, find_official_file, read_jsonl
from .link_utils import extract_pdf_candidate_urls


DEFAULT_USER_AGENT = "LitTraceQA-RawPDFDownloader/1.0 (+https://github.com/)"  # polite crawler identity
PDF_MAGIC_BYTES = b"%PDF"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw public PDFs from LitTraceQA paper metadata.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", default="raw_pdfs")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", type=_parse_bool, default=True)
    parser.add_argument("--overwrite", type=_parse_bool, default=False)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_paper_id(value: Any, fallback_index: int) -> str:
    raw = str(value or f"paper_{fallback_index:06d}").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe or f"paper_{fallback_index:06d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_pdf(content_type: str, body: bytes) -> bool:
    return "application/pdf" in content_type.lower() or body.startswith(PDF_MAGIC_BYTES)


def _request_once(url: str, user_agent: str, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/pdf,*/*;q=0.8"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        return {
            "body": body,
            "http_status": int(getattr(response, "status", response.getcode())),
            "content_type": response.headers.get("Content-Type", ""),
            "resolved_url": response.geturl(),
        }


def _download_candidate(
    candidate: dict[str, Any],
    user_agent: str,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    total_attempts = max(1, max_retries + 1)
    for attempt_no in range(1, total_attempts + 1):
        http_status: int | None = None
        content_type = ""
        resolved_url = ""
        error = ""
        body = b""
        try:
            result = _request_once(candidate["url"], user_agent, timeout_seconds)
            body = result["body"]
            http_status = result["http_status"]
            content_type = result["content_type"]
            resolved_url = result["resolved_url"]
            if http_status == 200 and _looks_like_pdf(content_type, body):
                attempts.append(
                    {
                        "source_type": candidate["source_type"],
                        "url": candidate["url"],
                        "resolved_url": resolved_url,
                        "http_status": http_status,
                        "content_type": content_type,
                        "attempt_no": attempt_no,
                        "error": "",
                    }
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                return (
                    {
                        "body": body,
                        "http_status": http_status,
                        "content_type": content_type,
                        "resolved_url": resolved_url,
                        "attempt_count": attempt_no,
                    },
                    attempts,
                )
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
                "source_type": candidate["source_type"],
                "url": candidate["url"],
                "resolved_url": resolved_url,
                "http_status": http_status,
                "content_type": content_type,
                "attempt_no": attempt_no,
                "error": error,
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if attempt_no < total_attempts:
            time.sleep(min(30.0, 2.0 ** (attempt_no - 1)))
    return None, attempts


def _build_manifest_row(
    paper: dict[str, Any],
    paper_id: str,
    status: str,
    local_path: Path,
    candidate: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    attempt_count: int = 0,
    error: str = "",
) -> dict[str, Any]:
    has_file = local_path.exists()
    return {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "status": status,
        "source_type": candidate.get("source_type", "") if candidate else "",
        "source_url": candidate.get("url", "") if candidate else "",
        "resolved_url": result.get("resolved_url", "") if result else "",
        "local_path": str(local_path),
        "http_status": result.get("http_status") if result else None,
        "content_type": result.get("content_type", "") if result else "",
        "file_size_bytes": local_path.stat().st_size if has_file else 0,
        "sha256": _sha256_file(local_path) if has_file else "",
        "downloaded_at": _now_iso(),
        "attempt_count": attempt_count,
        "error": error,
    }


def _write_report(
    report_path: Path,
    total_papers: int,
    attempted_papers: int,
    downloaded_count: int,
    skipped_count: int,
    failed_count: int,
    success_by_source: dict[str, int],
    output_dir: Path,
    manifest_path: Path,
    errors_path: Path,
) -> None:
    source_lines = [
        f"- direct_pdf: {success_by_source.get('direct_pdf', 0)}",
        f"- arxiv: {success_by_source.get('arxiv', 0)}",
        f"- openreview: {success_by_source.get('openreview', 0)}",
        f"- doi: {success_by_source.get('doi', 0)}",
        f"- direct_url: {success_by_source.get('direct_url', 0)}",
        f"- unknown: {success_by_source.get('unknown', 0)}",
    ]
    lines = [
        "# Raw PDF Download Report",
        "",
        "## 汇总",
        "",
        f"- 总 paper 数: {total_papers}",
        f"- 本次尝试 paper 数: {attempted_papers}",
        f"- 成功下载数: {downloaded_count}",
        f"- skipped 数: {skipped_count}",
        f"- failed 数: {failed_count}",
        "",
        "## 各来源成功数",
        "",
        *source_lines,
        "",
        "## 输出",
        "",
        f"- 输出目录: `{output_dir}`",
        f"- manifest 路径: `{manifest_path}`",
        f"- error log 路径: `{errors_path}`",
        "",
        "## 说明",
        "",
        "- 本脚本只下载 raw PDF，不做 PDF 结构化解析。",
        "- 本脚本不抽取 text、tables、figures、equations 或 citation contexts。",
        "- 本脚本不调用 LLM API，不使用 gold labels，不修改 metadata-only baseline 输出。",
        "- 本脚本不绕过 paywall，不使用 Sci-Hub，只尝试 metadata 中公开可访问的 DOI、arXiv、OpenReview、direct URL、PDF URL。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pdf_dir = output_dir / "pdf"
    failed_dir = output_dir / "failed"
    manifest_path = output_dir / "manifest.jsonl"
    errors_path = output_dir / "download_errors.jsonl"
    report_path = output_dir / "download_report.md"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)

    metadata_path = find_official_file(args.official_dir, "paper_metadata.jsonl")
    metadata_rows = read_jsonl(metadata_path)
    selected_rows = metadata_rows[: args.max_papers] if args.max_papers is not None else metadata_rows

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    success_by_source: dict[str, int] = {}

    print(f"读取 metadata: {metadata_path}，总数 {len(metadata_rows)}，本次处理 {len(selected_rows)}")
    for index, paper in enumerate(selected_rows, start=1):
        paper_id = _safe_paper_id(paper.get("paper_id"), index)
        local_path = pdf_dir / f"{paper_id}.pdf"
        candidates = extract_pdf_candidate_urls(paper)
        print(f"[{index}/{len(selected_rows)}] {paper_id}: 候选 URL {len(candidates)} 个")

        if local_path.exists() and args.resume and not args.overwrite:
            row = _build_manifest_row(paper, paper_id, "skipped", local_path, attempt_count=0, error="local pdf exists; resume enabled")
            append_jsonl(manifest_path, row)
            skipped_count += 1
            continue

        all_attempts: list[dict[str, Any]] = []
        success_candidate: dict[str, Any] | None = None
        success_result: dict[str, Any] | None = None
        for candidate in candidates:
            result, attempts = _download_candidate(
                candidate,
                user_agent=args.user_agent,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )
            all_attempts.extend(attempts)
            if result is not None:
                local_path.write_bytes(result["body"])
                success_candidate = candidate
                success_result = result
                break

        if success_candidate and success_result:
            source_type = success_candidate.get("source_type", "unknown")
            success_by_source[source_type] = success_by_source.get(source_type, 0) + 1
            row = _build_manifest_row(
                paper,
                paper_id,
                "downloaded",
                local_path,
                candidate=success_candidate,
                result=success_result,
                attempt_count=len(all_attempts),
                error="",
            )
            append_jsonl(manifest_path, row)
            downloaded_count += 1
            continue

        final_error = "no candidate urls found" if not candidates else "all candidate urls failed"
        error_row = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "candidate_urls": candidates,
            "attempts": all_attempts,
            "final_error": final_error,
        }
        append_jsonl(errors_path, error_row)
        row = _build_manifest_row(
            paper,
            paper_id,
            "failed",
            local_path,
            candidate=candidates[0] if candidates else None,
            attempt_count=len(all_attempts),
            error=final_error,
        )
        append_jsonl(manifest_path, row)
        failed_count += 1

    _write_report(
        report_path=report_path,
        total_papers=len(metadata_rows),
        attempted_papers=len(selected_rows),
        downloaded_count=downloaded_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        success_by_source=success_by_source,
        output_dir=output_dir,
        manifest_path=manifest_path,
        errors_path=errors_path,
    )
    print(
        {
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "manifest": str(manifest_path),
            "errors": str(errors_path),
            "report": str(report_path),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
