"""Rebuild symbolic records from cached Docling JSON without re-reading PDFs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl
from .transcription_backends.docling_backend import DoclingTranscriptionBackend
from .transcription_backends.standardizer import STANDARDIZER_VERSION, standardize_transcribed_document


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-standardize cached Docling outputs into an isolated root.")
    parser.add_argument("--source-root", required=True, help="Existing root containing <paper>/raw_docling_output.")
    parser.add_argument("--output-root", required=True, help="New root for standardized symbolic artifacts.")
    parser.add_argument("--paper-ids-input", default="", help="Optional candidate_papers.jsonl; only its paper IDs are rebuilt.")
    parser.add_argument("--only-paper-ids", default="", help="Optional comma-separated paper IDs.")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _status_complete(path: Path) -> bool:
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        status.get("status") == "complete"
        and status.get("transcription_backend") == "docling"
        and status.get("standardizer_version") == STANDARDIZER_VERSION
        and (path.parent / "symbolic_records.runtime.jsonl").exists()
    )


def _paper_ids(args: argparse.Namespace, source_root: Path) -> list[str]:
    selected = {part.strip() for part in args.only_paper_ids.split(",") if part.strip()}
    if args.paper_ids_input:
        selected.update(str(row.get("paper_id") or "").strip() for row in read_jsonl(args.paper_ids_input))
    if selected:
        return sorted(paper_id for paper_id in selected if paper_id)
    return sorted(path.name for path in source_root.iterdir() if path.is_dir())


def _link_cached_raw(source: Path, target: Path) -> None:
    raw_source = source / "raw_docling_output"
    raw_target = target / "raw_docling_output"
    if raw_target.exists() or raw_target.is_symlink():
        return
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(raw_source.resolve(), raw_target, target_is_directory=True)


def main() -> int:
    args = _args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    backend = DoclingTranscriptionBackend()
    report: list[dict[str, Any]] = []
    for paper_id in _paper_ids(args, source_root):
        source = source_root / paper_id
        raw_json = source / "raw_docling_output" / f"{paper_id}.docling.json"
        target = output_root / paper_id
        if not raw_json.exists():
            report.append({"paper_id": paper_id, "status": "skipped", "reason": "raw_docling_json_missing"})
            continue
        if not args.refresh and _status_complete(target / "artifact_status.json"):
            report.append({"paper_id": paper_id, "status": "reused"})
            continue
        try:
            _link_cached_raw(source, target)
            manifest_path = source / "document_manifest.json"
            pdf_path = ""
            if manifest_path.exists():
                pdf_path = str((json.loads(manifest_path.read_text(encoding="utf-8")) or {}).get("pdf_path") or "")
            document = backend.transcribe_pdf(paper_id, Path(pdf_path or f"{paper_id}.pdf"), target, overwrite=False)
            status = standardize_transcribed_document(document, target, overwrite=True, cache_policy="refresh")
            report.append({"paper_id": paper_id, "status": status.get("status"), "records": status.get("record_count")})
        except Exception as exc:
            report.append({"paper_id": paper_id, "status": "failed", "error": str(exc)})
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "restandardize_report.jsonl", report)
    counts: dict[str, int] = {}
    for row in report:
        counts[str(row.get("status") or "unknown")] = counts.get(str(row.get("status") or "unknown"), 0) + 1
    print(json.dumps({"papers": len(report), "status_counts": counts, "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
