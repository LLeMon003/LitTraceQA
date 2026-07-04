from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .data_io import read_jsonl


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 2}


def retrieve_contexts_for_query(
    query: str,
    candidate_records: list[dict[str, Any]],
    processed_pdf_root: str | Path,
    index_root: str | Path,
    context_selector: Any = None,
    top_n_text: int = 12,
    top_n_visual: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    selected_text: list[dict[str, Any]] = []
    selected_visual: list[dict[str, Any]] = []
    query_tokens = _tokens(query)
    for candidate in candidate_records:
        paper_id = str(candidate.get("paper_id", ""))
        paper_dir = Path(processed_pdf_root) / paper_id
        chunks_path = paper_dir / "chunks.jsonl"
        visuals_path = paper_dir / "visual_contexts.jsonl"
        chunks = read_jsonl(chunks_path) if chunks_path.exists() else []
        visuals = read_jsonl(visuals_path) if visuals_path.exists() else []
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            chunk_tokens = _tokens(text)
            overlap = len(query_tokens & chunk_tokens)
            type_boost = 2 if chunk.get("type") in {"table", "figure_caption", "equation"} else 0
            score = overlap + type_boost + min(len(text), 1200) / 12000.0
            selected_text.append({**chunk, "selection_score": score, "selection_method": "ocr_chunk_lexical"})
        for visual in visuals:
            caption_tokens = _tokens(str(visual.get("caption", "")))
            score = len(query_tokens & caption_tokens) + 1
            selected_visual.append({**visual, "selection_score": score, "selection_method": "ocr_visual_caption_lexical"})
    selected_text.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    selected_visual.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
    return {"selected_text_contexts": selected_text[:top_n_text], "selected_visual_contexts": selected_visual[:top_n_visual]}
