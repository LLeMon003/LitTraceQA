"""Two-stage candidate retrieval: BM25 top-N then Qwen metadata filtering.

Stage 1 runs the alias-aware BM25/hybrid retriever over the full metadata pool
with a large budget (default 200) so gold papers ranked far down (for example
q_029's four papers at ranks 1/12/129/150) are not discarded by a small
per-query budget.  Stage 2 asks the answer model to select the most relevant
paper_ids from that pool using question + bounded title/abstract projections.
The result is a new candidate_papers.jsonl compatible with the downstream
pipeline.  This is inference-only: only the question and the metadata pool are
used, never validation gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .data_io import read_jsonl, write_jsonl
from .metadata_index import build_metadata_records, retrieve_candidates
from .parser import extract_json_object
from .vlm_answer_client import VLMAnswerClient


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage BM25 + Qwen candidate retrieval.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    parser.add_argument("--bm25-top-k", type=int, default=200)
    parser.add_argument("--select-input-limit", type=int, default=60, help="Candidates shown to the LLM (abstracts for first 60).")
    parser.add_argument("--select-output-limit", type=int, default=12)
    parser.add_argument("--samples", type=int, default=1, help="Independent LLM selections per query; union the results.")
    parser.add_argument("--only-query-ids", default="")
    return parser.parse_args()


def _project_candidate(hit: dict[str, Any], rank: int, select_input_limit: int) -> dict[str, Any]:
    abstract = (hit.get("abstract") or "")[:220] if rank <= 60 else ""
    return {
        "rank": rank,
        "paper_id": hit.get("paper_id"),
        "title": (hit.get("title") or "")[:160],
        "venue": f"{hit.get('venue') or ''} {hit.get('year') or ''}".strip(),
        "abstract": abstract,
    }


def select_relevant_papers(
    client: VLMAnswerClient,
    question: str,
    candidates: list[dict[str, Any]],
    *,
    input_limit: int,
    output_limit: int,
) -> list[str]:
    """Ask Qwen to pick relevant paper_ids from the projected candidate list."""
    projected = [
        _project_candidate(hit, rank, input_limit)
        for rank, hit in enumerate(candidates[:input_limit], start=1)
    ]
    system = (
        "Select the papers most likely to be REQUIRED to answer the research question. "
        "A paper may be required for its method, result table or figure, or citation context. "
        "Use venue/year, method names, datasets, and explicit figure/table mentions to decide. "
        "Do not exclude a paper merely because its title is generic; the question may reference "
        "a specific table, figure, or citation inside it. "
        f"Return only the paper_ids of the most relevant papers (up to {output_limit}) in JSON: {{\"paper_ids\":[...]}}. "
        "The list must contain at least one paper_id and must not be empty."
    )
    user = json.dumps({"question": question, "candidates": projected}, ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    parsed: dict[str, Any] = {}
    for _attempt in range(3):
        result = client.generate_prediction(messages, max_tokens=800)
        try:
            parsed = extract_json_object(str(result["content"]))
            break
        except ValueError:
            continue
    return [str(value) for value in (parsed.get("paper_ids") or []) if isinstance(value, str)][:max(1, output_limit)]


def main() -> int:
    args = _args()
    config = load_pipeline_config(args.env_path)
    client = VLMAnswerClient(config)
    pool_rows = read_jsonl(Path(args.official_dir) / "data" / "paper_metadata.jsonl")
    records = build_metadata_records(pool_rows)
    inputs = read_jsonl(Path(args.official_dir) / "data" / "validation_inputs.jsonl")
    only = {value.strip() for value in args.only_query_ids.split(",") if value.strip()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for sample in inputs:
        qid = str(sample.get("query_id") or "")
        if only and qid not in only:
            continue
        hits = retrieve_candidates(
            str(sample.get("question") or ""), records, top_k=args.bm25_top_k,
            enable_query_decomposition=True, subquery_top_k=8,
        )
        question = str(sample.get("question") or "")
        selected_ids: list[str] = []
        for _sample_index in range(max(1, args.samples)):
            for paper_id in select_relevant_papers(
                client, question, hits,
                input_limit=args.select_input_limit, output_limit=args.select_output_limit,
            ):
                if paper_id not in selected_ids:
                    selected_ids.append(paper_id)
        selected_set = set(selected_ids)
        rank = 0
        for hit in hits:
            if hit.get("paper_id") not in selected_set:
                continue
            rank += 1
            out_rows.append({
                "query_id": qid,
                "rank": rank,
                "configured_top_k_papers": args.select_output_limit,
                "query_decomposition_enabled": True,
                "retrieval_source": "two_stage_bm25_qwen",
                "paper_id": hit.get("paper_id"),
                "title": hit.get("title"),
                "abstract": hit.get("abstract"),
                "authors": hit.get("authors") or [],
                "venue": hit.get("venue"),
                "year": hit.get("year"),
                "pdf_url": hit.get("pdf_url"),
                "source_url": hit.get("source_url"),
                "arxiv_id": hit.get("arxiv_id"),
                "doi": hit.get("doi"),
                "openreview_id": hit.get("openreview_id"),
                "anthology_id": hit.get("anthology_id"),
                "online_links": hit.get("online_links") or [],
                "matched_aliases": hit.get("matched_aliases") or [],
                "retrieval_method": "two_stage",
                "score": hit.get("score"),
                "stage2_bm25_rank": hit.get("rank") or 0,
            })
        trace.append({"query_id": qid, "bm25_candidates": len(hits), "selected": selected_ids})
    write_jsonl(output / "candidate_papers.jsonl", out_rows)
    write_jsonl(output / "retrieval_trace.jsonl", trace)
    print(json.dumps({"queries": len(trace), "candidate_rows": len(out_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
