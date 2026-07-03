from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import is_api_key_configured, load_config, mask_api_key
from .data_io import append_jsonl, find_official_file, read_jsonl, write_jsonl
from .llm_client import SiliconFlowClient
from .metadata_index import build_metadata_records, retrieve_candidates
from .parser import extract_json_object, make_fallback_prediction, normalize_prediction, validate_prediction_shape
from .prompt_builder import build_littraceqa_prompt


BASELINE_TYPE = "metadata_only_title_abstract"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata-only LitTraceQA API baseline.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", default="outputs/api_baseline")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--env-path", default=".env")
    return parser.parse_args()


def _load_done_predictions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        query_id = row.get("query_id")
        if isinstance(query_id, str):
            done.add(query_id)
    return done


def _write_report(
    output_dir: Path,
    *,
    processed: int,
    successful_api_calls: int,
    parse_failures: int,
    fallback_predictions: int,
    model: str,
    base_url: str,
    top_k: int,
    dry_run: bool,
    api_key_status: str,
) -> None:
    report = output_dir / "run_report.md"
    lines = [
        "# LitTraceQA Metadata-Only Baseline Report",
        "",
        f"- baseline type: `{BASELINE_TYPE}`",
        f"- processed query count: {processed}",
        f"- successful API call count: {successful_api_calls}",
        f"- parse failure count: {parse_failures}",
        f"- fallback prediction count: {fallback_predictions}",
        f"- model: `{model}`",
        f"- base url: `{base_url}`",
        f"- top_k: {top_k}",
        f"- dry_run: {dry_run}",
        f"- API key status: {api_key_status}",
        "",
        "## Output Paths",
        "",
        f"- predictions: `{output_dir / 'predictions.jsonl'}`",
        f"- raw responses: `{output_dir / 'raw_llm_responses.jsonl'}`",
        f"- candidate papers: `{output_dir / 'candidate_papers.jsonl'}`",
        f"- prompt previews: `{output_dir / 'prompt_previews.jsonl'}`",
        f"- errors: `{output_dir / 'errors.jsonl'}`",
        "",
        "## Local Evaluator",
        "",
        "```bash",
        f"python -m littrace_baseline.evaluate_local --official-dir official_dev --pred {output_dir / 'predictions.jsonl'}",
        "```",
        "",
        "## Scope",
        "",
        "This run did not access PDFs, URLs, DOI links, arXiv pages, OpenReview pages, or full paper text. "
        "It used only `title` and `abstract` fields from `paper_metadata.jsonl` for retrieval and prompting.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    official_dir = Path(args.official_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = find_official_file(official_dir, "validation_inputs.jsonl")
    metadata_path = find_official_file(official_dir, "paper_metadata.jsonl")
    samples = read_jsonl(inputs_path)
    if args.max_queries is not None:
        samples = samples[: args.max_queries]
    metadata_records = build_metadata_records(read_jsonl(metadata_path))

    predictions_path = output_dir / "predictions.jsonl"
    raw_path = output_dir / "raw_llm_responses.jsonl"
    candidates_path = output_dir / "candidate_papers.jsonl"
    previews_path = output_dir / "prompt_previews.jsonl"
    errors_path = output_dir / "errors.jsonl"

    if not args.resume:
        for path in [candidates_path, previews_path, errors_path]:
            path.write_text("", encoding="utf-8")
        if not args.dry_run:
            for path in [predictions_path, raw_path]:
                path.write_text("", encoding="utf-8")

    config = load_config(args.env_path)
    api_key_status = "<not required for dry-run>"
    client: SiliconFlowClient | None = None
    if not args.dry_run:
        if not config.env_exists:
            print(f"未检测到 .env 文件：{config.env_path}。停止 API 调用；请先补齐 SiliconFlow 配置。")
            return 2
        if not is_api_key_configured(config.api_key):
            print("已检测到 .env，但 SILICONFLOW_API_KEY 未补齐或仍是占位符。停止 API 调用。")
            print("请在 .env 中设置真实 key；不会输出或记录完整 API key。")
            return 2
        api_key_status = mask_api_key(config.api_key or "")
        print("已检测到 .env。")
        print(f"已检测到 SILICONFLOW_BASE_URL: {config.base_url}")
        print(f"已检测到 SILICONFLOW_MODEL: {config.model}")
        print(f"已检测到 SILICONFLOW_API_KEY: {api_key_status}")
        print("本次 baseline 为 metadata-only，只使用 title 和 abstract，不访问 PDF 或在线链接。")
        client = SiliconFlowClient(config)

    done = _load_done_predictions(predictions_path) if args.resume and not args.dry_run else set()
    processed = 0
    successful_api_calls = 0
    parse_failures = 0
    fallback_predictions = 0

    for sample in samples:
        query_id = str(sample.get("query_id", ""))
        if query_id in done:
            continue
        candidates = retrieve_candidates(str(sample.get("question", "")), metadata_records, top_k=args.top_k)
        append_jsonl(candidates_path, {"query_id": query_id, "question": sample.get("question"), "candidates": candidates})
        messages = build_littraceqa_prompt(sample, candidates)
        append_jsonl(
            previews_path,
            {
                "query_id": query_id,
                "messages": messages,
                "baseline_type": BASELINE_TYPE,
                "used_online_access": False,
            },
        )
        processed += 1
        if args.dry_run:
            continue

        top1 = candidates[0] if candidates else None
        candidate_ids = [str(candidate.get("paper_id", "")) for candidate in candidates]
        try:
            assert client is not None
            content, raw_response = client.chat_completions(messages)
            successful_api_calls += 1
            append_jsonl(raw_path, {"query_id": query_id, "raw_response": raw_response, "content": content})
            try:
                parsed = extract_json_object(content)
                prediction, normalization_errors = normalize_prediction(parsed, sample, candidate_ids)
                for error in normalization_errors:
                    append_jsonl(errors_path, error)
            except Exception as exc:
                parse_failures += 1
                fallback_predictions += 1
                append_jsonl(errors_path, {"query_id": query_id, "type": "parse_failure", "error": str(exc)})
                prediction = make_fallback_prediction(sample, top1)
        except Exception as exc:
            fallback_predictions += 1
            append_jsonl(errors_path, {"query_id": query_id, "type": "api_failure", "error": str(exc)})
            prediction = make_fallback_prediction(sample, top1)

        if not validate_prediction_shape(prediction):
            fallback_predictions += 1
            append_jsonl(errors_path, {"query_id": query_id, "type": "invalid_prediction_shape"})
            prediction = make_fallback_prediction(sample, top1)
        append_jsonl(predictions_path, prediction)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    _write_report(
        output_dir,
        processed=processed,
        successful_api_calls=successful_api_calls,
        parse_failures=parse_failures,
        fallback_predictions=fallback_predictions,
        model=config.model,
        base_url=config.base_url,
        top_k=args.top_k,
        dry_run=args.dry_run,
        api_key_status=api_key_status,
    )
    print(
        json.dumps(
            {
                "processed": processed,
                "successful_api_calls": successful_api_calls,
                "parse_failures": parse_failures,
                "fallback_predictions": fallback_predictions,
                "output_dir": str(output_dir),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

