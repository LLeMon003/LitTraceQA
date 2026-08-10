from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
VER2 = PARENT / "littraceqa_baseline_Ver.2"
UQ = PARENT / "littraceqa_baseline_uq_experiments"
EXPERIMENT_DEFAULT = ROOT / "outputs" / "experiments" / "llm_prompt_context_causal_ablation"
RUN_SOURCE = ROOT / "outputs" / "fresh_api_manual_20260718_010651"
INPUT = VER2 / "inputs" / "validation_inputs.jsonl"
METADATA = VER2 / "inputs" / "paper_metadata.jsonl"
ENV_FILE = UQ / ".env"
ENDPOINT = "https://api.siliconflow.cn/v1"
CURRENT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
STRONGER_MODEL = "deepseek-ai/DeepSeek-V3.2"
TEMPERATURE = 0
PILOT_IDS = [
    "q_006", "q_031", "q_039", "q_046",
    "q_020", "q_021", "q_024", "q_026",
    "q_022", "q_028", "q_052", "q_056",
]
CONDITIONS = {
    "current-c0": {"model": CURRENT_MODEL, "context": "C0"},
    "current-c1": {"model": CURRENT_MODEL, "context": "C1"},
    "stronger-c1": {"model": STRONGER_MODEL, "context": "C1"},
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: non-object JSONL row")
            rows.append(obj)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def by_id(rows: list[dict[str, Any]], field: str = "query_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field) or "")
        if not key or key in result:
            raise ValueError(f"missing or duplicate {field}: {key!r}")
        result[key] = row
    return result


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_FILE.is_file():
        for raw in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith("SILICONFLOW_") and value:
            values[key] = value
    return values


def api_key() -> str:
    key = load_env().get("SILICONFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("SILICONFLOW_API_KEY is not configured")
    return key


def api_request(method: str, path: str, payload: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, Any], str, float]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT.rstrip("/") + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    started = time.perf_counter()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            latency = time.perf_counter() - started
            return response.status, json.loads(raw), raw, latency
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        latency = time.perf_counter() - started
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"error": raw[:500]}
        return exc.code, parsed, raw, latency


def list_models(timeout: int) -> dict[str, Any]:
    status, parsed, raw, latency = api_request("GET", "/models", None, timeout)
    ids = []
    for item in parsed.get("data", []) if isinstance(parsed, dict) else []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return {
        "status_code": status,
        "latency_sec": latency,
        "raw_hash": sha_text(raw),
        "current_model_available": CURRENT_MODEL in ids,
        "stronger_model_available": STRONGER_MODEL in ids,
        "matching_model_ids": [x for x in ids if "DeepSeek" in x or "deepseek" in x],
    }


def model_availability(experiment_dir: Path, timeout: int) -> dict[str, Any]:
    path = experiment_dir / "api_run" / "manifests" / "model_availability.json"
    if path.is_file():
        cached = read_json(path)
        cached["source"] = "cached"
        return cached
    result = list_models(timeout)
    result.update({"created_at_utc": now(), "source": "provider", "model_list_requests_this_runner": 1})
    write_json(path, result)
    return result


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} block, found {count}")
    return text.replace(old, new, 1)


def lock_conditions(experiment_dir: Path) -> None:
    pilot = read_json(experiment_dir / "PILOT_SELECTION.json")["cases"]
    final_dir = experiment_dir / "prompt_manifests_final"
    manifest_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for case in pilot:
        qid = case["query_id"]
        base_p2 = read_json(experiment_dir / "prompt_manifests" / f"{qid}_p2.json")
        c0 = read_json(experiment_dir / "context_current" / f"{qid}.json")
        c1 = read_json(experiment_dir / "context_local" / f"{qid}.json")
        for condition, spec in CONDITIONS.items():
            messages = json.loads(json.dumps(base_p2["messages"], ensure_ascii=False))
            if spec["context"] == "C0":
                messages[-1]["content"] = replace_once(
                    messages[-1]["content"], c1["rendered_context"], c0["rendered_context"], f"{qid} C1 context"
                )
                context_hash = sha_text(json.dumps(c0, ensure_ascii=False, sort_keys=True))
            else:
                context_hash = sha_text(json.dumps(c1, ensure_ascii=False, sort_keys=True))
            record = {
                "query_id": qid,
                "answer_family": case["answer_family"],
                "condition": condition,
                "model": spec["model"],
                "temperature": TEMPERATURE,
                "messages": messages,
                "prompt_length_chars": sum(len(m["content"]) for m in messages),
                "prompt_hash": sha_text(prompt_text(messages)),
                "context_policy": spec["context"],
                "context_hash": context_hash,
                "input_hash": base_p2["input_hash"],
                "gold_used": False,
                "forbidden_fields_in_prompt": [],
            }
            write_json(final_dir / condition / f"{qid}.json", record)
            manifest_rows.append({k: v for k, v in record.items() if k != "messages"})
    by_q: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest_rows:
        by_q.setdefault(row["query_id"], {})[row["condition"]] = row
    for qid, rows in by_q.items():
        c0_prompt = read_json(final_dir / "current-c0" / f"{qid}.json")
        c1_prompt = read_json(final_dir / "current-c1" / f"{qid}.json")
        stronger_prompt = read_json(final_dir / "stronger-c1" / f"{qid}.json")
        c0_context = read_json(experiment_dir / "context_current" / f"{qid}.json")["rendered_context"]
        c1_context = read_json(experiment_dir / "context_local" / f"{qid}.json")["rendered_context"]
        c0_norm = prompt_text(c0_prompt["messages"]).replace(c0_context, "<CONTEXT>")
        c1_norm = prompt_text(c1_prompt["messages"]).replace(c1_context, "<CONTEXT>")
        s_norm = prompt_text(stronger_prompt["messages"]).replace(c1_context, "<CONTEXT>")
        diff_rows.append({
            "query_id": qid,
            "c0_c1_only_context_diff": c0_norm == c1_norm and rows["current-c0"]["context_hash"] != rows["current-c1"]["context_hash"],
            "current_stronger_only_model_diff": c1_norm == s_norm and rows["current-c1"]["prompt_hash"] == rows["stronger-c1"]["prompt_hash"] and rows["current-c1"]["model"] != rows["stronger-c1"]["model"],
            "current_c0_prompt_hash": rows["current-c0"]["prompt_hash"],
            "current_c1_prompt_hash": rows["current-c1"]["prompt_hash"],
            "stronger_c1_prompt_hash": rows["stronger-c1"]["prompt_hash"],
        })
    if len(manifest_rows) != 36 or not all(x["c0_c1_only_context_diff"] and x["current_stronger_only_model_diff"] for x in diff_rows):
        raise RuntimeError("condition lock failed allowed-difference checks")
    lock = {
        "created_at_utc": now(),
        "status": "LOCKED",
        "planned_successful_generation_records": 36,
        "maximum_successful_generation_records": 42,
        "conditions": CONDITIONS,
        "temperature": TEMPERATURE,
        "prompt_manifest_count": len(manifest_rows),
        "pilot_ids": PILOT_IDS,
        "allowed_difference_checks": diff_rows,
        "gold_used": False,
    }
    write_json(experiment_dir / "MANUAL_CONDITION_LOCK.json", lock)
    write_json(experiment_dir / "FINAL_PROMPT_CONDITION_LOCK.json", {"created_at_utc": now(), "cases": manifest_rows})
    (experiment_dir / "PROMPT_DIFF_AUDIT.md").write_text(
        "# Final Prompt Difference Audit\n\n"
        "The final lock uses the same structured family-specific prompt wording for CURRENT-C0 and CURRENT-C1. "
        "For each query, only the context block and context hash differ between those conditions.\n\n"
        "CURRENT-C1 and STRONGER-C1 use byte-identical prompt messages and context; only the model id differs.\n\n"
        f"Planned generation records: 36. Budget ceiling: 42. Gold and frozen answers were not loaded into prompt construction.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "LOCKED", "planned_pairs": 36, "queries": len(PILOT_IDS)}))


def condition_paths(experiment_dir: Path, condition: str) -> dict[str, Path]:
    api_dir = experiment_dir / "api_run"
    stem = condition.replace("-", "_")
    return {
        "raw": api_dir / "raw" / f"{stem}.jsonl",
        "parsed": api_dir / "parsed" / f"{stem}.jsonl",
        "processed": api_dir / "processed" / f"{stem}.jsonl",
        "log": api_dir / "logs" / f"{stem}.jsonl",
        "status": api_dir / "status" / f"{stem}.json",
        "manifest": api_dir / "manifests" / f"{stem}.json",
        "recovery_raw": api_dir / "recovery" / "raw" / f"{stem}.jsonl",
        "recovery_parsed": api_dir / "recovery" / "parsed" / f"{stem}.jsonl",
    }


def validate_prompt(record: dict[str, Any], condition: str, model: str) -> None:
    if record["condition"] != condition or record["model"] != model:
        raise ValueError(f"prompt condition/model mismatch for {record.get('query_id')}")
    text = prompt_text(record["messages"]).lower()
    forbidden = ["official gold", "expected option letter", "frozen checkpoint", "version 3 correction"]
    hits = [x for x in forbidden if x in text]
    if hits:
        raise ValueError(f"forbidden prompt phrase in {record.get('query_id')}: {hits}")


def normalize_processed(parsed: dict[str, Any], qid: str, condition: str, model: str, experiment_dir: Path) -> dict[str, Any]:
    source_path = str(VER2)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from src.formatter import normalize_prediction, postprocess_with_pdf_context  # type: ignore
    inputs = by_id(read_jsonl(INPUT))
    metadata = read_jsonl(METADATA)
    all_paper_ids = {str(x.get("paper_id") or "") for x in metadata}
    c0 = read_json(experiment_dir / "context_current" / f"{qid}.json")
    prediction = normalize_prediction(parsed, inputs[qid], c0["candidate_papers"], all_paper_ids)
    prediction = postprocess_with_pdf_context(prediction, inputs[qid], c0["pdf_context"])
    prediction["_experiment"] = {"condition": condition, "model": model}
    return prediction


def run_condition(experiment_dir: Path, condition: str, model: str, resume: bool, max_records: int | None, query_id: str | None, timeout: int, recover_failed: bool = False) -> None:
    paths = condition_paths(experiment_dir, condition)
    expected = model
    if condition in CONDITIONS and CONDITIONS[condition]["model"] != model:
        raise ValueError(f"{condition} must use {CONDITIONS[condition]['model']}")
    availability = model_availability(experiment_dir, timeout)
    if not availability.get("current_model_available") or (model == STRONGER_MODEL and not availability.get("stronger_model_available")):
        raise RuntimeError(f"model availability check failed: {availability}")
    raw_rows = read_jsonl(paths["raw"]) if resume else []
    recovery_raw_rows = read_jsonl(paths["recovery_raw"]) if resume else []
    parsed_rows = read_jsonl(paths["parsed"]) if resume else []
    processed_rows = read_jsonl(paths["processed"]) if resume else []
    all_attempt_rows = raw_rows + recovery_raw_rows
    attempted_pairs = {(r["condition"], r["query_id"]) for r in raw_rows}
    recovered_pairs = {(r["condition"], r["query_id"]) for r in recovery_raw_rows}
    valid_pairs = {(r["condition"], r["query_id"]) for r in all_attempt_rows if r.get("parse_valid")}
    targets = [query_id] if query_id else PILOT_IDS
    made = 0
    attempts_total = 0
    errors: list[dict[str, Any]] = []
    for qid in targets:
        pair = (condition, qid)
        if pair in valid_pairs:
            continue
        if recover_failed:
            if pair not in attempted_pairs or pair in recovered_pairs:
                continue
        elif pair in attempted_pairs:
            continue
        if max_records is not None and made >= max_records:
            break
        prompt = read_json(experiment_dir / "prompt_manifests_final" / condition / f"{qid}.json")
        validate_prompt(prompt, condition, expected)
        parsed_obj = None
        raw_response_obj: dict[str, Any] | None = None
        response_text = ""
        last_error = None
        latency_total = 0.0
        usage = None
        attempts = 0
        while attempts < 2 and parsed_obj is None:
            attempts += 1
            attempts_total += 1
            payload = {
                "model": model,
                "messages": prompt["messages"],
                "temperature": TEMPERATURE,
                "max_tokens": 6000,
                "response_format": {"type": "json_object"},
            }
            try:
                status, response_obj, raw_text, latency = api_request("POST", "/chat/completions", payload, timeout)
                latency_total += latency
                raw_response_obj = response_obj
                if status >= 400:
                    last_error = f"http_{status}"
                else:
                    choice = (response_obj.get("choices") or [{}])[0]
                    message = choice.get("message") if isinstance(choice, dict) else {}
                    response_text = str((message or {}).get("content") or "")
                    usage = response_obj.get("usage")
                    parsed_obj = extract_json_object(response_text)
                    if parsed_obj is None:
                        last_error = "malformed_json"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if parsed_obj is None and attempts < 2:
                time.sleep(1)
        raw_record = {
            "created_at_utc": now(),
            "query_id": qid,
            "answer_family": prompt["answer_family"],
            "condition": condition,
            "model": model,
            "temperature": TEMPERATURE,
            "prompt_hash": prompt["prompt_hash"],
            "context_hash": prompt["context_hash"],
            "input_hash": prompt["input_hash"],
            "prompt_length_chars": prompt["prompt_length_chars"],
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "latency_sec": latency_total,
            "response_text_hash": sha_text(response_text),
            "provider_response_hash": sha_text(json.dumps(raw_response_obj, ensure_ascii=False, sort_keys=True)) if raw_response_obj else None,
            "provider_response": raw_response_obj,
            "parse_valid": parsed_obj is not None,
            "error": last_error,
            "usage": usage,
            "gold_used": False,
            "execution_phase": "POST_TUN_NETWORK_RECOVERY" if recover_failed else "PRE_TUN_OR_STABLE_PRIMARY",
            "network_route": "direct_socket_proxy_bypass_tls_verified",
        }
        target_raw_rows = recovery_raw_rows if recover_failed else raw_rows
        target_raw_rows.append(raw_record)
        write_jsonl_atomic(paths["recovery_raw"] if recover_failed else paths["raw"], target_raw_rows)
        if parsed_obj is not None:
            parsed_record = {
                "query_id": qid,
                "answer_family": prompt["answer_family"],
                "condition": condition,
                "model": model,
                "parsed_json": parsed_obj,
                "parsed_hash": sha_text(json.dumps(parsed_obj, ensure_ascii=False, sort_keys=True)),
                "source_response_hash": raw_record["response_text_hash"],
                "execution_phase": raw_record["execution_phase"],
                "network_route": raw_record["network_route"],
            }
            if recover_failed:
                recovery_parsed_rows = read_jsonl(paths["recovery_parsed"])
                recovery_parsed_rows.append(parsed_record)
                write_jsonl_atomic(paths["recovery_parsed"], recovery_parsed_rows)
            parsed_rows = [r for r in parsed_rows if not (r.get("condition") == condition and r.get("query_id") == qid)]
            parsed_rows.append(parsed_record)
            write_jsonl_atomic(paths["parsed"], parsed_rows)
            processed_rows = [r for r in processed_rows if r.get("query_id") != qid]
            processed_rows.append(normalize_processed(parsed_obj, qid, condition, model, experiment_dir))
            write_jsonl_atomic(paths["processed"], processed_rows)
            made += 1
        else:
            errors.append({"query_id": qid, "error": last_error})
        log_rows = read_jsonl(paths["log"])
        log_rows.append({"time": now(), "query_id": qid, "condition": condition, "attempts": attempts, "parse_valid": parsed_obj is not None, "error": last_error})
        write_jsonl_atomic(paths["log"], log_rows)
        write_json(paths["status"], {"updated_at_utc": now(), "condition": condition, "records": len(raw_rows), "recovery_records": len(recovery_raw_rows), "parsed_records": len(parsed_rows), "last_query_id": qid})
    all_rows = raw_rows + recovery_raw_rows
    manifest = {
        "created_at_utc": now(),
        "condition": condition,
        "model": model,
        "temperature": TEMPERATURE,
        "records": len(all_rows),
        "primary_records": len(raw_rows),
        "post_tun_recovery_records": len(recovery_raw_rows),
        "parsed_records": len(parsed_rows),
        "processed_records": len(processed_rows),
        "unique_query_ids": len({r["query_id"] for r in all_rows}),
        "attempts": sum(int(r.get("attempts") or 0) for r in all_rows),
        "retries": sum(int(r.get("retry_count") or 0) for r in all_rows),
        "timeouts": sum("timeout" in str(r.get("error") or "").lower() for r in all_rows),
        "usage": summarize_usage(all_rows),
        "latency_sec": sum(float(r.get("latency_sec") or 0) for r in all_rows),
        "raw_sha256": sha_file(paths["raw"]) if paths["raw"].is_file() else None,
        "recovery_raw_sha256": sha_file(paths["recovery_raw"]) if paths["recovery_raw"].is_file() else None,
        "parsed_sha256": sha_file(paths["parsed"]) if paths["parsed"].is_file() else None,
        "processed_sha256": sha_file(paths["processed"]) if paths["processed"].is_file() else None,
        "errors": errors,
        "model_availability": availability,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps({"status": "CONDITION_DONE", "condition": condition, "records": len(all_rows), "parsed": len(parsed_rows), "new_successes": made, "errors": len(errors), "recover_failed": recover_failed}))


def summarize_usage(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    present = 0
    for row in raw_rows:
        usage = row.get("usage")
        if isinstance(usage, dict):
            present += 1
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return {"records_with_usage": present, **totals, "estimated_cost": None}


def dry_run(experiment_dir: Path) -> None:
    lock = read_json(experiment_dir / "MANUAL_CONDITION_LOCK.json")
    planned = []
    for condition, spec in CONDITIONS.items():
        for qid in PILOT_IDS:
            path = experiment_dir / "prompt_manifests_final" / condition / f"{qid}.json"
            prompt = read_json(path)
            validate_prompt(prompt, condition, spec["model"])
            planned.append((condition, qid, prompt["prompt_hash"]))
    if len(planned) != 36 or len(set((c, q) for c, q, _ in planned)) != 36:
        raise RuntimeError("dry run did not find exactly 36 unique condition/query pairs")
    print(json.dumps({"status": "DRY_RUN_PASS", "planned_pairs": len(planned), "lock_status": lock.get("status")}))


def status(experiment_dir: Path) -> None:
    result = {"conditions": {}, "total_successful_generation_records": 0, "budget_ceiling": 42}
    for condition in CONDITIONS:
        paths = condition_paths(experiment_dir, condition)
        raw_rows = read_jsonl(paths["raw"])
        recovery_raw_rows = read_jsonl(paths["recovery_raw"])
        all_rows = raw_rows + recovery_raw_rows
        parsed_rows = read_jsonl(paths["parsed"])
        result["conditions"][condition] = {
            "raw_records": len(all_rows),
            "primary_raw_records": len(raw_rows),
            "post_tun_recovery_records": len(recovery_raw_rows),
            "parsed_records": len(parsed_rows),
            "unique_query_ids": len({r.get("query_id") for r in all_rows}),
            "path": str(paths["raw"].relative_to(ROOT)),
        }
        result["total_successful_generation_records"] += len(parsed_rows)
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DEFAULT)
    parser.add_argument("--condition", choices=["current-c0", "current-c1", "stronger-c1", "all"])
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--query-id")
    parser.add_argument("--request-timeout-sec", type=int, default=180)
    parser.add_argument("--stall-timeout-sec", type=int, default=1200)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--recover-failed", action="store_true")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if args.lock:
        lock_conditions(experiment_dir)
        return
    if args.dry_run:
        dry_run(experiment_dir)
        return
    if args.status_only:
        status(experiment_dir)
        return
    if not args.condition:
        parser.error("--condition is required unless --lock, --dry-run, or --status-only is used")
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for condition in conditions:
        model = args.model or CONDITIONS[condition]["model"]
        run_condition(experiment_dir, condition, model, args.resume, args.max_records, args.query_id, args.request_timeout_sec, args.recover_failed)


if __name__ == "__main__":
    main()
