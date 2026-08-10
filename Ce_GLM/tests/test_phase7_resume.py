from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_ver2_reproduction as runner  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_runner(output: Path, resume_run: Path) -> runner.Runner:
    args = SimpleNamespace(
        mode="raw-fresh",
        output=output,
        start_stage=None,
        stop_stage=None,
        resume_manifest=None,
        resume_run=resume_run,
        verify_hashes=True,
        stall_timeout_sec=5.0,
        heartbeat_interval_sec=0.1,
    )
    return runner.Runner(args)


def test_resume_merge_skips_completed_and_preserves_order() -> None:
    test_root = WORKSPACE / "outputs" / "_phase7_resume_test"
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True)
    source = WORKSPACE / "outputs" / "fresh_api_manual_20260718_010651"
    interrupted = test_root / "interrupted"
    interrupted_raw_dir = interrupted / "work" / "raw_full_validation"
    interrupted_raw_dir.mkdir(parents=True)
    completed_ids = {
        "q_001", "q_002", "q_003", "q_004", "q_005", "q_006", "q_007", "q_008", "q_009", "q_010",
        "q_011", "q_012", "q_013", "q_014", "q_015", "q_017", "q_018", "q_019", "q_020", "q_021",
        "q_022", "q_023", "q_024", "q_025", "q_026", "q_027", "q_028", "q_029", "q_030", "q_031",
    }
    raw_rows = [row for row in read_jsonl(source / "work" / "raw_full_validation" / "raw_generation.jsonl") if row["query_id"] in completed_ids]
    run_rows = [row for row in read_jsonl(source / "work" / "raw_full_validation" / "run_log.jsonl") if row["query_id"] in completed_ids]
    runner.write_jsonl_atomic(interrupted_raw_dir / "raw_generation.jsonl", raw_rows)
    runner.write_jsonl_atomic(interrupted_raw_dir / "run_log.jsonl", run_rows)
    out = test_root / "predictions.jsonl"
    instance = make_runner(out, interrupted)
    instance.initialize()
    calls: list[list[str]] = []

    def fake_stream(name, command, cwd, stdout_path, stderr_path, heartbeat_files=None):
        calls.append(command)
        missing_input = Path(command[command.index("--input") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        rows = read_jsonl(missing_input)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = [{"query_id": row["query_id"], "answer": {}, "evidence": [], "gold_papers": []} for row in rows]
        runner.write_jsonl_atomic(output_dir / "stage_00_generated.jsonl", predictions)
        runner.write_jsonl_atomic(output_dir / "raw_generation.jsonl", [{"query_id": row["query_id"], "final_prediction": pred} for row, pred in zip(rows, predictions)])
        runner.write_jsonl_atomic(output_dir / "run_log.jsonl", [{"query_id": row["query_id"], "synthetic_test_log": True} for row in rows])
        stdout_path.write_text("synthetic stdout\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {"pid": 12345, "started_at_utc": runner.utc_now(), "exit_code": 0, "timed_out": False}

    instance.run_streaming_command = fake_stream  # type: ignore[method-assign]
    merged = instance.merge_partial_raw_generation(interrupted, instance.work_dir / "raw_full_validation")
    merged_rows = read_jsonl(merged)
    original_ids = [row["query_id"] for row in read_jsonl(runner.VER2 / "inputs" / "validation_inputs.jsonl")]
    merged_ids = [row["query_id"] for row in merged_rows]
    summary = json.loads((instance.work_dir / "raw_full_validation" / "resume_summary.json").read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert summary["completed_reused"] == 30
    assert summary["new_records_generated"] == 25
    assert summary["api_calls_avoided"] == 30
    assert merged_ids == original_ids
    assert len(set(merged_ids)) == 55
    assert "q_001" in summary["completed_ids"]
    assert "q_001" not in summary["remaining_ids"]


def test_duplicate_completed_records_are_rejected() -> None:
    source = WORKSPACE / "outputs" / "fresh_api_manual_20260718_010651"
    test_root = WORKSPACE / "outputs" / "_phase7_resume_duplicate_test"
    if test_root.exists():
        shutil.rmtree(test_root)
    shutil.copytree(source, test_root)
    raw = test_root / "work" / "raw_full_validation" / "raw_generation.jsonl"
    first = raw.read_text(encoding="utf-8").splitlines()[0]
    with raw.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(first + "\n")
    instance = make_runner(test_root / "predictions.jsonl", test_root)
    instance.initialize()
    try:
        instance.merge_partial_raw_generation(test_root, instance.work_dir / "raw_full_validation")
    except ValueError as exc:
        assert "duplicate query_id" in str(exc)
    else:
        raise AssertionError("duplicate partial query IDs were not rejected")


if __name__ == "__main__":
    test_resume_merge_skips_completed_and_preserves_order()
    test_duplicate_completed_records_are_rejected()
    print("phase7_resume_tests_passed")
