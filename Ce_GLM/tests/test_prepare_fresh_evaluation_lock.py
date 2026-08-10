import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_fresh_evaluation_lock.py"


def test_prepares_single_use_lock_without_embedding_gold_contents(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text("".join(json.dumps({"query_id": f"q_{index:03d}"}) + "\n" for index in range(55)), encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"opaque":"fixture"}\n', encoding="utf-8")
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("# fixture\n", encoding="utf-8")
    result = tmp_path / "result.json"
    lock = tmp_path / "lock.json"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--prediction", str(prediction), "--gold", str(gold), "--evaluator", str(evaluator), "--python", sys.executable, "--result", str(result), "--lock", str(lock)],
        check=True,
        text=True,
        capture_output=True,
    )

    value = json.loads(lock.read_text(encoding="utf-8"))
    assert value["record_count"] == 55
    assert "fixture" not in lock.read_text(encoding="utf-8")
    assert "LOCK_PREPARED" in completed.stdout
