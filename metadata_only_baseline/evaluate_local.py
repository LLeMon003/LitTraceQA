from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from .data_io import find_official_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LitTraceQA local evaluator.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--pred", default="outputs/api_baseline/predictions.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_dir = Path(args.official_dir)
    evaluator = official_dir / "scripts" / "evaluate.py"
    if not evaluator.exists():
        evaluator = official_dir / "evaluate.py"
    if not evaluator.exists():
        raise FileNotFoundError(f"Cannot find evaluate.py under {official_dir}")
    gold = find_official_file(official_dir, "validation.jsonl")
    pred = Path(args.pred)
    if not pred.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {pred}")
    script_text = evaluator.read_text(encoding="utf-8", errors="replace")
    if re.search(r"--gold\b", script_text) and re.search(r"--pred\b", script_text):
        cmd = [sys.executable, str(evaluator), "--gold", str(gold), "--pred", str(pred)]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode
    print("无法安全确认 evaluator 参数。建议手动运行：")
    print(f"{sys.executable} {evaluator} --gold {gold} --pred {pred}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
