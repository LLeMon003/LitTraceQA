from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(os.environ.get("PHASE7_PYTHON_EXE") or sys.executable)


def test_phase7_launcher_dry_run_starts_child_without_secret_output() -> None:
    command = [
        str(PYTHON_EXE),
        str(WORKSPACE / "scripts" / "phase7_launcher.py"),
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    report = json.loads((WORKSPACE / "outputs" / "fresh_api" / "environment_report.json").read_text(encoding="utf-8"))

    assert payload["executable"].lower() == str(PYTHON_EXE).lower()
    assert payload["cwd"].lower() == str(WORKSPACE).lower()
    assert payload["has_api_key"] is True
    assert payload["api_key_value_printed"] is False
    assert report["case_insensitive_duplicate_environment_keys"] == []
    assert report["canonical_path_key_present"] is True
    assert report["siliconflow_api_key_present"] is True
    assert report["secrets_printed"] is False
    assert "SILICONFLOW_API_KEY" not in completed.stdout
    assert "SILICONFLOW_API_KEY" not in completed.stderr


if __name__ == "__main__":
    test_phase7_launcher_dry_run_starts_child_without_secret_output()
    print("phase7_launcher_test_passed")
