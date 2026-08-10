from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
PYTHON_EXE = Path(os.environ.get("PHASE7_PYTHON_EXE") or sys.executable)
ENV_FILE = ROOT / "littraceqa_baseline_uq_experiments" / ".env"
FRESH_DIR = WORKSPACE / "outputs" / "fresh_api"
PREDICTIONS = FRESH_DIR / "predictions.jsonl"

PRELAUNCH_FILES = {
    "API_BLOCKED.json",
    "api_usage.json",
    "raw_fresh_stdout.log",
    "raw_fresh_stderr.log",
}
RUN_CREATED_NAMES = {
    "predictions.jsonl",
    "run_manifest.json",
    "freeze_manifest.json",
    "stage_audit.jsonl",
    "mc_decisions.jsonl",
    "environment_report.json",
    "generation_log_summary.json",
    "stage_manifests",
    "work",
    "logs",
}


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def normalized_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    selected: dict[str, tuple[str, str]] = {}
    for key, value in os.environ.items():
        canonical = "Path" if key.lower() == "path" else key
        selected[canonical.lower()] = (canonical, value)
    for key, value in read_dotenv(ENV_FILE).items():
        canonical = "Path" if key.lower() == "path" else key
        selected[canonical.lower()] = (canonical, value)
    for key, value in (extra or {}).items():
        canonical = "Path" if key.lower() == "path" else key
        selected[canonical.lower()] = (canonical, value)
    return {key: value for key, value in selected.values()}


def duplicate_keys_case_insensitive(keys: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for key in keys:
        lowered = key.lower()
        if lowered in seen and seen[lowered] != key:
            duplicates.append(f"{seen[lowered]}|{key}")
        else:
            seen[lowered] = key
    return duplicates


def clean_prelaunch_artifacts() -> list[str]:
    FRESH_DIR.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in FRESH_DIR.iterdir()}
    protected = sorted(existing & RUN_CREATED_NAMES)
    if protected:
        raise RuntimeError(f"Refusing to clean fresh_api because run artifacts exist: {protected}")
    unexpected = sorted(existing - PRELAUNCH_FILES)
    if unexpected:
        raise RuntimeError(f"Refusing to clean unexpected fresh_api artifacts: {unexpected}")
    removed: list[str] = []
    for name in sorted(existing & PRELAUNCH_FILES):
        target = FRESH_DIR / name
        if target.is_file():
            target.unlink()
            removed.append(str(target))
    return removed


def reproduction_command() -> list[str]:
    return [
        str(PYTHON_EXE),
        "scripts\\run_ver2_reproduction.py",
        "--mode",
        "raw-fresh",
        "--output",
        "outputs\\fresh_api\\predictions.jsonl",
        "--verify-hashes",
    ]


def dry_run_child_code() -> str:
    return (
        "import json, os, pathlib, sys; "
        "payload={"
        "'executable': sys.executable,"
        "'cwd': str(pathlib.Path.cwd()),"
        "'has_api_key': bool(os.environ.get('SILICONFLOW_API_KEY')),"
        "'api_key_value_printed': False,"
        "'argv': sys.argv[1:]"
        "}; "
        "print(json.dumps(payload, sort_keys=True))"
    )


def write_environment_report(path: Path, command: list[str], env: dict[str, str], dry_run: bool, removed: list[str]) -> None:
    report = {
        "schema_version": 1,
        "launcher": str(Path(__file__).resolve()),
        "python_executable": str(PYTHON_EXE),
        "working_directory": str(WORKSPACE),
        "command": command,
        "dry_run": dry_run,
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.is_file(),
        "siliconflow_api_key_present": bool(env.get("SILICONFLOW_API_KEY")),
        "secrets_printed": False,
        "case_insensitive_duplicate_environment_keys": duplicate_keys_case_insensitive(env.keys()),
        "canonical_path_key_present": "Path" in env,
        "removed_prelaunch_artifacts": removed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_child(command: list[str], env: dict[str, str]) -> int:
    completed = subprocess.run(command, cwd=WORKSPACE, env=env)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 launcher without PowerShell Start-Process.")
    parser.add_argument("--dry-run", action="store_true", help="Start a child Python process without network/API work.")
    parser.add_argument("--clean-prelaunch-artifacts", action="store_true")
    args = parser.parse_args()

    env = normalized_environment()
    duplicates = duplicate_keys_case_insensitive(env.keys())
    if duplicates:
        raise RuntimeError(f"Duplicate environment keys after normalization: {duplicates}")

    removed = clean_prelaunch_artifacts() if args.clean_prelaunch_artifacts and not args.dry_run else []
    if args.dry_run:
        command = [str(PYTHON_EXE), "-c", dry_run_child_code(), "--", "phase7-dry-run"]
    else:
        command = reproduction_command()
    write_environment_report(FRESH_DIR / "environment_report.json", command, env, args.dry_run, removed)
    return run_child(command, env)


if __name__ == "__main__":
    raise SystemExit(main())
