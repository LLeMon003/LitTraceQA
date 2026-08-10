from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "scripts"))

import run_ver2_reproduction as runner  # noqa: E402


def option_value(command: list[str], option: str) -> Path:
    return Path(command[command.index(option) + 1]).resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def test_raw_fresh_writes_are_isolated_to_v23_workspace() -> None:
    run_root = (WORKSPACE / "outputs" / "fresh_api_manual_20260718_010651").resolve()
    raw_dir = run_root / "work" / "raw_full_validation"
    args = SimpleNamespace(
        mode="raw-fresh",
        output=run_root / "predictions.jsonl",
        start_stage=None,
        stop_stage=None,
        resume_manifest=None,
        resume_run=run_root,
        verify_hashes=True,
        stall_timeout_sec=1200.0,
        heartbeat_interval_sec=30.0,
    )
    instance = runner.Runner(args)
    command = instance.raw_generation_command(raw_dir, raw_dir / "remaining_input.jsonl")

    assert "--baseline-extra" in command
    assert option_value(command, "--output-dir") == raw_dir.resolve()
    for option in ["--pdf-dir", "--pdf-text-dir", "--pdf-image-dir", "--pdf-vlm-cache-dir", "--pdf-manifest"]:
        target = option_value(command, option)
        assert is_relative_to(target, (run_root / "cache").resolve())
        assert not is_relative_to(target, runner.VER2.resolve())
    assert is_relative_to(option_value(command, "--object-index"), runner.VER2.resolve())


if __name__ == "__main__":
    test_raw_fresh_writes_are_isolated_to_v23_workspace()
    print("phase7_path_isolation_test_passed")
