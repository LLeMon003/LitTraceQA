import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_target_model_architecture1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("target_model_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_cache_loader_skips_other_phase_and_reuses_matching_batch(tmp_path: Path):
    module = load_runner()
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"phase": "calibration", "batch_index": 0, "raw": "{}"},
                {"phase": "holdout", "batch_index": 0, "raw": "{\"results\": []}"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert module.load_cached_raw(path, "calibration") == {0: "{}"}
    assert module.load_cached_raw(path, "holdout") == {0: '{"results": []}'}
