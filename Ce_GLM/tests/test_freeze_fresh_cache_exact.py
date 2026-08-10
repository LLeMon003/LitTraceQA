import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_fresh_cache_exact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fresh_cache_exact", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_prediction(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for number in range(55):
            handle.write(json.dumps({"query_id": f"synthetic_{number:03d}", "answer": {}}) + "\n")


def test_freeze_then_replay_is_byte_exact_and_non_answering(tmp_path: Path):
    module = load_module()
    source = tmp_path / "fresh.jsonl"
    cache = tmp_path / "fresh-cache"
    replay = tmp_path / "replayed.jsonl"
    write_prediction(source)

    frozen = module.freeze(source, cache)
    replayed = module.replay(cache, replay)

    assert frozen["classification"] == "FRESH_FROZEN_CACHE_EXACT_REPLAY"
    assert replayed["prediction_sha256"] == module.sha256(source) == module.sha256(replay)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider_calls"] == 0
    assert manifest["gold_used"] is False
    assert manifest["evaluator_invoked"] is False


def test_replay_rejects_tampered_cache_and_freeze_refuses_overwrite(tmp_path: Path):
    module = load_module()
    source = tmp_path / "fresh.jsonl"
    cache = tmp_path / "fresh-cache"
    write_prediction(source)
    module.freeze(source, cache)
    with pytest.raises(FileExistsError):
        module.freeze(source, cache)
    with (cache / "prediction.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError):
        module.replay(cache, tmp_path / "replayed.jsonl")
