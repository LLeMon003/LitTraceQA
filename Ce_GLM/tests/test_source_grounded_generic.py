import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_source_grounded_generic.py"


def load_module():
    spec = importlib.util.spec_from_file_location("source_grounded_generic", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_option_gold_key_guard_is_recursive_and_fail_closed():
    module = load_module()

    assert module.contains_gold_key({"multiple_choice": {"options": {"A": "opaque"}}}) is False
    assert module.contains_gold_key({"nested": [{"GOLD": "forbidden"}]}) is True


def test_mc_letter_reads_prediction_field_without_touching_option_inputs():
    module = load_module()

    assert module.mc_letter({"answer": {"multiple_choice": {"gold": "B"}}}) == "B"
    assert module.mc_letter({"answer": {}}) is None
