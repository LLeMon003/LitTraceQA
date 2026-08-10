import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ver2_reproduction.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("fresh_reproduction_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_provider():
    credential = SimpleNamespace(value="test-only-secret", variable_name="SILICONFLOW_API_KEY")
    return SimpleNamespace(
        credential=credential,
        endpoint="https://example.invalid/v1",
        model="example-model",
        status=lambda: {
            "credential_present": True,
            "variable_name": "SILICONFLOW_API_KEY",
            "source": "test",
            "endpoint": "https://example.invalid/v1",
            "model": "example-model",
        },
    )


def test_raw_fresh_preflight_uses_explicit_external_roots_without_secret_output(tmp_path, monkeypatch):
    module = load_runner_module()
    release_root = tmp_path / "release"
    source_root = tmp_path / "source"
    module.configure_external_roots(argparse.Namespace(release_root=release_root, source_root=source_root))
    for _, path in module.raw_fresh_requirements():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(module, "resolve_provider_config", fake_provider)
    assets = []
    for role, path in module.raw_fresh_requirements():
        root_name = "release_root" if path.is_relative_to(release_root) else "source_root"
        root = release_root if root_name == "release_root" else source_root
        assets.append({"role": role, "root": root_name, "relative_path": str(path.relative_to(root)), "sha256": module.sha256(path)})
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")

    report = module.raw_fresh_preflight(manifest)

    assert report["status"] == "ready"
    assert report["cache_boundary_used"] is False
    assert report["asset_manifest"]["verified_asset_count"] == 9
    assert "test-only-secret" not in str(report)


def test_raw_fresh_preflight_reports_explicit_model_without_secret_output(tmp_path, monkeypatch):
    module = load_runner_module()
    release_root = tmp_path / "release"
    source_root = tmp_path / "source"
    module.configure_external_roots(argparse.Namespace(release_root=release_root, source_root=source_root))
    assets = []
    for role, path in module.raw_fresh_requirements():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
        root_name = "release_root" if path.is_relative_to(release_root) else "source_root"
        root = release_root if root_name == "release_root" else source_root
        assets.append({"role": role, "root": root_name, "relative_path": str(path.relative_to(root)), "sha256": module.sha256(path)})
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")
    monkeypatch.setattr(module, "resolve_provider_config", fake_provider)

    report = module.raw_fresh_preflight(manifest, text_model="deepseek-ai/DeepSeek-V3.2")

    assert report["provider"]["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert report["provider"]["model_source"] == "explicit_child_override"
    assert "test-only-secret" not in str(report)


def test_raw_fresh_child_uses_central_resolver_without_overwriting_existing_key(monkeypatch):
    module = load_runner_module()
    runner = module.Runner.__new__(module.Runner)
    runner.child_environment = {"SILICONFLOW_API_KEY": "inherited-value"}
    monkeypatch.setattr(module, "resolve_provider_config", fake_provider)

    runner.configure_raw_fresh_credentials()

    assert runner.child_environment["SILICONFLOW_API_KEY"] == "inherited-value"
    assert runner.child_environment["SILICONFLOW_BASE_URL"] == "https://example.invalid/v1"
    assert runner.child_environment["SILICONFLOW_TEXT_MODEL"] == "example-model"
    assert "test-only-secret" not in str(runner.provider_status)


def test_raw_fresh_explicit_model_overrides_default_only_in_child_environment(monkeypatch):
    module = load_runner_module()
    runner = module.Runner.__new__(module.Runner)
    runner.args = argparse.Namespace(text_model="deepseek-ai/DeepSeek-V3.2")
    runner.child_environment = {}
    monkeypatch.setattr(module, "resolve_provider_config", fake_provider)

    runner.configure_raw_fresh_credentials()

    assert runner.child_environment["SILICONFLOW_TEXT_MODEL"] == "deepseek-ai/DeepSeek-V3.2"
    assert runner.child_environment["SILICONFLOW_API_KEY"] == "test-only-secret"
    assert "test-only-secret" not in str(runner.provider_status)
    assert runner.provider_status["model"] == "deepseek-ai/DeepSeek-V3.2"
    assert runner.provider_status["model_source"] == "explicit_child_override"


def test_raw_generation_heartbeats_raw_outputs_before_final_prediction(tmp_path, monkeypatch):
    module = load_runner_module()
    release_root = tmp_path / "release"
    monkeypatch.setattr(module, "VER2", release_root)
    runner = module.Runner.__new__(module.Runner)
    runner.args = argparse.Namespace(resume_run=None)
    runner.work_dir = tmp_path / "work"
    runner.raw_generation_command = lambda raw_dir, input_path: ["fixture"]
    captured = {}

    def command_stage(*args, **kwargs):
        captured["heartbeat_files"] = kwargs["heartbeat_files"]
        return tmp_path / "result.jsonl"

    runner.command_stage = command_stage

    result = runner.raw_generation_stage()

    assert result == tmp_path / "result.jsonl"
    assert captured["heartbeat_files"] == [
        runner.work_dir / "raw_full_validation" / "raw_generation.jsonl",
        runner.work_dir / "raw_full_validation" / "run_log.jsonl",
        runner.work_dir / "raw_full_validation" / "stage_00_generated.jsonl",
    ]
