from pathlib import Path

import pytest

from src.credential_resolver import CredentialUnavailable, resolve_credential, resolve_provider_config


def write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"SILICONFLOW_API_KEY='{value}'\n", encoding="utf-8")


def test_inherited_environment_wins_and_status_is_secret_safe(tmp_path: Path) -> None:
    configured = tmp_path / "configured.env"
    write_secret(configured, "file-secret-value")
    result = resolve_credential(environ={"SILICONFLOW_TOKEN": "env-secret-value", "LITTRACEQA_SECRETS_FILE": str(configured)}, home=tmp_path)
    assert result.source == "process_environment"
    assert result.variable_name == "SILICONFLOW_TOKEN"
    assert "env-secret-value" not in repr(result)
    assert "env-secret-value" not in str(result.status())


def test_configured_file_precedes_default_file(tmp_path: Path) -> None:
    configured = tmp_path / "configured.env"
    write_secret(configured, "configured-value")
    write_secret(tmp_path / ".littraceqa" / "secrets.env", "default-value")
    result = resolve_credential(environ={"LITTRACEQA_SECRETS_FILE": str(configured)}, home=tmp_path)
    assert result.source == "configured_secret_file"
    assert result.value == "configured-value"


def test_default_user_local_file_is_used(tmp_path: Path) -> None:
    write_secret(tmp_path / ".littraceqa" / "secrets.env", "default-value")
    result = resolve_credential(environ={}, home=tmp_path)
    assert result.source == "default_user_secret_file"
    assert result.variable_name == "SILICONFLOW_API_KEY"


def test_missing_file_and_missing_key_are_non_secret_diagnostics(tmp_path: Path) -> None:
    with pytest.raises(CredentialUnavailable) as missing_file:
        resolve_credential(environ={"LITTRACEQA_SECRETS_FILE": str(tmp_path / "missing.env")}, home=tmp_path)
    assert "missing.env" not in str(missing_file.value)
    empty = tmp_path / "empty.env"
    empty.write_text("SILICONFLOW_BASE_URL=https://example.invalid\n", encoding="utf-8")
    with pytest.raises(CredentialUnavailable) as missing_key:
        resolve_credential(environ={"LITTRACEQA_SECRETS_FILE": str(empty)}, home=tmp_path)
    assert "SILICONFLOW_BASE_URL" not in str(missing_key.value)


def test_provider_config_does_not_require_project_dotenv(tmp_path: Path) -> None:
    write_secret(tmp_path / ".littraceqa" / "secrets.env", "default-value")
    config = resolve_provider_config(environ={}, home=tmp_path)
    assert config.endpoint == "https://api.siliconflow.cn/v1"
    assert config.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert "default-value" not in str(config.status())
