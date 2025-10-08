"""Tests for the config validation helper."""

from __future__ import annotations

import pytest

from src.config import validate_settings_payload
from src.config.models import Environment


def test_validate_config_payload_success():
    ok, errors, settings = validate_settings_payload({"app_name": "Test App"})

    assert ok is True
    assert errors == []
    assert settings is not None
    assert settings.app_name == "Test App"


def test_validate_config_payload_failure_when_required_keys_missing():
    ok, errors, settings = validate_settings_payload(
        {
            "embedding_provider": "openai",
            "openai": {"api_key": ""},
        }
    )

    assert ok is False
    assert settings is None
    assert any("OpenAI API key" in message for message in errors)


@pytest.mark.parametrize(
    "base_env",
    [Environment.TESTING, Environment.DEVELOPMENT],
)
def test_validate_config_payload_respects_base_overrides(base_env: Environment):
    ok, errors, settings = validate_settings_payload({}, base={"environment": base_env})

    assert ok is True
    assert errors == []
    assert settings is not None
    assert settings.environment is base_env
