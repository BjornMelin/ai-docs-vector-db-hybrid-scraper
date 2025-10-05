"""Tests for the simplified security configuration model."""

from __future__ import annotations

import pytest

from src.config.security.config import SecurityConfig


def test_security_config_defaults() -> None:
    """Default values align with the security middleware expectations."""

    config = SecurityConfig()

    assert config.enabled is True
    assert config.enable_rate_limiting is True
    assert config.default_rate_limit == 100
    assert config.rate_limit_window == 60
    assert config.redis_url is None
    assert config.redis_password is None
    assert config.x_frame_options == "DENY"
    assert config.x_content_type_options == "nosniff"
    assert config.x_xss_protection == "1; mode=block"
    assert "self" in config.content_security_policy
    assert config.api_key_required is False
    assert config.api_keys == []


@pytest.mark.parametrize(
    "api_keys",
    [[], ["alpha"], ["alpha", "beta"]],
)
def test_api_keys_round_trip(api_keys: list[str]) -> None:
    """API key lists are preserved exactly."""

    config = SecurityConfig(api_key_required=bool(api_keys), api_keys=api_keys)
    assert config.api_key_required is bool(api_keys)
    assert config.api_keys == api_keys


@pytest.mark.parametrize(
    "rate_limit_window",
    [1, 60, 600],
)
def test_rate_limit_window_accepts_positive_values(rate_limit_window: int) -> None:
    """Positive rate limit windows are accepted."""

    config = SecurityConfig(rate_limit_window=rate_limit_window)
    assert config.rate_limit_window == rate_limit_window


def test_rate_limit_window_rejects_non_positive_values() -> None:
    """Non-positive windows raise validation errors."""

    with pytest.raises(ValueError):
        SecurityConfig(rate_limit_window=0)
    with pytest.raises(ValueError):
        SecurityConfig(rate_limit_window=-10)


def test_default_rate_limit_requires_positive_integer() -> None:
    """Default rate limit must be a positive integer."""

    with pytest.raises(ValueError):
        SecurityConfig(default_rate_limit=0)
    with pytest.raises(ValueError):
        SecurityConfig(default_rate_limit=-5)
