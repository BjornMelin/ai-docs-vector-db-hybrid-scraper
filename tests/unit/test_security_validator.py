"""Tests for the streamlined ML security validator."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.security.ml_security import MLSecurityValidator, SecurityCheckResult


@dataclass
class _SecuritySettings:
    enable_ml_input_validation: bool = True
    max_ml_input_size: int = 1024
    suspicious_patterns: list[str] = field(
        default_factory=lambda: ["<script", "DROP TABLE"]
    )
    enable_dependency_scanning: bool = True


@dataclass
class _Config:
    security: _SecuritySettings = field(default_factory=_SecuritySettings)


@pytest.fixture()
def validator(monkeypatch: pytest.MonkeyPatch) -> MLSecurityValidator:
    """Return a validator instance backed by an isolated config object."""
    config = _Config()
    monkeypatch.setattr("src.security.ml_security.get_config", lambda: config)
    return MLSecurityValidator()


def test_validate_input_accepts_small_payload(validator: MLSecurityValidator) -> None:
    """Valid payloads should pass when validation is enabled."""
    result = validator.validate_input({"text": "safe"}, {"text": str})
    assert isinstance(result, SecurityCheckResult)
    assert result.passed is True


def test_validate_input_rejects_suspicious_pattern(
    validator: MLSecurityValidator,
) -> None:
    """Suspicious patterns should trigger a failure result."""
    payload = {"text": "<script>alert('xss')</script>"}
    result = validator.validate_input(payload)
    assert result.passed is False
    assert "Suspicious pattern" in result.message


def test_dependency_check_skips_when_tool_missing(
    validator: MLSecurityValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependency scanning should skip gracefully when pip-audit is absent."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = validator.check_dependencies()
    assert result.passed is True
    assert "skipped" in result.message.lower()


def test_container_check_validates_image_name(
    validator: MLSecurityValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Container scanning should reject invalid image identifiers."""
    monkeypatch.setattr(
        "shutil.which", lambda tool: "/usr/bin/trivy" if tool == "trivy" else None
    )
    result = validator.check_container("invalid")
    assert result.passed is False
    assert "Invalid container image name" in result.message
