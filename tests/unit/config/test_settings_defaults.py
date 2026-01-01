"""Tests for configuration defaults and precedence resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from src.config.loader import Settings
from src.config.models import EmbeddingProvider, Environment


# pylint: disable=no-member  # Dynamic Pydantic models expose attributes at runtime during tests.


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure cached settings instances are cleared between tests."""
    monkeypatch.delenv("AI_DOCS_CACHE__TTL_SEARCH_RESULTS", raising=False)
    monkeypatch.delenv("AI_DOCS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("AI_DOCS_OPENAI__API_KEY", raising=False)
    monkeypatch.setattr("src.config.loader._ACTIVE_SETTINGS", None)


def _write_env_file(directory: Path, payload: str | None) -> None:
    """Create a .env file with the provided payload when requested."""
    if payload is None:
        return
    (directory / ".env").write_text(payload, encoding="utf-8")


@dataclass(frozen=True)
class TtlCase:
    """Test case definition for TTL precedence scenarios."""

    env_value: str | None
    env_file_value: str | None
    override_value: int | None
    expected: int


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(TtlCase("7200", None, None, 7200), id="env-only"),
        pytest.param(
            TtlCase(None, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=8100\n", None, 8100),
            id="env-file",
        ),
        pytest.param(TtlCase(None, None, 9000, 9000), id="override-only"),
        pytest.param(TtlCase(None, None, None, 3600), id="defaults"),
        pytest.param(
            TtlCase("600", "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n", 700, 700),
            id="override-wins",
        ),
        pytest.param(
            TtlCase("600", "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n", None, 600),
            id="env-beats-file",
        ),
        pytest.param(
            TtlCase(None, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n", 700, 700),
            id="override-beats-file",
        ),
    ],
)
def test_cache_ttl_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: TtlCase
) -> None:
    """Cache TTL values should resolve according to source precedence."""
    monkeypatch.chdir(tmp_path)
    if case.env_value is not None:
        monkeypatch.setenv("AI_DOCS_CACHE__TTL_SEARCH_RESULTS", case.env_value)
    _write_env_file(tmp_path, case.env_file_value)
    overrides: dict[str, Any] = {}
    if case.override_value is not None:
        overrides["cache"] = cast(Any, {"ttl_search_results": case.override_value})

    settings = Settings(**overrides)

    assert settings.cache.ttl_search_results == case.expected


def test_cache_ttl_negative_values_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative TTL values should fail validation with a ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(cache=cast(Any, {"ttl_search_results": -1}))


def test_chunk_overlap_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chunk overlap exceeding the chunk size should raise ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            environment=Environment.TESTING,
            chunking=cast(Any, {"chunk_size": 100, "chunk_overlap": 200}),
        )

    assert "chunk_overlap" in str(exc_info.value)


def test_openai_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI embedding provider should require an API key outside test mode."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        Settings(embedding_provider=EmbeddingProvider.OPENAI)

    assert "OpenAI API key required" in str(exc_info.value)


def test_testing_environment_skips_provider_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test environment should skip API key validation for convenience."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        environment=Environment.TESTING, embedding_provider=EmbeddingProvider.OPENAI
    )

    assert settings.embedding_provider is EmbeddingProvider.OPENAI


def test_env_file_invalid_value_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed numeric values in .env should raise ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    _write_env_file(tmp_path, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=not-a-number\n")

    with pytest.raises(ValidationError):
        Settings()


def test_programmatic_overrides_apply_when_no_other_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Programmatic overrides should update nested configuration values."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(cache=cast(Any, {"ttl_search_results": 4321}))

    assert settings.cache.ttl_search_results == 4321
