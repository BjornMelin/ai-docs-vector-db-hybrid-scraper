"""Executable runtime configuration contract tests."""

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from yaml import safe_load

from src.config.template_assets import load_builtin_template_assets


PROJECT_ROOT = Path(__file__).parents[3]


def _app_service() -> dict[str, Any]:
    compose = safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], compose["services"]["app"])


def test_app_loads_optional_local_environment_file() -> None:
    """Compose should forward canonical `.env` settings into the application."""
    assert _app_service()["env_file"] == [{"path": ".env", "required": False}]


def test_compose_overrides_only_container_network_paths() -> None:
    """Explicit values should only replace host-specific endpoints and paths."""
    environment = cast(list[str], _app_service()["environment"])
    assert {entry.partition("=")[0] for entry in environment} == {
        "AI_DOCS_QDRANT__URL",
        "AI_DOCS_CACHE__DRAGONFLY_URL",
        "AI_DOCS_FASTEMBED__CACHE_DIR",
    }


def test_claude_config_uses_stdio_and_canonical_settings() -> None:
    """Claude should launch stdio with only supported application variables."""
    payload = json.loads(
        (PROJECT_ROOT / "config/claude-mcp-config.example.json").read_text(
            encoding="utf-8"
        )
    )
    server = payload["mcpServers"]["ai-docs-vector-db-unified"]
    environment = cast(dict[str, str], server["env"])

    assert environment["FASTMCP_TRANSPORT"] == "stdio"
    assert {key for key in environment if key.startswith("AI_DOCS_")} == {
        "AI_DOCS_EMBEDDING_PROVIDER",
        "AI_DOCS_CRAWL_PROVIDER",
        "AI_DOCS_QDRANT__URL",
    }


def test_production_profile_defers_provider_credentials() -> None:
    """Production templates should validate without embedded fake credentials."""
    _, profiles = load_builtin_template_assets()
    overrides = cast(dict[str, Any], profiles["production"]["overrides"])

    assert overrides["cache"]["dragonfly_url"] == "redis://dragonfly:6379"
    assert "api_key" not in overrides["openai"]
    assert "api_key" not in overrides["browser"]["firecrawl"]
    assert "embedding_provider" not in overrides
    assert "crawl_provider" not in overrides
    assert overrides["observability"]["otlp_endpoint"] == "http://otel-collector:4317"


def test_default_runtime_excludes_optional_provider_and_transformer_stacks() -> None:
    """The default image should exclude agentic browser and reranking extras."""
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = cast(list[str], metadata["project"]["dependencies"])
    reranking = cast(
        list[str], metadata["project"]["optional-dependencies"]["reranking"]
    )
    agentic_browser = cast(
        list[str], metadata["project"]["optional-dependencies"]["agentic-browser"]
    )
    dev_dependencies = cast(list[str], metadata["dependency-groups"]["dev"])

    assert any(dependency.startswith("crawl4ai>=") for dependency in dependencies)
    assert not any(dependency.startswith("crawl4ai[") for dependency in dependencies)
    assert not any(
        dependency.lower().startswith("flagembedding") for dependency in dependencies
    )
    assert not any(dependency.startswith("scikit-learn") for dependency in dependencies)
    assert not any(dependency.startswith("browser-use") for dependency in dependencies)
    assert any(dependency.startswith("FlagEmbedding>=") for dependency in reranking)
    assert any(dependency.startswith("browser-use>=") for dependency in agentic_browser)
    assert any(
        dependency.startswith("browser-use>=") for dependency in dev_dependencies
    )
