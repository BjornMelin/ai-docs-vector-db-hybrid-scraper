"""Shared fixtures and helpers for MCP unit tests."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from src.config.models import CrawlProvider, EmbeddingProvider


@pytest.fixture(name="build_unified_mcp_config")
def fixture_build_unified_mcp_config() -> Callable[..., SimpleNamespace]:
    """Provide a builder for unified MCP configuration namespaces.

    Returns:
        Callable[..., SimpleNamespace]: Factory that mirrors the production
        configuration namespace with overridable keys for targeted scenarios.
    """

    def _build_config(
        *,
        openai_key: str | None = "sk-123",
        firecrawl_key: str | None = "fc-123",
        qdrant_url: str | None = "http://localhost:6333",
        providers: list[str] | None = None,
        crawling_providers: list[str] | None = None,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        crawl_provider: CrawlProvider = CrawlProvider.FIRECRAWL,
    ) -> SimpleNamespace:
        providers = providers or ["openai"]
        crawling_providers = crawling_providers or ["firecrawl"]
        return SimpleNamespace(
            get_active_providers=lambda: providers,
            openai=SimpleNamespace(api_key=openai_key),
            embedding=SimpleNamespace(provider=embedding_provider),
            crawling=SimpleNamespace(providers=crawling_providers),
            browser=SimpleNamespace(firecrawl=SimpleNamespace(api_key=firecrawl_key)),
            qdrant=SimpleNamespace(url=qdrant_url),
            crawl_provider=crawl_provider,
            cache=SimpleNamespace(
                enable_dragonfly_cache=False,
                dragonfly_url=None,
            ),
            monitoring=SimpleNamespace(
                enabled=False,
                include_system_metrics=False,
                system_metrics_interval=60,
            ),
        )

    return _build_config


@pytest.fixture(name="tool_module_factory")
def fixture_tool_module_factory() -> Callable[..., SimpleNamespace]:
    """Provide a factory that builds namespace proxies with registration hooks.

    Returns:
        Callable[..., SimpleNamespace]: Callable that creates a namespace exposing
        the desired registrar attribute while recording invocations in a shared
        list or raising provided exceptions.
    """

    def _factory(
        name: str,
        attr: str,
        registered: list[str],
        *,
        raises: Exception | None = None,
    ) -> SimpleNamespace:
        def register(mcp: Any, **kwargs: Any) -> None:
            """Record the module registration or simulate a failure path."""
            del mcp, kwargs
            if raises is not None:
                raise raises
            registered.append(name)

        return SimpleNamespace(**{attr: register})

    return _factory


_MODULE_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("search", "register_tools", False),
    ("documents", "register_tools", False),
    ("embeddings", "register_tools", False),
    ("lightweight_scrape", "register_tools", False),
    ("collection_management", "register_tools", False),
    ("projects", "register_tools", False),
    ("search_tools", "register_tools", False),
    ("query_processing_tools", "register_tools", False),
    ("payload_indexing", "register_tools", False),
    ("analytics", "register_tools", False),
    ("cache", "register_tools", False),
    ("utilities", "register_tools", False),
    ("content_intelligence", "register_tools", False),
    ("agentic_rag", "register_tools", True),
)


@pytest.fixture(name="build_tool_modules")
def fixture_build_tool_modules(
    tool_module_factory: Callable[..., SimpleNamespace],
) -> Callable[[list[str], dict[str, Any] | None], SimpleNamespace]:
    """Provide a builder for the tool registry namespace used across suites.

    Args:
        tool_module_factory: Callable fixture producing individual module proxies.

    Returns:
        Callable[[list[str], dict[str, Any] | None], SimpleNamespace]: Factory that
        assembles the tool namespace, allowing targeted overrides for failure cases
        while preserving default optional-module behaviour.
    """

    def _builder(
        registered: list[str],
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> SimpleNamespace:
        overrides = overrides or {}
        modules: dict[str, Any] = {}
        for name, attr, optional in _MODULE_SPECS:
            override = overrides.get(name, {})
            registrar_attr = override.get("attr", attr)
            raises: Exception | None
            if "raises" in override:
                raises = override["raises"]
            elif optional:
                raises = ImportError("optional dependency not available")
            else:
                raises = None
            modules[name] = tool_module_factory(
                name=name,
                attr=registrar_attr,
                registered=registered,
                raises=raises,
            )
        return SimpleNamespace(**modules)

    return _builder
