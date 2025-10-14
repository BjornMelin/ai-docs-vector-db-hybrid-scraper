"""Hypothesis strategies for configuration schema testing.

This module provides custom Hypothesis strategies for generating test data
that matches the structure and constraints of configuration models.
"""

import string
import tempfile
from pathlib import Path
from typing import Any

import hypothesis.strategies as st

from src.config.models import (
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    LogLevel,
)


# Basic type strategies
@st.composite
def api_keys(draw, prefix: str, min_length: int = 20, max_length: int = 64) -> str:
    """Generate valid API keys with proper prefixes."""
    suffix = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=min_length - len(prefix),
            max_size=max_length - len(prefix),
        )
    )
    return f"{prefix}{suffix}"


@st.composite
def openai_api_keys(draw) -> str:
    """Generate valid OpenAI API keys."""
    return draw(api_keys(prefix="sk-", min_length=20))


@st.composite
def firecrawl_api_keys(draw) -> str:
    """Generate valid Firecrawl API keys."""
    return draw(api_keys(prefix="fc-", min_length=20))


@st.composite
def redis_urls(draw) -> str:
    """Generate valid Redis URLs."""
    host = draw(
        st.sampled_from(
            ["localhost", "127.0.0.1", "redis.example.com", "cache.test.org"]
        )
    )
    port = draw(st.integers(min_value=1024, max_value=65535))
    password = draw(
        st.one_of(
            st.none(),
            st.text(
                alphabet=string.ascii_letters + string.digits + string.punctuation,
                min_size=8,
                max_size=32,
            ),
        )
    )

    if password:
        return f"redis://:{password}@{host}:{port}"
    return f"redis://{host}:{port}"


@st.composite
def http_urls(draw, schemes: list[str] | None = None) -> str:
    """Generate valid HTTP URLs."""
    if schemes is None:
        schemes = ["http", "https"]

    scheme = draw(st.sampled_from(schemes))
    # Use predefined valid hostnames to avoid invalid international domain names
    host = draw(
        st.sampled_from(
            [
                "localhost",
                "127.0.0.1",
                "example.com",
                "test.org",
                "api.example.com",
                "service.test.com",
            ]
        )
    )
    port = draw(st.one_of(st.none(), st.integers(min_value=1024, max_value=65535)))
    path = draw(
        st.one_of(
            st.just(""),
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20
            ).map(lambda x: f"/{x}"),
        )
    )

    url = f"{scheme}://{host}"
    if port:
        url += f":{port}"
    url += path

    return url


@st.composite
def file_paths(draw, relative: bool = False) -> Path:
    """Generate valid file paths."""
    components = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=3,  # Reduce nesting to avoid deep paths
        )
    )

    if relative:
        return Path(*components)
    # Use temporary directory for absolute paths to avoid permission issues
    temp_dir = Path(tempfile.gettempdir())
    return temp_dir / "test_config" / Path(*components)


@st.composite
def positive_integers(draw, min_value: int = 1, max_value: int = 1000000) -> int:
    """Generate positive integers within reasonable bounds."""
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def positive_floats(draw, min_value: float = 0.1, max_value: float = 10000.0) -> float:
    """Generate positive floats within reasonable bounds."""
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_infinity=False,
            allow_nan=False,
        )
    )


@st.composite
def percentage_floats(draw) -> float:
    """Generate percentage values between 0.0 and 1.0."""
    return draw(
        st.floats(min_value=0.0, max_value=1.0, allow_infinity=False, allow_nan=False)
    )


@st.composite
def temperature_floats(draw) -> float:
    """Generate temperature values for AI models (0.0 to 2.0)."""
    return draw(
        st.floats(min_value=0.0, max_value=2.0, allow_infinity=False, allow_nan=False)
    )


@st.composite
def port_numbers(draw) -> int:
    """Generate valid port numbers."""
    return draw(st.integers(min_value=1, max_value=65535))


@st.composite
def database_urls(draw) -> str:
    """Generate valid database URLs."""
    scheme = draw(
        st.sampled_from(["sqlite+aiosqlite", "postgresql+asyncpg", "mysql+aiomysql"])
    )

    if scheme.startswith("sqlite"):
        # SQLite file path
        path = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
                min_size=5,
                max_size=50,
            )
        )
        return f"{scheme}:///{path}.db"
    # Network database
    host = draw(
        st.one_of(
            st.just("localhost"),
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll",)),
                min_size=3,
                max_size=20,
            ),
        )
    )
    port = draw(st.integers(min_value=1024, max_value=65535))
    database = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=3,
            max_size=20,
        )
    )
    username = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=3,
            max_size=20,
        )
    )
    password = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + string.punctuation,
            min_size=8,
            max_size=32,
        )
    )

    return f"{scheme}://{username}:{password}@{host}:{port}/{database}"


@st.composite
def chunk_configurations(draw) -> dict[str, Any]:
    """Generate valid chunking configurations with proper constraints."""
    chunk_size = draw(st.integers(min_value=100, max_value=5000))
    chunk_overlap = draw(st.integers(min_value=0, max_value=chunk_size - 1))
    token_chunk_size = draw(st.integers(min_value=100, max_value=2000))
    token_chunk_overlap = draw(
        st.integers(min_value=0, max_value=max(token_chunk_size - 1, 0))
    )
    json_max_chars = draw(st.integers(min_value=max(chunk_size, 1000), max_value=50000))

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "strategy": draw(st.sampled_from(list(ChunkingStrategy))),
        "token_chunk_size": token_chunk_size,
        "token_chunk_overlap": token_chunk_overlap,
        "token_model": draw(
            st.sampled_from(["cl100k_base", "o200k_base", "gpt-4o-mini"])
        ),
        "json_max_chars": json_max_chars,
        "enable_semantic_html_segmentation": draw(st.booleans()),
        "normalize_html_text": draw(st.booleans()),
    }


@st.composite
def openai_configurations(draw) -> dict[str, Any]:
    """Generate valid OpenAI configurations."""
    return {
        "api_key": draw(st.one_of(st.none(), openai_api_keys())),
        "model": draw(
            st.sampled_from(
                [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ]
            )
        ),
        "dimensions": draw(st.integers(min_value=512, max_value=3072)),
        "batch_size": draw(st.integers(min_value=1, max_value=2048)),
        "max_requests_per_minute": draw(st.integers(min_value=100, max_value=10000)),
        "cost_per_million_tokens": draw(st.floats(min_value=0.001, max_value=1.0)),
    }


@st.composite
def qdrant_configurations(draw) -> dict[str, Any]:
    """Generate valid Qdrant configurations."""
    return {
        "url": draw(http_urls()),
        "api_key": draw(st.one_of(st.none(), st.text(min_size=20, max_size=64))),
        "timeout": draw(positive_floats(min_value=1.0, max_value=300.0)),
        "collection_name": draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
                min_size=1,
                max_size=50,
            )
        ),
        "batch_size": draw(st.integers(min_value=1, max_value=1000)),
        "prefer_grpc": draw(st.booleans()),
        "grpc_port": draw(port_numbers()),
        "use_grpc": draw(st.booleans()),
        "enable_grouping": draw(st.booleans()),
        "group_by_field": draw(
            st.text(
                alphabet=string.ascii_lowercase + string.digits + "_",
                min_size=1,
                max_size=30,
            )
        ),
        "group_size": draw(st.integers(min_value=1, max_value=10)),
        "groups_limit_multiplier": draw(
            st.floats(
                min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
            )
        ),
    }


@st.composite
def cache_configurations(draw) -> dict[str, Any]:
    """Generate valid cache configurations."""
    return {
        "enable_caching": draw(st.booleans()),
        "enable_dragonfly_cache": draw(st.booleans()),
        "dragonfly_url": draw(redis_urls()),
        "dragonfly_password": draw(
            st.one_of(st.none(), st.text(min_size=8, max_size=32))
        ),
        "dragonfly_database": draw(st.integers(min_value=0, max_value=15)),
        "ttl_embeddings": draw(positive_integers(min_value=60, max_value=86400)),
        "ttl_crawl": draw(positive_integers(min_value=60, max_value=86400)),
        "ttl_queries": draw(positive_integers(min_value=60, max_value=86400)),
        "ttl_search_results": draw(positive_integers(min_value=60, max_value=86400)),
        "cache_ttl_seconds": draw(
            st.dictionaries(
                keys=st.sampled_from(
                    ["search_results", "embeddings", "collections", "queries"]
                ),
                values=st.integers(min_value=60, max_value=86400),
                min_size=1,
                max_size=10,
            )
        ),
    }


@st.composite
def circuit_breaker_configurations(draw) -> dict[str, Any]:
    """Generate valid circuit breaker configurations."""
    return {
        "failure_threshold": draw(st.integers(min_value=1, max_value=20)),
        "recovery_timeout": draw(positive_floats(min_value=5.0, max_value=300.0)),
        "service_overrides": draw(
            st.dictionaries(
                keys=st.sampled_from(["openai", "firecrawl", "qdrant", "redis"]),
                values=st.dictionaries(
                    keys=st.sampled_from(["failure_threshold", "recovery_timeout"]),
                    values=st.one_of(
                        st.integers(min_value=1, max_value=20),
                        st.floats(min_value=5.0, max_value=300.0),
                    ),
                    min_size=1,
                    max_size=2,
                ),
                min_size=0,
                max_size=4,
            )
        ),
    }


@st.composite
def rag_configurations(draw) -> dict[str, Any]:
    """Generate valid RAG configurations."""
    return {
        "enable_rag": draw(st.booleans()),
        "model": draw(
            st.sampled_from(
                [
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "gpt-4-turbo",
                    "claude-3-sonnet",
                    "claude-3-haiku",
                ]
            )
        ),
        "temperature": draw(temperature_floats()),
        "max_tokens": draw(st.integers(min_value=100, max_value=4000)),
        "timeout_seconds": draw(positive_floats(min_value=5.0, max_value=120.0)),
        "max_context_length": draw(st.integers(min_value=1000, max_value=8000)),
        "max_results_for_context": draw(st.integers(min_value=1, max_value=20)),
        "min_confidence_threshold": draw(percentage_floats()),
        "include_sources": draw(st.booleans()),
        "include_confidence_score": draw(st.booleans()),
        "enable_answer_metrics": draw(st.booleans()),
        "enable_caching": draw(st.booleans()),
        "cache_ttl_seconds": draw(positive_integers(min_value=300, max_value=86400)),
        "parallel_processing": draw(st.booleans()),
    }


@st.composite
def documentation_sites(draw) -> dict[str, Any]:
    """Generate valid documentation site configurations."""
    return {
        "name": draw(st.text(min_size=1, max_size=100)),
        "url": draw(http_urls(schemes=["https"])),
        "max_pages": draw(positive_integers(min_value=1, max_value=1000)),
        "max_depth": draw(positive_integers(min_value=1, max_value=10)),
        "priority": draw(st.sampled_from(["low", "medium", "high"])),
    }


@st.composite
def complete_configurations(draw) -> dict[str, Any]:
    """Generate complete configuration objects with all components."""
    # First generate providers to coordinate API keys
    embedding_provider = draw(st.sampled_from(list(EmbeddingProvider)))
    crawl_provider = draw(st.sampled_from(list(CrawlProvider)))

    config = {
        "environment": draw(st.sampled_from(list(Environment))),
        "debug": draw(st.booleans()),
        "log_level": draw(st.sampled_from(list(LogLevel))),
        "app_name": draw(st.text(min_size=1, max_size=100)),
        "version": draw(st.text(min_size=1, max_size=20)),
        "embedding_provider": embedding_provider,
        "crawl_provider": crawl_provider,
        "cache": draw(cache_configurations()),
        "qdrant": draw(qdrant_configurations()),
        "openai": draw(openai_configurations()),
        "chunking": draw(chunk_configurations()),
        "rag": draw(rag_configurations()),
        "documentation_sites": draw(
            st.lists(documentation_sites(), min_size=0, max_size=5)
        ),
        "data_dir": draw(file_paths()),
        "cache_dir": draw(file_paths()),
        "logs_dir": draw(file_paths()),
    }

    browser_config: dict[str, Any] = {
        "lightweight": {},
        "crawl4ai": {},
        "playwright": {},
        "browser_use": {},
        "firecrawl": {},
        "router": {},
    }

    if crawl_provider == CrawlProvider.FIRECRAWL:
        browser_config["firecrawl"] = {
            "api_key": draw(firecrawl_api_keys()),
            "api_url": draw(http_urls()),
            "default_formats": draw(
                st.lists(
                    st.sampled_from(["markdown", "html", "text"]),
                    min_size=1,
                    max_size=3,
                    unique=True,
                )
            ),
            "timeout_seconds": draw(st.integers(min_value=30, max_value=600)),
        }

    config["browser"] = browser_config

    if embedding_provider == EmbeddingProvider.OPENAI:
        config.setdefault("openai", {})
        config["openai"]["api_key"] = draw(openai_api_keys())

    return config


# Mutation testing strategies - for generating invalid configurations
@st.composite
def invalid_api_keys(draw) -> str:
    """Generate invalid API keys for mutation testing."""
    return draw(
        st.one_of(
            st.text(min_size=1, max_size=10),  # Too short
            st.text(alphabet="invalid", min_size=5, max_size=20),  # Wrong prefix
            st.just(""),  # Empty
            st.text(alphabet=" \t\n", min_size=5, max_size=20),  # Whitespace only
        )
    )


@st.composite
def invalid_urls(draw) -> str:
    """Generate invalid URLs for mutation testing."""
    return draw(
        st.one_of(
            st.just("not-a-url"),
            st.just("http://"),
            st.just("://missing-scheme"),
            st.just("ftp://unsupported-scheme.com"),
            st.text(alphabet="spaces not allowed", min_size=10, max_size=20),
        )
    )


@st.composite
def invalid_positive_integers(draw) -> int:
    """Generate invalid positive integers for mutation testing."""
    return draw(
        st.one_of(
            st.integers(max_value=0),  # Non-positive (including 0 and negative)
            st.just(-1),  # Explicitly negative
            st.just(-100),  # More negative
        )
    )


@st.composite
def invalid_chunk_configurations(draw) -> dict[str, Any]:
    """Generate invalid chunking configurations for mutation testing."""
    base_config = dict(draw(chunk_configurations()))
    chunk_size = base_config["chunk_size"]

    violation_type = draw(
        st.sampled_from(
            [
                "overlap_too_large",
                "token_overlap_too_large",
                "json_too_small",
                "negative_values",
            ]
        )
    )

    if violation_type == "overlap_too_large":
        base_config["chunk_overlap"] = chunk_size
        return base_config
    if violation_type == "token_overlap_too_large":
        base_config["token_chunk_overlap"] = base_config["token_chunk_size"]
        return base_config
    if violation_type == "json_too_small":
        base_config["json_max_chars"] = max(1, chunk_size - 1)
        return base_config
    # negative_values
    return {
        "chunk_size": draw(st.integers(max_value=0)),
        "chunk_overlap": draw(st.integers(max_value=-1)),
        "token_chunk_size": draw(st.integers(max_value=0)),
        "token_chunk_overlap": draw(st.integers(max_value=0)),
        "token_model": "",
        "json_max_chars": draw(st.integers(max_value=0)),
    }
