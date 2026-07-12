"""Application settings definitions and providers."""

# pylint: disable=global-statement

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import (  # pyright: ignore[reportMissingImports]
    Field,
    ValidationError,
    model_validator,
)
from pydantic_settings import (  # pyright: ignore[reportMissingImports]
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)

from src import __version__
from src.config.browser import BrowserAutomationConfig

from .models import (
    AgenticConfig,
    CacheConfig,
    ChunkingConfig,
    ChunkingStrategy,
    CircuitBreakerConfig,
    CrawlProvider,
    DatabaseConfig,
    DeploymentConfig,
    DocumentationSite,
    EmbeddingConfig,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    HyDEConfig,
    LogLevel,
    MCPClientConfig,
    MonitoringConfig,
    ObservabilityConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    QueryProcessingConfig,
    RAGConfig,
    ReRankingConfig,
    SearchStrategy,
)
from .security.config import SecurityConfig
from .template_utils import merge_overrides


_ACTIVE_CONFIG_PATH = Path("config.json")


class ActivatedConfigSettingsSource(JsonConfigSettingsSource):
    """Load and strictly validate the profile activated in ``config.json``."""

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        """Return the activated configuration after canonical field validation."""
        try:
            payload = super()._read_file(file_path)
        except json.JSONDecodeError as exc:
            msg = f"Invalid activated configuration {file_path}: {exc}"
            raise SettingsError(msg) from exc

        if not isinstance(payload, dict):
            msg = f"Activated configuration {file_path} must contain a JSON object."
            raise SettingsError(msg)

        unknown_paths = _unknown_setting_paths(payload)
        if unknown_paths:
            paths = ", ".join(unknown_paths)
            msg = f"Unsupported activated configuration field(s): {paths}"
            raise SettingsError(msg)
        return payload


class Settings(BaseSettings):
    """Normalized application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="AI_DOCS_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=False,
        env_ignore_empty=True,
        arbitrary_types_allowed=True,
    )

    # Core application metadata
    app_name: str = Field(
        default="AI Documentation Vector DB", description="Application name"
    )
    version: str = Field(default=__version__, description="Application version")
    mode: str = Field(default="production", description="Deployment mode label")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug features")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    # Provider selection
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.FASTEMBED, description="Embedding provider"
    )
    crawl_provider: CrawlProvider = Field(
        default=CrawlProvider.CRAWL4AI, description="Crawling provider"
    )

    # Nested configuration sections
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Cache configuration"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig, description="Qdrant configuration"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="OpenAI configuration"
    )
    fastembed: FastEmbedConfig = Field(
        default_factory=FastEmbedConfig, description="FastEmbed configuration"
    )
    browser: BrowserAutomationConfig = Field(
        default_factory=BrowserAutomationConfig,
        description="Browser automation configuration",
    )
    mcp_client: MCPClientConfig = Field(
        default_factory=MCPClientConfig,  # type: ignore[call-arg]
        description="MCP client configuration",
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Document chunking settings"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding configuration"
    )
    hyde: HyDEConfig = Field(
        default_factory=HyDEConfig, description="HyDE configuration"
    )
    agentic: AgenticConfig = Field(
        default_factory=AgenticConfig, description="Agentic workflow configuration"
    )
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    reranking: ReRankingConfig = Field(
        default_factory=ReRankingConfig, description="Re-ranking configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )
    query_processing: QueryProcessingConfig = Field(
        default_factory=QueryProcessingConfig,
        description="Query processing configuration",
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="Observability configuration"
    )
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig,
        description="Deployment orchestration settings",
    )
    documentation_sites: list[DocumentationSite] = Field(
        default_factory=list, description="Documentation sites to crawl"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define precedence: init, environment, .env, activated file, defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ActivatedConfigSettingsSource(
                settings_cls,
                json_file=_ACTIVE_CONFIG_PATH,
                json_file_encoding="utf-8",
            ),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> Settings:
        """Validate that provider API keys are configured when required."""
        if (
            self.embedding.retrieval_mode
            in {SearchStrategy.SPARSE, SearchStrategy.HYBRID}
            and not self.fastembed.sparse_model
        ):
            msg = "Sparse or hybrid retrieval requires fastembed.sparse_model"
            raise ValueError(msg)
        if self.environment == Environment.TESTING:
            return self
        openai_api_key = getattr(self.openai, "api_key", None)
        firecrawl_settings = getattr(self.browser, "firecrawl", None)
        firecrawl_api_key = getattr(firecrawl_settings, "api_key", None)

        if self.embedding_provider is EmbeddingProvider.OPENAI and not openai_api_key:
            msg = "OpenAI API key required when using OpenAI embedding provider"
            raise ValueError(msg)
        if self.crawl_provider is CrawlProvider.FIRECRAWL and not firecrawl_api_key:
            msg = "Firecrawl API key required when using Firecrawl provider"
            raise ValueError(msg)
        return self

    def is_development(self) -> bool:
        """Return True when running in development environment."""
        return self.environment is Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Return True when running in production environment."""
        return self.environment is Environment.PRODUCTION

    def get_effective_chunking_strategy(self) -> ChunkingStrategy:
        """Return the configured chunking strategy."""
        return getattr(self.chunking, "strategy", ChunkingStrategy.BASIC)

    def get_effective_search_strategy(self) -> SearchStrategy:
        """Return the configured search strategy."""
        return self.embedding.retrieval_mode

    def get_feature_flags(self) -> dict[str, bool]:
        """Return the active feature flags for the unified application."""
        return {
            "comprehensive_observability": bool(
                getattr(self.observability, "enabled", False)
            ),
        }


def _resolve_schema(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a local JSON Schema reference."""
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    resolved = definitions.get(reference.rsplit("/", maxsplit=1)[-1], {})
    return {
        **resolved,
        **{key: value for key, value in schema.items() if key != "$ref"},
    }


def _find_unknown_setting_paths(
    value: Any,
    schema: dict[str, Any],
    definitions: dict[str, Any],
    path: tuple[str, ...] = (),
) -> list[str]:
    """Return configuration paths that are absent from the Settings schema."""
    schema = _resolve_schema(schema, definitions)

    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            resolved = _resolve_schema(alternative, definitions)
            if isinstance(value, dict) and (
                resolved.get("type") == "object" or "properties" in resolved
            ):
                return _find_unknown_setting_paths(value, resolved, definitions, path)
            if isinstance(value, list) and resolved.get("type") == "array":
                return _find_unknown_setting_paths(value, resolved, definitions, path)
        return []

    if isinstance(value, dict):
        properties = schema.get("properties")
        additional = schema.get("additionalProperties")
        if isinstance(properties, dict):
            unknown: list[str] = []
            for key, item in value.items():
                child_path = (*path, str(key))
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    unknown.extend(
                        _find_unknown_setting_paths(
                            item, child_schema, definitions, child_path
                        )
                    )
                elif isinstance(additional, dict):
                    unknown.extend(
                        _find_unknown_setting_paths(
                            item, additional, definitions, child_path
                        )
                    )
                else:
                    unknown.append(".".join(child_path))
            return unknown
        if isinstance(additional, dict):
            return [
                unknown
                for key, item in value.items()
                for unknown in _find_unknown_setting_paths(
                    item, additional, definitions, (*path, str(key))
                )
            ]

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [
            unknown
            for index, item in enumerate(value)
            for unknown in _find_unknown_setting_paths(
                item, schema["items"], definitions, (*path, str(index))
            )
        ]

    return []


def _unknown_setting_paths(payload: dict[str, Any]) -> list[str]:
    """Validate payload keys without tightening mixed runtime environments."""
    schema = Settings.model_json_schema()
    return _find_unknown_setting_paths(payload, schema, schema.get("$defs", {}))


def ensure_runtime_directories(settings: Settings) -> None:
    """Create runtime directories required by the application."""
    for directory in (settings.data_dir, settings.cache_dir, settings.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides: Any) -> Settings:
    """Instantiate settings from the environment without caching."""
    settings = Settings(**overrides)
    ensure_runtime_directories(settings)
    return settings


_ACTIVE_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings instance."""
    global _ACTIVE_SETTINGS
    if _ACTIVE_SETTINGS is None:
        _ACTIVE_SETTINGS = load_settings()
    return _ACTIVE_SETTINGS


def refresh_settings(
    *,
    settings: Settings | None = None,
    **overrides: Any,
) -> Settings:
    """Replace the cached settings instance.

    Args:
        settings: Optional pre-built settings instance to promote to the cache.
        **overrides: Keyword arguments forwarded to ``load_settings``.

    Returns:
        The newly cached settings instance.
    """
    global _ACTIVE_SETTINGS
    if settings is not None and overrides:
        msg = "Provide either a concrete settings instance or overrides, not both."
        raise ValueError(msg)
    if settings is not None:
        _ACTIVE_SETTINGS = settings
        return _ACTIVE_SETTINGS
    if overrides:
        _ACTIVE_SETTINGS = load_settings(**overrides)
        return _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = load_settings()
    return _ACTIVE_SETTINGS


def validate_settings_payload(
    payload: dict[str, Any], *, base: dict[str, Any] | None = None
) -> tuple[bool, list[str], Settings | None]:
    """Validate configuration data using the Settings model."""
    merged = Settings.model_construct().model_dump(mode="python")
    if base:
        merged = merge_overrides(merged, base)
    merged = merge_overrides(merged, payload)

    unknown_paths = _unknown_setting_paths(merged)
    if unknown_paths:
        return (
            False,
            [f"{path}: Extra inputs are not permitted" for path in unknown_paths],
            None,
        )

    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(part) for part in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
        return False, errors, None
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        return False, [str(exc)], None

    return True, [], settings


def load_settings_from_file(path: Path) -> Settings:
    """Load settings overrides from a JSON or YAML file."""
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    try:
        if suffix in {".json"}:
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore import  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "Loading YAML configurations requires PyYAML. "
                    "Install with `pip install pyyaml`."
                ) from exc
            try:
                payload = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML: {exc}") from exc
        else:
            msg = f"Unsupported configuration file format: {path.suffix}"
            raise ValueError(msg)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration file: {exc}") from exc

    if not isinstance(payload, dict):
        msg = "Configuration file must define a JSON/YAML object"
        raise TypeError(msg)

    unknown_paths = _unknown_setting_paths(payload)
    if unknown_paths:
        paths = ", ".join(unknown_paths)
        raise ValueError(f"Unsupported configuration field(s): {paths}")

    return load_settings(**payload)


__all__ = [
    "Settings",
    "ensure_runtime_directories",
    "get_settings",
    "load_settings",
    "load_settings_from_file",
    "refresh_settings",
    "validate_settings_payload",
]
