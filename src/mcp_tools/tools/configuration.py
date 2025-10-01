"""Minimal configuration management using Pydantic BaseSettings."""

import logging
from typing import Any

from fastmcp import Context
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Database
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Search settings
    default_search_limit: int = Field(default=10, ge=1, le=100)
    default_search_strategy: str = Field(default="hybrid")

    # Performance
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    request_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")


def register_tools(mcp, client_manager: ClientManager):
    """Register configuration management tools."""

    @mcp.tool()
    async def get_config(
        key: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get configuration value(s).

        Args:
            key: Specific config key (optional, returns all if None)
            ctx: MCP context

        Returns:
            Configuration value(s)
        """
        try:
            settings = AppSettings()

            if key:
                value = getattr(settings, key, None)
                if value is None:
                    if ctx:
                        await ctx.warning(f"Config key '{key}' not found")
                    return {"key": key, "value": None, "found": False}

                if ctx:
                    await ctx.info(f"Retrieved config: {key}")

                return {"key": key, "value": value, "found": True}

            # Return all non-sensitive config
            config_dict = settings.model_dump(
                exclude={
                    "openai_api_key",
                    "tavily_api_key",
                    "anthropic_api_key",
                    "qdrant_api_key",
                }
            )

            if ctx:
                await ctx.info("Retrieved all configuration")

            return {"config": config_dict}

        except Exception as e:
            logger.exception("Failed to get config")
            if ctx:
                await ctx.error(f"Config error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def validate_config(
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Validate current configuration.

        Args:
            ctx: MCP context

        Returns:
            Validation results
        """
        try:
            settings = AppSettings()

            # Check required keys
            issues = []
            if not settings.qdrant_url:
                issues.append("QDRANT_URL not configured")

            # Check optional but recommended
            warnings = []
            if not settings.openai_api_key:
                warnings.append("OPENAI_API_KEY not set (embedding features limited)")
            if not settings.tavily_api_key:
                warnings.append("TAVILY_API_KEY not set (web search unavailable)")

            is_valid = len(issues) == 0

            result = {
                "valid": is_valid,
                "issues": issues,
                "warnings": warnings,
                "settings_found": len(
                    [k for k, v in settings.model_dump().items() if v is not None]
                ),
            }

            if ctx:
                if is_valid:
                    await ctx.info("Configuration is valid")
                else:
                    await ctx.error(f"Configuration has {len(issues)} issues")

            return result

        except Exception as e:
            logger.exception("Config validation failed")
            if ctx:
                await ctx.error(f"Validation error: {e}")
            return {
                "valid": False,
                "error": str(e),
            }
