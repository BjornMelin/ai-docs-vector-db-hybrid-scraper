"""Utility tools for MCP server."""

import logging

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import ConfigValidationResponse, GenericDictResponse


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register utility tools with the MCP server."""

    @mcp.tool()
    async def estimate_costs(
        text_count: int,
        average_length: int = 1000,
        include_storage: bool = True,
        ctx: Context = None,
    ) -> GenericDictResponse:
        """Estimate costs for processing documents.

        Calculates embedding generation and storage costs based on
        current pricing models.
        """
        if ctx:
            await ctx.info(
                f"Estimating costs for {text_count} texts with average length "
                f"{average_length}"
            )

        try:
            # Estimate tokens (rough approximation)
            total_chars = text_count * average_length
            estimated_tokens = total_chars / 4  # Rough char-to-token ratio

            # Calculate costs
            embedding_cost = estimated_tokens * 0.00002 / 1000  # $0.02 per 1M tokens

            costs = {
                "text_count": text_count,
                "estimated_tokens": int(estimated_tokens),
                "embedding_cost": round(embedding_cost, 4),
                "provider": "openai/text-embedding-3-small",
            }

            if ctx:
                await ctx.debug(
                    f"Estimated {estimated_tokens} tokens, "
                    f"embedding cost: ${embedding_cost:.4f}"
                )

            if include_storage:
                # Assume 1536 dimensions, 4 bytes per float
                storage_bytes = text_count * 1536 * 4
                storage_gb = storage_bytes / 1e9
                storage_cost = storage_gb * 0.20  # $0.20 per GB/month estimate

                costs["storage_gb"] = round(storage_gb, 4)
                costs["storage_cost_monthly"] = round(storage_cost, 4)
                costs["total_cost"] = round(embedding_cost + storage_cost, 4)

                if ctx:
                    await ctx.debug(
                        f"Storage: {storage_gb:.4f} GB, "
                        f"monthly cost: ${storage_cost:.4f}"
                    )

            if ctx:
                await ctx.info(
                    f"Cost estimation completed. Total estimated cost: "
                    f"${costs.get('total_cost', costs['embedding_cost']):.4f}"
                )

            return GenericDictResponse(**costs)

        except (AttributeError, OSError, PermissionError):
            if ctx:
                await ctx.error("Failed to estimate costs")
            logger.exception("Failed to estimate costs")
            raise

    @mcp.tool()
    async def validate_configuration(ctx: Context = None) -> ConfigValidationResponse:
        """Validate system configuration and API keys.

        Checks all required configuration and returns validation results.
        """
        if ctx:
            await ctx.info("Starting configuration validation")

        try:
            config_status = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "config": {},
            }

            # Check Qdrant
            config_status["config"]["qdrant_url"] = (
                client_manager.unified_config.qdrant.url
            )
            if ctx:
                await ctx.debug("Qdrant URL configured")

            # Check API keys
            if not client_manager.unified_config.openai.api_key:
                config_status["warnings"].append("OpenAI API key not configured")
                if ctx:
                    await ctx.warning("OpenAI API key not configured")
            else:
                config_status["config"]["openai"] = "configured"
                if ctx:
                    await ctx.debug("OpenAI API key configured")

            if not client_manager.unified_config.firecrawl.api_key:
                config_status["warnings"].append("Firecrawl API key not configured")
                if ctx:
                    await ctx.warning("Firecrawl API key not configured")
            else:
                config_status["config"]["firecrawl"] = "configured"
                if ctx:
                    await ctx.debug("Firecrawl API key configured")

            # Check cache configuration
            config_status["config"]["cache"] = {
                "l1_enabled": True,
                "l1_max_items": client_manager.unified_config.cache.max_items,
                "l2_enabled": client_manager.unified_config.cache.redis_url is not None,
            }
            if ctx:
                await ctx.debug("Cache configuration: L1 enabled, L2 enabled")

            # Determine overall validity
            if config_status["errors"]:
                config_status["valid"] = False

            if ctx:
                await ctx.info(
                    "Configuration validation completed. Valid: "
                    f"{config_status['valid']}, Warnings"
                )

            return ConfigValidationResponse(
                status="success" if config_status["valid"] else "error",
                errors=config_status["errors"] or None,
                details=config_status,
            )

        except (TimeoutError, OSError, PermissionError):
            if ctx:
                await ctx.error("Failed to validate configuration")
            logger.exception("Failed to validate configuration")
            raise
