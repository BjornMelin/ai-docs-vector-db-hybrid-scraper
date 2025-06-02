"""Utility tools for MCP server."""

import logging
from typing import Any

from ...infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register utility tools with the MCP server."""

    @mcp.tool()
    async def estimate_costs(
        text_count: int, average_length: int = 1000, include_storage: bool = True
    ) -> dict[str, Any]:
        """
        Estimate costs for processing documents.

        Calculates embedding generation and storage costs based on
        current pricing models.
        """
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

        if include_storage:
            # Assume 1536 dimensions, 4 bytes per float
            storage_bytes = text_count * 1536 * 4
            storage_gb = storage_bytes / 1e9
            storage_cost = storage_gb * 0.20  # $0.20 per GB/month estimate

            costs["storage_gb"] = round(storage_gb, 4)
            costs["storage_cost_monthly"] = round(storage_cost, 4)
            costs["total_cost"] = round(embedding_cost + storage_cost, 4)

        return costs

    @mcp.tool()
    async def validate_configuration() -> dict[str, Any]:
        """
        Validate system configuration and API keys.

        Checks all required configuration and returns validation results.
        """
        config_status = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config": {},
        }

        # Check Qdrant
        config_status["config"]["qdrant_url"] = client_manager.unified_config.qdrant.url

        # Check API keys
        if not client_manager.unified_config.openai.api_key:
            config_status["warnings"].append("OpenAI API key not configured")
        else:
            config_status["config"]["openai"] = "configured"

        if not client_manager.unified_config.firecrawl.api_key:
            config_status["warnings"].append("Firecrawl API key not configured")
        else:
            config_status["config"]["firecrawl"] = "configured"

        # Check cache configuration
        config_status["config"]["cache"] = {
            "l1_enabled": True,
            "l1_max_items": client_manager.unified_config.cache.max_items,
            "l2_enabled": client_manager.unified_config.cache.redis_url is not None,
        }

        # Determine overall validity
        if config_status["errors"]:
            config_status["valid"] = False

        return config_status
