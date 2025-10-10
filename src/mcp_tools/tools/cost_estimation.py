"""Minimal cost estimation with real calculations."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Context


logger = logging.getLogger(__name__)
# Current API pricing (as of 2024)
PRICING = {
    "openai": {
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        "text-embedding-3-small": {"input": 0.00002 / 1000},
        "text-embedding-3-large": {"input": 0.00013 / 1000},
    },
    "anthropic": {
        "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        "claude-3-sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    },
    "tavily": {
        "search": 0.005,  # per search
    },
}


def register_tools(mcp):
    """Register cost estimation tools."""

    @mcp.tool()
    async def estimate_cost(
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Estimate cost for API usage.

        Args:
            provider: API provider (openai, anthropic, tavily)
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            ctx: MCP context

        Returns:
            Cost estimation
        """

        try:
            provider = provider.lower()
            if provider not in PRICING:
                return {
                    "error": f"Unknown provider: {provider}",
                    "available": list(PRICING.keys()),
                }

            if model not in PRICING[provider]:
                return {
                    "error": f"Unknown model: {model}",
                    "available": list(PRICING[provider].keys()),
                }

            pricing = PRICING[provider][model]

            # Calculate cost
            if "input" in pricing and "output" in pricing:
                input_cost = input_tokens * pricing["input"]
                output_cost = output_tokens * pricing["output"]
                total_cost = input_cost + output_cost

                result = {
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost_usd": round(input_cost, 6),
                    "output_cost_usd": round(output_cost, 6),
                    "total_cost_usd": round(total_cost, 6),
                }
            else:
                # Flat rate pricing (e.g., Tavily)
                total_cost = pricing.get("search", 0)
                result = {
                    "provider": provider,
                    "model": model,
                    "total_cost_usd": total_cost,
                }

            if ctx:
                await ctx.info(f"Estimated cost: ${result['total_cost_usd']}")

            return result

        except Exception as e:
            logger.exception("Cost estimation failed")
            if ctx:
                await ctx.error(f"Estimation error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_pricing(
        provider: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get current API pricing information.

        Args:
            provider: Specific provider (optional, returns all if None)
            ctx: MCP context

        Returns:
            Pricing information
        """

        try:
            if provider:
                provider = provider.lower()
                if provider not in PRICING:
                    return {
                        "error": f"Unknown provider: {provider}",
                        "available": list(PRICING.keys()),
                    }

                result = {"provider": provider, "pricing": PRICING[provider]}
            else:
                result = {"pricing": PRICING}

            if ctx:
                await ctx.info("Retrieved pricing information")

            return result

        except Exception as e:
            logger.exception("Failed to get pricing")
            if ctx:
                await ctx.error(f"Pricing error: {e}")
            return {"error": str(e)}
