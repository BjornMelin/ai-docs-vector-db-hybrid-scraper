"""Embedding management tools for MCP server."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...

from ...infrastructure.client_manager import ClientManager
from ..models.requests import EmbeddingRequest

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register embedding management tools with the MCP server."""

    from ..models.responses import EmbeddingGenerationResponse
    from ..models.responses import EmbeddingProviderInfo

    @mcp.tool()
    async def generate_embeddings(
        request: EmbeddingRequest, ctx: Context = None
    ) -> EmbeddingGenerationResponse:
        """
        Generate embeddings using the optimal provider.

        Automatically selects the best embedding model based on cost,
        performance, and availability.
        """
        if ctx:
            await ctx.info(
                f"Generating embeddings for {len(request.texts)} texts using model: {request.model or 'default'}"
            )

        try:
            # Get embedding manager from client manager
            embedding_manager = await client_manager.get_embedding_manager()

            if ctx:
                await ctx.debug(
                    f"Using batch size: {request.batch_size}, sparse embeddings: {request.generate_sparse}"
                )

            # Generate embeddings
            result = await embedding_manager.generate_embeddings(
                texts=request.texts,
                model=request.model,
                batch_size=request.batch_size,
                generate_sparse=request.generate_sparse,
            )

            # Get provider info
            provider_info = embedding_manager.get_current_provider_info()

            if ctx:
                await ctx.info(
                    f"Successfully generated embeddings for {len(request.texts)} texts using provider: {provider_info.get('name', 'unknown')}"
                )

            return EmbeddingGenerationResponse(
                embeddings=result.embeddings,
                sparse_embeddings=result.sparse_embeddings,
                model=result.model,
                provider=provider_info.get("name", "unknown"),
                cost_estimate=(result.total_tokens * 0.00002)
                if result.total_tokens
                else None,
                total_tokens=result.total_tokens,
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Embedding generation failed: {e}")
            logger.error(f"Embedding generation failed: {e}")
            raise

    @mcp.tool()
    async def list_embedding_providers(ctx: Context = None) -> list[EmbeddingProviderInfo]:
        """
        List available embedding providers and their capabilities.

        Returns information about supported models, costs, and current status.
        """
        if ctx:
            await ctx.info("Retrieving available embedding providers")

        try:
            # Get embedding manager from client manager
            embedding_manager = await client_manager.get_embedding_manager()

            # Get available providers
            providers = []

            # OpenAI provider
            if embedding_manager._openai_available():
                providers.append(
                    {
                        "name": "openai",
                        "models": [
                            {
                                "name": "text-embedding-3-small",
                                "dimensions": 1536,
                                "max_tokens": 8191,
                                "cost_per_million": 0.02,
                            },
                            {
                                "name": "text-embedding-3-large",
                                "dimensions": 3072,
                                "max_tokens": 8191,
                                "cost_per_million": 0.13,
                            },
                            {
                                "name": "text-embedding-ada-002",
                                "dimensions": 1536,
                                "max_tokens": 8191,
                                "cost_per_million": 0.10,
                            },
                        ],
                        "status": "available",
                        "features": ["dense", "async", "batching"],
                    }
                )
                if ctx:
                    await ctx.debug("OpenAI provider available")

            # FastEmbed provider
            providers.append(
                {
                    "name": "fastembed",
                    "models": [
                        {
                            "name": "BAAI/bge-small-en-v1.5",
                            "dimensions": 384,
                            "max_tokens": 512,
                            "cost_per_million": 0.0,
                        },
                        {
                            "name": "BAAI/bge-base-en-v1.5",
                            "dimensions": 768,
                            "max_tokens": 512,
                            "cost_per_million": 0.0,
                        },
                        {
                            "name": "sentence-transformers/all-MiniLM-L6-v2",
                            "dimensions": 384,
                            "max_tokens": 256,
                            "cost_per_million": 0.0,
                        },
                    ],
                    "status": "available",
                    "features": ["dense", "sparse", "local", "cpu", "gpu"],
                }
            )
            if ctx:
                await ctx.debug("FastEmbed provider available")

            if ctx:
                await ctx.info(f"Found {len(providers)} available embedding providers")

            return [EmbeddingProviderInfo(**p) for p in providers]

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list embedding providers: {e}")
            logger.error(f"Failed to list embedding providers: {e}")
            raise
