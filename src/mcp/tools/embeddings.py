"""Embedding management tools for MCP server."""

import logging
from typing import Any

from ...infrastructure.client_manager import ClientManager
from ...services.embeddings.manager import EmbeddingManager
from ..models.requests import EmbeddingRequest

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register embedding management tools with the MCP server."""

    @mcp.tool()
    async def generate_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        """
        Generate embeddings using the optimal provider.

        Automatically selects the best embedding model based on cost,
        performance, and availability.
        """
        try:
            # Initialize embedding manager on-demand
            embedding_manager = EmbeddingManager(client_manager)

            # Generate embeddings
            result = await embedding_manager.generate_embeddings(
                texts=request.texts,
                model=request.model,
                batch_size=request.batch_size,
                generate_sparse=request.generate_sparse,
            )

            # Get provider info
            provider_info = embedding_manager.get_current_provider_info()

            return {
                "embeddings": result.embeddings,
                "sparse_embeddings": result.sparse_embeddings,
                "count": len(result.embeddings),
                "dimensions": result.dimensions,
                "provider": provider_info.get("name", "unknown"),
                "model": result.model,
                "cost_estimate": result.total_tokens * 0.00002
                if result.total_tokens
                else None,
                "total_tokens": result.total_tokens,
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    @mcp.tool()
    async def list_embedding_providers() -> list[dict[str, Any]]:
        """
        List available embedding providers and their capabilities.

        Returns information about supported models, costs, and current status.
        """
        try:
            # Initialize embedding manager
            embedding_manager = EmbeddingManager(client_manager)

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

            return providers

        except Exception as e:
            logger.error(f"Failed to list embedding providers: {e}")
            raise
