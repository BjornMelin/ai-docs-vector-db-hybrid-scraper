"""Query Processing Pipeline Factory for MCP server."""

import logging

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.services.query_processing.orchestrator import SearchOrchestrator
from src.services.query_processing.pipeline import QueryProcessingPipeline


logger = logging.getLogger(__name__)


class QueryProcessingPipelineFactory:
    """Factory for creating and initializing query processing pipelines."""

    def __init__(self, client_manager: ClientManager):
        """Initialize factory with client manager."""
        self.client_manager = client_manager

    async def create_pipeline(
        self, ctx: Context | None = None
    ) -> QueryProcessingPipeline:
        """Create and initialize query processing pipeline."""
        try:
            # Get required services
            # embedding_manager = await self.client_manager.get_embedding_manager()
            # qdrant_service = await self.client_manager.get_qdrant_service()
            # hyde_engine = await self.client_manager.get_hyde_engine()

            # Get cache manager if available
            try:
                # cache_manager = await self.client_manager.get_cache_manager()
                pass  # Placeholder for the commented code
            except (ConnectionError, OSError, RuntimeError, TimeoutError):
                if ctx:
                    await ctx.debug(
                        "Cache manager not available, proceeding without caching"
                    )

            # Create orchestrator
            orchestrator = SearchOrchestrator(
                cache_size=1000,
                enable_performance_optimization=True,
            )

            # Create pipeline
            pipeline = QueryProcessingPipeline(orchestrator=orchestrator)
            await pipeline.initialize()

        except (AttributeError, ImportError, OSError):
            if ctx:
                await ctx.error("Failed to initialize query processing pipeline")
            logger.exception("Pipeline initialization failed")
            raise
