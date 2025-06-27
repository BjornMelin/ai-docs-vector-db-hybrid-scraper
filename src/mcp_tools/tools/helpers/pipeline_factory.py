"""Query Processing Pipeline Factory for MCP server."""

import logging  # noqa: PLC0415
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


from ....infrastructure.client_manager import ClientManager
from ....services.query_processing.pipeline import QueryProcessingPipeline


logger = logging.getLogger(__name__)


class QueryProcessingPipelineFactory:
    """Factory for creating and initializing query processing pipelines."""

    def __init__(self, client_manager: ClientManager):
        """Initialize factory with client manager."""
        self.client_manager = client_manager

    async def create_pipeline(
        self, ctx: "Context | None" = None
    ) -> QueryProcessingPipeline:
        """Create and initialize query processing pipeline."""
        try:
            # Get required services
            embedding_manager = await self.client_manager.get_embedding_manager()
            qdrant_service = await self.client_manager.get_qdrant_service()
            hyde_engine = await self.client_manager.get_hyde_engine()

            # Get cache manager if available
            cache_manager = None
            try:
                cache_manager = await self.client_manager.get_cache_manager()
            except Exception:
                if ctx:
                    await ctx.debug(
                        "Cache manager not available, proceeding without caching"
                    )

            # Create orchestrator
            from ....services.query_processing.orchestrator import (  # noqa: PLC0415
                QueryProcessingOrchestrator,
            )

            orchestrator = QueryProcessingOrchestrator(
                embedding_manager=embedding_manager,
                qdrant_service=qdrant_service,
                hyde_engine=hyde_engine,
                cache_manager=cache_manager,
            )

            # Create pipeline
            pipeline = QueryProcessingPipeline(orchestrator=orchestrator)
            await pipeline.initialize()

            return pipeline

        except Exception as e:
            if ctx:
                await ctx.error("Failed to initialize query processing pipeline")
            logger.exception("Pipeline initialization failed")
            raise
