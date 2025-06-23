"""Additional tool registration functions for query processing MCP tools."""

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

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


from .pipeline_factory import QueryProcessingPipelineFactory

logger = logging.getLogger(__name__)


def register_pipeline_health_tool(mcp, factory: QueryProcessingPipelineFactory):
    """Register the pipeline health check tool."""

    @mcp.tool()
    async def get_processing_pipeline_health(ctx: Context) -> dict:
        """
        Get health status of the advanced query processing pipeline.

        Returns detailed health information for all pipeline components including
        orchestrator, intent classifier, preprocessor, and strategy selector.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline health check {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await factory.create_pipeline(ctx)

            # Perform health check
            health_status = await pipeline.health_check()

            await ctx.info(
                f"Pipeline health check {request_id} completed: "
                f"status={'healthy' if health_status.get('pipeline_healthy') else 'unhealthy'}"
            )

            return health_status

        except Exception as e:
            await ctx.error(f"Pipeline health check {request_id} failed: {e}")
            logger.exception(f"Pipeline health check failed: {e}")
            return {
                "pipeline_healthy": False,
                "error": str(e),
                "components": {"health_check": "failed"},
            }


def register_pipeline_metrics_tool(mcp, factory: QueryProcessingPipelineFactory):
    """Register the pipeline metrics tool."""

    @mcp.tool()
    async def get_processing_pipeline_metrics(ctx: Context) -> dict:
        """
        Get performance metrics from the advanced query processing pipeline.

        Returns comprehensive performance statistics including processing times,
        strategy usage, success rates, and fallback utilization.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline metrics collection {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await factory.create_pipeline(ctx)

            # Get performance metrics
            metrics = pipeline.get_performance_metrics()

            await ctx.info(f"Pipeline metrics collection {request_id} completed")

            return metrics

        except Exception as e:
            await ctx.error(f"Pipeline metrics collection {request_id} failed: {e}")
            logger.exception(f"Pipeline metrics collection failed: {e}")
            return {"error": str(e), "metrics_available": False}


def register_pipeline_warmup_tool(mcp, factory: QueryProcessingPipelineFactory):
    """Register the pipeline warm-up tool."""

    @mcp.tool()
    async def warm_up_processing_pipeline(ctx: Context) -> dict:
        """
        Warm up the advanced query processing pipeline.

        Pre-loads models and caches by processing test queries to ensure
        optimal performance for subsequent real queries.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting pipeline warm-up {request_id}")

        try:
            # Get query processing pipeline
            pipeline = await factory.create_pipeline(ctx)

            # Warm up pipeline
            await pipeline.warm_up()

            await ctx.info(f"Pipeline warm-up {request_id} completed successfully")

            return {"status": "success", "message": "Pipeline warmed up successfully"}

        except Exception as e:
            await ctx.warning(f"Pipeline warm-up {request_id} had issues: {e}")
            logger.warning(f"Pipeline warm-up failed: {e}")
            return {
                "status": "partial_success",
                "message": f"Pipeline warm-up completed with issues: {e}",
            }
