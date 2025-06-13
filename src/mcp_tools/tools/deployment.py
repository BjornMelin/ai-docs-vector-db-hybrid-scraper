"""Deployment and alias management tools for MCP server."""

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


from ...config.enums import SearchStrategy
from ...infrastructure.client_manager import ClientManager
from ..models.requests import SearchRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register deployment and alias management tools with the MCP server."""

    # Import search utility
    from ._search_utils import search_documents_core

    @mcp.tool()
    async def search_with_alias(
        query: str,
        alias: str = "documentation",
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        enable_reranking: bool = False,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """
        Search using collection alias for zero-downtime updates.

        Aliases allow instant switching between collection versions without
        affecting search availability.
        """
        # Get actual collection from alias
        alias_manager = await client_manager.get_alias_manager()
        collection = await alias_manager.get_collection_for_alias(alias)
        if not collection:
            raise ValueError(f"Alias {alias} not found")

        # Perform search on actual collection
        request = SearchRequest(
            query=query,
            collection=collection,
            limit=limit,
            strategy=strategy,
            enable_reranking=enable_reranking,
        )

        return await search_documents_core(request, client_manager, ctx)

    from ..models.responses import ABTestAnalysisResponse
    from ..models.responses import AliasesResponse
    from ..models.responses import CanaryStatusResponse
    from ..models.responses import OperationStatus

    @mcp.tool()
    async def list_aliases(ctx=None) -> AliasesResponse:
        """
        List all collection aliases and their targets.

        Returns a mapping of alias names to collection names.
        """
        if ctx:
            await ctx.info("Retrieving all collection aliases")

        try:
            alias_manager = await client_manager.get_alias_manager()
            aliases = await alias_manager.list_aliases()

            if ctx:
                await ctx.info(f"Successfully retrieved {len(aliases)} aliases")

            return AliasesResponse(aliases=aliases)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list aliases: {e}")
            logger.error(f"Failed to list aliases: {e}")
            raise

    @mcp.tool()
    async def create_alias(
        alias_name: str,
        collection_name: str,
        force: bool = False,
        ctx=None,
    ) -> OperationStatus:
        """
        Create or update an alias to point to a collection.

        Args:
            alias_name: Name of the alias
            collection_name: Collection to point to
            force: If True, overwrite existing alias

        Returns:
            Status information
        """
        if ctx:
            await ctx.info(
                f"Creating alias {alias_name} -> {collection_name}, force={force}"
            )

        try:
            alias_manager = await client_manager.get_alias_manager()
            success = await alias_manager.create_alias(
                alias_name=alias_name,
                collection_name=collection_name,
                force=force,
            )

            status = "success" if success else "error"
            if ctx:
                if success:
                    await ctx.info(
                        f"Successfully created alias {alias_name} -> {collection_name}"
                    )
                else:
                    await ctx.warning(
                        f"Failed to create alias {alias_name} -> {collection_name}"
                    )

            return OperationStatus(
                status=status,
                details={"alias": alias_name, "collection": collection_name},
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to create alias {alias_name}: {e}")
            logger.error(f"Failed to create alias: {e}")
            raise

    @mcp.tool()
    async def deploy_new_index(
        alias: str,
        source: str,
        validation_queries: "list[str] | None" = None,
        rollback_on_failure: bool = True,
        ctx: Context = None,
    ) -> OperationStatus:
        """
        Deploy new index version with zero downtime using blue-green deployment.

        Creates a new collection, populates it, validates, and atomically switches
        the alias. Includes automatic rollback on failure.

        Args:
            alias: Alias to update
            source: Data source (e.g., "collection:docs_v1" or "crawl:new")
            validation_queries: Queries to validate deployment
            rollback_on_failure: Whether to rollback on validation failure

        Returns:
            Deployment status with details
        """
        if ctx:
            await ctx.info(f"Starting blue-green deployment for alias {alias}")

        # Default validation queries if none provided
        if not validation_queries:
            validation_queries = [
                "python asyncio",
                "react hooks",
                "fastapi authentication",
            ]

        blue_green = await client_manager.get_blue_green_deployment()
        result = await blue_green.deploy_new_version(
            alias_name=alias,
            data_source=source,
            validation_queries=validation_queries,
            rollback_on_failure=rollback_on_failure,
        )

        if ctx:
            await ctx.info(
                f"Deployment completed successfully. "
                f"Alias {alias} now points to {result['new_collection']}"
            )

        return OperationStatus(status="success", details=result)

    @mcp.tool()
    async def start_ab_test(
        experiment_name: str,
        control_collection: str,
        treatment_collection: str,
        traffic_split: float = 0.5,
        metrics: "list[str] | None" = None,
        ctx=None,
    ) -> OperationStatus:
        """
        Start A/B test between two collections.

        Enables testing new embeddings, chunking strategies, or configurations
        on live traffic with automatic metrics collection.

        Args:
            experiment_name: Name of the experiment
            control_collection: Control (baseline) collection
            treatment_collection: Treatment (test) collection
            traffic_split: Percentage of traffic to treatment (0-1)
            metrics: Metrics to track (default: latency, relevance, clicks)

        Returns:
            Experiment ID and status
        """
        if ctx:
            await ctx.info(
                f"Starting A/B test: {experiment_name} ({control_collection} vs {treatment_collection})"
            )

        try:
            ab_testing = await client_manager.get_ab_testing()
            experiment_id = await ab_testing.create_experiment(
                experiment_name=experiment_name,
                control_collection=control_collection,
                treatment_collection=treatment_collection,
                traffic_split=traffic_split,
                metrics_to_track=metrics,
            )

            if ctx:
                await ctx.info(
                    f"A/B test started successfully with ID: {experiment_id}"
                )

            return OperationStatus(
                status="started",
                details={
                    "experiment_id": experiment_id,
                    "control": control_collection,
                    "treatment": treatment_collection,
                    "traffic_split": traffic_split,
                },
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to start A/B test {experiment_name}: {e}")
            logger.error(f"Failed to start A/B test: {e}")
            raise

    @mcp.tool()
    async def analyze_ab_test(experiment_id: str, ctx=None) -> ABTestAnalysisResponse:
        """
        Analyze results of an A/B test experiment.

        Returns statistical analysis including p-values, confidence intervals,
        and improvement metrics for each tracked metric.

        Args:
            experiment_id: ID of the experiment to analyze

        Returns:
            Detailed analysis results
        """
        if ctx:
            await ctx.info(f"Analyzing A/B test experiment: {experiment_id}")

        try:
            ab_testing = await client_manager.get_ab_testing()
            result = await ab_testing.analyze_experiment(experiment_id)

            if ctx:
                await ctx.info(
                    f"A/B test analysis completed for experiment {experiment_id}"
                )

            return ABTestAnalysisResponse(**result)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to analyze A/B test {experiment_id}: {e}")
            logger.error(f"Failed to analyze A/B test: {e}")
            raise

    @mcp.tool()
    async def start_canary_deployment(
        alias: str,
        new_collection: str,
        stages: "list[dict] | None" = None,
        auto_rollback: bool = True,
        ctx: Context = None,
    ) -> OperationStatus:
        """
        Start canary deployment with gradual traffic rollout.

        Progressively shifts traffic to new collection with health monitoring
        and automatic rollback on errors.

        Args:
            alias: Alias to update
            new_collection: New collection to deploy
            stages: Custom deployment stages (default: 5% -> 25% -> 50% -> 100%)
            auto_rollback: Whether to auto-rollback on failure

        Returns:
            Deployment ID and status
        """
        if ctx:
            await ctx.info(f"Starting canary deployment for alias {alias}")

        # Validate stages if provided
        if stages:
            for i, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    raise ValueError(f"Stage {i} must be a dictionary")
                if "percentage" not in stage:
                    raise ValueError(f"Stage {i} missing required 'percentage' field")
                if "duration_minutes" not in stage:
                    raise ValueError(
                        f"Stage {i} missing required 'duration_minutes' field"
                    )

                percentage = stage["percentage"]
                if (
                    not isinstance(percentage, int | float)
                    or percentage < 0
                    or percentage > 100
                ):
                    raise ValueError(f"Stage {i} percentage must be between 0 and 100")

                duration = stage["duration_minutes"]
                if not isinstance(duration, int | float) or duration <= 0:
                    raise ValueError(f"Stage {i} duration_minutes must be positive")

        canary = await client_manager.get_canary_deployment()
        deployment_id = await canary.start_canary(
            alias_name=alias,
            new_collection=new_collection,
            stages=stages,
            auto_rollback=auto_rollback,
        )

        if ctx:
            await ctx.info(f"Canary deployment started with ID: {deployment_id}")

        return OperationStatus(
            status="started",
            details={
                "deployment_id": deployment_id,
                "alias": alias,
                "new_collection": new_collection,
            },
        )

    @mcp.tool()
    async def get_canary_status(deployment_id: str, ctx=None) -> CanaryStatusResponse:
        """
        Get current status of a canary deployment.

        Shows current stage, traffic percentage, and health metrics.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Current deployment status
        """
        if ctx:
            await ctx.info(f"Retrieving canary deployment status: {deployment_id}")

        try:
            canary = await client_manager.get_canary_deployment()
            status = await canary.get_deployment_status(deployment_id)

            if ctx:
                await ctx.info(
                    f"Successfully retrieved status for deployment {deployment_id}"
                )

            return CanaryStatusResponse(**status)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to get canary status for {deployment_id}: {e}")
            logger.error(f"Failed to get canary status: {e}")
            raise

    @mcp.tool()
    async def pause_canary(deployment_id: str, ctx=None) -> OperationStatus:
        """
        Pause a canary deployment.

        Stops progression through stages but maintains current traffic split.

        Args:
            deployment_id: ID of the deployment to pause

        Returns:
            Status message
        """
        if ctx:
            await ctx.info(f"Pausing canary deployment: {deployment_id}")

        try:
            canary = await client_manager.get_canary_deployment()
            success = await canary.pause_deployment(deployment_id)
            status = "paused" if success else "failed"

            if ctx:
                if success:
                    await ctx.info(f"Successfully paused deployment {deployment_id}")
                else:
                    await ctx.warning(f"Failed to pause deployment {deployment_id}")

            return OperationStatus(
                status=status, details={"deployment_id": deployment_id}
            )

        except Exception as e:
            if ctx:
                await ctx.error(
                    f"Failed to pause canary deployment {deployment_id}: {e}"
                )
            logger.error(f"Failed to pause canary deployment: {e}")
            raise

    @mcp.tool()
    async def resume_canary(deployment_id: str, ctx=None) -> OperationStatus:
        """
        Resume a paused canary deployment.

        Continues progression through remaining stages.

        Args:
            deployment_id: ID of the deployment to resume

        Returns:
            Status message
        """
        if ctx:
            await ctx.info(f"Resuming canary deployment: {deployment_id}")

        try:
            canary = await client_manager.get_canary_deployment()
            success = await canary.resume_deployment(deployment_id)
            status = "resumed" if success else "failed"

            if ctx:
                if success:
                    await ctx.info(f"Successfully resumed deployment {deployment_id}")
                else:
                    await ctx.warning(f"Failed to resume deployment {deployment_id}")

            return OperationStatus(
                status=status, details={"deployment_id": deployment_id}
            )

        except Exception as e:
            if ctx:
                await ctx.error(
                    f"Failed to resume canary deployment {deployment_id}: {e}"
                )
            logger.error(f"Failed to resume canary deployment: {e}")
            raise
