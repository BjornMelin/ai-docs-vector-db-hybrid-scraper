"""Deployment and alias management tools for MCP server."""

import logging
from typing import Any

from fastmcp import Context

from ...config.enums import SearchStrategy
from ..models.requests import SearchRequest
from ..models.responses import SearchResult

# Handle both module and script imports
try:
    from infrastructure.client_manager import ClientManager
except ImportError:
    from ...infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register deployment and alias management tools with the MCP server."""

    # Import search tools to get search_documents
    from .search import register_tools as register_search_tools

    search_tools_registry = {}
    register_search_tools(
        type(
            "MockMCP",
            (),
            {"tool": lambda f: search_tools_registry.update({f.__name__: f}) or f},
        )(),
        client_manager,
    )
    search_documents = search_tools_registry["search_documents"]

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
        collection = await client_manager.alias_manager.get_collection_for_alias(alias)
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

        return await search_documents(request, ctx)

    @mcp.tool()
    async def list_aliases() -> dict[str, str]:
        """
        List all collection aliases and their targets.

        Returns a mapping of alias names to collection names.
        """
        return await client_manager.alias_manager.list_aliases()

    @mcp.tool()
    async def create_alias(
        alias_name: str,
        collection_name: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Create or update an alias to point to a collection.

        Args:
            alias_name: Name of the alias
            collection_name: Collection to point to
            force: If True, overwrite existing alias

        Returns:
            Status information
        """
        success = await client_manager.alias_manager.create_alias(
            alias_name=alias_name,
            collection_name=collection_name,
            force=force,
        )

        return {
            "success": success,
            "alias": alias_name,
            "collection": collection_name,
        }

    @mcp.tool()
    async def deploy_new_index(
        alias: str,
        source: str,
        validation_queries: list[str] | None = None,
        rollback_on_failure: bool = True,
        ctx: Context = None,
    ) -> dict[str, Any]:
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

        result = await client_manager.blue_green.deploy_new_version(
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

        return result

    @mcp.tool()
    async def start_ab_test(
        experiment_name: str,
        control_collection: str,
        treatment_collection: str,
        traffic_split: float = 0.5,
        metrics: list[str] | None = None,
    ) -> dict[str, str]:
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
        experiment_id = await client_manager.ab_testing.create_experiment(
            experiment_name=experiment_name,
            control_collection=control_collection,
            treatment_collection=treatment_collection,
            traffic_split=traffic_split,
            metrics_to_track=metrics,
        )

        return {
            "experiment_id": experiment_id,
            "status": "started",
            "control": control_collection,
            "treatment": treatment_collection,
            "traffic_split": traffic_split,
        }

    @mcp.tool()
    async def analyze_ab_test(experiment_id: str) -> dict[str, Any]:
        """
        Analyze results of an A/B test experiment.

        Returns statistical analysis including p-values, confidence intervals,
        and improvement metrics for each tracked metric.

        Args:
            experiment_id: ID of the experiment to analyze

        Returns:
            Detailed analysis results
        """
        return client_manager.ab_testing.analyze_experiment(experiment_id)

    @mcp.tool()
    async def start_canary_deployment(
        alias: str,
        new_collection: str,
        stages: list[dict] | None = None,
        auto_rollback: bool = True,
        ctx: Context = None,
    ) -> dict[str, Any]:
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

        deployment_id = await client_manager.canary.start_canary(
            alias_name=alias,
            new_collection=new_collection,
            stages=stages,
            auto_rollback=auto_rollback,
        )

        if ctx:
            await ctx.info(f"Canary deployment started with ID: {deployment_id}")

        return {
            "deployment_id": deployment_id,
            "status": "started",
            "alias": alias,
            "new_collection": new_collection,
        }

    @mcp.tool()
    async def get_canary_status(deployment_id: str) -> dict[str, Any]:
        """
        Get current status of a canary deployment.

        Shows current stage, traffic percentage, and health metrics.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Current deployment status
        """
        return await client_manager.canary.get_deployment_status(deployment_id)

    @mcp.tool()
    async def pause_canary(deployment_id: str) -> dict[str, str]:
        """
        Pause a canary deployment.

        Stops progression through stages but maintains current traffic split.

        Args:
            deployment_id: ID of the deployment to pause

        Returns:
            Status message
        """
        success = await client_manager.canary.pause_deployment(deployment_id)

        return {
            "status": "paused" if success else "failed",
            "deployment_id": deployment_id,
        }

    @mcp.tool()
    async def resume_canary(deployment_id: str) -> dict[str, str]:
        """
        Resume a paused canary deployment.

        Continues progression through remaining stages.

        Args:
            deployment_id: ID of the deployment to resume

        Returns:
            Status message
        """
        success = await client_manager.canary.resume_deployment(deployment_id)

        return {
            "status": "resumed" if success else "failed",
            "deployment_id": deployment_id,
        }
