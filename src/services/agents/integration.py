"""Integration layer for unified agentic system coordination.

This module provides a unified interface that integrates the agent coordination,
agentic vector management, and tool orchestration systems into a cohesive
autonomous AI platform based on I4 research findings.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager
from src.services.agents.coordination import (
    CoordinationStrategy,
    ParallelAgentCoordinator,
    TaskDefinition,
)
from src.services.agents.tool_orchestration import (
    AdvancedToolOrchestrator,
    ToolCapability,
    ToolDefinition,
    ToolOrchestrationPlan,
    ToolPriority,
)
from src.services.vector_db.agentic_manager import (
    AgentCollectionConfig,
    AgenticVectorManager,
    OptimizationStrategy,
)


logger = logging.getLogger(__name__)


class UnifiedAgentRequest(BaseModel):
    """Unified request for agentic system operations."""

    request_id: str = Field(..., description="Unique request identifier")
    goal: str = Field(..., description="High-level goal description")
    context: dict[str, Any] = Field(default_factory=dict, description="Request context")

    # Vector database requirements
    vector_requirements: dict[str, Any] | None = Field(
        None, description="Vector database requirements"
    )
    collection_preferences: dict[str, Any] | None = Field(
        None, description="Collection preferences"
    )

    # Tool orchestration preferences
    tool_preferences: dict[str, Any] | None = Field(
        None, description="Tool execution preferences"
    )
    optimization_target: str = Field(
        "balanced", description="Optimization target: speed, quality, cost, balanced"
    )

    # Coordination constraints
    coordination_strategy: str = Field("adaptive", description="Coordination strategy")
    max_execution_time_seconds: float = Field(
        120.0, description="Maximum execution time"
    )
    quality_threshold: float = Field(0.8, description="Minimum quality threshold")

    # Resource constraints
    max_parallel_agents: int = Field(5, description="Maximum parallel agents")
    resource_limits: dict[str, float] = Field(
        default_factory=dict, description="Resource limits"
    )


class UnifiedAgentResponse(BaseModel):
    """Unified response from agentic system operations."""

    request_id: str = Field(..., description="Original request identifier")
    success: bool = Field(..., description="Whether the operation succeeded")
    execution_time_seconds: float = Field(..., description="Total execution time")

    # Results
    results: dict[str, Any] = Field(
        default_factory=dict, description="Operation results"
    )
    vector_results: dict[str, Any] | None = Field(
        None, description="Vector database results"
    )
    tool_results: dict[str, Any] | None = Field(
        None, description="Tool execution results"
    )
    coordination_results: dict[str, Any] | None = Field(
        None, description="Coordination results"
    )

    # Quality metrics
    quality_score: float = Field(0.0, description="Overall quality score")
    confidence: float = Field(0.0, description="Confidence in results")
    completeness: float = Field(0.0, description="Result completeness")

    # Performance metrics
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )
    resource_usage: dict[str, float] = Field(
        default_factory=dict, description="Resource usage"
    )

    # Error handling
    error: str | None = Field(None, description="Error message if failed")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")

    # Recommendations
    optimization_recommendations: list[str] = Field(
        default_factory=list, description="Optimization suggestions"
    )


class AgenticSystemStatus(BaseModel):
    """Status of the unified agentic system."""

    # System health
    overall_health: str = Field(..., description="Overall system health")
    coordinator_status: dict[str, Any] = Field(..., description="Coordinator status")
    vector_manager_status: dict[str, Any] = Field(
        ..., description="Vector manager status"
    )
    orchestrator_status: dict[str, Any] = Field(
        ..., description="Tool orchestrator status"
    )

    # Performance metrics
    active_requests: int = Field(0, description="Number of active requests")
    total_requests_24h: int = Field(0, description="Total requests in last 24 hours")
    avg_response_time_ms: float = Field(0.0, description="Average response time")
    success_rate_24h: float = Field(0.0, description="Success rate in last 24 hours")

    # Resource utilization
    resource_utilization: dict[str, float] = Field(
        default_factory=dict, description="Resource utilization"
    )
    capacity_remaining: dict[str, float] = Field(
        default_factory=dict, description="Remaining capacity"
    )

    # System insights
    top_capabilities: list[str] = Field(
        default_factory=list, description="Most used capabilities"
    )
    optimization_opportunities: list[str] = Field(
        default_factory=list, description="Optimization opportunities"
    )


class UnifiedAgenticSystem:
    """Unified agentic system integrating coordination, vector management,
    and orchestration.

    This system provides a single interface for complex autonomous AI operations,
    combining the capabilities of:
    - Multi-agent coordination for parallel task execution
    - Intelligent vector database management for dynamic data handling
    - Advanced tool orchestration for complex workflow automation

    Based on I4 Vector Database Modernization research findings.
    """

    def __init__(
        self, client_manager: ClientManager, config: dict[str, Any] | None = None
    ):
        """Initialize the unified agentic system.

        Args:
            client_manager: Client manager for resource access
            config: Optional system configuration
        """
        self.client_manager = client_manager
        self.config = config or {}
        self._initialized = False

        # Initialize subsystems
        self.coordinator = ParallelAgentCoordinator(
            client_manager=client_manager,
            max_concurrent_agents=self.config.get("max_concurrent_agents", 10),
        )

        self.vector_manager = AgenticVectorManager(
            client_manager=client_manager, config=self.config.get("vector_config", {})
        )

        self.orchestrator = AdvancedToolOrchestrator(
            client_manager=client_manager,
            max_parallel_executions=self.config.get("max_parallel_tools", 10),
            default_timeout_seconds=self.config.get("default_timeout", 30.0),
        )

        # Request tracking
        self.active_requests: dict[str, dict[str, Any]] = {}
        self.request_history: list[UnifiedAgentResponse] = []
        self.performance_metrics: dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_execution_time": 0.0,
            "last_24h_requests": [],
        }

        logger.info("UnifiedAgenticSystem initialized")

    async def initialize(self) -> None:
        """Initialize the unified agentic system."""
        if self._initialized:
            return

        try:
            # Initialize subsystems
            await self.coordinator.initialize()
            await self.vector_manager.initialize()

            # Register default tools with orchestrator
            await self._register_default_tools()

            self._initialized = True
            logger.info("UnifiedAgenticSystem fully initialized")

        except Exception as e:
            logger.error(
                "Failed to initialize UnifiedAgenticSystem: %s", e, exc_info=True
            )
            raise

    async def execute_unified_request(
        self, request: UnifiedAgentRequest
    ) -> UnifiedAgentResponse:
        """Execute a unified agentic request.

        Args:
            request: Unified agent request

        Returns:
            Unified agent response with results and metrics
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        request_start = datetime.now(tz=datetime.timezone.utc)

        logger.info(
            "Executing unified request %s: %s", request.request_id, request.goal
        )

        # Track active request
        self.active_requests[request.request_id] = {
            "start_time": request_start,
            "goal": request.goal,
            "status": "initializing",
        }

        try:
            # Phase 1: Vector Database Preparation
            vector_results = await self._prepare_vector_environment(request)

            # Phase 2: Tool Chain Composition
            orchestration_plan = await self._compose_tool_chain(request, vector_results)

            # Phase 3: Agent Coordination Setup
            coordination_tasks = await self._create_coordination_tasks(
                request, vector_results, orchestration_plan
            )

            # Phase 4: Unified Execution
            execution_results = await self._execute_unified_workflow(
                request, coordination_tasks, orchestration_plan
            )

            # Phase 5: Results Integration
            integrated_results = await self._integrate_results(
                request, vector_results, execution_results
            )

            execution_time = time.time() - start_time

            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                request, integrated_results, execution_time
            )

            # Create response
            response = UnifiedAgentResponse(
                request_id=request.request_id,
                success=True,
                execution_time_seconds=execution_time,
                results=integrated_results,
                vector_results=vector_results,
                tool_results=execution_results.get("tool_results"),
                coordination_results=execution_results.get("coordination_results"),
                quality_score=quality_metrics["quality_score"],
                confidence=quality_metrics["confidence"],
                completeness=quality_metrics["completeness"],
                performance_metrics=quality_metrics["performance_metrics"],
                resource_usage=quality_metrics["resource_usage"],
                optimization_recommendations=(
                    await self._generate_optimization_recommendations(
                        request, execution_results, quality_metrics
                    )
                ),
            )

            # Update metrics and cleanup
            await self._update_system_metrics(response)
            self.active_requests.pop(request.request_id, None)

            logger.info(
                "Unified request %s completed successfully in %.2fs",
                request.request_id,
                execution_time,
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Unified request %s failed: %s", request.request_id, e, exc_info=True
            )

            # Create error response
            error_response = UnifiedAgentResponse(
                request_id=request.request_id,
                success=False,
                execution_time_seconds=execution_time,
                error=str(e),
                warnings=[f"Request failed after {execution_time:.2f}s"],
            )

            # Update metrics and cleanup
            await self._update_system_metrics(error_response)
            self.active_requests.pop(request.request_id, None)

            return error_response

    async def get_system_status(self) -> AgenticSystemStatus:
        """Get comprehensive system status.

        Returns:
            Current system status and metrics
        """
        try:
            # Get subsystem statuses
            coordinator_status = await self.coordinator.get_coordination_status()
            vector_status = await self.vector_manager.get_system_status()
            orchestrator_status = await self.orchestrator.get_orchestration_status()

            # Calculate system health
            health_scores = []
            if coordinator_status.get("health_score"):
                health_scores.append(coordinator_status["health_score"])
            if vector_status.get("health_score"):
                health_scores.append(vector_status["health_score"])
            if orchestrator_status.get("healthy_tools", 0) > 0:
                health_scores.append(
                    orchestrator_status["healthy_tools"]
                    / max(1, orchestrator_status["registered_tools"])
                )

            overall_health_score = (
                sum(health_scores) / len(health_scores) if health_scores else 0.0
            )
            overall_health = (
                "healthy"
                if overall_health_score > 0.8
                else "degraded"
                if overall_health_score > 0.5
                else "critical"
            )

            # Calculate 24h metrics
            cutoff_time = datetime.now(tz=datetime.timezone.utc) - timedelta(hours=24)
            recent_requests = [
                req
                for req in self.request_history
                if hasattr(req, "start_time") and req.start_time > cutoff_time
            ]

            success_rate_24h = (
                sum(1 for req in recent_requests if req.success) / len(recent_requests)
                if recent_requests
                else 0.0
            )

            avg_response_time = (
                sum(req.execution_time_seconds for req in recent_requests)
                / len(recent_requests)
                * 1000
                if recent_requests
                else 0.0
            )

            # Get top capabilities
            capability_usage = {}
            for req in recent_requests[-100:]:  # Last 100 requests
                if req.tool_results and "capabilities_used" in req.tool_results:
                    for cap in req.tool_results["capabilities_used"]:
                        capability_usage[cap] = capability_usage.get(cap, 0) + 1

            top_capabilities = sorted(
                capability_usage.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return AgenticSystemStatus(
                overall_health=overall_health,
                coordinator_status=coordinator_status,
                vector_manager_status=vector_status,
                orchestrator_status=orchestrator_status,
                active_requests=len(self.active_requests),
                total_requests_24h=len(recent_requests),
                avg_response_time_ms=avg_response_time,
                success_rate_24h=success_rate_24h,
                resource_utilization={
                    "coordinator_load": coordinator_status.get("current_load", 0.0),
                    "vector_collections": vector_status.get("active_collections", 0),
                    "tool_executions": orchestrator_status.get("active_executions", 0),
                },
                capacity_remaining={
                    "agent_slots": max(
                        0,
                        coordinator_status.get("max_agents", 10)
                        - coordinator_status.get("active_agents", 0),
                    ),
                    "tool_slots": max(
                        0,
                        orchestrator_status.get("max_executions", 10)
                        - orchestrator_status.get("active_executions", 0),
                    ),
                },
                top_capabilities=[cap for cap, _ in top_capabilities],
                optimization_opportunities=await self._identify_system_optimizations(),
            )

        except Exception as e:
            logger.error("Failed to get system status: %s", e, exc_info=True)

            return AgenticSystemStatus(
                overall_health="unknown",
                coordinator_status={"error": str(e)},
                vector_manager_status={"error": str(e)},
                orchestrator_status={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Cleanup system resources."""
        try:
            # Cancel active requests
            for request_id in list(self.active_requests.keys()):
                logger.warning("Cancelling active request %s", request_id)

            # Cleanup subsystems
            await self.coordinator.cleanup()
            await self.vector_manager.cleanup()

            self._initialized = False
            logger.info("UnifiedAgenticSystem cleaned up")

        except Exception as e:
            logger.error("Error during cleanup: %s", e, exc_info=True)

    # Private implementation methods

    async def _prepare_vector_environment(
        self, request: UnifiedAgentRequest
    ) -> dict[str, Any]:
        """Prepare vector database environment for the request."""
        vector_results = {"collections_created": [], "optimizations_applied": []}

        try:
            if request.vector_requirements:
                # Create or optimize collections as needed
                for collection_req in request.vector_requirements.get(
                    "collections", []
                ):
                    config = AgentCollectionConfig(
                        collection_name=collection_req["name"],
                        vector_size=collection_req.get("vector_size", 1536),
                        distance_metric=collection_req.get("distance", "Cosine"),
                        optimization_strategy=OptimizationStrategy(
                            collection_req.get("optimization", "balanced")
                        ),
                    )

                    collection_name = await self.vector_manager.create_agent_collection(
                        config
                    )
                    vector_results["collections_created"].append(collection_name)

                # Apply optimizations
                if request.collection_preferences:
                    for collection_name in vector_results["collections_created"]:
                        opt_result = await self.vector_manager.optimize_collection(
                            collection_name,
                            OptimizationStrategy(request.optimization_target),
                        )
                        vector_results["optimizations_applied"].append(opt_result)

            return vector_results

        except Exception as e:
            logger.exception("Vector environment preparation failed")
            return {"error": str(e)}

    async def _compose_tool_chain(
        self, request: UnifiedAgentRequest, vector_results: dict[str, Any]
    ) -> ToolOrchestrationPlan:
        """Compose tool orchestration plan for the request."""
        try:
            constraints = {
                "timeout_seconds": request.max_execution_time_seconds,
                "quality_threshold": request.quality_threshold,
                "max_parallel_tools": request.max_parallel_agents,
            }

            preferences = {"optimize_for": request.optimization_target}

            # Add vector context to constraints
            if vector_results.get("collections_created"):
                constraints["vector_collections"] = vector_results[
                    "collections_created"
                ]

            return await self.orchestrator.compose_tool_chain(
                goal=request.goal, constraints=constraints, preferences=preferences
            )

        except Exception as e:
            logger.exception("Tool chain composition failed")
            raise

    async def _create_coordination_tasks(
        self,
        request: UnifiedAgentRequest,
        vector_results: dict[str, Any],
        orchestration_plan: ToolOrchestrationPlan,
    ) -> list[TaskDefinition]:
        """Create coordination tasks from orchestration plan."""
        try:
            tasks = []

            # Convert orchestration nodes to coordination tasks
            for node in orchestration_plan.nodes:
                task_context = {
                    "tool_id": node.tool_id,
                    "execution_mode": node.execution_mode.value,
                    "input_mapping": node.input_mapping,
                    "output_mapping": node.output_mapping,
                }

                if vector_results.get("collections_created"):
                    task_context["vector_collections"] = vector_results[
                        "collections_created"
                    ]

                task = TaskDefinition(
                    task_id=node.node_id,
                    description=f"Execute tool {node.tool_id}",
                    priority=1.0,  # Will be adjusted based on tool priority
                    estimated_duration_seconds=30.0,  # Default estimation
                    dependencies=node.depends_on,
                    context=task_context,
                )

                tasks.append(task)

            return tasks

        except Exception as e:
            logger.exception("Coordination task creation failed")
            raise

    async def _execute_unified_workflow(
        self,
        request: UnifiedAgentRequest,
        coordination_tasks: list[TaskDefinition],
        orchestration_plan: ToolOrchestrationPlan,
    ) -> dict[str, Any]:
        """Execute the unified workflow."""
        try:
            # Determine coordination strategy
            if request.coordination_strategy == "parallel":
                strategy = CoordinationStrategy.PARALLEL
            elif request.coordination_strategy == "sequential":
                strategy = CoordinationStrategy.SEQUENTIAL
            elif request.coordination_strategy == "hierarchical":
                strategy = CoordinationStrategy.HIERARCHICAL
            else:  # adaptive
                strategy = CoordinationStrategy.ADAPTIVE

            # Execute coordination
            coordination_result = await self.coordinator.execute_coordinated_workflow(
                tasks=coordination_tasks,
                strategy=strategy,
                timeout_seconds=request.max_execution_time_seconds,
            )

            # Execute tool orchestration in parallel
            tool_result = await self.orchestrator.execute_tool_chain(
                plan=orchestration_plan,
                input_data=request.context,
                timeout_seconds=request.max_execution_time_seconds,
            )

            return {
                "coordination_results": coordination_result,
                "tool_results": tool_result,
                "integration_success": True,
            }

        except Exception as e:
            logger.exception("Unified workflow execution failed")
            return {"error": str(e), "integration_success": False}

    async def _integrate_results(
        self,
        request: UnifiedAgentRequest,
        vector_results: dict[str, Any],
        execution_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Integrate results from all subsystems."""
        try:
            integrated = {
                "goal": request.goal,
                "request_id": request.request_id,
                "vector_operations": vector_results,
                "agent_coordination": execution_results.get("coordination_results", {}),
                "tool_orchestration": execution_results.get("tool_results", {}),
                "integration_metadata": {
                    "systems_involved": [
                        "coordinator",
                        "vector_manager",
                        "orchestrator",
                    ],
                    "integration_time": datetime.now(
                        tz=datetime.timezone.utc
                    ).isoformat(),
                    "success": execution_results.get("integration_success", False),
                },
            }

            # Extract final results
            if execution_results.get("tool_results", {}).get("results"):
                integrated["final_results"] = execution_results["tool_results"][
                    "results"
                ]

            if execution_results.get("coordination_results", {}).get("results"):
                integrated["coordination_outputs"] = execution_results[
                    "coordination_results"
                ]["results"]

            return integrated

        except Exception as e:
            logger.exception("Results integration failed")
            return {"error": str(e)}

    async def _calculate_quality_metrics(
        self,
        request: UnifiedAgentRequest,
        results: dict[str, Any],
        execution_time: float,
    ) -> dict[str, Any]:
        """Calculate quality metrics for the execution."""
        try:
            # Base quality calculation
            quality_score = 0.8  # Default
            confidence = 0.7  # Default
            completeness = 0.9  # Default

            # Adjust based on results
            if results.get("integration_metadata", {}).get("success"):
                quality_score += 0.1

            if execution_time < request.max_execution_time_seconds * 0.5:
                quality_score += 0.05  # Bonus for fast execution

            if results.get("final_results"):
                completeness = 1.0
                confidence += 0.1

            # Performance metrics
            performance_metrics = {
                "execution_time_seconds": execution_time,
                "time_efficiency": min(
                    1.0, request.max_execution_time_seconds / execution_time
                ),
                "quality_threshold_met": quality_score >= request.quality_threshold,
                "systems_utilized": len(
                    [
                        s
                        for s in ["coordinator", "vector_manager", "orchestrator"]
                        if s
                        in results.get("integration_metadata", {}).get(
                            "systems_involved", []
                        )
                    ]
                ),
            }

            # Resource usage (estimated)
            resource_usage = {
                "cpu_time_seconds": execution_time * 0.8,  # Estimated
                "memory_mb": 128.0,  # Estimated
                "api_calls": len(results.get("coordination_outputs", {})),
                "vector_operations": len(results.get("vector_operations", {})),
            }

            return {
                "quality_score": min(1.0, quality_score),
                "confidence": min(1.0, confidence),
                "completeness": completeness,
                "performance_metrics": performance_metrics,
                "resource_usage": resource_usage,
            }

        except Exception as e:
            logger.exception("Quality metrics calculation failed")
            return {
                "quality_score": 0.5,
                "confidence": 0.5,
                "completeness": 0.5,
                "performance_metrics": {},
                "resource_usage": {},
            }

    async def _generate_optimization_recommendations(
        self,
        request: UnifiedAgentRequest,
        execution_results: dict[str, Any],
        quality_metrics: dict[str, Any],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        try:
            # Performance-based recommendations
            if (
                quality_metrics.get("performance_metrics", {}).get(
                    "execution_time_seconds", 0
                )
                > request.max_execution_time_seconds * 0.8
            ):
                recommendations.append(
                    "Consider increasing timeout or optimizing for speed"
                )

            if quality_metrics.get("quality_score", 0) < request.quality_threshold:
                recommendations.append(
                    "Quality threshold not met - consider quality optimization"
                )

            # System-specific recommendations
            if not execution_results.get("integration_success"):
                recommendations.append(
                    "Integration issues detected - review system coordination"
                )

            if len(quality_metrics.get("resource_usage", {})) > 0:
                high_resource_usage = any(
                    v > 100
                    for v in quality_metrics["resource_usage"].values()
                    if isinstance(v, int | float)
                )
                if high_resource_usage:
                    recommendations.append(
                        "High resource usage detected - consider resource optimization"
                    )

            # Default recommendation if none generated
            if not recommendations:
                recommendations.append("System performing optimally")

            return recommendations

        except Exception as e:
            logger.exception("Recommendation generation failed")
            return ["Unable to generate recommendations"]

    async def _register_default_tools(self) -> None:
        """Register default tools with the orchestrator."""
        try:
            # Search tool
            search_tool = ToolDefinition(
                tool_id="vector_search",
                name="Vector Search",
                description="Semantic vector search capability",
                capabilities={ToolCapability.SEARCH},
                priority=ToolPriority.HIGH,
                estimated_duration_ms=500.0,
                resource_requirements={"cpu": 0.2, "memory": 64.0},
                dependencies=[],
                success_rate=0.95,
            )

            # Analysis tool
            analysis_tool = ToolDefinition(
                tool_id="content_analysis",
                name="Content Analysis",
                description="Content analysis and processing",
                capabilities={ToolCapability.ANALYSIS},
                priority=ToolPriority.NORMAL,
                estimated_duration_ms=1000.0,
                resource_requirements={"cpu": 0.4, "memory": 128.0},
                dependencies=[],
                success_rate=0.90,
            )

            # Generation tool
            generation_tool = ToolDefinition(
                tool_id="content_generation",
                name="Content Generation",
                description="AI-powered content generation",
                capabilities={ToolCapability.GENERATION},
                priority=ToolPriority.NORMAL,
                estimated_duration_ms=2000.0,
                resource_requirements={"cpu": 0.6, "memory": 256.0},
                dependencies=[],
                success_rate=0.85,
            )

            # Register tools
            await self.orchestrator.register_tool(search_tool)
            await self.orchestrator.register_tool(analysis_tool)
            await self.orchestrator.register_tool(generation_tool)

            logger.info("Default tools registered with orchestrator")

        except Exception as e:
            logger.exception("Failed to register default tools")

    async def _update_system_metrics(self, response: UnifiedAgentResponse) -> None:
        """Update system performance metrics."""
        try:
            self.performance_metrics["total_requests"] += 1

            if response.success:
                self.performance_metrics["successful_requests"] += 1

            # Update moving average of execution time
            current_avg = self.performance_metrics["avg_execution_time"]
            new_avg = (
                current_avg * (self.performance_metrics["total_requests"] - 1)
                + response.execution_time_seconds
            ) / self.performance_metrics["total_requests"]
            self.performance_metrics["avg_execution_time"] = new_avg

            # Track 24h requests
            response.start_time = datetime.now(
                tz=datetime.timezone.utc
            )  # Add timestamp
            self.performance_metrics["last_24h_requests"].append(response)

            # Keep only last 24 hours
            cutoff = datetime.now(tz=datetime.timezone.utc) - timedelta(hours=24)
            self.performance_metrics["last_24h_requests"] = [
                req
                for req in self.performance_metrics["last_24h_requests"]
                if hasattr(req, "start_time") and req.start_time > cutoff
            ]

            # Store in history (keep last 1000)
            self.request_history.append(response)
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]

        except Exception as e:
            logger.exception("Failed to update system metrics")

    async def _identify_system_optimizations(self) -> list[str]:
        """Identify system-wide optimization opportunities."""
        try:
            optimizations = []

            # Check recent performance
            if len(self.request_history) > 10:
                recent_requests = self.request_history[-10:]
                avg_time = sum(r.execution_time_seconds for r in recent_requests) / len(
                    recent_requests
                )
                success_rate = sum(1 for r in recent_requests if r.success) / len(
                    recent_requests
                )

                if avg_time > 60.0:  # More than 1 minute average
                    optimizations.append(
                        "High average execution time - consider performance tuning"
                    )

                if success_rate < 0.9:  # Less than 90% success rate
                    optimizations.append(
                        "Low success rate - review error handling and fallbacks"
                    )

            # Resource utilization checks
            if len(self.active_requests) > 8:  # High load
                optimizations.append(
                    "High concurrent load - consider scaling resources"
                )

            # Default message if no optimizations needed
            if not optimizations:
                optimizations.append("System operating efficiently")

            return optimizations

        except Exception as e:
            logger.exception("Failed to identify optimizations")
            return ["Unable to analyze optimizations"]
