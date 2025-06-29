"""Migration utility for transitioning from ToolCompositionEngine to NativeToolAgent.

This module provides a gradual migration path that allows both systems to coexist
during the transition, with automatic fallback and performance comparison.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .core import BaseAgentDependencies
from .native_tool_agent import ExecutionResult, NativeToolAgent
from .tool_composition import ToolCompositionEngine


logger = logging.getLogger(__name__)


class MigrationConfig(BaseModel):
    """Configuration for the migration process."""

    # Migration strategy
    native_agent_percentage: float = Field(
        default=0.1,
        description="Percentage of requests to route to native agent (0.0-1.0)",
    )

    # Performance thresholds for automatic switching
    performance_threshold_ms: float = Field(
        default=1000.0, description="Latency threshold for preferring native agent"
    )
    quality_threshold: float = Field(
        default=0.8, description="Quality threshold for preferring native agent"
    )

    # Safety settings
    enable_fallback: bool = Field(
        default=True,
        description="Enable fallback to legacy engine on native agent failure",
    )
    enable_parallel_execution: bool = Field(
        default=False, description="Execute both engines in parallel for comparison"
    )

    # Monitoring
    collect_metrics: bool = Field(default=True)
    log_comparisons: bool = Field(default=True)


class ComparisonResult(BaseModel):
    """Result from comparing native agent vs legacy engine."""

    native_success: bool
    legacy_success: bool

    native_latency_ms: float
    legacy_latency_ms: float

    native_result: dict[str, Any] = Field(default_factory=dict)
    legacy_result: dict[str, Any] = Field(default_factory=dict)

    performance_improvement: float = Field(
        description="Percentage improvement of native vs legacy"
    )

    recommended_engine: str = Field(
        description="Recommended engine based on performance"
    )

    comparison_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: float = Field(default_factory=time.time)


class HybridToolOrchestrator:
    """Hybrid orchestrator that manages transition from legacy to native agent.

    This orchestrator intelligently routes requests between the legacy ToolCompositionEngine
    and the new NativeToolAgent based on configuration and performance metrics.
    """

    def __init__(self, client_manager, config: MigrationConfig = None):
        """Initialize the hybrid orchestrator.

        Args:
            client_manager: Client manager for accessing services
            config: Migration configuration
        """
        self.client_manager = client_manager
        self.config = config or MigrationConfig()

        # Initialize both systems
        self.legacy_engine = ToolCompositionEngine(client_manager)
        self.native_agent = NativeToolAgent()

        # Performance tracking
        self.execution_history: list[ComparisonResult] = []
        self.performance_stats = {
            "native_agent": {"executions": 0, "successes": 0, "total_latency": 0.0},
            "legacy_engine": {"executions": 0, "successes": 0, "total_latency": 0.0},
        }

        self._initialized = False

    async def initialize(self, deps: BaseAgentDependencies) -> None:
        """Initialize both orchestration systems."""
        if self._initialized:
            return

        # Initialize legacy engine
        await self.legacy_engine.initialize()

        # Initialize native agent
        await self.native_agent.initialize(deps)

        self._initialized = True
        logger.info("Hybrid tool orchestrator initialized successfully")

    async def orchestrate(
        self,
        goal: str,
        constraints: dict[str, Any],
        deps: BaseAgentDependencies,
        force_engine: str | None = None,
    ) -> dict[str, Any]:
        """Main orchestration method with intelligent routing.

        Args:
            goal: High-level goal description
            constraints: Performance and quality constraints
            deps: Agent dependencies
            force_engine: Force specific engine ("native" or "legacy")

        Returns:
            Orchestration result with metadata about which engine was used
        """
        if not self._initialized:
            await self.initialize(deps)

        # Determine which engine to use
        selected_engine = force_engine or self._select_engine(goal, constraints)

        start_time = time.time()

        try:
            if selected_engine == "native":
                result = await self._execute_native(goal, constraints, deps)
            elif selected_engine == "legacy":
                result = await self._execute_legacy(goal, constraints, deps)
            elif selected_engine == "parallel":
                result = await self._execute_parallel_comparison(
                    goal, constraints, deps
                )
            else:
                # Fallback to legacy
                result = await self._execute_legacy(goal, constraints, deps)

            # Add metadata about engine selection
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["selected_engine"] = selected_engine
            result["metadata"]["orchestrator"] = "hybrid"
            result["metadata"]["migration_config"] = self.config.model_dump()

            return result

        except Exception as e:
            logger.error(f"Orchestration failed with {selected_engine} engine: {e}")

            # Try fallback if enabled
            if self.config.enable_fallback and selected_engine == "native":
                logger.info("Falling back to legacy engine")
                return await self._execute_legacy(goal, constraints, deps)
            raise

    def _select_engine(self, goal: str, constraints: dict[str, Any]) -> str:
        """Intelligently select which engine to use."""

        # Check if parallel comparison is enabled
        if self.config.enable_parallel_execution:
            return "parallel"

        # Use percentage-based routing
        import random

        if random.random() < self.config.native_agent_percentage:
            return "native"

        # Check performance history for intelligent selection
        if len(self.execution_history) > 10:
            recent_comparisons = self.execution_history[-10:]
            native_wins = sum(
                1 for comp in recent_comparisons if comp.recommended_engine == "native"
            )

            if native_wins >= 7:  # Native agent performing well
                return "native"

        return "legacy"

    async def _execute_native(
        self, goal: str, constraints: dict[str, Any], deps: BaseAgentDependencies
    ) -> dict[str, Any]:
        """Execute using the native Pydantic-AI agent."""

        result = await self.native_agent.orchestrate_tools(goal, constraints, deps)

        # Update performance stats
        self.performance_stats["native_agent"]["executions"] += 1
        self.performance_stats["native_agent"]["total_latency"] += (
            result.actual_latency_ms
        )

        if result.success:
            self.performance_stats["native_agent"]["successes"] += 1

        # Convert to expected format
        return {
            "success": result.success,
            "execution_id": result.plan_id,
            "results": result.results,
            "error": result.error_details,
            "metadata": {
                "total_execution_time_ms": result.actual_latency_ms,
                "tools_used": result.tools_executed,
                "engine": "native_agent",
                "performance_metrics": result.performance_metrics,
                "learned_insights": result.learned_insights,
            },
        }

    async def _execute_legacy(
        self, goal: str, constraints: dict[str, Any], deps: BaseAgentDependencies
    ) -> dict[str, Any]:
        """Execute using the legacy ToolCompositionEngine."""

        # Convert to legacy format
        available_tools = None  # Use all available tools
        chain = await self.legacy_engine.compose_tool_chain(
            goal, constraints, available_tools
        )

        # Execute the chain
        input_data = {"query": goal, "collection": "documentation"}
        result = await self.legacy_engine.execute_tool_chain(chain, input_data)

        # Update performance stats
        self.performance_stats["legacy_engine"]["executions"] += 1
        latency = result["metadata"]["total_execution_time_ms"]
        self.performance_stats["legacy_engine"]["total_latency"] += latency

        if result["success"]:
            self.performance_stats["legacy_engine"]["successes"] += 1

        # Add engine metadata
        result["metadata"]["engine"] = "legacy_engine"

        return result

    async def _execute_parallel_comparison(
        self, goal: str, constraints: dict[str, Any], deps: BaseAgentDependencies
    ) -> dict[str, Any]:
        """Execute both engines in parallel for performance comparison."""

        # Execute both in parallel
        native_task = self._execute_native(goal, constraints, deps)
        legacy_task = self._execute_legacy(goal, constraints, deps)

        native_result, legacy_result = await asyncio.gather(
            native_task, legacy_task, return_exceptions=True
        )

        # Handle exceptions
        native_success = not isinstance(native_result, Exception)
        legacy_success = not isinstance(legacy_result, Exception)

        if isinstance(native_result, Exception):
            native_result = {
                "success": False,
                "error": str(native_result),
                "metadata": {"total_execution_time_ms": 0.0},
            }

        if isinstance(legacy_result, Exception):
            legacy_result = {
                "success": False,
                "error": str(legacy_result),
                "metadata": {"total_execution_time_ms": 0.0},
            }

        # Create comparison
        comparison = self._create_comparison(
            native_result, legacy_result, native_success, legacy_success
        )

        # Store comparison for learning
        self.execution_history.append(comparison)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        # Log comparison if enabled
        if self.config.log_comparisons:
            logger.info(
                f"Engine comparison: {comparison.recommended_engine} recommended "
                f"(native: {comparison.native_latency_ms:.1f}ms, "
                f"legacy: {comparison.legacy_latency_ms:.1f}ms)"
            )

        # Return the better result
        if comparison.recommended_engine == "native" and native_success:
            native_result["metadata"]["comparison"] = comparison.model_dump()
            return native_result
        legacy_result["metadata"]["comparison"] = comparison.model_dump()
        return legacy_result

    def _create_comparison(
        self,
        native_result: dict[str, Any],
        legacy_result: dict[str, Any],
        native_success: bool,
        legacy_success: bool,
    ) -> ComparisonResult:
        """Create a comparison result from parallel execution."""

        native_latency = native_result["metadata"]["total_execution_time_ms"]
        legacy_latency = legacy_result["metadata"]["total_execution_time_ms"]

        # Calculate performance improvement
        if legacy_latency > 0:
            improvement = ((legacy_latency - native_latency) / legacy_latency) * 100
        else:
            improvement = 0.0

        # Determine recommended engine
        recommended = "legacy"  # Default

        if native_success and not legacy_success:
            recommended = "native"
        elif native_success and legacy_success:
            # Both succeeded, compare performance
            if (
                native_latency < legacy_latency * 0.8
                or native_latency < self.config.performance_threshold_ms
            ):  # Native is significantly faster
                recommended = "native"

        return ComparisonResult(
            native_success=native_success,
            legacy_success=legacy_success,
            native_latency_ms=native_latency,
            legacy_latency_ms=legacy_latency,
            native_result=native_result.get("results", {}),
            legacy_result=legacy_result.get("results", {}),
            performance_improvement=improvement,
            recommended_engine=recommended,
        )

    def get_migration_metrics(self) -> dict[str, Any]:
        """Get comprehensive migration metrics."""

        # Calculate success rates
        native_stats = self.performance_stats["native_agent"]
        legacy_stats = self.performance_stats["legacy_engine"]

        native_success_rate = (
            native_stats["successes"] / native_stats["executions"]
            if native_stats["executions"] > 0
            else 0.0
        )

        legacy_success_rate = (
            legacy_stats["successes"] / legacy_stats["executions"]
            if legacy_stats["executions"] > 0
            else 0.0
        )

        # Calculate average latencies
        native_avg_latency = (
            native_stats["total_latency"] / native_stats["executions"]
            if native_stats["executions"] > 0
            else 0.0
        )

        legacy_avg_latency = (
            legacy_stats["total_latency"] / legacy_stats["executions"]
            if legacy_stats["executions"] > 0
            else 0.0
        )

        # Analysis of comparisons
        comparison_analysis = {}
        if self.execution_history:
            recent_comparisons = self.execution_history[-50:]
            native_recommendations = sum(
                1 for comp in recent_comparisons if comp.recommended_engine == "native"
            )

            comparison_analysis = {
                "total_comparisons": len(self.execution_history),
                "recent_native_recommendations": native_recommendations,
                "recent_native_percentage": native_recommendations
                / len(recent_comparisons),
                "avg_performance_improvement": sum(
                    comp.performance_improvement for comp in recent_comparisons
                )
                / len(recent_comparisons),
            }

        return {
            "migration_config": self.config.model_dump(),
            "performance_stats": {
                "native_agent": {
                    "executions": native_stats["executions"],
                    "success_rate": native_success_rate,
                    "avg_latency_ms": native_avg_latency,
                },
                "legacy_engine": {
                    "executions": legacy_stats["executions"],
                    "success_rate": legacy_success_rate,
                    "avg_latency_ms": legacy_avg_latency,
                },
            },
            "comparison_analysis": comparison_analysis,
            "migration_readiness": self._assess_migration_readiness(),
            "recommendations": self._generate_migration_recommendations(),
        }

    def _assess_migration_readiness(self) -> dict[str, Any]:
        """Assess readiness for full migration to native agent."""

        if len(self.execution_history) < 20:
            return {
                "ready": False,
                "reason": "Insufficient comparison data",
                "confidence": 0.0,
            }

        recent_comparisons = self.execution_history[-20:]

        # Check success rate parity
        native_successes = sum(1 for comp in recent_comparisons if comp.native_success)
        native_success_rate = native_successes / len(recent_comparisons)

        # Check performance advantage
        performance_wins = sum(
            1 for comp in recent_comparisons if comp.recommended_engine == "native"
        )
        performance_advantage = performance_wins / len(recent_comparisons)

        # Overall readiness score
        readiness_score = (native_success_rate * 0.6) + (performance_advantage * 0.4)

        ready = (
            native_success_rate >= 0.9
            and performance_advantage >= 0.6
            and readiness_score >= 0.8
        )

        return {
            "ready": ready,
            "confidence": readiness_score,
            "native_success_rate": native_success_rate,
            "performance_advantage": performance_advantage,
            "criteria_met": {
                "success_rate": native_success_rate >= 0.9,
                "performance": performance_advantage >= 0.6,
                "overall_score": readiness_score >= 0.8,
            },
        }

    def _generate_migration_recommendations(self) -> list[str]:
        """Generate recommendations for migration strategy."""

        readiness = self._assess_migration_readiness()
        recommendations = []

        if not readiness["ready"]:
            if readiness["native_success_rate"] < 0.9:
                recommendations.append(
                    "Improve native agent reliability before increasing traffic percentage"
                )

            if readiness["performance_advantage"] < 0.6:
                recommendations.append(
                    "Optimize native agent performance to achieve consistent advantages"
                )

            recommendations.append(
                f"Gradually increase native_agent_percentage from "
                f"{self.config.native_agent_percentage} to 0.5"
            )
        else:
            recommendations.extend(
                [
                    "Native agent is ready for full migration",
                    "Consider setting native_agent_percentage to 1.0",
                    "Plan deprecation timeline for legacy ToolCompositionEngine",
                    "Enable comprehensive monitoring during full migration",
                ]
            )

        return recommendations

    async def update_migration_config(self, new_config: MigrationConfig) -> None:
        """Update migration configuration."""
        self.config = new_config
        logger.info(
            f"Migration config updated: native percentage = {new_config.native_agent_percentage}"
        )
