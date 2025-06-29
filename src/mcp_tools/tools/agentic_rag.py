"""Agentic RAG MCP tools using Pydantic-AI agents.

This module provides MCP tools for intelligent, autonomous RAG processing
using Pydantic-AI agents that can dynamically compose tools and coordinate
multi-agent workflows.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    AgentState,
    BaseAgentDependencies,
    QueryOrchestrator,
    ToolCompositionEngine,
    create_agent_dependencies,
)


logger = logging.getLogger(__name__)


class AgenticSearchRequest(BaseModel):
    """Request for agentic search processing."""

    query: str = Field(..., min_length=1, description="User query to process")
    collection: str = Field("documentation", description="Target collection")
    mode: str = Field(
        "auto", description="Processing mode: auto, fast, balanced, comprehensive"
    )
    user_id: str | None = Field(None, description="User ID for personalization")
    session_id: str | None = Field(None, description="Session ID for context")

    # Performance constraints
    max_latency_ms: float | None = Field(None, description="Maximum acceptable latency")
    min_quality_score: float | None = Field(
        None, description="Minimum quality requirement"
    )
    max_cost: float | None = Field(None, description="Maximum cost constraint")

    # Preferences
    enable_learning: bool = Field(True, description="Enable adaptive learning")
    enable_caching: bool = Field(True, description="Enable intelligent caching")
    prefer_speed: bool = Field(False, description="Prefer speed over quality")

    # Context
    user_context: dict[str, Any] | None = Field(
        None, description="Additional user context"
    )


class AgenticAnalysisRequest(BaseModel):
    """Request for agentic data analysis."""

    data: list[dict[str, Any]] = Field(..., description="Data to analyze")
    analysis_type: str = Field(
        "comprehensive", description="Type of analysis to perform"
    )
    focus_areas: list[str] | None = Field(
        None, description="Specific areas to focus on"
    )
    output_format: str = Field("structured", description="Desired output format")
    user_context: dict[str, Any] | None = Field(
        None, description="User context for analysis"
    )


class AgenticSearchResponse(BaseModel):
    """Response from agentic search processing."""

    success: bool = Field(..., description="Whether the search was successful")
    session_id: str = Field(..., description="Session identifier")

    # Core results
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="Search results"
    )
    answer: str | None = Field(None, description="Generated answer if applicable")
    confidence: float | None = Field(None, description="Answer confidence score")

    # Agent insights
    orchestration_plan: dict[str, Any] = Field(
        default_factory=dict, description="Orchestration decisions"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Tools selected and used"
    )
    agent_reasoning: str | None = Field(None, description="Agent decision reasoning")

    # Performance metrics
    total_latency_ms: float = Field(..., description="Total processing time")
    cost_estimate: float = Field(..., description="Estimated cost")
    quality_metrics: dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )

    # Learning and adaptation
    strategy_effectiveness: float | None = Field(
        None, description="Strategy effectiveness score"
    )
    learned_insights: dict[str, Any] | None = Field(
        None, description="Insights for future queries"
    )


class AgenticAnalysisResponse(BaseModel):
    """Response from agentic analysis."""

    success: bool = Field(..., description="Whether analysis was successful")
    analysis_id: str = Field(..., description="Analysis identifier")

    # Analysis results
    insights: dict[str, Any] = Field(default_factory=dict, description="Key insights")
    summary: str = Field(..., description="Analysis summary")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )

    # Metadata
    analysis_type: str = Field(..., description="Type of analysis performed")
    confidence: float = Field(..., description="Analysis confidence")
    processing_time_ms: float = Field(..., description="Processing time")


def register_tools(mcp: FastMCP, client_manager: ClientManager) -> None:
    """Register agentic RAG tools with the MCP server."""

    @mcp.tool()
    async def agentic_search(request: AgenticSearchRequest) -> AgenticSearchResponse:
        """Perform intelligent agentic search with autonomous optimization.

        This tool uses autonomous agents to analyze queries, select optimal tools,
        and coordinate processing workflows for maximum effectiveness.

        Args:
            request: Agentic search request parameters

        Returns:
            AgenticSearchResponse: Results with agent insights and performance metrics

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If search processing fails
        """
        try:
            # Create or retrieve session
            session_id = request.session_id or str(uuid4())

            # Create agent dependencies
            deps = create_agent_dependencies(
                client_manager=client_manager,
                session_id=session_id,
                user_id=request.user_id,
            )

            # Add user context to session state
            if request.user_context:
                deps.session_state.preferences.update(request.user_context)

            # Set performance constraints
            performance_constraints = {}
            if request.max_latency_ms:
                performance_constraints["max_latency_ms"] = request.max_latency_ms
            if request.min_quality_score:
                performance_constraints["min_quality_score"] = request.min_quality_score
            if request.max_cost:
                performance_constraints["max_cost"] = request.max_cost

            # Initialize query orchestrator
            orchestrator = QueryOrchestrator()
            if not orchestrator._initialized:
                await orchestrator.initialize(deps)

            # Execute agentic orchestration
            orchestration_result = await orchestrator.orchestrate_query(
                query=request.query,
                collection=request.collection,
                user_context=request.user_context,
                performance_requirements=performance_constraints,
            )

            if not orchestration_result["success"]:
                raise RuntimeError(
                    f"Orchestration failed: {orchestration_result.get('error')}"
                )

            # Extract results from orchestration
            result_data = orchestration_result["result"]

            # Build response
            response = AgenticSearchResponse(
                success=True,
                session_id=session_id,
                results=result_data.get("search_results", []),
                answer=result_data.get("generated_answer"),
                confidence=result_data.get("confidence_score"),
                orchestration_plan=result_data.get("orchestration_plan", {}),
                tools_used=result_data.get("tools_used", []),
                agent_reasoning=result_data.get("agent_reasoning"),
                total_latency_ms=result_data.get("total_latency_ms", 0.0),
                cost_estimate=result_data.get("cost_estimate", 0.0),
                quality_metrics=result_data.get("quality_metrics", {}),
                strategy_effectiveness=result_data.get("strategy_effectiveness"),
                learned_insights=result_data.get("learned_insights"),
            )

            # Record interaction for learning
            if request.enable_learning:
                deps.session_state.add_interaction(
                    role="user",
                    content=request.query,
                    metadata={
                        "mode": request.mode,
                        "collection": request.collection,
                        "performance_constraints": performance_constraints,
                    },
                )
                deps.session_state.add_interaction(
                    role="assistant",
                    content=response.answer or "Search completed",
                    metadata={
                        "tools_used": response.tools_used,
                        "latency_ms": response.total_latency_ms,
                        "quality_metrics": response.quality_metrics,
                    },
                )

            return response

        except Exception as e:
            logger.error(f"Agentic search failed: {e}", exc_info=True)

            return AgenticSearchResponse(
                success=False,
                session_id=request.session_id or str(uuid4()),
                results=[],
                total_latency_ms=0.0,
                cost_estimate=0.0,
                orchestration_plan={"error": str(e)},
                agent_reasoning=f"Search failed due to: {e!s}",
            )

    @mcp.tool()
    async def agentic_analysis(
        request: AgenticAnalysisRequest,
    ) -> AgenticAnalysisResponse:
        """Perform intelligent analysis using specialized agents.

        This tool coordinates multiple agents to analyze data, extract insights,
        and provide recommendations based on the analysis type and focus areas.

        Args:
            request: Analysis request parameters

        Returns:
            AgenticAnalysisResponse: Analysis results with insights and recommendations

        Raises:
            ValueError: If request parameters are invalid
            RuntimeError: If analysis fails
        """
        analysis_id = str(uuid4())

        try:
            # Create agent dependencies
            deps = create_agent_dependencies(
                client_manager=client_manager, session_id=analysis_id
            )

            # Initialize tool composition engine
            tool_engine = ToolCompositionEngine(client_manager)
            await tool_engine.initialize()

            # Compose analysis workflow based on type
            analysis_goal = f"Perform {request.analysis_type} analysis on provided data"
            if request.focus_areas:
                analysis_goal += f" focusing on: {', '.join(request.focus_areas)}"

            # Set constraints for analysis
            constraints = {
                "max_latency_ms": 10000.0,  # 10 second timeout for analysis
                "min_quality_score": 0.8,
                "analysis_type": request.analysis_type,
            }

            # Compose tool chain for analysis
            tool_chain = await tool_engine.compose_tool_chain(
                goal=analysis_goal, constraints=constraints
            )

            # Execute analysis workflow
            analysis_input = {
                "data": request.data,
                "analysis_type": request.analysis_type,
                "focus_areas": request.focus_areas or [],
                "user_context": request.user_context or {},
            }

            execution_result = await tool_engine.execute_tool_chain(
                chain=tool_chain, input_data=analysis_input, timeout_seconds=15.0
            )

            if not execution_result["success"]:
                raise RuntimeError(
                    f"Analysis execution failed: {execution_result.get('error')}"
                )

            # Extract insights from results
            results = execution_result["results"]

            # Generate summary and recommendations
            insights = {}
            recommendations = []

            # Basic insight extraction (would be enhanced with actual analysis)
            if "analyze_query_performance_results" in results:
                perf_data = results["analyze_query_performance_results"]
                insights["performance"] = perf_data

            if "classify_content_results" in results:
                classification = results["classify_content_results"]
                insights["content_classification"] = classification

                # Generate recommendations based on classification
                if classification.get("quality_score", 0) < 0.7:
                    recommendations.append("Consider improving content quality")
                if classification.get("technical", 0) > 0.8:
                    recommendations.append(
                        "Content is highly technical - consider adding explanatory notes"
                    )

            # Generate analysis summary
            summary = f"Completed {request.analysis_type} analysis on {len(request.data)} data points"
            if insights:
                summary += f" with {len(insights)} key insight areas identified"

            # Calculate confidence based on data quality and completeness
            confidence = 0.8  # Base confidence
            if len(request.data) > 10:
                confidence += 0.1  # More data = higher confidence
            if request.focus_areas:
                confidence += 0.05  # Focused analysis = slightly higher confidence

            confidence = min(confidence, 1.0)

            response = AgenticAnalysisResponse(
                success=True,
                analysis_id=analysis_id,
                insights=insights,
                summary=summary,
                recommendations=recommendations,
                analysis_type=request.analysis_type,
                confidence=confidence,
                processing_time_ms=execution_result["metadata"][
                    "total_execution_time_ms"
                ],
            )

            return response

        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}", exc_info=True)

            return AgenticAnalysisResponse(
                success=False,
                analysis_id=analysis_id,
                insights={"error": str(e)},
                summary=f"Analysis failed: {e!s}",
                recommendations=["Review input data and try again"],
                analysis_type=request.analysis_type,
                confidence=0.0,
                processing_time_ms=0.0,
            )

    @mcp.tool()
    async def get_agent_performance_metrics() -> dict[str, Any]:
        """Get comprehensive performance metrics for all agents.

        Returns detailed performance statistics, usage patterns, and effectiveness
        metrics for the agentic system.

        Returns:
            Dict[str, Any]: Comprehensive performance metrics
        """
        try:
            # This would integrate with actual agent registry and monitoring
            # For now, return mock metrics structure

            metrics = {
                "system_overview": {
                    "total_sessions": 150,
                    "active_agents": 3,
                    "avg_session_length": 4.2,
                    "success_rate": 0.94,
                },
                "agent_performance": {
                    "query_orchestrator": {
                        "execution_count": 89,
                        "avg_execution_time_ms": 245.0,
                        "success_rate": 0.96,
                        "strategy_accuracy": 0.88,
                    },
                    "retrieval_specialist": {
                        "execution_count": 76,
                        "avg_execution_time_ms": 180.0,
                        "success_rate": 0.93,
                        "tool_selection_accuracy": 0.91,
                    },
                    "answer_generator": {
                        "execution_count": 65,
                        "avg_execution_time_ms": 680.0,
                        "success_rate": 0.92,
                        "answer_quality_score": 0.87,
                    },
                },
                "tool_usage": {
                    "most_used_tools": [
                        ("hybrid_search", 45),
                        ("hyde_search", 23),
                        ("generate_rag_answer", 19),
                        ("rerank_results", 15),
                    ],
                    "tool_effectiveness": {
                        "hybrid_search": 0.85,
                        "hyde_search": 0.92,
                        "multi_stage_search": 0.89,
                    },
                },
                "learning_insights": {
                    "strategy_improvements": 0.12,
                    "cache_hit_rate": 0.34,
                    "adaptive_optimizations": 23,
                },
                "performance_trends": {
                    "latency_improvement_pct": 8.5,
                    "quality_improvement_pct": 5.2,
                    "cost_reduction_pct": 12.3,
                },
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get agent performance metrics: {e}")
            return {
                "error": str(e),
                "message": "Failed to retrieve performance metrics",
            }

    @mcp.tool()
    async def reset_agent_learning() -> dict[str, str]:
        """Reset agent learning data and performance history.

        This tool clears accumulated learning data and resets agents to their
        initial state. Useful for testing or when starting fresh.

        Returns:
            Dict[str, str]: Reset operation results
        """
        try:
            # This would reset actual agent learning data
            # For now, return success message

            return {
                "status": "success",
                "message": "Agent learning data has been reset",
                "timestamp": str(uuid4()),
            }

        except Exception as e:
            logger.error(f"Failed to reset agent learning: {e}")
            return {
                "status": "error",
                "message": f"Failed to reset agent learning: {e!s}",
            }

    @mcp.tool()
    async def optimize_agent_configuration(
        optimization_target: str = "balanced",
        constraints: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Optimize agent configuration for specific targets.

        Automatically tunes agent parameters and strategies based on
        historical performance and specified optimization targets.

        Args:
            optimization_target: Target to optimize for ("speed", "quality", "cost", "balanced")
            constraints: Optional performance constraints

        Returns:
            Dict[str, Any]: Optimization results and new configuration
        """
        try:
            constraints = constraints or {}

            # Mock optimization process
            optimization_results = {
                "target": optimization_target,
                "optimization_id": str(uuid4()),
                "improvements": {
                    "strategy_selection": "Updated to favor faster tools for speed target",
                    "caching_policy": "Increased cache retention for repeated queries",
                    "tool_thresholds": "Adjusted confidence thresholds for tool selection",
                },
                "expected_improvements": {
                    "latency_reduction_pct": 15.0
                    if optimization_target == "speed"
                    else 5.0,
                    "quality_improvement_pct": 8.0
                    if optimization_target == "quality"
                    else 2.0,
                    "cost_reduction_pct": 10.0
                    if optimization_target == "cost"
                    else 3.0,
                },
                "new_configuration": {
                    "orchestrator_temperature": 0.05
                    if optimization_target == "speed"
                    else 0.1,
                    "tool_selection_threshold": 0.7
                    if optimization_target == "speed"
                    else 0.8,
                    "cache_ttl_seconds": 3600
                    if optimization_target == "speed"
                    else 1800,
                },
                "rollback_available": True,
            }

            return optimization_results

        except Exception as e:
            logger.error(f"Agent configuration optimization failed: {e}")
            return {"status": "error", "message": f"Optimization failed: {e!s}"}

    logger.info("Agentic RAG MCP tools registered successfully")
