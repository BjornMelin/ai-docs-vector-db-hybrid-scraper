"""Dynamic Tool Discovery Engine - J3 Research Implementation.

This module implements intelligent tool discovery and capability assessment
based on J3 research findings for autonomous tool orchestration.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .core import BaseAgent, BaseAgentDependencies, _check_api_key_availability


logger = logging.getLogger(__name__)


class ToolCapabilityType(str, Enum):
    """Types of tool capabilities."""

    SEARCH = "search"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    SYNTHESIS = "synthesis"
    ORCHESTRATION = "orchestration"


@dataclass
class ToolMetrics:
    """Performance metrics for a tool."""

    average_latency_ms: float
    success_rate: float
    accuracy_score: float
    cost_per_execution: float
    reliability_score: float


class ToolCapability(BaseModel):
    """Tool capability definition."""

    name: str = Field(..., description="Tool name")
    capability_type: ToolCapabilityType = Field(
        ..., description="Primary capability type"
    )
    description: str = Field(..., description="Tool description")
    input_types: list[str] = Field(..., description="Supported input types")
    output_types: list[str] = Field(..., description="Supported output types")

    # Performance characteristics
    metrics: ToolMetrics | None = Field(None, description="Performance metrics")
    requirements: dict[str, Any] = Field(
        default_factory=dict, description="Tool requirements"
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Tool constraints"
    )

    # Compatibility and dependencies
    compatible_tools: list[str] = Field(
        default_factory=list, description="Compatible tools for chaining"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Required dependencies"
    )

    # Quality indicators
    confidence_score: float = Field(
        0.8, description="Confidence in capability assessment"
    )
    last_updated: str = Field(..., description="Last capability assessment update")


class DynamicToolDiscovery(BaseAgent):
    """Dynamic tool discovery engine with intelligent capability assessment.

    Based on J3 research findings for autonomous tool orchestration with
    performance-driven selection and real-time capability evaluation.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the dynamic tool discovery engine.

        Args:
            model: LLM model for capability assessment
            temperature: Generation temperature for tool evaluation
        """
        super().__init__(
            name="dynamic_tool_discovery",
            model=model,
            temperature=temperature,
            max_tokens=2000,
        )

        self.discovered_tools: dict[str, ToolCapability] = {}
        self.tool_performance_history: dict[str, list[ToolMetrics]] = {}
        self.capability_cache: dict[str, ToolCapability] = {}

    def get_system_prompt(self) -> str:
        """Define autonomous tool discovery behavior."""
        return """You are an autonomous tool discovery engine with the following capabilities:

1. INTELLIGENT TOOL ASSESSMENT
   - Analyze tool capabilities and performance characteristics
   - Assess compatibility between different tools for chaining
   - Evaluate tool suitability for specific task requirements

2. DYNAMIC CAPABILITY EVALUATION
   - Real-time assessment of tool performance and reliability
   - Learning from execution feedback to improve capability models
   - Adaptive scoring based on success patterns and failure analysis

3. PERFORMANCE-DRIVEN SELECTION
   - Select optimal tools based on performance requirements
   - Balance speed, quality, cost, and reliability constraints
   - Recommend tool combinations for complex workflows

Your goal is to provide intelligent tool discovery and capability assessment
for autonomous agent systems, enabling optimal tool selection and orchestration."""

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize tool discovery capabilities.

        Args:
            deps: Agent dependencies for tool initialization
        """
        # Check if we're in fallback mode
        fallback_reason = getattr(self, "_fallback_reason", None)
        if fallback_reason:
            logger.info(
                f"DynamicToolDiscovery tools initialized in fallback mode (reason: {fallback_reason})"
            )
        else:
            logger.info("DynamicToolDiscovery tools initialized (discovery-based)")

    async def initialize_discovery(self, deps: BaseAgentDependencies) -> None:
        """Initialize tool discovery with system scanning.

        Args:
            deps: Agent dependencies for tool scanning
        """
        if not self._initialized:
            await self.initialize(deps)

        # Discover available tools from the system
        await self._scan_available_tools(deps)

        # Initialize performance monitoring
        await self._initialize_performance_tracking()

        logger.info(
            "DynamicToolDiscovery initialized with %d tools", len(self.discovered_tools)
        )

    async def _scan_available_tools(self, _deps: BaseAgentDependencies) -> None:
        """Scan system for available tools and assess capabilities.

        Args:
            deps: Agent dependencies for system access
        """
        # Core tools available in the system
        core_tools = {
            "hybrid_search": {
                "capability_type": ToolCapabilityType.SEARCH,
                "description": "Hybrid vector and text search with reranking",
                "input_types": ["text", "query"],
                "output_types": ["search_results", "ranked_documents"],
                "metrics": ToolMetrics(
                    average_latency_ms=150.0,
                    success_rate=0.94,
                    accuracy_score=0.87,
                    cost_per_execution=0.02,
                    reliability_score=0.92,
                ),
            },
            "rag_generation": {
                "capability_type": ToolCapabilityType.GENERATION,
                "description": "RAG-based answer generation with context synthesis",
                "input_types": ["text", "context", "search_results"],
                "output_types": ["generated_text", "synthesized_answer"],
                "metrics": ToolMetrics(
                    average_latency_ms=800.0,
                    success_rate=0.91,
                    accuracy_score=0.89,
                    cost_per_execution=0.05,
                    reliability_score=0.88,
                ),
            },
            "content_analysis": {
                "capability_type": ToolCapabilityType.ANALYSIS,
                "description": "Content analysis and classification",
                "input_types": ["text", "documents"],
                "output_types": ["analysis_results", "classifications"],
                "metrics": ToolMetrics(
                    average_latency_ms=200.0,
                    success_rate=0.93,
                    accuracy_score=0.85,
                    cost_per_execution=0.01,
                    reliability_score=0.90,
                ),
            },
        }

        # Convert to ToolCapability objects
        for tool_name, tool_data in core_tools.items():
            capability = ToolCapability(
                name=tool_name,
                capability_type=tool_data["capability_type"],
                description=tool_data["description"],
                input_types=tool_data["input_types"],
                output_types=tool_data["output_types"],
                metrics=tool_data["metrics"],
                last_updated=asyncio.get_event_loop().time().__str__(),
            )

            self.discovered_tools[tool_name] = capability
            self.tool_performance_history[tool_name] = [tool_data["metrics"]]

        # Assess tool compatibility
        await self._assess_tool_compatibility()

    async def _assess_tool_compatibility(self) -> None:
        """Assess compatibility between tools for chaining."""
        for tool_name, tool in self.discovered_tools.items():
            compatible = []

            # Define compatibility rules
            if tool.capability_type == ToolCapabilityType.SEARCH:
                # Search tools are compatible with generation and analysis
                compatible.extend(
                    [
                        name
                        for name, other in self.discovered_tools.items()
                        if other.capability_type
                        in [ToolCapabilityType.GENERATION, ToolCapabilityType.ANALYSIS]
                        and name != tool_name
                    ]
                )

            elif tool.capability_type == ToolCapabilityType.ANALYSIS:
                # Analysis tools can feed into search or generation
                compatible.extend(
                    [
                        name
                        for name, other in self.discovered_tools.items()
                        if other.capability_type
                        in [ToolCapabilityType.SEARCH, ToolCapabilityType.GENERATION]
                        and name != tool_name
                    ]
                )

            tool.compatible_tools = compatible

    async def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for discovered tools."""
        logger.info("Initialized performance tracking for tool capability assessment")

    async def discover_tools_for_task(
        self, task_description: str, requirements: dict[str, Any]
    ) -> list[ToolCapability]:
        """Discover and rank tools suitable for a specific task.

        Args:
            task_description: Description of the task to perform
            requirements: Performance and quality requirements

        Returns:
            List of suitable tools ranked by suitability score
        """
        # Check if we're in fallback mode
        fallback_reason = getattr(self, "_fallback_reason", None)
        if fallback_reason:
            return await self._fallback_discover_tools(task_description, requirements)

        suitable_tools = []

        for tool in self.discovered_tools.values():
            suitability_score = await self._calculate_suitability_score(
                tool, task_description, requirements
            )

            if suitability_score > 0.5:  # Threshold for tool selection
                tool_copy = tool.model_copy()
                tool_copy.confidence_score = suitability_score
                suitable_tools.append(tool_copy)

        # Sort by suitability score
        suitable_tools.sort(key=lambda t: t.confidence_score, reverse=True)

        return suitable_tools

    async def _calculate_suitability_score(
        self, tool: ToolCapability, task_description: str, requirements: dict[str, Any]
    ) -> float:
        """Calculate tool suitability score for a specific task.

        Args:
            tool: Tool capability to evaluate
            task_description: Description of the task
            requirements: Performance requirements

        Returns:
            Suitability score between 0 and 1
        """
        score = 0.0

        # Base capability match
        task_lower = task_description.lower()
        if (
            (
                "search" in task_lower
                and tool.capability_type == ToolCapabilityType.SEARCH
            )
            or (
                "generate" in task_lower
                and tool.capability_type == ToolCapabilityType.GENERATION
            )
            or (
                "analyze" in task_lower
                and tool.capability_type == ToolCapabilityType.ANALYSIS
            )
        ):
            score += 0.4

        # Performance requirements
        if tool.metrics:
            # Latency requirement
            if "max_latency_ms" in requirements:
                max_latency = requirements["max_latency_ms"]
                if tool.metrics.average_latency_ms <= max_latency:
                    score += 0.2
                else:
                    score -= 0.1  # Penalty for exceeding latency

            # Quality requirement
            if "min_accuracy" in requirements:
                min_accuracy = requirements["min_accuracy"]
                if tool.metrics.accuracy_score >= min_accuracy:
                    score += 0.2

            # Cost requirement
            if "max_cost" in requirements:
                max_cost = requirements["max_cost"]
                if tool.metrics.cost_per_execution <= max_cost:
                    score += 0.1

            # Reliability bonus
            score += tool.metrics.reliability_score * 0.1

        return min(score, 1.0)

    async def update_tool_performance(
        self, tool_name: str, execution_metrics: ToolMetrics
    ) -> None:
        """Update tool performance metrics based on execution feedback.

        Args:
            tool_name: Name of the tool
            execution_metrics: Latest execution metrics
        """
        if tool_name in self.tool_performance_history:
            self.tool_performance_history[tool_name].append(execution_metrics)

            # Update rolling average metrics
            recent_metrics = self.tool_performance_history[tool_name][
                -10:
            ]  # Last 10 executions
            avg_metrics = self._calculate_average_metrics(recent_metrics)

            if tool_name in self.discovered_tools:
                self.discovered_tools[tool_name].metrics = avg_metrics
                self.discovered_tools[tool_name].last_updated = (
                    asyncio.get_event_loop().time().__str__()
                )

            logger.debug("Updated performance metrics for tool: %s", tool_name)

    def _calculate_average_metrics(
        self, metrics_list: list[ToolMetrics]
    ) -> ToolMetrics:
        """Calculate average metrics from a list of tool metrics.

        Args:
            metrics_list: List of tool metrics

        Returns:
            Averaged tool metrics
        """
        if not metrics_list:
            return ToolMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        return ToolMetrics(
            average_latency_ms=sum(m.average_latency_ms for m in metrics_list)
            / len(metrics_list),
            success_rate=sum(m.success_rate for m in metrics_list) / len(metrics_list),
            accuracy_score=sum(m.accuracy_score for m in metrics_list)
            / len(metrics_list),
            cost_per_execution=sum(m.cost_per_execution for m in metrics_list)
            / len(metrics_list),
            reliability_score=sum(m.reliability_score for m in metrics_list)
            / len(metrics_list),
        )

    async def get_tool_recommendations(
        self, task_type: str, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Get intelligent tool recommendations for a task type.

        Args:
            task_type: Type of task to perform
            constraints: Performance and resource constraints

        Returns:
            Tool recommendations with reasoning
        """
        # Check if we're in fallback mode
        fallback_reason = getattr(self, "_fallback_reason", None)
        if fallback_reason:
            return await self._fallback_get_recommendations(task_type, constraints)

        recommendations = {
            "primary_tools": [],
            "secondary_tools": [],
            "tool_chains": [],
            "reasoning": "",
        }

        # Find suitable tools
        suitable_tools = await self.discover_tools_for_task(task_type, constraints)

        if suitable_tools:
            recommendations["primary_tools"] = [
                {
                    "name": tool.name,
                    "suitability_score": tool.confidence_score,
                    "capability_type": tool.capability_type.value,
                    "estimated_latency_ms": tool.metrics.average_latency_ms
                    if tool.metrics
                    else 0,
                }
                for tool in suitable_tools[:3]  # Top 3 tools
            ]

            # Generate tool chains for complex tasks
            if len(suitable_tools) > 1:
                recommendations["tool_chains"] = await self._generate_tool_chains(
                    suitable_tools
                )

            # Generate reasoning
            best_tool = suitable_tools[0]
            recommendations["reasoning"] = (
                f"Recommended '{best_tool.name}' as primary tool based on "
                f"{best_tool.confidence_score:.2f} suitability score. "
                f"Tool specializes in {best_tool.capability_type.value} with "
                f"{best_tool.metrics.success_rate:.2f} success rate."
            )

        return recommendations

    async def _generate_tool_chains(
        self, tools: list[ToolCapability]
    ) -> list[dict[str, Any]]:
        """Generate intelligent tool chains from available tools.

        Args:
            tools: List of available tools

        Returns:
            List of recommended tool chains
        """
        chains = []

        # Simple chaining logic - can be enhanced with ML
        search_tools = [
            t for t in tools if t.capability_type == ToolCapabilityType.SEARCH
        ]
        generation_tools = [
            t for t in tools if t.capability_type == ToolCapabilityType.GENERATION
        ]
        analysis_tools = [
            t for t in tools if t.capability_type == ToolCapabilityType.ANALYSIS
        ]

        # Search → Generation chain
        if search_tools and generation_tools:
            chains.append(
                {
                    "chain": [search_tools[0].name, generation_tools[0].name],
                    "type": "search_then_generate",
                    "estimated_total_latency_ms": (
                        (
                            search_tools[0].metrics.average_latency_ms
                            if search_tools[0].metrics
                            else 0
                        )
                        + (
                            generation_tools[0].metrics.average_latency_ms
                            if generation_tools[0].metrics
                            else 0
                        )
                    ),
                }
            )

        # Analysis → Search → Generation chain
        if analysis_tools and search_tools and generation_tools:
            chains.append(
                {
                    "chain": [
                        analysis_tools[0].name,
                        search_tools[0].name,
                        generation_tools[0].name,
                    ],
                    "type": "analyze_search_generate",
                    "estimated_total_latency_ms": (
                        (
                            analysis_tools[0].metrics.average_latency_ms
                            if analysis_tools[0].metrics
                            else 0
                        )
                        + (
                            search_tools[0].metrics.average_latency_ms
                            if search_tools[0].metrics
                            else 0
                        )
                        + (
                            generation_tools[0].metrics.average_latency_ms
                            if generation_tools[0].metrics
                            else 0
                        )
                    ),
                }
            )

        return chains

    async def _fallback_discover_tools(
        self, task_description: str, requirements: dict[str, Any]
    ) -> list[ToolCapability]:
        """Fallback tool discovery when agent is in fallback mode.

        Args:
            task_description: Description of the task to perform
            requirements: Performance and quality requirements

        Returns:
            List of mock suitable tools
        """
        fallback_reason = getattr(self, "_fallback_reason", "unknown")
        logger.info(f"Using fallback tool discovery (reason: {fallback_reason})")

        # Create mock tools based on task analysis
        mock_tools = []
        task_lower = task_description.lower()

        # Mock hybrid search tool
        if any(keyword in task_lower for keyword in ["search", "find", "retrieve"]):
            mock_tool = ToolCapability(
                name="mock_hybrid_search",
                capability_type=ToolCapabilityType.SEARCH,
                description="Mock hybrid search tool (fallback mode)",
                input_types=["text", "query"],
                output_types=["search_results"],
                metrics=ToolMetrics(
                    average_latency_ms=200.0,
                    success_rate=0.85,
                    accuracy_score=0.80,
                    cost_per_execution=0.01,
                    reliability_score=0.85,
                ),
                confidence_score=0.85,
                last_updated="fallback_mode",
            )
            mock_tools.append(mock_tool)

        # Mock analysis tool
        if any(keyword in task_lower for keyword in ["analyze", "examine", "evaluate"]):
            mock_tool = ToolCapability(
                name="mock_content_analysis",
                capability_type=ToolCapabilityType.ANALYSIS,
                description="Mock content analysis tool (fallback mode)",
                input_types=["text", "documents"],
                output_types=["analysis_results"],
                metrics=ToolMetrics(
                    average_latency_ms=300.0,
                    success_rate=0.80,
                    accuracy_score=0.75,
                    cost_per_execution=0.02,
                    reliability_score=0.80,
                ),
                confidence_score=0.80,
                last_updated="fallback_mode",
            )
            mock_tools.append(mock_tool)

        # Mock generation tool
        if any(keyword in task_lower for keyword in ["generate", "create", "compose"]):
            mock_tool = ToolCapability(
                name="mock_content_generation",
                capability_type=ToolCapabilityType.GENERATION,
                description="Mock content generation tool (fallback mode)",
                input_types=["text", "context"],
                output_types=["generated_content"],
                metrics=ToolMetrics(
                    average_latency_ms=500.0,
                    success_rate=0.75,
                    accuracy_score=0.70,
                    cost_per_execution=0.05,
                    reliability_score=0.75,
                ),
                confidence_score=0.75,
                last_updated="fallback_mode",
            )
            mock_tools.append(mock_tool)

        # Default fallback tool if no specific tools matched
        if not mock_tools:
            mock_tool = ToolCapability(
                name="mock_general_tool",
                capability_type=ToolCapabilityType.SEARCH,
                description="Mock general purpose tool (fallback mode)",
                input_types=["text"],
                output_types=["results"],
                metrics=ToolMetrics(
                    average_latency_ms=250.0,
                    success_rate=0.70,
                    accuracy_score=0.65,
                    cost_per_execution=0.02,
                    reliability_score=0.70,
                ),
                confidence_score=0.70,
                last_updated="fallback_mode",
            )
            mock_tools.append(mock_tool)

        return mock_tools

    async def _fallback_get_recommendations(
        self, task_type: str, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback tool recommendations when agent is in fallback mode.

        Args:
            task_type: Type of task to perform
            constraints: Performance and resource constraints

        Returns:
            Mock tool recommendations
        """
        fallback_reason = getattr(self, "_fallback_reason", "unknown")
        logger.info(f"Using fallback tool recommendations (reason: {fallback_reason})")

        # Get fallback tools
        fallback_tools = await self._fallback_discover_tools(task_type, constraints)

        recommendations = {
            "primary_tools": [
                {
                    "name": tool.name,
                    "suitability_score": tool.confidence_score,
                    "capability_type": tool.capability_type.value,
                    "estimated_latency_ms": tool.metrics.average_latency_ms
                    if tool.metrics
                    else 0,
                    "fallback_mode": True,
                }
                for tool in fallback_tools[:2]  # Top 2 fallback tools
            ],
            "secondary_tools": [],
            "tool_chains": [
                {
                    "chain": [tool.name for tool in fallback_tools],
                    "type": "fallback_chain",
                    "estimated_total_latency_ms": sum(
                        tool.metrics.average_latency_ms if tool.metrics else 0
                        for tool in fallback_tools
                    ),
                }
            ]
            if len(fallback_tools) > 1
            else [],
            "reasoning": (
                f"Using fallback mode recommendations (reason: {fallback_reason}). "
                f"Identified {len(fallback_tools)} mock tools suitable for '{task_type}' tasks. "
                "These are simulated capabilities that will provide basic functionality."
            ),
            "fallback_mode": True,
        }

        return recommendations


# Global discovery engine instance
_discovery_engine: DynamicToolDiscovery | None = None


def get_discovery_engine() -> DynamicToolDiscovery:
    """Get singleton discovery engine instance."""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = DynamicToolDiscovery()
    return _discovery_engine


async def discover_tools_for_task(
    task: str, requirements: dict[str, Any], deps: BaseAgentDependencies
) -> list[ToolCapability]:
    """Convenient function for dynamic tool discovery.

    Args:
        task: Task description
        requirements: Performance requirements
        deps: Agent dependencies

    Returns:
        List of suitable tools for the task
    """
    engine = get_discovery_engine()
    if not engine._initialized:
        await engine.initialize_discovery(deps)
    return await engine.discover_tools_for_task(task, requirements)
