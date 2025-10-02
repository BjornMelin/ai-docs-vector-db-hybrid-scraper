"""Dynamic Tool Discovery Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core import BaseAgent, BaseAgentDependencies


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


@dataclass(slots=True)
class ToolMetrics:
    """Performance metrics for a tool."""

    average_latency_ms: float
    success_rate: float
    accuracy_score: float
    cost_per_execution: float
    reliability_score: float


class ToolCapability(BaseModel):
    """Tool capability definition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(...)
    capability_type: ToolCapabilityType = Field(...)
    description: str = Field(...)
    input_types: list[str] = Field(...)
    output_types: list[str] = Field(...)

    metrics: ToolMetrics | None = Field(None)
    requirements: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    compatible_tools: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)

    confidence_score: float = Field(0.8)
    last_updated: str = Field(...)


class DynamicToolDiscovery(BaseAgent):
    """Dynamic tool discovery engine with capability assessment."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1) -> None:
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
        return (
            "You are a tool discovery engine: assess capabilities, compatibility, "
            "and performance to recommend optimal tools and chains."
        )

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize discovery (noop if in fallback)."""
        reason = getattr(self, "_fallback_reason", None)
        if reason:
            logger.info("DynamicToolDiscovery in fallback mode (reason: %s)", reason)
        else:
            logger.info("DynamicToolDiscovery tools initialized")

    async def initialize_discovery(self, deps: BaseAgentDependencies) -> None:
        """Initialize discovery by scanning system and setting compatibility."""
        if not self._initialized:
            await self.initialize(deps)
        await self._scan_available_tools()
        await self._assess_tool_compatibility()
        logger.info(
            "DynamicToolDiscovery initialized with %d tools", len(self.discovered_tools)
        )

    async def _scan_available_tools(self) -> None:
        """Scan system for available tools and seed capabilities."""
        now = datetime.now(tz=UTC).isoformat()

        core_tools: dict[str, dict[str, Any]] = {
            "hybrid_search": {
                "capability_type": ToolCapabilityType.SEARCH,
                "description": "Hybrid vector and text search with reranking",
                "input_types": ["text", "query"],
                "output_types": ["search_results", "ranked_documents"],
                "metrics": ToolMetrics(150.0, 0.94, 0.87, 0.02, 0.92),
            },
            "rag_generation": {
                "capability_type": ToolCapabilityType.GENERATION,
                "description": "RAG-based answer generation with context synthesis",
                "input_types": ["text", "context", "search_results"],
                "output_types": ["generated_text", "synthesized_answer"],
                "metrics": ToolMetrics(800.0, 0.91, 0.89, 0.05, 0.88),
            },
            "content_analysis": {
                "capability_type": ToolCapabilityType.ANALYSIS,
                "description": "Content analysis and classification",
                "input_types": ["text", "documents"],
                "output_types": ["analysis_results", "classifications"],
                "metrics": ToolMetrics(200.0, 0.93, 0.85, 0.01, 0.90),
            },
        }

        for name, data in core_tools.items():
            cap = ToolCapability(
                name=name,
                capability_type=data["capability_type"],
                description=data["description"],
                input_types=data["input_types"],
                output_types=data["output_types"],
                metrics=data["metrics"],
                confidence_score=0.8,  # explicit for Pyright
                last_updated=now,
            )
            self.discovered_tools[name] = cap
            self.tool_performance_history[name] = [data["metrics"]]

    async def _assess_tool_compatibility(self) -> None:
        """Assess compatibility between tools for chaining."""
        for tool_name, tool in self.discovered_tools.items():
            compatible: list[str] = []
            if tool.capability_type == ToolCapabilityType.SEARCH:
                compatible.extend(
                    [
                        n
                        for n, other in self.discovered_tools.items()
                        if other.capability_type
                        in (ToolCapabilityType.GENERATION, ToolCapabilityType.ANALYSIS)
                        and n != tool_name
                    ]
                )
            elif tool.capability_type == ToolCapabilityType.ANALYSIS:
                compatible.extend(
                    [
                        n
                        for n, other in self.discovered_tools.items()
                        if other.capability_type
                        in (ToolCapabilityType.SEARCH, ToolCapabilityType.GENERATION)
                        and n != tool_name
                    ]
                )
            tool.compatible_tools = compatible

    async def discover_tools_for_task(
        self, task_description: str, requirements: dict[str, Any]
    ) -> list[ToolCapability]:
        """Return suitable tools ranked by suitability score."""
        suitable: list[ToolCapability] = []
        for tool in self.discovered_tools.values():
            score = await self._calculate_suitability_score(
                tool, task_description, requirements
            )
            if score > 0.5:
                copy = tool.model_copy()
                copy.confidence_score = score
                suitable.append(copy)
        suitable.sort(key=lambda t: t.confidence_score, reverse=True)
        return suitable

    async def _calculate_suitability_score(
        self, tool: ToolCapability, task_description: str, requirements: dict[str, Any]
    ) -> float:
        """Score suitability without long boolean chains (R0916)."""
        s = task_description.lower()

        def match(kind: ToolCapabilityType, *keywords: str) -> bool:
            return any(k in s for k in keywords) and tool.capability_type == kind

        cap_match = any(
            [
                match(ToolCapabilityType.SEARCH, "search", "find", "lookup"),
                match(ToolCapabilityType.GENERATION, "generate", "compose", "answer"),
                match(ToolCapabilityType.ANALYSIS, "analyze", "classify", "assess"),
            ]
        )

        score = 0.4 if cap_match else 0.0

        if tool.metrics:
            if "max_latency_ms" in requirements:
                score += (
                    0.2
                    if tool.metrics.average_latency_ms <= requirements["max_latency_ms"]
                    else -0.1
                )
            if (
                "min_accuracy" in requirements
                and tool.metrics.accuracy_score >= requirements["min_accuracy"]
            ):
                score += 0.2
            if (
                "max_cost" in requirements
                and tool.metrics.cost_per_execution <= requirements["max_cost"]
            ):
                score += 0.1
            score += tool.metrics.reliability_score * 0.1

        return min(score, 1.0)

    async def get_tool_recommendations(
        self, task_type: str, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Return recommendations with reasoning and optional tool chains."""
        recommendations: dict[str, Any] = {
            "primary_tools": [],
            "secondary_tools": [],
            "tool_chains": [],
            "reasoning": "",
        }
        suitable = await self.discover_tools_for_task(task_type, constraints)
        if suitable:
            recommendations["primary_tools"] = [
                {
                    "name": t.name,
                    "suitability_score": t.confidence_score,
                    "capability_type": t.capability_type.value,
                    "estimated_latency_ms": (
                        t.metrics.average_latency_ms if t.metrics else 0.0
                    ),
                }
                for t in suitable[:3]
            ]
            if len(suitable) > 1:
                recommendations["tool_chains"] = await self._generate_tool_chains(
                    suitable
                )
            best = suitable[0]
            sr = best.metrics.success_rate if best.metrics else 0.0
            recommendations["reasoning"] = (
                f"Recommended {best.name} ({best.capability_type.value}) with "
                f"{best.confidence_score:.2f} suitability; success≈{sr:.2f}."
            )
        return recommendations

    async def _generate_tool_chains(
        self, tools: list[ToolCapability]
    ) -> list[dict[str, Any]]:
        """Generate compatible tool chains from available tools."""
        chains: list[dict[str, Any]] = []
        search = [t for t in tools if t.capability_type == ToolCapabilityType.SEARCH]
        gen = [t for t in tools if t.capability_type == ToolCapabilityType.GENERATION]
        ana = [t for t in tools if t.capability_type == ToolCapabilityType.ANALYSIS]

        if search and gen:
            chains.append(
                {
                    "chain": [search[0].name, gen[0].name],
                    "type": "search_then_generate",
                    "estimated_total_latency_ms": (
                        (
                            search[0].metrics.average_latency_ms
                            if search[0].metrics
                            else 0
                        )
                        + (gen[0].metrics.average_latency_ms if gen[0].metrics else 0)
                    ),
                }
            )
        if ana and search and gen:
            chains.append(
                {
                    "chain": [ana[0].name, search[0].name, gen[0].name],
                    "type": "analyze_search_generate",
                    "estimated_total_latency_ms": sum(
                        (t.metrics.average_latency_ms if t.metrics else 0.0)
                        for t in (ana[0], search[0], gen[0])
                    ),
                }
            )
        return chains
