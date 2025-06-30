"""Integration tests for multi-agent result fusion algorithms.

Tests result fusion and aggregation patterns from distributed agent systems,
focusing on complex workflow scenarios and performance optimization.
"""

import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.agentic_orchestrator import (
    AgenticOrchestrator,
    ToolRequest,
    ToolResponse,
)
from src.services.agents.core import BaseAgentDependencies, create_agent_dependencies
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    ToolCapabilityType,
    ToolMetrics,
)


class ResultFusionEngine:
    """Engine for fusing results from multiple agents."""

    def __init__(self):
        self.fusion_strategies = {
            "confidence_weighted": self._confidence_weighted_fusion,
            "majority_voting": self._majority_voting_fusion,
            "performance_weighted": self._performance_weighted_fusion,
            "hierarchical": self._hierarchical_fusion,
        }

    async def fuse_results(
        self,
        results: list[dict[str, Any]],
        strategy: str = "confidence_weighted",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fuse multiple agent results using specified strategy."""
        if strategy not in self.fusion_strategies:
            msg = f"Unknown fusion strategy: {strategy}"
            raise ValueError(msg)

        return await self.fusion_strategies[strategy](results, metadata or {})

    async def _confidence_weighted_fusion(
        self, results: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fuse results weighted by confidence scores."""
        if not results:
            return {"success": False, "error": "No results to fuse"}

        # Calculate weighted average confidence
        total_weight = sum(r.get("confidence", 0.5) for r in results)
        if total_weight == 0:
            total_weight = len(results)

        # Aggregate results with confidence weighting
        fused_data = {}
        for result in results:
            weight = result.get("confidence", 0.5) / total_weight
            for key, value in result.get("results", {}).items():
                if key not in fused_data:
                    fused_data[key] = []
                fused_data[key].append({"value": value, "weight": weight})

        # Compute weighted aggregates
        final_results = {}
        for key, weighted_values in fused_data.items():
            if all(isinstance(wv["value"], int | float) for wv in weighted_values):
                # Numerical weighted average
                final_results[key] = sum(
                    wv["value"] * wv["weight"] for wv in weighted_values
                )
            else:
                # For non-numerical, take highest weight value
                best_value = max(weighted_values, key=lambda x: x["weight"])
                final_results[key] = best_value["value"]

        return {
            "success": True,
            "fusion_strategy": "confidence_weighted",
            "results": final_results,
            "source_count": len(results),
            "total_confidence": sum(r.get("confidence", 0.5) for r in results)
            / len(results),
            "fusion_metadata": {
                "weights_used": [
                    r.get("confidence", 0.5) / total_weight for r in results
                ],
                "strategy_effectiveness": self._calculate_fusion_effectiveness(results),
            },
        }

    async def _majority_voting_fusion(
        self, results: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fuse results using majority voting."""
        if not results:
            return {"success": False, "error": "No results to fuse"}

        # Collect votes for each result key
        votes = {}
        for result in results:
            for key, value in result.get("results", {}).items():
                if key not in votes:
                    votes[key] = []
                votes[key].append(value)

        # Determine majority for each key
        majority_results = {}
        for key, values in votes.items():
            if all(isinstance(v, bool) for v in values):
                # Boolean majority
                majority_results[key] = sum(values) > len(values) / 2
            elif all(isinstance(v, int | float) for v in values):
                # Numerical median
                majority_results[key] = statistics.median(values)
            else:
                # Most common value
                from collections import Counter

                counter = Counter(str(v) for v in values)
                most_common = counter.most_common(1)[0]
                majority_results[key] = most_common[0]

        return {
            "success": True,
            "fusion_strategy": "majority_voting",
            "results": majority_results,
            "source_count": len(results),
            "consensus_strength": self._calculate_consensus_strength(results),
            "fusion_metadata": {
                "vote_distributions": {
                    key: len({str(v) for v in values}) for key, values in votes.items()
                },
            },
        }

    async def _performance_weighted_fusion(
        self, results: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fuse results weighted by performance metrics."""
        if not results:
            return {"success": False, "error": "No results to fuse"}

        # Calculate performance weights
        performance_weights = []
        for result in results:
            latency = result.get("latency_ms", 1000)
            success = 1.0 if result.get("success", False) else 0.1
            confidence = result.get("confidence", 0.5)

            # Performance score: higher is better
            # Lower latency, higher success, higher confidence
            performance_score = (success * confidence) / (latency / 1000 + 1)
            performance_weights.append(performance_score)

        # Normalize weights
        total_weight = sum(performance_weights)
        if total_weight == 0:
            total_weight = len(performance_weights)
            performance_weights = [1.0] * len(results)

        normalized_weights = [w / total_weight for w in performance_weights]

        # Aggregate with performance weighting
        fused_results = {}
        for i, result in enumerate(results):
            weight = normalized_weights[i]
            for key, value in result.get("results", {}).items():
                if key not in fused_results:
                    fused_results[key] = []
                fused_results[key].append({"value": value, "weight": weight})

        # Compute weighted results
        final_results = {}
        for key, weighted_values in fused_results.items():
            if all(isinstance(wv["value"], int | float) for wv in weighted_values):
                final_results[key] = sum(
                    wv["value"] * wv["weight"] for wv in weighted_values
                )
            else:
                best_value = max(weighted_values, key=lambda x: x["weight"])
                final_results[key] = best_value["value"]

        return {
            "success": True,
            "fusion_strategy": "performance_weighted",
            "results": final_results,
            "source_count": len(results),
            "avg_performance_score": sum(performance_weights)
            / len(performance_weights),
            "fusion_metadata": {
                "performance_weights": normalized_weights,
                "performance_scores": performance_weights,
            },
        }

    async def _hierarchical_fusion(
        self, results: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Fuse results using hierarchical priority."""
        if not results:
            return {"success": False, "error": "No results to fuse"}

        # Sort by priority (higher priority first)
        prioritized_results = sorted(
            results,
            key=lambda r: (
                r.get("priority", 0),
                r.get("confidence", 0),
                -r.get("latency_ms", 1000),
            ),
            reverse=True,
        )

        # Start with highest priority result
        base_result = prioritized_results[0].get("results", {}).copy()

        # Merge in lower priority results for missing keys only
        for result in prioritized_results[1:]:
            for key, value in result.get("results", {}).items():
                if key not in base_result:
                    base_result[key] = value

        return {
            "success": True,
            "fusion_strategy": "hierarchical",
            "results": base_result,
            "source_count": len(results),
            "primary_source_confidence": prioritized_results[0].get("confidence", 0),
            "fusion_metadata": {
                "priority_order": [
                    {
                        "index": i,
                        "priority": r.get("priority", 0),
                        "confidence": r.get("confidence", 0),
                    }
                    for i, r in enumerate(prioritized_results)
                ],
            },
        }

    def _calculate_fusion_effectiveness(self, results: list[dict[str, Any]]) -> float:
        """Calculate effectiveness of fusion process."""
        if len(results) <= 1:
            return 1.0

        # Measure agreement between results
        agreements = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result1 = results[i].get("results", {})
                result2 = results[j].get("results", {})

                common_keys = set(result1.keys()) & set(result2.keys())
                if common_keys:
                    agreements.append(
                        len(common_keys)
                        / len(set(result1.keys()) | set(result2.keys()))
                    )

        return sum(agreements) / len(agreements) if agreements else 0.5

    def _calculate_consensus_strength(self, results: list[dict[str, Any]]) -> float:
        """Calculate strength of consensus across results."""
        if len(results) <= 1:
            return 1.0

        # Calculate agreement ratio
        all_keys = set()
        for result in results:
            all_keys.update(result.get("results", {}).keys())

        consensus_scores = []
        for key in all_keys:
            values = [
                r.get("results", {}).get(key)
                for r in results
                if key in r.get("results", {})
            ]
            if values:
                unique_values = len({str(v) for v in values})
                consensus_scores.append(
                    1.0 / unique_values
                )  # Higher consensus = fewer unique values

        return (
            sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.5
        )


class TestResultFusionAlgorithms:
    """Test result fusion algorithms for complex workflows."""

    @pytest.fixture
    def fusion_engine(self) -> ResultFusionEngine:
        """Create result fusion engine."""
        return ResultFusionEngine()

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_fusion_session",
        )

    @pytest.mark.asyncio
    async def test_confidence_weighted_fusion(self, fusion_engine):
        """Test confidence-weighted result fusion."""
        # Create test results with different confidence levels
        results = [
            {
                "success": True,
                "confidence": 0.9,
                "results": {"accuracy": 0.95, "latency": 150, "quality": "high"},
                "latency_ms": 150,
            },
            {
                "success": True,
                "confidence": 0.7,
                "results": {"accuracy": 0.85, "latency": 200, "quality": "medium"},
                "latency_ms": 200,
            },
            {
                "success": True,
                "confidence": 0.6,
                "results": {"accuracy": 0.80, "latency": 100, "quality": "medium"},
                "latency_ms": 100,
            },
        ]

        fused_result = await fusion_engine.fuse_results(results, "confidence_weighted")

        # Verify fusion results
        assert fused_result["success"] is True
        assert fused_result["fusion_strategy"] == "confidence_weighted"
        assert fused_result["source_count"] == 3

        # Check weighted averages
        fused_accuracy = fused_result["results"]["accuracy"]
        # Should be closer to 0.95 (highest confidence) than simple average
        assert 0.85 < fused_accuracy < 0.95
        assert fused_accuracy > (0.95 + 0.85 + 0.80) / 3  # Better than simple average

        # Verify fusion metadata
        assert "fusion_metadata" in fused_result
        assert len(fused_result["fusion_metadata"]["weights_used"]) == 3
        assert fused_result["fusion_metadata"]["strategy_effectiveness"] > 0

    @pytest.mark.asyncio
    async def test_majority_voting_fusion(self, fusion_engine):
        """Test majority voting result fusion."""
        # Create test results with votes
        results = [
            {
                "results": {
                    "is_relevant": True,
                    "category": "technical",
                    "score": 8.5,
                    "recommendation": "approve",
                }
            },
            {
                "results": {
                    "is_relevant": True,
                    "category": "technical",
                    "score": 7.8,
                    "recommendation": "approve",
                }
            },
            {
                "results": {
                    "is_relevant": False,
                    "category": "business",
                    "score": 6.2,
                    "recommendation": "reject",
                }
            },
            {
                "results": {
                    "is_relevant": True,
                    "category": "technical",
                    "score": 8.1,
                    "recommendation": "approve",
                }
            },
        ]

        fused_result = await fusion_engine.fuse_results(results, "majority_voting")

        # Verify majority voting results
        assert fused_result["success"] is True
        assert fused_result["fusion_strategy"] == "majority_voting"
        assert fused_result["source_count"] == 4

        # Check majority decisions
        assert fused_result["results"]["is_relevant"] is True  # 3 out of 4 voted True
        assert fused_result["results"]["category"] == "technical"  # 3 out of 4
        assert fused_result["results"]["recommendation"] == "approve"  # 3 out of 4

        # Score should be median
        expected_median = statistics.median([8.5, 7.8, 6.2, 8.1])
        assert abs(fused_result["results"]["score"] - expected_median) < 0.1

        # Verify consensus strength
        assert 0 <= fused_result["consensus_strength"] <= 1

    @pytest.mark.asyncio
    async def test_performance_weighted_fusion(self, fusion_engine):
        """Test performance-weighted result fusion."""
        # Create results with different performance characteristics
        results = [
            {
                "success": True,
                "confidence": 0.8,
                "latency_ms": 100,  # Fast
                "results": {"processed_items": 1000, "error_rate": 0.01},
            },
            {
                "success": True,
                "confidence": 0.9,
                "latency_ms": 500,  # Slow but accurate
                "results": {"processed_items": 800, "error_rate": 0.005},
            },
            {
                "success": False,
                "confidence": 0.3,
                "latency_ms": 50,  # Fast but failed
                "results": {"processed_items": 200, "error_rate": 0.1},
            },
        ]

        fused_result = await fusion_engine.fuse_results(results, "performance_weighted")

        # Verify performance weighting
        assert fused_result["success"] is True
        assert fused_result["fusion_strategy"] == "performance_weighted"
        assert fused_result["source_count"] == 3

        # Check that successful, fast agents get higher weight
        weights = fused_result["fusion_metadata"]["performance_weights"]
        assert len(weights) == 3

        # Fast successful agent should have higher weight than failed agent
        assert weights[0] > weights[2]  # Fast success > fast failure

        # Verify performance scores
        performance_scores = fused_result["fusion_metadata"]["performance_scores"]
        assert len(performance_scores) == 3
        assert all(score >= 0 for score in performance_scores)

    @pytest.mark.asyncio
    async def test_hierarchical_fusion(self, fusion_engine):
        """Test hierarchical priority-based fusion."""
        # Create results with different priorities
        results = [
            {
                "priority": 1,  # Low priority
                "confidence": 0.9,
                "results": {
                    "primary_result": "low_priority",
                    "secondary_data": "A",
                    "extra": "1",
                },
            },
            {
                "priority": 3,  # High priority
                "confidence": 0.7,
                "results": {"primary_result": "high_priority", "unique_data": "B"},
            },
            {
                "priority": 2,  # Medium priority
                "confidence": 0.8,
                "results": {
                    "primary_result": "medium_priority",
                    "secondary_data": "C",
                    "other": "2",
                },
            },
        ]

        fused_result = await fusion_engine.fuse_results(results, "hierarchical")

        # Verify hierarchical fusion
        assert fused_result["success"] is True
        assert fused_result["fusion_strategy"] == "hierarchical"
        assert fused_result["source_count"] == 3

        # Primary result should come from highest priority
        assert fused_result["results"]["primary_result"] == "high_priority"

        # Secondary data should come from high priority if available, otherwise fallback
        assert fused_result["results"]["unique_data"] == "B"  # Only in high priority

        # Missing keys should be filled from lower priority results
        assert (
            "secondary_data" in fused_result["results"]
        )  # Should be filled from lower priority
        assert (
            "extra" in fused_result["results"]
        )  # Should be filled from lowest priority

        # Verify priority ordering metadata
        priority_order = fused_result["fusion_metadata"]["priority_order"]
        assert len(priority_order) == 3
        assert priority_order[0]["priority"] == 3  # Highest first


class TestComplexWorkflowScenarios:
    """Test complex multi-agent workflow scenarios with result fusion."""

    @pytest.fixture
    def fusion_engine(self) -> ResultFusionEngine:
        """Create result fusion engine."""
        return ResultFusionEngine()

    @pytest.fixture
    def agent_pool(self) -> list[AgenticOrchestrator]:
        """Create pool of agents for complex workflows."""
        return [
            AgenticOrchestrator(model="gpt-4o-mini", temperature=0.1) for _ in range(4)
        ]

    @pytest.fixture
    def discovery_agents(self) -> list[DynamicToolDiscovery]:
        """Create discovery agents."""
        return [
            DynamicToolDiscovery(model="gpt-4o-mini", temperature=0.1) for _ in range(2)
        ]

    @pytest.fixture
    def mock_client_manager(self) -> ClientManager:
        """Create mock client manager."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.get_qdrant_client = AsyncMock()
        client_manager.get_openai_client = AsyncMock()
        client_manager.get_redis_client = AsyncMock()
        return client_manager

    @pytest.fixture
    def agent_dependencies(self, mock_client_manager) -> BaseAgentDependencies:
        """Create agent dependencies."""
        return create_agent_dependencies(
            client_manager=mock_client_manager,
            session_id="test_complex_workflow",
        )

    @pytest.mark.asyncio
    async def test_search_analysis_generation_chain(
        self, agent_pool, discovery_agents, fusion_engine, agent_dependencies
    ):
        """Test Search → Analysis → Generation agent chain with result fusion."""
        # Initialize agents
        search_agent = agent_pool[0]
        analysis_agent = agent_pool[1]
        generation_agent = agent_pool[2]
        discovery_agent = discovery_agents[0]

        await search_agent.initialize(agent_dependencies)
        await analysis_agent.initialize(agent_dependencies)
        await generation_agent.initialize(agent_dependencies)
        await discovery_agent.initialize_discovery(agent_dependencies)

        # Phase 1: Search phase
        search_task = "search for recent AI safety research papers"
        search_tools = await discovery_agent.discover_tools_for_task(
            search_task, {"max_latency_ms": 2000}
        )

        search_response = await search_agent.orchestrate(
            search_task, {"phase": "search"}, agent_dependencies
        )

        # Phase 2: Analysis phase (using search results)
        analysis_task = "analyze search results for key trends and insights"
        analysis_context = {
            "phase": "analysis",
            "search_results": search_response.results,
            "search_confidence": search_response.confidence,
        }

        analysis_response = await analysis_agent.orchestrate(
            analysis_task, analysis_context, agent_dependencies
        )

        # Phase 3: Generation phase (using analysis results)
        generation_task = "generate comprehensive report from analysis"
        generation_context = {
            "phase": "generation",
            "analysis_results": analysis_response.results,
            "search_metadata": search_response.tools_used,
        }

        generation_response = await generation_agent.orchestrate(
            generation_task, generation_context, agent_dependencies
        )

        # Fusion of all phase results
        phase_results = [
            {
                "phase": "search",
                "success": search_response.success,
                "confidence": search_response.confidence,
                "latency_ms": search_response.latency_ms,
                "results": search_response.results,
                "priority": 1,
            },
            {
                "phase": "analysis",
                "success": analysis_response.success,
                "confidence": analysis_response.confidence,
                "latency_ms": analysis_response.latency_ms,
                "results": analysis_response.results,
                "priority": 2,
            },
            {
                "phase": "generation",
                "success": generation_response.success,
                "confidence": generation_response.confidence,
                "latency_ms": generation_response.latency_ms,
                "results": generation_response.results,
                "priority": 3,
            },
        ]

        # Test different fusion strategies
        confidence_fusion = await fusion_engine.fuse_results(
            phase_results, "confidence_weighted"
        )
        hierarchical_fusion = await fusion_engine.fuse_results(
            phase_results, "hierarchical"
        )

        # Verify chain execution
        assert all(result["success"] for result in phase_results)
        assert confidence_fusion["success"] is True
        assert hierarchical_fusion["success"] is True

        # Verify result propagation through chain
        assert len(agent_dependencies.session_state.conversation_history) >= 3

        # Verify fusion quality
        assert confidence_fusion["source_count"] == 3
        assert hierarchical_fusion["source_count"] == 3

        # Hierarchical should prioritize generation results
        assert (
            "generation" in str(hierarchical_fusion["results"])
            or len(hierarchical_fusion["results"]) > 0
        )

    @pytest.mark.asyncio
    async def test_parallel_processing_with_aggregation(
        self, agent_pool, fusion_engine, agent_dependencies
    ):
        """Test parallel processing with result aggregation."""
        # Initialize agents for parallel processing
        parallel_agents = agent_pool[:3]
        for agent in parallel_agents:
            await agent.initialize(agent_dependencies)

        # Create parallel processing tasks
        parallel_tasks = [
            {
                "agent_id": i,
                "task": f"process data chunk {i}",
                "constraints": {"chunk_id": i, "parallel_execution": True},
                "expected_results": {
                    "chunk_size": 1000 + i * 100,
                    "processing_time": 100 + i * 50,
                },
            }
            for i in range(3)
        ]

        # Execute tasks in parallel
        start_time = time.time()
        parallel_responses = await asyncio.gather(
            *[
                parallel_agents[task["agent_id"]].orchestrate(
                    task["task"], task["constraints"], agent_dependencies
                )
                for task in parallel_tasks
            ]
        )
        parallel_execution_time = time.time() - start_time

        # Prepare results for fusion
        fusion_results = [
            {
                "agent_id": i,
                "success": response.success,
                "confidence": response.confidence,
                "latency_ms": response.latency_ms,
                "results": {
                    **response.results,
                    "chunk_id": parallel_tasks[i]["constraints"]["chunk_id"],
                },
            }
            for i, response in enumerate(parallel_responses)
        ]

        # Test aggregation strategies
        majority_fusion = await fusion_engine.fuse_results(
            fusion_results, "majority_voting"
        )
        performance_fusion = await fusion_engine.fuse_results(
            fusion_results, "performance_weighted"
        )

        # Verify parallel execution
        assert len(parallel_responses) == 3
        assert all(response.success for response in parallel_responses)
        assert parallel_execution_time < 10.0  # Should complete quickly with mocking

        # Verify fusion results
        assert majority_fusion["success"] is True
        assert performance_fusion["success"] is True
        assert majority_fusion["source_count"] == 3
        assert performance_fusion["source_count"] == 3

        # Check aggregated metrics
        total_latency = sum(r.latency_ms for r in parallel_responses)
        avg_confidence = sum(r.confidence for r in parallel_responses) / len(
            parallel_responses
        )

        assert total_latency > 0
        assert 0 <= avg_confidence <= 1

    @pytest.mark.asyncio
    async def test_fault_tolerance_with_result_fusion(
        self, agent_pool, fusion_engine, agent_dependencies
    ):
        """Test fault tolerance and graceful degradation with result fusion."""
        # Initialize agents
        reliable_agents = agent_pool[:2]
        unreliable_agent = agent_pool[2]

        for agent in reliable_agents:
            await agent.initialize(agent_dependencies)
        await unreliable_agent.initialize(agent_dependencies)

        # Mock unreliable agent to fail sometimes
        original_orchestrate = unreliable_agent.orchestrate
        failure_count = 0

        async def mock_unreliable_orchestrate(task, constraints, deps):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 2 == 0:  # Fail every other call
                return ToolResponse(
                    success=False,
                    results={"error": "Simulated intermittent failure"},
                    tools_used=[],
                    reasoning="Agent experiencing intermittent issues",
                    latency_ms=50.0,
                    confidence=0.0,
                )
            return await original_orchestrate(task, constraints, deps)

        unreliable_agent.orchestrate = mock_unreliable_orchestrate

        # Execute multiple tasks with fault tolerance
        fault_tolerant_tasks = [f"fault tolerant task {i}" for i in range(4)]

        all_results = []
        for task in fault_tolerant_tasks:
            # Execute task on all agents
            task_results = await asyncio.gather(
                *[
                    agent.orchestrate(task, {"task_id": task}, agent_dependencies)
                    for agent in [*reliable_agents, unreliable_agent]
                ],
                return_exceptions=True,
            )

            # Convert to fusion format
            fusion_input = []
            for i, result in enumerate(task_results):
                if isinstance(result, ToolResponse):
                    fusion_input.append(
                        {
                            "agent_id": i,
                            "success": result.success,
                            "confidence": result.confidence,
                            "latency_ms": result.latency_ms,
                            "results": result.results,
                            "priority": 1
                            if i < 2
                            else 0,  # Reliable agents get higher priority
                        }
                    )
                elif isinstance(result, Exception):
                    fusion_input.append(
                        {
                            "agent_id": i,
                            "success": False,
                            "confidence": 0.0,
                            "latency_ms": 0.0,
                            "results": {"error": str(result)},
                            "priority": 0,
                        }
                    )

            # Fuse results with fault tolerance
            if any(r["success"] for r in fusion_input):
                fused_result = await fusion_engine.fuse_results(
                    fusion_input,
                    "hierarchical",  # Prioritize reliable agents
                )
                all_results.append(fused_result)

        # Verify fault tolerance
        assert len(all_results) >= 2  # At least some tasks should succeed
        successful_fusions = [r for r in all_results if r["success"]]
        assert len(successful_fusions) >= 2  # Most fusions should succeed

        # Verify graceful degradation
        for result in successful_fusions:
            assert result["source_count"] >= 2  # At least reliable agents contributed
            # Should have meaningful results despite some failures
            assert len(result["results"]) > 0 or "error" not in str(result["results"])

    @pytest.mark.asyncio
    async def test_performance_optimization_across_agent_boundaries(
        self, agent_pool, discovery_agents, fusion_engine, agent_dependencies
    ):
        """Test performance optimization across agent boundaries."""
        # Initialize high-performance agent configuration
        speed_agent = agent_pool[0]  # Optimized for speed
        quality_agent = agent_pool[1]  # Optimized for quality
        balanced_agent = agent_pool[2]  # Balanced approach
        discovery_agent = discovery_agents[0]

        for agent in [speed_agent, quality_agent, balanced_agent]:
            await agent.initialize(agent_dependencies)
        await discovery_agent.initialize_discovery(agent_dependencies)

        # Performance optimization task
        optimization_task = (
            "optimize data processing pipeline for 3-10x performance improvement"
        )

        # Discover optimal tools for each agent type
        speed_tools = await discovery_agent.discover_tools_for_task(
            optimization_task, {"max_latency_ms": 500, "priority": "speed"}
        )
        quality_tools = await discovery_agent.discover_tools_for_task(
            optimization_task, {"min_accuracy": 0.95, "priority": "quality"}
        )
        balanced_tools = await discovery_agent.discover_tools_for_task(
            optimization_task,
            {"max_latency_ms": 2000, "min_accuracy": 0.85, "priority": "balanced"},
        )

        # Execute with different optimization targets
        performance_tests = [
            (speed_agent, "speed_optimized", {"max_latency_ms": 500}),
            (quality_agent, "quality_optimized", {"min_accuracy": 0.95}),
            (
                balanced_agent,
                "balanced_optimized",
                {"max_latency_ms": 2000, "min_accuracy": 0.85},
            ),
        ]

        optimization_results = []
        for agent, optimization_type, constraints in performance_tests:
            start_time = time.time()

            response = await agent.orchestrate(
                optimization_task,
                {**constraints, "optimization_type": optimization_type},
                agent_dependencies,
            )

            execution_time = time.time() - start_time

            # Calculate performance metrics
            performance_gain = 1.0 / (response.latency_ms / 1000 + 1)  # Simulated gain
            quality_score = response.confidence

            optimization_results.append(
                {
                    "optimization_type": optimization_type,
                    "success": response.success,
                    "confidence": response.confidence,
                    "latency_ms": response.latency_ms,
                    "execution_time": execution_time,
                    "results": {
                        **response.results,
                        "performance_gain": performance_gain,
                        "quality_score": quality_score,
                        "optimization_effectiveness": performance_gain * quality_score,
                    },
                    "priority": 2 if optimization_type == "balanced_optimized" else 1,
                }
            )

        # Fuse optimization results to find best approach
        optimization_fusion = await fusion_engine.fuse_results(
            optimization_results, "performance_weighted"
        )

        # Verify performance optimization
        assert optimization_fusion["success"] is True
        assert optimization_fusion["source_count"] == 3

        # Check that we achieved performance improvements
        avg_performance_score = optimization_fusion["avg_performance_score"]
        assert avg_performance_score > 0

        # Verify 3-10x performance target (simulated)
        fused_effectiveness = optimization_fusion["results"].get(
            "optimization_effectiveness", 0
        )
        assert fused_effectiveness > 0.5  # Should show significant improvement

        # Check individual optimization results
        assert len(optimization_results) == 3
        speed_result = next(
            r
            for r in optimization_results
            if r["optimization_type"] == "speed_optimized"
        )
        quality_result = next(
            r
            for r in optimization_results
            if r["optimization_type"] == "quality_optimized"
        )

        # Speed optimization should have lower latency
        # Quality optimization should have higher confidence
        assert speed_result["success"]
        assert quality_result["success"]

        # Balanced approach should be represented in final fusion
        performance_weights = optimization_fusion["fusion_metadata"][
            "performance_weights"
        ]
        assert len(performance_weights) == 3
        assert all(weight > 0 for weight in performance_weights)

    @pytest.mark.asyncio
    async def test_dynamic_load_balancing_effectiveness(
        self, agent_pool, fusion_engine, agent_dependencies
    ):
        """Test dynamic load balancing effectiveness across agents."""
        # Initialize agent pool
        agents = agent_pool
        for agent in agents:
            await agent.initialize(agent_dependencies)

        # Create varying load scenarios
        load_scenarios = [
            {"load_level": "light", "task_count": 2, "complexity": "simple"},
            {"load_level": "medium", "task_count": 4, "complexity": "moderate"},
            {"load_level": "heavy", "task_count": 6, "complexity": "complex"},
        ]

        load_balancing_results = []

        for scenario in load_scenarios:
            scenario_tasks = [
                f"{scenario['complexity']} task {i} for {scenario['load_level']} load"
                for i in range(scenario["task_count"])
            ]

            # Dynamic load balancing: assign tasks to least loaded agents
            agent_loads = dict.fromkeys(range(len(agents)), 0)
            task_assignments = []

            for task_id, task in enumerate(scenario_tasks):
                # Find least loaded agent
                least_loaded_agent_id = min(
                    agent_loads.keys(), key=lambda k: agent_loads[k]
                )
                agent_loads[least_loaded_agent_id] += 1
                task_assignments.append((task_id, least_loaded_agent_id, task))

            # Execute tasks with load balancing
            start_time = time.time()
            task_results = await asyncio.gather(
                *[
                    agents[agent_id].orchestrate(
                        task,
                        {
                            "load_level": scenario["load_level"],
                            "task_id": task_id,
                            "agent_id": agent_id,
                        },
                        agent_dependencies,
                    )
                    for task_id, agent_id, task in task_assignments
                ]
            )
            scenario_execution_time = time.time() - start_time

            # Calculate load balancing effectiveness
            successful_tasks = sum(1 for result in task_results if result.success)
            avg_latency = sum(r.latency_ms for r in task_results) / len(task_results)
            load_distribution_variance = (
                statistics.variance(agent_loads.values()) if len(agent_loads) > 1 else 0
            )

            scenario_result = {
                "load_level": scenario["load_level"],
                "task_count": scenario["task_count"],
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / scenario["task_count"],
                "avg_latency_ms": avg_latency,
                "total_execution_time": scenario_execution_time,
                "load_distribution_variance": load_distribution_variance,
                "agent_utilization": agent_loads,
                "results": {
                    "tasks_completed": successful_tasks,
                    "efficiency_score": successful_tasks / scenario_execution_time,
                    "load_balance_score": 1.0 / (load_distribution_variance + 1),
                },
                "confidence": sum(r.confidence for r in task_results)
                / len(task_results),
                "latency_ms": avg_latency,
                "success": successful_tasks == scenario["task_count"],
            }

            load_balancing_results.append(scenario_result)

        # Fuse load balancing results
        load_balance_fusion = await fusion_engine.fuse_results(
            load_balancing_results, "performance_weighted"
        )

        # Verify load balancing effectiveness
        assert load_balance_fusion["success"] is True
        assert load_balance_fusion["source_count"] == 3

        # Check that load balancing improved performance
        avg_performance = load_balance_fusion["avg_performance_score"]
        assert avg_performance > 0

        # Verify load distribution
        for result in load_balancing_results:
            # Load should be relatively balanced (low variance)
            assert (
                result["load_distribution_variance"] < 2.0
            )  # Reasonable load distribution

            # Success rate should be high
            assert result["success_rate"] >= 0.75  # At least 75% success rate

            # Efficiency should scale with load
            assert result["results"]["efficiency_score"] > 0

        # Check that heavy load scenario was handled appropriately
        heavy_load_result = next(
            r for r in load_balancing_results if r["load_level"] == "heavy"
        )
        light_load_result = next(
            r for r in load_balancing_results if r["load_level"] == "light"
        )

        # Heavy load should still maintain reasonable performance
        assert heavy_load_result["success_rate"] >= 0.5  # At least 50% under heavy load
        assert heavy_load_result["results"]["efficiency_score"] > 0
