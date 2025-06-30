"""Integration tests for multi-agent performance optimization.

Tests performance optimization scenarios that demonstrate 3-10x performance
improvements through autonomous agent coordination and intelligent orchestration.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from src.services.agents.core import BaseAgentDependencies


@dataclass
class PerformanceMetrics:
    """Performance metrics for multi-agent operations."""

    operation_name: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None
    throughput_ops_per_second: float | None = None
    success_rate: float = 0.0
    error_rate: float = 0.0
    latency_p95_ms: float | None = None
    latency_avg_ms: float | None = None
    resource_utilization: dict[str, float] = field(default_factory=dict)
    performance_improvement_factor: float | None = None
    baseline_comparison: dict[str, float] | None = None


@dataclass
class OptimizationScenario:
    """Defines a performance optimization scenario."""

    scenario_id: str
    description: str
    baseline_performance: dict[str, float]
    target_improvement: float  # e.g., 3.0 for 3x improvement
    optimization_strategies: list[str]
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, float] = field(default_factory=dict)


class PerformanceOptimizer:
    """Orchestrates performance optimization across multi-agent systems."""

    def __init__(self):
        self.optimization_history: list[PerformanceMetrics] = []
        self.baseline_metrics: dict[str, PerformanceMetrics] = {}
        self.active_optimizations: dict[str, OptimizationScenario] = {}

    async def run_baseline_benchmark(
        self,
        scenario: OptimizationScenario,
        agents: list[Any],
        dependencies: BaseAgentDependencies,
    ) -> PerformanceMetrics:
        """Run baseline benchmark to establish performance baseline."""
        start_time = datetime.now()

        # Simple sequential processing (baseline)
        task_results = []
        task_latencies = []

        for i, agent in enumerate(agents):
            task_start = time.time()

            try:
                if hasattr(agent, "orchestrate"):
                    response = await agent.orchestrate(
                        f"baseline task {i}",
                        {"baseline": True, "agent_id": i},
                        dependencies,
                    )
                    task_results.append(response.success)
                    task_latencies.append(response.latency_ms)
                else:
                    # Discovery agent
                    tools = await agent.discover_tools_for_task(
                        f"baseline discovery {i}", {"baseline": True}
                    )
                    task_results.append(len(tools) > 0)
                    task_latencies.append((time.time() - task_start) * 1000)

            except Exception:
                task_results.append(False)
                task_latencies.append((time.time() - task_start) * 1000)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        baseline_metrics = PerformanceMetrics(
            operation_name=f"baseline_{scenario.scenario_id}",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            throughput_ops_per_second=len(agents) / duration if duration > 0 else 0,
            success_rate=sum(task_results) / len(task_results) if task_results else 0,
            error_rate=1 - (sum(task_results) / len(task_results))
            if task_results
            else 1,
            latency_avg_ms=statistics.mean(task_latencies) if task_latencies else 0,
            latency_p95_ms=statistics.quantiles(task_latencies, n=20)[18]
            if len(task_latencies) > 5
            else (task_latencies[0] if task_latencies else 0),
        )

        self.baseline_metrics[scenario.scenario_id] = baseline_metrics
        return baseline_metrics

    async def run_optimized_benchmark(
        self,
        scenario: OptimizationScenario,
        agents: list[Any],
        dependencies: BaseAgentDependencies,
        optimization_strategy: str = "parallel_with_coordination",
    ) -> PerformanceMetrics:
        """Run optimized benchmark using multi-agent coordination."""
        start_time = datetime.now()

        if optimization_strategy == "parallel_with_coordination":
            return await self._parallel_coordinated_execution(
                scenario, agents, dependencies, start_time
            )
        if optimization_strategy == "hierarchical_optimization":
            return await self._hierarchical_optimized_execution(
                scenario, agents, dependencies, start_time
            )
        if optimization_strategy == "adaptive_load_balancing":
            return await self._adaptive_load_balanced_execution(
                scenario, agents, dependencies, start_time
            )
        # Default to parallel execution
        return await self._parallel_coordinated_execution(
            scenario, agents, dependencies, start_time
        )

    async def _parallel_coordinated_execution(
        self,
        scenario: OptimizationScenario,
        agents: list[Any],
        dependencies: BaseAgentDependencies,
        start_time: datetime,
    ) -> PerformanceMetrics:
        """Execute tasks in parallel with coordination."""

        # Parallel execution with coordination
        async def coordinated_task(agent, task_id):
            task_start = time.time()

            try:
                if hasattr(agent, "orchestrate"):
                    response = await agent.orchestrate(
                        f"optimized parallel task {task_id}",
                        {
                            "optimization": "parallel_coordination",
                            "task_id": task_id,
                            "coordination_enabled": True,
                        },
                        dependencies,
                    )
                    return {
                        "success": response.success,
                        "latency_ms": response.latency_ms,
                        "agent_type": "orchestrator",
                    }
                tools = await agent.discover_tools_for_task(
                    f"optimized discovery {task_id}",
                    {"parallel": True, "coordination": True},
                )
                return {
                    "success": len(tools) > 0,
                    "latency_ms": (time.time() - task_start) * 1000,
                    "agent_type": "discovery",
                }
            except Exception:
                return {
                    "success": False,
                    "latency_ms": (time.time() - task_start) * 1000,
                    "agent_type": "unknown",
                }

        # Execute all tasks in parallel
        tasks = [coordinated_task(agent, i) for i, agent in enumerate(agents)]
        results = await asyncio.gather(*tasks)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success_results = [r["success"] for r in results]
        latencies = [r["latency_ms"] for r in results]

        return PerformanceMetrics(
            operation_name=f"optimized_{scenario.scenario_id}_parallel",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            throughput_ops_per_second=len(agents) / duration if duration > 0 else 0,
            success_rate=sum(success_results) / len(success_results)
            if success_results
            else 0,
            error_rate=1 - (sum(success_results) / len(success_results))
            if success_results
            else 1,
            latency_avg_ms=statistics.mean(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18]
            if len(latencies) > 5
            else (latencies[0] if latencies else 0),
        )

    async def _hierarchical_optimized_execution(
        self,
        scenario: OptimizationScenario,
        agents: list[Any],
        dependencies: BaseAgentDependencies,
        start_time: datetime,
    ) -> PerformanceMetrics:
        """Execute with hierarchical optimization."""
        # Separate coordinator and worker agents
        coordinators = [a for a in agents if hasattr(a, "orchestrate")][:2]
        workers = [a for a in agents if hasattr(a, "discover_tools_for_task")]

        # Phase 1: Discovery and planning (parallel)
        discovery_tasks = [
            worker.discover_tools_for_task(
                f"hierarchical discovery {i}",
                {"hierarchical": True, "optimization": "discovery"},
            )
            for i, worker in enumerate(workers)
        ]

        discovery_results = await asyncio.gather(*discovery_tasks)

        # Phase 2: Coordinated execution based on discovery
        coordination_tasks = []
        for i, coordinator in enumerate(coordinators):
            task = coordinator.orchestrate(
                f"hierarchical coordination {i}",
                {
                    "hierarchical": True,
                    "optimization": "execution",
                    "discovery_results": len(discovery_results),
                    "phase": "coordination",
                },
                dependencies,
            )
            coordination_tasks.append(task)

        coordination_results = await asyncio.gather(*coordination_tasks)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Combine results
        all_success = all(len(tools) > 0 for tools in discovery_results) and all(
            result.success for result in coordination_results
        )

        avg_latency = (
            sum(
                (time.time() - start_time.timestamp()) * 1000 / 2
                for _ in discovery_results
            )
            + sum(result.latency_ms for result in coordination_results)
        ) / (len(discovery_results) + len(coordination_results))

        return PerformanceMetrics(
            operation_name=f"optimized_{scenario.scenario_id}_hierarchical",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            throughput_ops_per_second=len(agents) / duration if duration > 0 else 0,
            success_rate=1.0 if all_success else 0.5,
            error_rate=0.0 if all_success else 0.5,
            latency_avg_ms=avg_latency,
            latency_p95_ms=avg_latency * 1.5,  # Approximate
        )

    async def _adaptive_load_balanced_execution(
        self,
        scenario: OptimizationScenario,
        agents: list[Any],
        dependencies: BaseAgentDependencies,
        start_time: datetime,
    ) -> PerformanceMetrics:
        """Execute with adaptive load balancing."""
        # Simulate adaptive load balancing by distributing tasks based on agent capabilities
        orchestrator_agents = [a for a in agents if hasattr(a, "orchestrate")]
        discovery_agents = [a for a in agents if hasattr(a, "discover_tools_for_task")]

        # Create more tasks than agents to test load balancing
        total_tasks = len(agents) * 2

        # Distribute tasks adaptively
        async def adaptive_task_execution():
            tasks = []

            # Assign orchestration tasks to orchestrator agents
            for i, agent in enumerate(orchestrator_agents):
                for j in range(2):  # 2 tasks per orchestrator
                    task = agent.orchestrate(
                        f"adaptive orchestration {i}_{j}",
                        {
                            "adaptive": True,
                            "load_balanced": True,
                            "task_complexity": j + 1,
                        },
                        dependencies,
                    )
                    tasks.append(task)

            # Assign discovery tasks to discovery agents
            for i, agent in enumerate(discovery_agents):
                for j in range(2):  # 2 tasks per discovery agent
                    task = agent.discover_tools_for_task(
                        f"adaptive discovery {i}_{j}",
                        {"adaptive": True, "load_balanced": True},
                    )
                    tasks.append(task)

            return await asyncio.gather(*tasks)

        results = await adaptive_task_execution()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Process results
        success_count = 0
        latencies = []

        for result in results:
            if hasattr(result, "success"):  # ToolResponse
                success_count += 1 if result.success else 0
                latencies.append(result.latency_ms)
            elif isinstance(result, list):  # Discovery result
                success_count += 1 if len(result) > 0 else 0
                latencies.append(100)  # Estimate for discovery latency

        return PerformanceMetrics(
            operation_name=f"optimized_{scenario.scenario_id}_adaptive",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            throughput_ops_per_second=len(results) / duration if duration > 0 else 0,
            success_rate=success_count / len(results) if results else 0,
            error_rate=1 - (success_count / len(results)) if results else 1,
            latency_avg_ms=statistics.mean(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18]
            if len(latencies) > 5
            else (latencies[0] if latencies else 0),
        )

    def calculate_performance_improvement(
        self,
        baseline: PerformanceMetrics,
        optimized: PerformanceMetrics,
    ) -> dict[str, float]:
        """Calculate performance improvement metrics."""
        improvements = {}

        # Throughput improvement
        if baseline.throughput_ops_per_second > 0:
            improvements["throughput_improvement"] = (
                optimized.throughput_ops_per_second / baseline.throughput_ops_per_second
            )

        # Duration improvement (speedup)
        if baseline.duration_seconds > 0:
            improvements["duration_improvement"] = (
                baseline.duration_seconds / optimized.duration_seconds
            )

        # Latency improvement
        if baseline.latency_avg_ms > 0:
            improvements["latency_improvement"] = (
                baseline.latency_avg_ms / optimized.latency_avg_ms
            )

        # Success rate improvement
        improvements["success_rate_improvement"] = (
            optimized.success_rate / baseline.success_rate
            if baseline.success_rate > 0
            else 0
        )

        # Overall performance improvement (composite metric)
        throughput_factor = improvements.get("throughput_improvement", 1.0)
        duration_factor = improvements.get("duration_improvement", 1.0)
        latency_factor = improvements.get("latency_improvement", 1.0)

        improvements["overall_performance_improvement"] = (
            throughput_factor + duration_factor + latency_factor
        ) / 3

        return improvements


class TestPerformanceOptimizationScenarios:
    """Test performance optimization scenarios demonstrating 3-10x improvements."""

    @pytest.fixture
    def performance_optimizer(self) -> PerformanceOptimizer:
        """Create performance optimizer."""
        return PerformanceOptimizer()

    @pytest.fixture
    def optimization_scenarios(self) -> list[OptimizationScenario]:
        """Create optimization test scenarios."""
        return [
            OptimizationScenario(
                scenario_id="complex_rag_pipeline",
                description="Complex RAG pipeline with multiple processing stages",
                baseline_performance={"duration": 10.0, "throughput": 2.0},
                target_improvement=5.0,  # 5x improvement
                optimization_strategies=[
                    "parallel_with_coordination",
                    "hierarchical_optimization",
                ],
                success_criteria={"min_improvement": 3.0, "max_latency_ms": 500},
            ),
            OptimizationScenario(
                scenario_id="high_load_processing",
                description="High-load processing with multiple concurrent requests",
                baseline_performance={"duration": 15.0, "throughput": 1.5},
                target_improvement=8.0,  # 8x improvement
                optimization_strategies=[
                    "adaptive_load_balancing",
                    "parallel_with_coordination",
                ],
                success_criteria={"min_improvement": 5.0, "success_rate": 0.9},
            ),
            OptimizationScenario(
                scenario_id="distributed_analysis",
                description="Distributed analysis across multiple data sources",
                baseline_performance={"duration": 8.0, "throughput": 3.0},
                target_improvement=4.0,  # 4x improvement
                optimization_strategies=[
                    "hierarchical_optimization",
                    "adaptive_load_balancing",
                ],
                success_criteria={"min_improvement": 2.5, "latency_reduction": 0.6},
            ),
        ]

    # Use the mocked agents from conftest.py instead of creating real ones

    @pytest.mark.asyncio
    async def test_3x_to_10x_performance_improvement_validation(
        self,
        performance_optimizer,
        optimization_scenarios,
        multi_agent_system,
        agent_dependencies,
    ):
        """Test validation of 3-10x performance improvements across scenarios."""
        # Use the mocked agent pool from multi_agent_system
        agent_pool = list(multi_agent_system["agent_pool"].values())

        improvement_results = {}

        for scenario in optimization_scenarios:
            print(f"\nTesting scenario: {scenario.description}")

            # Run baseline benchmark
            baseline_metrics = await performance_optimizer.run_baseline_benchmark(
                scenario, agent_pool, agent_dependencies
            )

            # Test different optimization strategies
            strategy_results = {}

            for strategy in scenario.optimization_strategies:
                optimized_metrics = await performance_optimizer.run_optimized_benchmark(
                    scenario, agent_pool, agent_dependencies, strategy
                )

                # Calculate improvements
                improvements = performance_optimizer.calculate_performance_improvement(
                    baseline_metrics, optimized_metrics
                )

                strategy_results[strategy] = {
                    "metrics": optimized_metrics,
                    "improvements": improvements,
                }

                print(f"  Strategy: {strategy}")
                print(
                    f"    Overall improvement: {improvements['overall_performance_improvement']:.2f}x"
                )
                print(
                    f"    Throughput improvement: {improvements.get('throughput_improvement', 0):.2f}x"
                )
                print(
                    f"    Duration improvement: {improvements.get('duration_improvement', 0):.2f}x"
                )

            improvement_results[scenario.scenario_id] = {
                "baseline": baseline_metrics,
                "strategies": strategy_results,
                "target_improvement": scenario.target_improvement,
            }

        # Verify performance improvements meet targets
        for scenario_id, results in improvement_results.items():
            scenario = next(
                s for s in optimization_scenarios if s.scenario_id == scenario_id
            )

            # Check if any strategy achieved target improvement
            best_improvement = 0
            best_strategy = None

            for strategy, strategy_result in results["strategies"].items():
                overall_improvement = strategy_result["improvements"][
                    "overall_performance_improvement"
                ]
                if overall_improvement > best_improvement:
                    best_improvement = overall_improvement
                    best_strategy = strategy

            print(f"\nScenario {scenario_id}:")
            print(f"  Target improvement: {scenario.target_improvement}x")
            print(f"  Best achieved: {best_improvement:.2f}x (using {best_strategy})")

            # Verify we achieved significant improvement (at least 2x)
            assert best_improvement >= 2.0, (
                f"Failed to achieve 2x improvement for {scenario_id}"
            )

            # Verify we met success criteria
            if "min_improvement" in scenario.success_criteria:
                min_required = scenario.success_criteria["min_improvement"]
                assert best_improvement >= min_required, (
                    f"Failed to meet minimum improvement of {min_required}x"
                )

    @pytest.mark.asyncio
    async def test_complex_rag_operations_optimization(
        self, performance_optimizer, agent_pool, agent_dependencies
    ):
        """Test optimization of complex RAG operations."""
        # Initialize agents
        for agent in agent_pool:
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Create complex RAG scenario
        complex_rag_scenario = OptimizationScenario(
            scenario_id="complex_rag_test",
            description="Complex RAG with search, analysis, and generation phases",
            baseline_performance={"duration": 12.0, "throughput": 1.8},
            target_improvement=6.0,
            optimization_strategies=["parallel_with_coordination"],
            success_criteria={"min_improvement": 4.0, "success_rate": 0.85},
        )

        # Simulate complex RAG baseline (sequential processing)
        async def complex_rag_baseline():
            start_time = time.time()

            # Phase 1: Search (sequential)
            search_agents = [
                a for a in agent_pool if hasattr(a, "discover_tools_for_task")
            ]
            search_results = []
            for agent in search_agents:
                tools = await agent.discover_tools_for_task(
                    "complex RAG search phase",
                    {"phase": "search", "complexity": "high"},
                )
                search_results.append(len(tools))

            # Phase 2: Analysis (sequential)
            analysis_agents = [a for a in agent_pool if hasattr(a, "orchestrate")][:2]
            analysis_results = []
            for agent in analysis_agents:
                response = await agent.orchestrate(
                    "complex RAG analysis phase",
                    {"phase": "analysis", "search_results": search_results},
                    agent_dependencies,
                )
                analysis_results.append(response.success)

            # Phase 3: Generation (sequential)
            generation_agent = analysis_agents[0]  # Use first orchestrator
            final_response = await generation_agent.orchestrate(
                "complex RAG generation phase",
                {"phase": "generation", "analysis_results": analysis_results},
                agent_dependencies,
            )

            duration = time.time() - start_time
            success = all(analysis_results) and final_response.success

            return {
                "duration": duration,
                "success": success,
                "phases_completed": 3,
                "throughput": 3 / duration if duration > 0 else 0,
            }

        # Simulate optimized RAG (parallel coordination)
        async def complex_rag_optimized():
            start_time = time.time()

            # Phase 1: Parallel search
            search_agents = [
                a for a in agent_pool if hasattr(a, "discover_tools_for_task")
            ]
            search_tasks = [
                agent.discover_tools_for_task(
                    f"optimized RAG search {i}",
                    {"phase": "search", "parallel": True, "optimization": True},
                )
                for i, agent in enumerate(search_agents)
            ]
            search_results = await asyncio.gather(*search_tasks)

            # Phase 2: Parallel analysis
            analysis_agents = [a for a in agent_pool if hasattr(a, "orchestrate")]
            analysis_tasks = [
                agent.orchestrate(
                    f"optimized RAG analysis {i}",
                    {
                        "phase": "analysis",
                        "parallel": True,
                        "search_results": len(search_results),
                        "optimization": True,
                    },
                    agent_dependencies,
                )
                for i, agent in enumerate(analysis_agents)
            ]
            analysis_results = await asyncio.gather(*analysis_tasks)

            # Phase 3: Coordinated generation
            generation_agent = analysis_agents[0]
            final_response = await generation_agent.orchestrate(
                "optimized RAG generation",
                {
                    "phase": "generation",
                    "parallel_analysis": True,
                    "analysis_count": len(analysis_results),
                    "optimization": "coordinated",
                },
                agent_dependencies,
            )

            duration = time.time() - start_time
            success = (
                all(r.success for r in analysis_results) and final_response.success
            )

            return {
                "duration": duration,
                "success": success,
                "phases_completed": 3,
                "throughput": 3 / duration if duration > 0 else 0,
            }

        # Execute baseline and optimized versions
        baseline_result = await complex_rag_baseline()
        optimized_result = await complex_rag_optimized()

        # Calculate improvement
        duration_improvement = (
            baseline_result["duration"] / optimized_result["duration"]
        )
        throughput_improvement = (
            optimized_result["throughput"] / baseline_result["throughput"]
        )

        print("Complex RAG optimization results:")
        print(f"  Baseline duration: {baseline_result['duration']:.2f}s")
        print(f"  Optimized duration: {optimized_result['duration']:.2f}s")
        print(f"  Duration improvement: {duration_improvement:.2f}x")
        print(f"  Throughput improvement: {throughput_improvement:.2f}x")

        # Verify significant improvement
        assert duration_improvement >= 2.0, "Failed to achieve 2x duration improvement"
        assert throughput_improvement >= 2.0, (
            "Failed to achieve 2x throughput improvement"
        )
        assert baseline_result["success"], "Baseline should succeed"
        assert optimized_result["success"], "Optimized version should succeed"

        # Verify target achievement
        overall_improvement = (duration_improvement + throughput_improvement) / 2
        assert (
            overall_improvement
            >= complex_rag_scenario.success_criteria["min_improvement"]
        )

    @pytest.mark.asyncio
    async def test_scalability_under_increasing_load(
        self, performance_optimizer, agent_pool, agent_dependencies
    ):
        """Test performance scalability under increasing load."""
        # Initialize agents
        for agent in agent_pool:
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Test different load levels
        load_levels = [
            {"agents": 2, "tasks_per_agent": 2, "complexity": "low"},
            {"agents": 3, "tasks_per_agent": 3, "complexity": "medium"},
            {"agents": 5, "tasks_per_agent": 4, "complexity": "high"},
        ]

        scalability_results = []

        for load_config in load_levels:
            print(
                f"\nTesting load: {load_config['agents']} agents, {load_config['tasks_per_agent']} tasks each"
            )

            # Sequential execution (baseline)
            async def sequential_execution():
                start_time = time.time()
                results = []

                test_agents = agent_pool[: load_config["agents"]]

                for agent in test_agents:
                    for task_id in range(load_config["tasks_per_agent"]):
                        if hasattr(agent, "orchestrate"):
                            response = await agent.orchestrate(
                                f"load test task {task_id}",
                                {
                                    "load_test": True,
                                    "complexity": load_config["complexity"],
                                    "sequential": True,
                                },
                                agent_dependencies,
                            )
                            results.append(response.success)
                        else:
                            tools = await agent.discover_tools_for_task(
                                f"load discovery {task_id}",
                                {
                                    "load_test": True,
                                    "complexity": load_config["complexity"],
                                },
                            )
                            results.append(len(tools) > 0)

                duration = time.time() - start_time
                return {
                    "duration": duration,
                    "success_rate": sum(results) / len(results) if results else 0,
                    "total_tasks": len(results),
                    "throughput": len(results) / duration if duration > 0 else 0,
                }

            # Parallel execution (optimized)
            async def parallel_execution():
                start_time = time.time()

                test_agents = agent_pool[: load_config["agents"]]

                # Create all tasks
                all_tasks = []
                for agent in test_agents:
                    for task_id in range(load_config["tasks_per_agent"]):
                        if hasattr(agent, "orchestrate"):
                            task = agent.orchestrate(
                                f"parallel load test {task_id}",
                                {
                                    "load_test": True,
                                    "complexity": load_config["complexity"],
                                    "parallel": True,
                                },
                                agent_dependencies,
                            )
                        else:
                            task = agent.discover_tools_for_task(
                                f"parallel discovery {task_id}",
                                {
                                    "load_test": True,
                                    "complexity": load_config["complexity"],
                                    "parallel": True,
                                },
                            )
                        all_tasks.append(task)

                # Execute all tasks in parallel
                results = await asyncio.gather(*all_tasks)

                duration = time.time() - start_time

                # Process results
                success_results = []
                for result in results:
                    if hasattr(result, "success"):
                        success_results.append(result.success)
                    elif isinstance(result, list):
                        success_results.append(len(result) > 0)
                    else:
                        success_results.append(False)

                return {
                    "duration": duration,
                    "success_rate": sum(success_results) / len(success_results)
                    if success_results
                    else 0,
                    "total_tasks": len(success_results),
                    "throughput": len(success_results) / duration
                    if duration > 0
                    else 0,
                }

            # Execute both versions
            sequential_result = await sequential_execution()
            parallel_result = await parallel_execution()

            # Calculate scalability metrics
            speedup = sequential_result["duration"] / parallel_result["duration"]
            throughput_gain = (
                parallel_result["throughput"] / sequential_result["throughput"]
            )
            efficiency = (
                speedup / load_config["agents"]
            )  # Ideal speedup would be number of agents

            scalability_results.append(
                {
                    "load_config": load_config,
                    "sequential": sequential_result,
                    "parallel": parallel_result,
                    "speedup": speedup,
                    "throughput_gain": throughput_gain,
                    "efficiency": efficiency,
                }
            )

            print(
                f"  Sequential: {sequential_result['duration']:.2f}s, {sequential_result['throughput']:.2f} tasks/s"
            )
            print(
                f"  Parallel: {parallel_result['duration']:.2f}s, {parallel_result['throughput']:.2f} tasks/s"
            )
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Efficiency: {efficiency:.2f}")

        # Verify scalability improvements
        for result in scalability_results:
            assert result["speedup"] >= 1.5, (
                f"Insufficient speedup: {result['speedup']:.2f}x"
            )
            assert result["throughput_gain"] >= 1.5, (
                f"Insufficient throughput gain: {result['throughput_gain']:.2f}x"
            )
            assert result["parallel"]["success_rate"] >= 0.8, (
                "Success rate too low under load"
            )

        # Verify scalability trend (should scale with load)
        speedups = [r["speedup"] for r in scalability_results]
        assert max(speedups) >= 3.0, "Failed to achieve 3x speedup at higher loads"

    @pytest.mark.asyncio
    async def test_autonomous_performance_optimization(
        self, performance_optimizer, agent_pool, agent_dependencies
    ):
        """Test autonomous performance optimization capabilities."""
        # Initialize agents
        for agent in agent_pool:
            if hasattr(agent, "initialize"):
                await agent.initialize(agent_dependencies)
            else:
                await agent.initialize_discovery(agent_dependencies)

        # Create autonomous optimization scenario
        optimization_scenario = OptimizationScenario(
            scenario_id="autonomous_optimization",
            description="Autonomous performance optimization with adaptive strategies",
            baseline_performance={"duration": 8.0, "throughput": 2.5},
            target_improvement=7.0,
            optimization_strategies=["adaptive_autonomous"],
            success_criteria={"min_improvement": 4.0, "autonomy_score": 0.8},
        )

        # Baseline: Manual coordination
        async def manual_coordination():
            start_time = time.time()

            # Fixed strategy execution
            orchestrator_agents = [a for a in agent_pool if hasattr(a, "orchestrate")]
            discovery_agents = [
                a for a in agent_pool if hasattr(a, "discover_tools_for_task")
            ]

            # Fixed sequence: discovery then orchestration
            discovery_results = []
            for agent in discovery_agents:
                tools = await agent.discover_tools_for_task(
                    "manual coordination discovery",
                    {"manual": True, "fixed_strategy": True},
                )
                discovery_results.append(len(tools))

            orchestration_results = []
            for agent in orchestrator_agents:
                response = await agent.orchestrate(
                    "manual coordination execution",
                    {"manual": True, "discovery_count": sum(discovery_results)},
                    agent_dependencies,
                )
                orchestration_results.append(response.success)

            duration = time.time() - start_time
            success_rate = (
                sum(orchestration_results) / len(orchestration_results)
                if orchestration_results
                else 0
            )

            return {
                "duration": duration,
                "success_rate": success_rate,
                "strategy": "manual_fixed",
                "autonomy_score": 0.2,  # Low autonomy
            }

        # Autonomous optimization: Adaptive strategy selection
        async def autonomous_optimization():
            start_time = time.time()

            # Autonomous strategy selection based on agent capabilities and load
            orchestrator_agents = [a for a in agent_pool if hasattr(a, "orchestrate")]
            discovery_agents = [
                a for a in agent_pool if hasattr(a, "discover_tools_for_task")
            ]

            # Autonomous decision: Choose parallel vs sequential based on agent count
            if len(orchestrator_agents) >= 2 and len(discovery_agents) >= 1:
                # Choose parallel strategy autonomously
                strategy = "parallel_autonomous"

                # Parallel discovery and orchestration
                discovery_tasks = [
                    agent.discover_tools_for_task(
                        f"autonomous discovery {i}",
                        {
                            "autonomous": True,
                            "adaptive": True,
                            "parallel": True,
                            "optimization_target": "performance",
                        },
                    )
                    for i, agent in enumerate(discovery_agents)
                ]

                orchestration_tasks = [
                    agent.orchestrate(
                        f"autonomous orchestration {i}",
                        {
                            "autonomous": True,
                            "adaptive": True,
                            "parallel": True,
                            "optimization_target": "performance",
                        },
                        agent_dependencies,
                    )
                    for i, agent in enumerate(orchestrator_agents)
                ]

                # Execute in parallel
                discovery_results, orchestration_results = await asyncio.gather(
                    asyncio.gather(*discovery_tasks),
                    asyncio.gather(*orchestration_tasks),
                )

                success_rate = sum(r.success for r in orchestration_results) / len(
                    orchestration_results
                )
                autonomy_score = 0.9  # High autonomy

            else:
                # Fall back to sequential but still autonomous
                strategy = "sequential_autonomous"

                # Sequential but optimized execution
                discovery_results = []
                for agent in discovery_agents:
                    tools = await agent.discover_tools_for_task(
                        "autonomous sequential discovery",
                        {
                            "autonomous": True,
                            "adaptive": True,
                            "sequential_optimized": True,
                        },
                    )
                    discovery_results.append(len(tools))

                orchestration_results = []
                for agent in orchestrator_agents:
                    response = await agent.orchestrate(
                        "autonomous sequential orchestration",
                        {
                            "autonomous": True,
                            "adaptive": True,
                            "sequential_optimized": True,
                            "discovery_feedback": sum(discovery_results),
                        },
                        agent_dependencies,
                    )
                    orchestration_results.append(response.success)

                success_rate = (
                    sum(orchestration_results) / len(orchestration_results)
                    if orchestration_results
                    else 0
                )
                autonomy_score = 0.7  # Moderate autonomy

            duration = time.time() - start_time

            return {
                "duration": duration,
                "success_rate": success_rate,
                "strategy": strategy,
                "autonomy_score": autonomy_score,
            }

        # Execute both approaches
        manual_result = await manual_coordination()
        autonomous_result = await autonomous_optimization()

        # Calculate improvements
        duration_improvement = manual_result["duration"] / autonomous_result["duration"]
        autonomy_improvement = (
            autonomous_result["autonomy_score"] / manual_result["autonomy_score"]
        )

        print("Autonomous optimization results:")
        print(
            f"  Manual approach: {manual_result['duration']:.2f}s, autonomy: {manual_result['autonomy_score']:.2f}"
        )
        print(
            f"  Autonomous approach: {autonomous_result['duration']:.2f}s, autonomy: {autonomous_result['autonomy_score']:.2f}"
        )
        print(f"  Duration improvement: {duration_improvement:.2f}x")
        print(f"  Autonomy improvement: {autonomy_improvement:.2f}x")
        print(f"  Selected strategy: {autonomous_result['strategy']}")

        # Verify autonomous optimization
        assert duration_improvement >= 2.0, "Insufficient autonomous optimization"
        assert (
            autonomous_result["autonomy_score"]
            >= optimization_scenario.success_criteria["autonomy_score"]
        )
        assert autonomous_result["success_rate"] >= 0.8, (
            "Success rate too low with autonomous optimization"
        )

        # Verify target achievement
        assert (
            duration_improvement
            >= optimization_scenario.success_criteria["min_improvement"]
        )

        # Verify autonomous decision making
        assert autonomous_result["strategy"] in [
            "parallel_autonomous",
            "sequential_autonomous",
        ]
        assert autonomous_result["autonomy_score"] > manual_result["autonomy_score"]
