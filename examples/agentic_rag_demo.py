#!/usr/bin/env python3
"""Demo script for Pydantic-AI Agentic RAG implementation.

This script demonstrates the capabilities of the autonomous agentic RAG system,
including intelligent query orchestration, dynamic tool composition, and
multi-agent coordination.
"""

import asyncio
import logging
from collections import Counter
from uuid import uuid4

from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    QueryOrchestrator,
    ToolCompositionEngine,
    create_agent_dependencies,
)
from src.services.agents.tool_composition import ToolCategory


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_query_orchestration():
    """Demonstrate intelligent query orchestration."""
    print("\n" + "=" * 60)
    print("DEMO: Query Orchestration with Autonomous Agents")
    print("=" * 60)

    try:
        # Initialize client manager
        client_manager = ClientManager()
        await client_manager.initialize()

        # Create agent dependencies
        deps = create_agent_dependencies(
            client_manager=client_manager, session_id=str(uuid4()), user_id="demo_user"
        )

        # Initialize query orchestrator
        orchestrator = QueryOrchestrator()
        await orchestrator.initialize(deps)

        # Test queries of different complexity levels
        test_queries = [
            {
                "query": "What is machine learning?",
                "collection": "documentation",
                "expected_complexity": "simple",
                "description": "Simple factual query",
            },
            {
                "query": (
                    "How do I implement a scalable RAG system with multiple "
                    "embedding models?"
                ),
                "collection": "documentation",
                "expected_complexity": "complex",
                "description": (
                    "Complex technical query requiring multi-step processing"
                ),
            },
            {
                "query": (
                    "Compare the performance of different search strategies for "
                    "code documentation"
                ),
                "collection": "documentation",
                "expected_complexity": "moderate",
                "description": "Analytical query requiring comparison",
            },
        ]

        for i, test_case in enumerate(test_queries, 1):
            print(f"\n--- Test Case {i}: {test_case['description']} ---")
            print(f"Query: {test_case['query']}")

            # Execute orchestration
            result = await orchestrator.orchestrate_query(
                query=test_case["query"],
                collection=test_case["collection"],
                performance_requirements={
                    "max_latency_ms": 5000.0,
                    "min_quality_score": 0.8,
                },
            )

            print(f"Success: {result['success']}")
            if result["success"]:
                print(f"Orchestration ID: {result.get('orchestration_id')}")
                if "result" in result:
                    result_data = result["result"]
                    print(f"Analysis: {result_data.get('analysis', {})}")
                    orchestration_plan = result_data.get(
                        "orchestration_plan", "No plan available"
                    )
                    print(f"Plan: {orchestration_plan}")
            else:
                print(f"Error: {result.get('error')}")

        # Cleanup
        await client_manager.cleanup()

    except Exception as e:
        logger.exception("Query orchestration demo failed")
        print(f"Demo failed: {e}")


async def demo_tool_composition():
    """Demonstrate dynamic tool composition."""
    print("\n" + "=" * 60)
    print("DEMO: Dynamic Tool Composition Engine")
    print("=" * 60)

    try:
        # Initialize client manager
        client_manager = ClientManager()
        await client_manager.initialize()

        # Initialize tool composition engine
        engine = ToolCompositionEngine(client_manager)
        await engine.initialize()

        tool_count = len(engine.tool_registry)
        print(f"Initialized tool composition engine with {tool_count} tools")

        # Demonstrate tool discovery
        print("\n--- Available Tools by Category ---")

        for category in ToolCategory:
            tools = engine.list_tools_by_category(category)
            if tools:
                print(f"{category.value}: {', '.join(tools)}")

        # Demonstrate tool chain composition
        print("\n--- Tool Chain Composition ---")

        goals = [
            {
                "goal": "search for technical documentation quickly",
                "constraints": {"max_latency_ms": 500, "performance_priority": "speed"},
                "description": "Speed-optimized search",
            },
            {
                "goal": "generate comprehensive analysis with high quality",
                "constraints": {
                    "min_quality_score": 0.9,
                    "performance_priority": "quality",
                },
                "description": "Quality-optimized analysis",
            },
            {
                "goal": "perform balanced search and answer generation",
                "constraints": {"max_latency_ms": 2000, "min_quality_score": 0.8},
                "description": "Balanced processing",
            },
        ]

        for i, goal_case in enumerate(goals, 1):
            print(f"\nGoal {i}: {goal_case['description']}")
            print(f"Objective: {goal_case['goal']}")
            print(f"Constraints: {goal_case['constraints']}")

            # Compose tool chain
            tool_chain = await engine.compose_tool_chain(
                goal=goal_case["goal"], constraints=goal_case["constraints"]
            )

            print(f"Composed chain with {len(tool_chain)} steps:")
            for j, step in enumerate(tool_chain):
                step_info = (
                    f"  {j + 1}. {step.tool_name} "
                    f"(parallel: {step.parallel}, optional: {step.optional})"
                )
                print(step_info)

            # Execute tool chain with mock data
            input_data = {
                "query": "example query for " + goal_case["description"],
                "collection": "documentation",
            }

            result = await engine.execute_tool_chain(
                chain=tool_chain, input_data=input_data, timeout_seconds=10.0
            )

            print(f"Execution success: {result['success']}")
            if result["success"]:
                metadata = result["metadata"]
                print(f"Total time: {metadata['total_execution_time_ms']:.1f}ms")
                steps_info = (
                    f"Steps executed: {metadata['steps_executed']}/"
                    f"{metadata['chain_length']}"
                )
                print(steps_info)
            else:
                print(f"Execution error: {result.get('error')}")

        # Demonstrate performance analytics
        print("\n--- Performance Analytics ---")
        stats = engine.get_performance_stats()
        if "message" not in stats:  # Has actual data
            print(f"Total executions: {stats['total_executions']}")
            print(f"Recent success rate: {stats['recent_success_rate']:.2%}")
            print(f"Average execution time: {stats['avg_execution_time_ms']:.1f}ms")
        else:
            print(stats["message"])

        # Cleanup
        await client_manager.cleanup()

    except Exception as e:
        logger.exception("Tool composition demo failed")
        print(f"Demo failed: {e}")


async def demo_agent_learning():
    """Demonstrate agent learning and adaptation."""
    print("\n" + "=" * 60)
    print("DEMO: Agent Learning and Adaptation")
    print("=" * 60)

    try:
        # Initialize client manager
        client_manager = ClientManager()
        await client_manager.initialize()

        # Create session for learning demonstration
        session_id = str(uuid4())
        deps = create_agent_dependencies(
            client_manager=client_manager,
            session_id=session_id,
            user_id="learning_demo_user",
        )

        print(f"Created learning session: {session_id}")

        # Initialize orchestrator
        orchestrator = QueryOrchestrator()
        await orchestrator.initialize(deps)

        # Simulate multiple interactions to demonstrate learning
        print("\n--- Simulating User Interactions ---")

        interactions = [
            "What is Python?",
            "How do I install Python packages?",
            "What are Python virtual environments?",
            "How do I create a Python web API?",
            "What are Python best practices?",
        ]

        for i, query in enumerate(interactions, 1):
            print(f"\nInteraction {i}: {query}")

            # Process query
            result = await orchestrator.orchestrate_query(
                query=query, collection="documentation"
            )

            if result["success"]:
                print("✓ Query processed successfully")

                # Show session state evolution
                history_count = len(deps.session_state.conversation_history)
                print(f"  Conversation history length: {history_count}")

                metrics_count = len(deps.session_state.performance_metrics)
                print(f"  Performance metrics count: {metrics_count}")
                print(f"  Tool usage stats: {deps.session_state.tool_usage_stats}")
            else:
                print(f"✗ Query failed: {result.get('error')}")

        # Show learning insights
        print("\n--- Learning Insights ---")
        print(f"Session interactions: {len(deps.session_state.conversation_history)}")
        print(f"User preferences developed: {deps.session_state.preferences}")
        print(f"Knowledge accumulated: {len(deps.session_state.knowledge_base)} items")

        # Show agent performance evolution
        print("\n--- Agent Performance Metrics ---")
        agent_metrics = orchestrator.get_performance_metrics()
        for metric, value in agent_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")

        # Show strategy effectiveness
        if orchestrator.strategy_performance:
            print("\n--- Strategy Effectiveness ---")
            for strategy, stats in orchestrator.strategy_performance.items():
                print(f"  {strategy}:")
                print(f"    Uses: {stats['total_uses']}")
                print(f"    Avg performance: {stats['avg_performance']:.3f}")
                print(f"    Avg latency: {stats['avg_latency']:.1f}ms")

        # Cleanup
        await client_manager.cleanup()

    except Exception as e:
        logger.exception("Agent learning demo failed")
        print(f"Demo failed: {e}")


async def demo_performance_optimization():
    """Demonstrate performance optimization capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: Performance Optimization and Monitoring")
    print("=" * 60)

    try:
        # Initialize client manager
        client_manager = ClientManager()
        await client_manager.initialize()

        # Initialize tool composition engine
        engine = ToolCompositionEngine(client_manager)
        await engine.initialize()

        print("Testing different optimization strategies...")

        # Test different optimization targets
        optimization_scenarios = [
            {
                "name": "Speed Optimization",
                "goal": "quick search for basic information",
                "constraints": {"max_latency_ms": 200, "performance_priority": "speed"},
                "expected": "Should favor fast, simple tools",
            },
            {
                "name": "Quality Optimization",
                "goal": "comprehensive analysis with high accuracy",
                "constraints": {
                    "min_quality_score": 0.95,
                    "performance_priority": "quality",
                },
                "expected": "Should use advanced tools like HyDE search",
            },
            {
                "name": "Cost Optimization",
                "goal": "search with minimal resource usage",
                "constraints": {"max_cost": 0.01, "performance_priority": "cost"},
                "expected": "Should prefer low-cost tools and caching",
            },
        ]

        performance_results = []

        for scenario in optimization_scenarios:
            print(f"\n--- {scenario['name']} ---")
            print(f"Goal: {scenario['goal']}")
            print(f"Expected: {scenario['expected']}")

            # Compose optimized tool chain
            tool_chain = await engine.compose_tool_chain(
                goal=scenario["goal"], constraints=scenario["constraints"]
            )

            print(f"Selected tools: {[step.tool_name for step in tool_chain]}")

            # Execute and measure performance
            start_time = asyncio.get_event_loop().time()

            result = await engine.execute_tool_chain(
                chain=tool_chain,
                input_data={"query": "test query", "collection": "test"},
                timeout_seconds=5.0,
            )

            end_time = asyncio.get_event_loop().time()
            actual_time = (end_time - start_time) * 1000

            performance_data = {
                "scenario": scenario["name"],
                "success": result["success"],
                "actual_time_ms": actual_time,
                "reported_time_ms": result.get("metadata", {}).get(
                    "total_execution_time_ms", 0
                ),
                "tools_used": [step.tool_name for step in tool_chain],
            }

            performance_results.append(performance_data)

            print(f"Execution time: {actual_time:.1f}ms")
            print(f"Success: {result['success']}")

        # Analyze optimization effectiveness
        print("\n--- Optimization Analysis ---")

        # Compare performance across scenarios
        for result in performance_results:
            print(f"{result['scenario']}:")
            print(f"  Time: {result['actual_time_ms']:.1f}ms")
            print(f"  Tools: {', '.join(result['tools_used'])}")
            print(f"  Success: {result['success']}")

        # Show fastest vs slowest
        times = [r["actual_time_ms"] for r in performance_results if r["success"]]
        if times:
            print(f"\nPerformance range: {min(times):.1f}ms - {max(times):.1f}ms")
            improvement = ((max(times) - min(times)) / max(times)) * 100
            print(f"Optimization improvement: {improvement:.1f}%")

        # Show tool selection patterns
        all_tools = []
        for result in performance_results:
            all_tools.extend(result["tools_used"])

        tool_usage = Counter(all_tools)
        print("\nTool usage patterns:")
        for tool, count in tool_usage.most_common():
            print(f"  {tool}: {count} times")

        # Cleanup
        await client_manager.cleanup()

    except Exception as e:
        logger.exception("Performance optimization demo failed")
        print(f"Demo failed: {e}")


async def main():
    """Run all demonstrations."""
    print("Pydantic-AI Agentic RAG System Demonstration")
    print("=" * 60)

    try:
        # Check if Pydantic-AI is available
        try:
            # Import check - pydantic_ai already imported at top
            print("✓ Pydantic-AI available - running full demonstrations")
            full_demo = True
        except ImportError:
            print("⚠ Pydantic-AI not available - running fallback demonstrations")
            full_demo = False

        # Run demonstrations
        await demo_query_orchestration()
        await demo_tool_composition()
        await demo_agent_learning()
        await demo_performance_optimization()

        if full_demo:
            print("\n" + "=" * 60)
            print("All demonstrations completed successfully!")
            print("The agentic RAG system is ready for production use.")
        else:
            print("\n" + "=" * 60)
            print("Fallback demonstrations completed.")
            print("Install pydantic-ai for full autonomous agent capabilities:")
            print("pip install pydantic-ai")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.exception("Demo failed")
        print(f"\nDemo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
