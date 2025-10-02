"""Query Orchestrator agent.

Analyzes queries, delegates to specialists, and coordinates multi-stage
retrieval. Operates with Pydantic-AI when available, otherwise provides a
deterministic fallback.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from ._compat import load_pydantic_ai
from .core import BaseAgent, BaseAgentDependencies


logger = logging.getLogger(__name__)

_PYA_AVAILABLE, _AgentCls, _RunCtx = load_pydantic_ai()


class QueryOrchestrator(BaseAgent):
    """Master agent that coordinates query processing workflow."""

    def __init__(self, model: str = "gpt-4") -> None:
        super().__init__(
            name="query_orchestrator", model=model, temperature=0.1, max_tokens=1500
        )
        self.strategy_performance: dict[str, dict[str, float]] = {}

    def get_system_prompt(self) -> str:
        return (
            "You are a Query Orchestrator responsible for coordinating "
            "query processing. Analyze, delegate, coordinate retrieval, and "
            "optimize for latency and cost."
        )

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize tool functions if Pydantic-AI is available."""
        fallback_reason = getattr(self, "_fallback_reason", None)
        if not _PYA_AVAILABLE or self.agent is None:
            reason = fallback_reason or "pydantic_ai_unavailable"
            logger.warning("QueryOrchestrator using fallback (reason: %s)", reason)
            return

        @_AgentCls.tool  # type: ignore[attr-defined]
        async def analyze_query_intent(
            ctx: _RunCtx[BaseAgentDependencies],  # type: ignore[valid-type]
            query: str,
            user_context: dict[str, Any] | None = None,  # noqa: ARG001
        ) -> dict[str, Any]:
            ctx.deps.session_state.increment_tool_usage("analyze_query_intent")
            query_lower = query.lower()

            complexity_indicators = {
                "simple": ["what is", "who is", "when did", "where is", "define"],
                "moderate": [
                    "how to",
                    "why does",
                    "compare",
                    "difference between",
                ],
                "complex": ["analyze", "evaluate", "recommend", "strategy"],
            }

            complexity = "moderate"
            for level, indicators in complexity_indicators.items():
                if any(ind in query_lower for ind in indicators):
                    complexity = level
                    break

            domains = {
                "technical": ["code", "programming", "api", "database", "algorithm"],
                "business": ["market", "revenue", "strategy", "customer", "sales"],
                "academic": ["research", "study", "theory", "paper"],
            }

            domain = "general"
            for dname, keywords in domains.items():
                if any(k in query_lower for k in keywords):
                    domain = dname
                    break

            if complexity == "simple":
                strategy = "fast"
            elif complexity == "complex":
                strategy = "comprehensive"
            else:
                strategy = "balanced"

            multi_step_indicators = [
                "and then",
                "after that",
                "step by step",
                "process",
                "first",
                "second",
                "finally",
            ]
            requires_multi_step = any(
                ind in query_lower for ind in multi_step_indicators
            )

            return {
                "query": query,
                "complexity": complexity,
                "domain": domain,
                "strategy": strategy,
                "requires_multi_step": requires_multi_step,
                "confidence": 0.8,
                "reasoning": f"Classified as {complexity} based on query patterns",
                "estimated_tokens": len(query.split()) * 4,
                "recommended_tools": self._recommend_tools(complexity, domain),
            }

        logger.info("QueryOrchestrator tools initialized")

    def _recommend_tools(self, complexity: str, domain: str) -> list[str]:
        """Recommend tools based on query characteristics."""
        base_tools = ["hybrid_search"]
        if complexity == "complex":
            base_tools.extend(["hyde_search", "multi_stage_search"])
        if domain == "technical":
            base_tools.append("content_classification")
        return base_tools

    def _estimate_completion_time(
        self, agent_type: str, task_data: dict[str, Any]
    ) -> float:
        """Estimate task completion time for an agent type."""
        base_times = {
            "retrieval_specialist": 2.0,
            "answer_generator": 3.0,
            "tool_selector": 1.0,
        }
        base_time = base_times.get(agent_type, 2.0)
        complexity_multipliers = {"simple": 0.5, "moderate": 1.0, "complex": 2.0}
        complexity = task_data.get("complexity", "moderate")
        return base_time * complexity_multipliers.get(complexity, 1.0)

    def _get_strategy_recommendation(self, stats: dict[str, float]) -> str:
        """Get recommendation based on strategy performance."""
        avg_performance = stats.get("avg_performance", 0.5)
        if avg_performance > 0.8:
            return "continue_using"
        if avg_performance > 0.6:
            return "monitor_performance"
        return "consider_alternative"

    async def orchestrate_query(
        self,
        query: str,
        collection: str = "documentation",
        user_context: dict[str, Any] | None = None,
        performance_requirements: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """High-level orchestration for a complete query workflow."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        execution_context = {
            "query": query,
            "collection": collection,
            "user_context": user_context or {},
            "performance_requirements": performance_requirements or {},
            "orchestration_id": str(uuid4()),
        }

        logger.info("Starting query orchestration for: %s", query[:50])

        try:
            if _PYA_AVAILABLE and self.agent is not None:
                result = await self.agent.run(  # type: ignore[attr-defined]
                    f"Orchestrate processing for this query: {query}\n"
                    f"Collection: {collection}\n"
                    f"User context: {user_context}\n"
                    f"Performance requirements: {performance_requirements}\n"
                    "Provide a complete orchestration plan and execute it."
                )

                return {
                    "success": True,
                    "result": result.data,
                    "orchestration_id": execution_context["orchestration_id"],
                }

            return await self._fallback_orchestration(execution_context)

        except Exception as exc:  # pragma: no cover - safety
            logger.exception("Query orchestration failed")
            return {
                "success": False,
                "error": str(exc),
                "orchestration_id": execution_context["orchestration_id"],
            }

    async def _fallback_orchestration(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fallback orchestration when Pydantic-AI is not available."""
        query = context["query"]
        fallback_reason = getattr(self, "_fallback_reason", "unknown")

        logger.info("Using fallback query orchestration (reason: %s)", fallback_reason)

        ql = query.lower()
        if any(k in ql for k in ["what is", "who is", "define"]):
            complexity, strategy, tools = "simple", "fast", ["basic_search"]
        elif any(k in ql for k in ["analyze", "compare", "evaluate", "complex"]):
            complexity, strategy, tools = (
                "complex",
                "comprehensive",
                ["advanced_search", "content_analysis", "synthesis"],
            )
        else:
            complexity, strategy, tools = (
                "moderate",
                "balanced",
                ["hybrid_search", "content_analysis"],
            )

        if any(k in ql for k in ["code", "programming", "api"]):
            domain = "technical"
            tools.append("technical_analyzer")
        elif any(k in ql for k in ["business", "market", "revenue"]):
            domain = "business"
            tools.append("business_analyzer")
        else:
            domain = "general"

        analysis = {
            "complexity": complexity,
            "domain": domain,
            "strategy": strategy,
            "recommended_tools": tools,
            "confidence": 0.7,
            "reasoning": (
                f"Fallback analysis based on keyword patterns "
                f"(reason: {fallback_reason})"
            ),
            "fallback_mode": True,
        }

        orchestration_plan = {
            "steps": [
                f"1. Analyze query: '{query}' (complexity: {complexity})",
                f"2. Apply {strategy} strategy",
                f"3. Use tools: {', '.join(tools)}",
                "4. Generate response with fallback capabilities",
            ],
            "estimated_time_seconds": 2.0
            if complexity == "simple"
            else 5.0
            if complexity == "moderate"
            else 8.0,
            "quality_expectation": "basic" if fallback_reason else "standard",
        }

        return {
            "success": True,
            "result": {
                "analysis": analysis,
                "orchestration_plan": orchestration_plan,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "mock_results": {
                    "query_processed": query,
                    "tools_executed": tools,
                    "processing_time_ms": 100.0,
                    "confidence": 0.7,
                    "fallback_response": (f"Mock orchestrated response for: {query}"),
                },
            },
            "orchestration_id": context["orchestration_id"],
        }
