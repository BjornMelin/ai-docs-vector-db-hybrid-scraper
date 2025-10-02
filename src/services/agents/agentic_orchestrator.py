"""Agentic orchestrator for dynamic tool composition.

Works with or without ``pydantic_ai`` installed. When unavailable, a deterministic
fallback path returns structured mock results so upstream systems remain stable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from ._compat import load_pydantic_ai
from .core import BaseAgent, BaseAgentDependencies
from .dynamic_tool_discovery import DynamicToolDiscovery


logger = logging.getLogger(__name__)

_PYA_AVAILABLE, _AgentCls, _RunCtx = load_pydantic_ai()


class ToolRequest(BaseModel):
    """Request for autonomous tool orchestration."""

    task: str = Field(..., description="Task description or query")
    constraints: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    """Response from autonomous tool orchestration."""

    success: bool = Field(...)
    results: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)
    reasoning: str = Field(...)
    latency_ms: float = Field(...)
    confidence: float = Field(...)


class AgenticOrchestrator(BaseAgent):
    """Pydantic-AI orchestrator for tool composition."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1) -> None:
        super().__init__(
            name="agentic_orchestrator",
            model=model,
            temperature=temperature,
            max_tokens=1500,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are a tool orchestrator. Analyze the task, select optimal tools, "
            "execute them (sequential/parallel), and explain your reasoning."
        )

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:  # noqa: ARG002
        """Register orchestration helpers on the pydantic-ai agent if available."""
        fallback_reason = getattr(self, "_fallback_reason", None)
        if not _PYA_AVAILABLE or self.agent is None:
            logger.warning(
                "AgenticOrchestrator fallback mode active (reason: %s)",
                fallback_reason or "pydantic_ai_unavailable",
            )
            return

        @self.agent.tool_plain  # type: ignore[reportAttributeAccessIssue]
        async def orchestrate_tools(request: ToolRequest) -> ToolResponse:
            return await self._orchestrate_autonomous(request, deps)

        @self.agent.tool_plain  # type: ignore[reportAttributeAccessIssue]
        async def discover_available_tools() -> dict[str, Any]:
            return await self._discover_tools(deps)

        @self.agent.tool_plain  # type: ignore[reportAttributeAccessIssue]
        async def execute_tool_chain(
            tools: list[str], input_data: dict[str, Any]
        ) -> dict[str, Any]:
            return await self._execute_chain(tools, input_data, deps)

        logger.info("AgenticOrchestrator initialized with Pydantic-AI tools")

    async def orchestrate(
        self, task: str, constraints: dict[str, Any], deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Main entry point for tool orchestration."""
        if not self._initialized:
            await self.initialize(deps)

        request = ToolRequest(
            task=task,
            constraints=constraints,
            context={"session_id": deps.session_state.session_id},
        )

        if not _PYA_AVAILABLE or self.agent is None:
            return await self._fallback_orchestrate(request, deps)

        try:
            result = await self.agent.run(  # type: ignore[attr-defined]
                f"Orchestrate tools for task: {task}", deps=deps
            )
            data = getattr(result, "data", None)
            if isinstance(data, ToolResponse):
                response = data
            else:
                response = ToolResponse(
                    success=True,
                    results={"agent_response": str(data)},
                    tools_used=["agent"],
                    reasoning="Pydantic-AI agent orchestration",
                    latency_ms=0.0,
                    confidence=0.8,
                )

            deps.session_state.increment_tool_usage("agentic_orchestrator")
            deps.session_state.add_interaction(
                role="orchestrator",
                content=response.reasoning,
                metadata={
                    "tools_used": response.tools_used,
                    "latency_ms": response.latency_ms,
                    "confidence": response.confidence,
                },
            )
            return response

        except (RuntimeError, ValueError) as exc:
            logger.exception("Orchestration failed")
            return ToolResponse(
                success=False,
                results={"error": str(exc)},
                tools_used=[],
                reasoning=f"Orchestration failed: {exc}",
                latency_ms=0.0,
                confidence=0.0,
            )

    async def _orchestrate_autonomous(
        self, request: ToolRequest, deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Core orchestration logic (when pydantic-ai available)."""
        start = time.time()
        try:
            available = await self._discover_tools(deps)
            selected = self._select_tools_for_task(
                request.task, available, request.constraints
            )
            results = await self._execute_chain(
                selected,
                {
                    "task": request.task,
                    "constraints": request.constraints,
                    **request.context,
                },
                deps,
            )
            latency_ms = (time.time() - start) * 1000.0
            return ToolResponse(
                success=True,
                results=results,
                tools_used=selected,
                reasoning=self._generate_reasoning(request.task, selected, results),
                latency_ms=latency_ms,
                confidence=self._calculate_confidence(results, selected),
            )
        except Exception as exc:  # pragma: no cover - safety
            latency_ms = (time.time() - start) * 1000.0
            return ToolResponse(
                success=False,
                results={"error": str(exc)},
                tools_used=[],
                reasoning=f"Orchestration failed due to: {exc}",
                latency_ms=latency_ms,
                confidence=0.0,
            )

    async def _discover_tools(self, deps: BaseAgentDependencies) -> dict[str, Any]:
        """Discover tools via the shared DynamicToolDiscovery engine."""
        engine = DynamicToolDiscovery()
        await engine.initialize_discovery(deps)
        # Return a light-weight index for selection heuristics.
        return {
            t.name: {
                "capability": t.capability_type.value,
                "description": t.description,
                "latency_ms": (t.metrics.average_latency_ms if t.metrics else 0.0),
            }
            for t in engine.discovered_tools.values()
        }

    def _select_tools_for_task(
        self, task: str, available: dict[str, Any], constraints: dict[str, Any]
    ) -> list[str]:
        """Select tools using simple keyword heuristics plus constraints."""
        tl = task.lower()
        selected: list[str] = []

        if (
            any(k in tl for k in ["search", "find", "lookup"])
            and "hybrid_search" in available
        ):
            selected.append("hybrid_search")
        if any(k in tl for k in ["generate", "answer", "explain"]):
            if "hybrid_search" in available and "hybrid_search" not in selected:
                selected.append("hybrid_search")
            if "rag_generation" in available:
                selected.append("rag_generation")
        if (
            any(k in tl for k in ["analyze", "classify", "assess"])
            and "content_analysis" in available
        ):
            selected.append("content_analysis")
        if not selected and "hybrid_search" in available:
            selected.append("hybrid_search")

        if constraints.get("max_latency_ms", 5000) < 1000:
            selected = [t for t in selected if t != "rag_generation"]
        return selected

    async def _execute_chain(
        self,
        tools: list[str],
        input_data: dict[str, Any],
        _deps: BaseAgentDependencies,
    ) -> dict[str, Any]:
        """Execute a chain of tools sequentially."""
        results: dict[str, Any] = {}
        context = dict(input_data)
        for name in tools:
            try:
                out = await self._execute_tool(name, context)
                results[f"{name}_result"] = out
                context.update(out)
            except Exception as exc:  # pragma: no cover - safety
                logger.warning("Tool %s failed: %s", name, exc)
                results[f"{name}_error"] = str(exc)
        return results

    async def _execute_tool(
        self, tool_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock tool execution (replace with MCP integration)."""
        await asyncio.sleep(0.1)
        return {
            "tool": tool_name,
            "result": f"Mock result from {tool_name}",
            "input_keys": sorted(context.keys()),
            "loop_time": asyncio.get_running_loop().time(),
        }

    def _generate_reasoning(
        self, task: str, tools_used: list[str], results: dict[str, Any]
    ) -> str:
        """Generate reasoning text for orchestration decisions."""
        reason = (
            f"For task '{task}', selected {len(tools_used)} tool(s): "
            f"{', '.join(tools_used)}. "
        )
        if "hybrid_search" in tools_used:
            reason += "Used hybrid search for information retrieval. "
        if "rag_generation" in tools_used:
            reason += "Applied RAG generation for comprehensive answers. "
        if "content_analysis" in tools_used:
            reason += "Performed content analysis for deeper insights. "
        ok = len([k for k in results if not k.endswith("_error")])
        reason += f"Successfully executed {ok} steps."
        return reason

    def _calculate_confidence(
        self, results: dict[str, Any], tools_used: list[str]
    ) -> float:
        """Calculate confidence score based on tool execution results."""
        if not results or not tools_used:
            return 0.0
        successful = len([k for k in results if not k.endswith("_error")])
        base = successful / len(tools_used) if tools_used else 0.0
        if successful > 1:
            base = min(base * 1.1, 1.0)
        return round(base, 2)

    async def _fallback_orchestrate(
        self, request: ToolRequest, deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Heuristic fallback orchestration used when pydantic-ai isn't available."""
        fallback_reason = getattr(self, "_fallback_reason", "unknown")
        logger.warning("Using fallback orchestrator (reason: %s)", fallback_reason)
        start = time.time()
        tl = request.task.lower()

        if any(k in tl for k in ["search", "find", "retrieve"]):
            tools, result_data, conf = (
                ["mock_search_tool"],
                {
                    "search_results": f"Mock search results for: {request.task}",
                    "result_count": 5,
                    "search_type": "fallback_search",
                },
                0.7,
            )
            strategy = "Fallback search orchestration"
        elif any(k in tl for k in ["analyze", "examine", "evaluate"]):
            tools, result_data, conf = (
                ["mock_analysis_tool"],
                {
                    "analysis_results": f"Mock analysis of: {request.task}",
                    "confidence_score": 0.65,
                    "analysis_type": "fallback_analysis",
                },
                0.65,
            )
            strategy = "Fallback analysis orchestration"
        elif any(k in tl for k in ["generate", "create", "compose"]):
            tools, result_data, conf = (
                ["mock_generation_tool"],
                {
                    "generated_content": (
                        f"Mock generated content for: {request.task}"
                    ),
                    "word_count": 150,
                    "generation_type": "fallback_generation",
                },
                0.6,
            )
            strategy = "Fallback generation orchestration"
        else:
            tools, result_data, conf = (
                ["mock_general_tool"],
                {
                    "general_response": f"Mock response for task: {request.task}",
                    "task_type": "general",
                    "processing_mode": "fallback",
                },
                0.5,
            )
            strategy = "Fallback general orchestration"

        latency_ms = (time.time() - start) * 1000.0
        response = ToolResponse(
            success=True,
            results={
                "fallback_mode": True,
                "fallback_reason": fallback_reason,
                **result_data,
            },
            tools_used=tools,
            reasoning=f"{strategy} (fallback: {fallback_reason})",
            latency_ms=latency_ms,
            confidence=conf,
        )
        deps.session_state.increment_tool_usage("agentic_orchestrator")
        deps.session_state.add_interaction(
            role="orchestrator",
            content=response.reasoning,
            metadata={
                "tools_used": response.tools_used,
                "latency_ms": response.latency_ms,
                "confidence": response.confidence,
                "fallback_mode": True,
                "fallback_reason": fallback_reason,
            },
        )
        return response
