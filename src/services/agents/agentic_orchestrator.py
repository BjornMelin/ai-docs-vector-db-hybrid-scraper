"""Pure Pydantic-AI native agentic orchestrator.

This module implements the clean, optimal solution defined in our
comprehensive research,
replacing the 950-line ToolCompositionEngine with ~150-300 lines of native patterns.

Based on research findings:
- G1-G5: Pydantic-AI native capabilities validation (98% confidence)
- J3: Dynamic Tool Composition Engine research (96% confidence)
- COMPREHENSIVE_SYNTHESIS_REPORT.md: Complete modernization strategy
"""

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field


try:
    from pydantic_ai import Agent, RunContext

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None

from .core import BaseAgent, BaseAgentDependencies


logger = logging.getLogger(__name__)


class ToolRequest(BaseModel):
    """Request for autonomous tool orchestration."""

    task: str = Field(..., description="Task description or query")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Performance constraints"
    )
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class ToolResponse(BaseModel):
    """Response from autonomous tool orchestration."""

    success: bool = Field(..., description="Whether orchestration succeeded")
    results: dict[str, Any] = Field(
        default_factory=dict, description="Orchestration results"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Tools selected and executed"
    )
    reasoning: str = Field(..., description="Agent decision reasoning")
    latency_ms: float = Field(..., description="Total execution time")
    confidence: float = Field(..., description="Confidence in results")


class AgenticOrchestrator(BaseAgent):
    """Pure Pydantic-AI native orchestrator for autonomous tool composition.

    This replaces the 950-line ToolCompositionEngine with intelligent,
    autonomous orchestration using native Pydantic-AI patterns only.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the agentic orchestrator.

        Args:
            model: LLM model for autonomous decisions
            temperature: Generation temperature for tool selection
        """
        super().__init__(
            name="agentic_orchestrator",
            model=model,
            temperature=temperature,
            max_tokens=1500,
        )

    def get_system_prompt(self) -> str:
        """Define autonomous orchestration behavior."""
        return """You are an autonomous tool orchestrator with the following
capabilities:

1. INTELLIGENT ANALYSIS
   - Analyze user requests to understand intent and requirements
   - Assess available tools and their capabilities dynamically
   - Select optimal tool combinations based on context and constraints

2. AUTONOMOUS EXECUTION
   - Choose execution strategies (sequential, parallel, or hybrid)
   - Make real-time decisions based on intermediate results
   - Adapt approach if initial tools don't provide expected results

3. PERFORMANCE OPTIMIZATION
   - Balance speed, quality, and resource constraints
   - Learn from execution patterns to improve future decisions
   - Provide clear reasoning for tool selection and orchestration choices

Your goal is to provide the most effective tool orchestration for each task while
maintaining high performance and reliability. Always explain your reasoning."""

    async def initialize_tools(self, deps: BaseAgentDependencies) -> None:
        """Initialize with dynamic tool discovery."""
        # Check fallback status
        fallback_reason = getattr(self, "_fallback_reason", None)

        if not PYDANTIC_AI_AVAILABLE or self.agent is None:
            logger.warning(
                "AgenticOrchestrator using fallback mode (reason: %s)",
                fallback_reason or "pydantic_ai_unavailable"
            )
            return

        # Set up native Pydantic-AI tool discovery and orchestration
        @self.agent.tool_plain
        async def orchestrate_tools(request: ToolRequest) -> ToolResponse:
            """Main orchestration tool for autonomous task execution."""
            return await self._orchestrate_autonomous(request, deps)

        @self.agent.tool_plain
        async def discover_available_tools() -> dict[str, Any]:
            """Discover and assess available MCP tools dynamically."""
            return await self._discover_tools(deps)

        @self.agent.tool_plain
        async def execute_tool_chain(
            tools: list[str], input_data: dict[str, Any]
        ) -> dict[str, Any]:
            """Execute selected tools with intelligent chaining."""
            return await self._execute_chain(tools, input_data, deps)

        logger.info("AgenticOrchestrator initialized with native Pydantic-AI patterns")

    async def orchestrate(
        self, task: str, constraints: dict[str, Any], deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Main entry point for autonomous tool orchestration.

        Args:
            task: Task description or query
            constraints: Performance and quality constraints
            deps: Agent dependencies

        Returns:
            Orchestration response with results and reasoning
        """
        if not self._initialized:
            await self.initialize(deps)

        request = ToolRequest(
            task=task,
            constraints=constraints,
            context={"session_id": deps.session_state.session_id},
        )

        if not PYDANTIC_AI_AVAILABLE or self.agent is None:
            return await self._fallback_orchestrate(request, deps)

        # Execute using native Pydantic-AI agent
        try:
            result = await self.agent.run(
                f"Orchestrate tools for task: {task}", deps=deps
            )

            # Parse response into structured format
            if hasattr(result, "data") and isinstance(result.data, ToolResponse):
                response = result.data
            else:
                # Create response from agent output
                response = ToolResponse(
                    success=True,
                    results={"agent_response": str(result.data)},
                    tools_used=["autonomous_agent"],
                    reasoning="Native Pydantic-AI agent orchestration",
                    latency_ms=0.0,
                    confidence=0.8,
                )

            # Update session state
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

        except Exception as e:
            logger.exception("Autonomous orchestration failed")
            return ToolResponse(
                success=False,
                results={"error": str(e)},
                tools_used=[],
                reasoning=f"Orchestration failed: {e}",
                latency_ms=0.0,
                confidence=0.0,
            )

    async def _orchestrate_autonomous(
        self, request: ToolRequest, deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Core autonomous orchestration logic."""
        start_time = time.time()

        try:
            # Step 1: Analyze task and discover tools
            available_tools = await self._discover_tools(deps)

            # Step 2: Intelligent tool selection based on task
            selected_tools = self._select_tools_for_task(
                request.task, available_tools, request.constraints
            )

            # Step 3: Execute tool chain
            results = await self._execute_chain(
                selected_tools,
                {
                    "task": request.task,
                    "constraints": request.constraints,
                    **request.context,
                },
                deps,
            )

            latency_ms = (time.time() - start_time) * 1000

            return ToolResponse(
                success=True,
                results=results,
                tools_used=selected_tools,
                reasoning=self._generate_reasoning(
                    request.task, selected_tools, results
                ),
                latency_ms=latency_ms,
                confidence=self._calculate_confidence(results, selected_tools),
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ToolResponse(
                success=False,
                results={"error": str(e)},
                tools_used=[],
                reasoning=f"Orchestration failed due to: {e}",
                latency_ms=latency_ms,
                confidence=0.0,
            )

    async def _discover_tools(self, deps: BaseAgentDependencies) -> dict[str, Any]:
        """Dynamic tool discovery based on J3 research findings."""
        # This would integrate with actual MCP tool registry
        # For now, return core tools available in the system
        return {
            "hybrid_search": {
                "capabilities": ["search", "retrieval"],
                "performance": {"latency_ms": 150, "accuracy": 0.85},
                "description": "Hybrid vector and text search",
            },
            "rag_generation": {
                "capabilities": ["generation", "synthesis"],
                "performance": {"latency_ms": 800, "accuracy": 0.90},
                "description": "RAG-based answer generation",
            },
            "content_analysis": {
                "capabilities": ["analysis", "classification"],
                "performance": {"latency_ms": 200, "accuracy": 0.88},
                "description": "Content analysis and classification",
            },
        }

    def _select_tools_for_task(
        self, task: str, available_tools: dict[str, Any], constraints: dict[str, Any]
    ) -> list[str]:
        """Intelligent tool selection based on task analysis."""
        task_lower = task.lower()
        selected = []

        # Rule-based tool selection (would be enhanced with ML in production)
        if any(keyword in task_lower for keyword in ["search", "find", "lookup"]):
            selected.append("hybrid_search")

        if any(keyword in task_lower for keyword in ["generate", "answer", "explain"]):
            if "hybrid_search" not in selected:
                selected.append("hybrid_search")
            selected.append("rag_generation")

        if any(keyword in task_lower for keyword in ["analyze", "classify", "assess"]):
            selected.append("content_analysis")

        # Fallback to search if no specific tools selected
        if not selected:
            selected.append("hybrid_search")

        # Apply constraints (e.g., max latency)
        if constraints.get("max_latency_ms", 5000) < 1000:
            # Remove slow tools for speed requirements
            selected = [t for t in selected if t != "rag_generation"]

        return selected

    async def _execute_chain(
        self, tools: list[str], input_data: dict[str, Any], deps: BaseAgentDependencies
    ) -> dict[str, Any]:
        """Execute selected tools with intelligent chaining."""
        results = {}
        context = input_data.copy()

        for tool_name in tools:
            try:
                # Mock execution - would integrate with actual MCP tools
                tool_result = await self._execute_tool(tool_name, context, deps)
                results[f"{tool_name}_result"] = tool_result

                # Update context for next tool
                context.update(tool_result)

            except Exception as e:
                logger.warning("Tool %s failed: %s", tool_name, e)
                results[f"{tool_name}_error"] = str(e)

        return results

    async def _execute_tool(
        self, tool_name: str, context: dict[str, Any], deps: BaseAgentDependencies
    ) -> dict[str, Any]:
        """Execute individual tool (mock implementation)."""
        # Simulate tool execution
        await asyncio.sleep(0.1)

        return {
            "tool": tool_name,
            "result": f"Mock result from {tool_name}",
            "input_keys": list(context.keys()),
            "timestamp": asyncio.get_event_loop().time(),
        }

    def _generate_reasoning(
        self, task: str, tools_used: list[str], results: dict[str, Any]
    ) -> str:
        """Generate clear reasoning for tool selection and execution."""
        reasoning = (
            f"For task '{task}', I selected {len(tools_used)} tools: "
            f"{', '.join(tools_used)}. "
        )

        if "hybrid_search" in tools_used:
            reasoning += "Used hybrid search for information retrieval. "
        if "rag_generation" in tools_used:
            reasoning += "Applied RAG generation for comprehensive answers. "
        if "content_analysis" in tools_used:
            reasoning += "Performed content analysis for deeper insights. "

        successful_tools = len([k for k in results if not k.endswith("_error")])
        reasoning += f"Successfully executed {successful_tools} tools."

        return reasoning

    def _calculate_confidence(
        self, results: dict[str, Any], tools_used: list[str]
    ) -> float:
        """Calculate confidence score based on execution success."""
        if not results:
            return 0.0

        successful = len([k for k in results if not k.endswith("_error")])
        total = len(tools_used)

        if total == 0:
            return 0.0

        base_confidence = successful / total

        # Boost confidence for multi-tool success
        if successful > 1:
            base_confidence = min(base_confidence * 1.1, 1.0)

        return round(base_confidence, 2)

    async def _fallback_orchestrate(
        self, request: ToolRequest, deps: BaseAgentDependencies
    ) -> ToolResponse:
        """Fallback orchestration when Pydantic-AI unavailable."""
        fallback_reason = getattr(self, "_fallback_reason", "unknown")
        logger.warning(
            "Using fallback orchestration mode (reason: %s)", fallback_reason
        )
        start_time = time.time()

        # Enhanced fallback logic with context-aware responses
        task_lower = request.task.lower()

        # Determine appropriate fallback response based on task type
        if any(keyword in task_lower for keyword in ["search", "find", "retrieve"]):
            fallback_tools = ["mock_search_tool"]
            result_data = {
                "search_results": f"Mock search results for: {request.task}",
                "result_count": 5,
                "search_type": "fallback_search",
            }
            reasoning = f"Fallback search orchestration for task: {request.task}"
            confidence = 0.7
        elif any(
            keyword in task_lower for keyword in ["analyze", "examine", "evaluate"]
        ):
            fallback_tools = ["mock_analysis_tool"]
            result_data = {
                "analysis_results": f"Mock analysis of: {request.task}",
                "confidence_score": 0.65,
                "analysis_type": "fallback_analysis",
            }
            reasoning = f"Fallback analysis orchestration for task: {request.task}"
            confidence = 0.65
        elif any(
            keyword in task_lower for keyword in ["generate", "create", "compose"]
        ):
            fallback_tools = ["mock_generation_tool"]
            result_data = {
                "generated_content": f"Mock generated content for: {request.task}",
                "word_count": 150,
                "generation_type": "fallback_generation",
            }
            reasoning = f"Fallback generation orchestration for task: {request.task}"
            confidence = 0.6
        else:
            fallback_tools = ["mock_general_tool"]
            result_data = {
                "general_response": f"Mock response for task: {request.task}",
                "task_type": "general",
                "processing_mode": "fallback",
            }
            reasoning = f"Fallback general orchestration for task: {request.task}"
            confidence = 0.5

        latency_ms = (time.time() - start_time) * 1000

        response = ToolResponse(
            success=True,
            results={
                "fallback_mode": True,
                "fallback_reason": fallback_reason,
                **result_data,
            },
            tools_used=fallback_tools,
            reasoning=f"{reasoning} (fallback mode - reason: {fallback_reason})",
            latency_ms=latency_ms,
            confidence=confidence,
        )

        # Update session state even in fallback mode
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


# Global orchestrator instance for singleton pattern
_orchestrator_instance: AgenticOrchestrator | None = None


def get_orchestrator() -> AgenticOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgenticOrchestrator()
    return _orchestrator_instance


async def orchestrate_tools(
    task: str, constraints: dict[str, Any], deps: BaseAgentDependencies
) -> ToolResponse:
    """Convenient function for tool orchestration."""
    orchestrator = get_orchestrator()
    return await orchestrator.orchestrate(task, constraints, deps)
