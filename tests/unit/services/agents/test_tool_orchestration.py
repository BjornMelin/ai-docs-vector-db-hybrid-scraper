"""Unit tests for the advanced tool orchestration workflow."""

from __future__ import annotations

import asyncio

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.agents.tool_orchestration import (
    AdvancedToolOrchestrator,
    ToolCapability,
    ToolDefinition,
    ToolPriority,
)


@pytest.mark.asyncio
async def test_orchestrator_register_compose_execute() -> None:
    """Validate end-to-end orchestration plan execution."""
    orch = AdvancedToolOrchestrator(ClientManager(), max_parallel_executions=4)

    async def fast_exec(_in: dict) -> dict:
        """Return a deterministic response after a short pause."""
        await asyncio.sleep(0.01)
        return {"ok": True, "quality_score": 0.95, "confidence": 0.9}

    await orch.register_tool(
        ToolDefinition(
            tool_id="vector_search",
            name="Vector Search",
            description="Search",
            capabilities={ToolCapability.SEARCH},
            priority=ToolPriority.HIGH,
            estimated_duration_ms=20.0,
            resource_requirements={"cpu": 0.1},
            dependencies=[],
            success_rate=0.98,
            executor=fast_exec,
        )
    )
    await orch.register_tool(
        ToolDefinition(
            tool_id="content_generation",
            name="Content Generation",
            description="Gen",
            capabilities={ToolCapability.GENERATION},
            priority=ToolPriority.NORMAL,
            estimated_duration_ms=30.0,
            resource_requirements={"cpu": 0.2},
            dependencies=[],
            success_rate=0.95,
            executor=fast_exec,
        )
    )

    plan = await orch.compose_tool_chain(
        goal="search then generate an answer",
        constraints={"timeout_seconds": 5.0, "max_parallel_tools": 4},
        preferences={"optimize_for": "balanced"},
    )
    out = await orch.execute_tool_chain(plan, {"q": "hi"}, timeout_seconds=2.0)
    assert out["success"] is True
    assert out["metadata"]["total_nodes"] >= 1
