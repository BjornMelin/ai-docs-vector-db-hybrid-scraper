"""Unit tests for the agent coordinator service."""

from __future__ import annotations

import pytest

from src.services.agents.coordination import (
    CoordinationStrategy,
    ParallelAgentCoordinator,
    TaskDefinition,
    TaskPriority,
)
from src.services.agents.core import BaseAgent, BaseAgentDependencies


class EchoAgent(BaseAgent):
    """Minimal agent implementation used to exercise coordinator flows."""

    def get_system_prompt(self) -> str:
        """Return a stable system prompt for the echo agent."""
        return "You are an echo agent for testing."

    async def initialize_tools(
        self, deps: BaseAgentDependencies
    ) -> None:  # pragma: no cover - minimal
        self._initialized = True

    async def execute(
        self, task: str, deps: BaseAgentDependencies, context: dict | None = None
    ) -> dict:
        return {
            "success": True,
            "result": f"ok:{task}",
            "metadata": {"agent": self.name},
        }


@pytest.mark.asyncio
async def test_coordinator_sequential_exec():
    coord = ParallelAgentCoordinator(max_parallel_agents=2)

    agent = EchoAgent(name="echo", model="noop", temperature=0.0, max_tokens=1)
    await coord.register_agent(agent, capabilities=["do"])

    task = TaskDefinition(
        task_id="t1",
        description="do something",
        priority=TaskPriority.NORMAL,
        estimated_duration_ms=10.0,
        dependencies=[],
        required_capabilities=["do"],
        input_data={"x": 1},
    )

    result = await coord.execute_coordinated_workflow(
        [task], CoordinationStrategy.SEQUENTIAL, timeout_seconds=3.0
    )
    assert result["success"] is True
    assert "task_results" in result
    assert any(v.get("success") for v in result["task_results"].values())

    await coord.stop_coordination()
