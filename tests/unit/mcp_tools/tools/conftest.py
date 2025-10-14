"""Shared fixtures and stubs for MCP tools test suite."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


def _install_agents_stub() -> None:
    """Install a lightweight stub for ``src.services.agents`` if missing."""

    if "src.services.agents" in sys.modules:
        return

    module = types.ModuleType("src.services.agents")

    class QueryOrchestrator:
        """Minimal query orchestrator stub used in tests."""

        def __init__(self) -> None:
            self.is_initialized = False
            self._history: list[tuple[str, dict[str, object]]] = []

        async def initialize(self, deps: object) -> None:
            self.is_initialized = True
            # pylint: disable=attribute-defined-outside-init
            self._deps = deps  # pragma: no cover - debug aid

        async def orchestrate_query(self, **kwargs) -> dict[str, object]:  # type: ignore[override]
            self._history.append(("query", kwargs))
            return {
                "success": True,
                "results": [],
                "orchestration_plan": {},
                "tools_used": [],
                "reasoning": "stub",
                "metrics": {},
            }

    def create_agent_dependencies(**kwargs) -> SimpleNamespace:  # type: ignore[override]
        session_state = SimpleNamespace(preferences={})
        return SimpleNamespace(session_state=session_state, kwargs=kwargs)

    async def orchestrate_tools(
        *_args, **_kwargs
    ) -> dict[str, object]:  # pragma: no cover - simple stub
        return {}

    module.QueryOrchestrator = QueryOrchestrator  # type: ignore[attr-defined]
    module.create_agent_dependencies = create_agent_dependencies  # type: ignore[attr-defined]
    module.orchestrate_tools = orchestrate_tools  # type: ignore[attr-defined]

    sys.modules["src.services.agents"] = module


_install_agents_stub()


@pytest.fixture(autouse=True)
def _ensure_agent_stubs():
    """Ensure the agent services module is stubbed before each test."""

    _install_agents_stub()
    yield
