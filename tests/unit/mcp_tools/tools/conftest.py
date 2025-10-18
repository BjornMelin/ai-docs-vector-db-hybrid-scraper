"""Shared fixtures and stubs for MCP tools test suite."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest


def _install_agents_stub() -> None:
    """Install a lightweight stub for ``src.services.agents`` if missing."""
    if "src.services.agents" in sys.modules:
        return
    try:
        import importlib

        importlib.import_module("src.services.agents")
    except ImportError:
        pass
    else:
        return

    module = types.ModuleType("src.services.agents")

    @dataclass
    class GraphSearchOutcome:
        """Minimal search outcome payload."""

        success: bool
        session_id: str
        answer: str | None
        confidence: float | None
        results: list[Any]
        tools_used: list[str]
        reasoning: list[str]
        metrics: dict[str, Any]
        errors: list[dict[str, Any]]

    @dataclass
    class GraphAnalysisOutcome:
        """Minimal analysis outcome payload."""

        success: bool
        analysis_id: str
        summary: str
        insights: dict[str, Any]
        recommendations: list[str]
        confidence: float | None
        metrics: dict[str, Any]
        errors: list[dict[str, Any]]

    class GraphRunner:
        """Minimal GraphRunner facade for tooling tests."""

        COMPONENTS: ClassVar[tuple[None, None, None]] = (None, None, None)

        @classmethod
        def build_components(cls) -> tuple[None, None, None, GraphRunner]:
            """Return stubbed components and runner instance."""
            return (*cls.COMPONENTS, cls())

        async def run_search(
            self, **_kwargs: Any
        ) -> GraphSearchOutcome:  # pragma: no cover - unused default
            """Fallback run implementation returning empty success."""
            return GraphSearchOutcome(
                success=True,
                session_id="noop",
                answer=None,
                confidence=None,
                results=[],
                tools_used=[],
                reasoning=[],
                metrics={},
                errors=[],
            )

        async def run_analysis(
            self, **_kwargs: Any
        ) -> GraphAnalysisOutcome:  # pragma: no cover - unused default
            """Fallback analysis implementation returning empty success."""
            return GraphAnalysisOutcome(
                success=True,
                analysis_id="noop",
                summary="",
                insights={},
                recommendations=[],
                confidence=None,
                metrics={},
                errors=[],
            )

    module.GraphRunner = GraphRunner  # type: ignore[attr-defined]
    module.GraphSearchOutcome = GraphSearchOutcome  # type: ignore[attr-defined]
    module.GraphAnalysisOutcome = GraphAnalysisOutcome  # type: ignore[attr-defined]

    sys.modules["src.services.agents"] = module


_install_agents_stub()


@pytest.fixture(autouse=True)
def _ensure_agent_stubs():
    """Ensure the agent services module is stubbed before each test."""
    _install_agents_stub()
    yield
