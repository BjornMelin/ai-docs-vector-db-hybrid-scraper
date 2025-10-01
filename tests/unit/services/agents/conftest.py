"""Pytest configuration and lightweight infra stubs for unit tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest


# src.config stub
config_mod = cast(Any, types.ModuleType("src.config"))


def get_config() -> dict[str, Any]:
    return {"env": "test"}


config_mod.get_config = get_config
sys.modules["src.config"] = config_mod

# src.infrastructure.client_manager stub
infra_mod = cast(Any, types.ModuleType("src.infrastructure.client_manager"))


class ClientManager:
    def __init__(self) -> None:  # pragma: no cover - trivial
        self.name = "test-client-manager"


infra_mod.ClientManager = ClientManager
sys.modules["src.infrastructure.client_manager"] = infra_mod

# src.services.cache.patterns stub (CircuitBreakerPattern)
cb_mod = cast(Any, types.ModuleType("src.services.cache.patterns"))


class CircuitBreakerPattern:
    """Minimal async-friendly circuit breaker stub."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        expected_exception: type[BaseException] = Exception,
    ) -> None:
        self._open = False
        self._expected = expected_exception

    def is_open(self) -> bool:
        return self._open

    async def call(self, func: Callable[[], Any]) -> Any:
        try:
            res = func()
            if hasattr(res, "__await__"):
                return await res  # coroutine support
            return res
        except self._expected:
            self._open = True
            raise


cb_mod.CircuitBreakerPattern = CircuitBreakerPattern
sys.modules["src.services.cache.patterns"] = cb_mod

# src.services.observability.tracking stub
obs_mod = cast(Any, types.ModuleType("src.services.observability.tracking"))


class PerformanceTracker:  # pragma: no cover - trivial
    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, name: str) -> None:
        self.events.append(name)


obs_mod.PerformanceTracker = PerformanceTracker
sys.modules["src.services.observability.tracking"] = obs_mod

# src.services.agents.core stub
core_mod = cast(Any, types.ModuleType("src.services.agents.core"))


@dataclass(slots=True)
class AgentState:
    session_id: str

    def increment_tool_usage(self, _name: str) -> None:  # pragma: no cover - trivial
        return

    def add_interaction(
        self, *, role: str, content: str, metadata: dict[str, Any]
    ) -> None:  # pragma: no cover - trivial
        return


class BaseAgentDependencies:  # pragma: no cover - trivial
    def __init__(
        self, client_manager: Any, config: Any, session_state: AgentState
    ) -> None:
        self.client_manager = client_manager
        self.config = config
        self.session_state = session_state


class BaseAgent:
    """Minimal BaseAgent with the surface used by tests."""

    def __init__(
        self, name: str, model: str, temperature: float, max_tokens: int
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = True
        self.agent = None  # pydantic-ai Agent in production
        self.is_initialized = True

    def get_system_prompt(self) -> str:  # pragma: no cover
        return ""

    async def initialize(
        self, _deps: BaseAgentDependencies
    ) -> None:  # pragma: no cover - minimal
        self._initialized = True

    async def initialize_tools(
        self, _deps: BaseAgentDependencies
    ) -> None:  # pragma: no cover - minimal
        return

    async def execute(
        self, task: str, _deps: BaseAgentDependencies, _context: dict | None = None
    ) -> dict:  # pragma: no cover
        return {"success": True, "result": f"ok:{task}"}


def create_agent_dependencies(client_manager: Any) -> BaseAgentDependencies:
    """Assemble agent dependencies using local stubs."""
    return BaseAgentDependencies(
        client_manager, get_config(), AgentState("test-session")
    )


core_mod.AgentState = AgentState
core_mod.BaseAgentDependencies = BaseAgentDependencies
core_mod.BaseAgent = BaseAgent
core_mod.create_agent_dependencies = create_agent_dependencies

sys.modules["src.services.agents.core"] = core_mod


@pytest.fixture
def mock_dependencies() -> BaseAgentDependencies:
    """Return agent dependencies backed by lightweight stubs."""
    mock_client_manager = Mock(spec=ClientManager)
    config = get_config()
    session_state = AgentState(session_id=str(uuid4()))

    return BaseAgentDependencies(
        client_manager=mock_client_manager,
        config=config,
        session_state=session_state,
    )
