"""Tests for the MCP tool execution service."""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest
from mcp.types import CallToolResult


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/services/agents/tool_execution_service.py"
)

_original_modules: dict[str, types.ModuleType | None] = {}
for module_name in (
    "src.services",
    "src.services.agents",
    "src.services.agents.tool_execution_service",
    "src.infrastructure.client_manager",
    "src.services.monitoring.telemetry_repository",
):
    _original_modules[module_name] = sys.modules.pop(module_name, None)

infra_stub = types.ModuleType("src.infrastructure.client_manager")


class ClientManager:  # noqa: D401 - lightweight stub
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            mcp_client=SimpleNamespace(
                request_timeout_ms=1000,
                servers=[SimpleNamespace(name="primary")],
            )
        )
        self.config.servers = [SimpleNamespace(name="primary")]
        self._client = SimpleNamespace()

    async def get_mcp_client(self) -> object:  # noqa: D401
        return self._client


setattr(infra_stub, "ClientManager", ClientManager)
sys.modules["src.infrastructure.client_manager"] = infra_stub

telemetry_stub = types.ModuleType("src.services.monitoring.telemetry_repository")


class _TelemetryRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []

    def increment_counter(
        self, name: str, *, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        self.calls.append((name, tags or {}))

    def record_observation(
        self, name: str, value: float, *, tags: dict[str, str] | None = None
    ) -> None:
        self.observations.append((name, value, tags or {}))

    def export_snapshot(self) -> dict[str, object]:
        counters: dict[str, list[dict[str, object]]] = {}
        for metric, tags in self.calls:
            counters.setdefault(metric, []).append({"value": 1, "tags": tags})
        histograms: dict[str, list[dict[str, object]]] = {}
        for metric, value, tags in self.observations:
            histograms.setdefault(metric, []).append(
                {"count": 1, "sum": value, "tags": tags}
            )
        return {"counters": counters, "histograms": histograms}

    def reset(self) -> None:
        self.calls.clear()
        self.observations.clear()


_repository = _TelemetryRepository()


def _get_telemetry_repository() -> _TelemetryRepository:
    return _repository


setattr(telemetry_stub, "get_telemetry_repository", _get_telemetry_repository)
sys.modules["src.services.monitoring.telemetry_repository"] = telemetry_stub

services_stub = types.ModuleType("src.services")
errors_stub = types.ModuleType("src.services.errors")


class APIError(Exception):
    """Minimal stub matching APIError signature."""


setattr(errors_stub, "APIError", APIError)
setattr(services_stub, "errors", errors_stub)
sys.modules["src.services"] = services_stub
sys.modules["src.services.errors"] = errors_stub

_spec = spec_from_file_location("_tool_execution_service_under_test", MODULE_PATH)
assert _spec and _spec.loader
_module = module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

for name, original in _original_modules.items():
    if original is not None:
        sys.modules[name] = original
    else:
        sys.modules.pop(name, None)

ToolExecutionService = _module.ToolExecutionService
ToolExecutionFailure = _module.ToolExecutionFailure
ToolExecutionInvalidArgument = _module.ToolExecutionInvalidArgument
ToolExecutionTimeout = _module.ToolExecutionTimeout
get_telemetry_repository = _get_telemetry_repository


@pytest.mark.asyncio
async def test_execute_tool_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful execution should record telemetry and return result."""

    manager = ClientManager()
    client = await manager.get_mcp_client()
    service = ToolExecutionService(manager, max_attempts=1)
    telemetry = get_telemetry_repository()  # pylint: disable=no-member
    telemetry.reset()

    class DummySession:
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            read_timeout_seconds=None,
        ):  # noqa: D401,ANN001
            assert tool_name == "demo"
            assert arguments == {"foo": "bar"}
            assert read_timeout_seconds == pytest.approx(1.0)
            return CallToolResult(
                content=[], structuredContent={"value": 1}, isError=False
            )

    @asynccontextmanager
    async def fake_open_session(client_arg: object, server_name: str):  # noqa: D401
        assert client_arg is client
        assert server_name == "primary"
        yield DummySession()

    monkeypatch.setattr(
        _module.ToolExecutionService, "_open_session", staticmethod(fake_open_session)
    )

    result = await service.execute_tool("demo", arguments={"foo": "bar"})
    assert result.server_name == "primary"
    snapshot = cast(Dict[str, Any], telemetry.export_snapshot())
    counters = cast(Dict[str, Any], snapshot.get("counters", {}))
    histograms = cast(Dict[str, Any], snapshot.get("histograms", {}))
    assert counters.get("mcp_tool_calls_total")
    assert histograms.get("mcp_tool_latency_ms")


@pytest.mark.asyncio
async def test_execute_tool_invalid_arguments() -> None:
    """Non-mapping arguments should raise a validation error."""

    service = ToolExecutionService(ClientManager())
    with pytest.raises(ToolExecutionInvalidArgument):
        await service.execute_tool("demo", arguments=[1, 2])


@pytest.mark.asyncio
async def test_execute_tool_reports_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote tool errors should surface as ToolExecutionFailure."""

    manager = ClientManager()
    service = ToolExecutionService(manager, max_attempts=1, backoff_seconds=0)
    telemetry = get_telemetry_repository()
    telemetry.reset()

    class DummySession:
        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            read_timeout_seconds=None,
        ):  # noqa: D401,ANN001
            return CallToolResult(content=[], structuredContent=None, isError=True)

    @asynccontextmanager
    async def fake_open_session(client_arg: object, server_name: str):  # noqa: D401
        yield DummySession()

    monkeypatch.setattr(
        _module.ToolExecutionService, "_open_session", staticmethod(fake_open_session)
    )

    with pytest.raises(ToolExecutionFailure):
        await service.execute_tool("demo", arguments={})

    counters_snapshot = cast(Dict[str, Any], telemetry.export_snapshot())
    counters = cast(
        list[dict[str, Any]],
        counters_snapshot.get("counters", {}).get("mcp_tool_errors_total", []),
    )
    assert counters
    assert counters[0]["value"] >= 1


@pytest.mark.asyncio
async def test_execute_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts should raise ToolExecutionTimeout after retries."""

    manager = ClientManager()
    service = ToolExecutionService(manager, max_attempts=2, backoff_seconds=0)

    class DummySession:
        async def call_tool(self, *_: Any, **__: Any):  # noqa: D401
            raise TimeoutError()

    @asynccontextmanager
    async def fake_open_session(client_arg: object, server_name: str):  # noqa: D401
        yield DummySession()

    monkeypatch.setattr(
        _module.ToolExecutionService, "_open_session", staticmethod(fake_open_session)
    )

    with pytest.raises(ToolExecutionTimeout):
        await service.execute_tool("demo", arguments={})
