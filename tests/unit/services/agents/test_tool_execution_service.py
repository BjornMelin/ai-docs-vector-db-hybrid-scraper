"""Tests for the MCP tool execution service."""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    "src.services.observability.tracking",
    "src.services.errors",
):
    _original_modules[module_name] = sys.modules.pop(module_name, None)

infra_stub: Any = types.ModuleType("src.infrastructure.client_manager")


class ClientManager:  # noqa: D401 - lightweight stub
    def __init__(
        self, servers: list[str] | None = None, *, timeout_ms: int = 1000
    ) -> None:
        server_names = servers or ["primary"]
        self.config = SimpleNamespace(
            mcp_client=SimpleNamespace(
                request_timeout_ms=timeout_ms,
                servers=[SimpleNamespace(name=name) for name in server_names],
            )
        )
        self._client = SimpleNamespace(
            connections={name: object() for name in server_names}
        )

    async def get_mcp_client(self) -> object:  # noqa: D401
        return self._client


infra_stub.ClientManager = ClientManager
sys.modules["src.infrastructure.client_manager"] = infra_stub

tracking_stub: Any = types.ModuleType("src.services.observability.tracking")


class _Tracker:
    def __init__(self) -> None:
        self.operations: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.operations.clear()


_tracker = _Tracker()


def _record_ai_operation(**payload: Any) -> None:
    _tracker.operations.append(payload)


def _get_ai_tracker() -> _Tracker:
    return _tracker


tracking_stub.record_ai_operation = _record_ai_operation
tracking_stub.get_ai_tracker = _get_ai_tracker
sys.modules["src.services.observability.tracking"] = tracking_stub

services_stub: Any = types.ModuleType("src.services")
errors_stub: Any = types.ModuleType("src.services.errors")


class APIError(Exception):
    """Minimal stub matching APIError signature."""


errors_stub.APIError = APIError
services_stub.errors = errors_stub
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
ToolExecutionError = _module.ToolExecutionError
recorded_operations = _tracker.operations


@pytest.fixture(autouse=True)
def _reset_tracker() -> Any:
    """Ensure tracker state is cleared between tests."""

    _tracker.reset()
    yield
    _tracker.reset()


@pytest.mark.asyncio
async def test_execute_tool_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful execution should record telemetry and return result."""

    manager = ClientManager()
    client = await manager.get_mcp_client()
    service = ToolExecutionService(manager, max_attempts=1)
    _tracker.reset()

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
    assert len(recorded_operations) == 1
    call_operation = recorded_operations[0]
    assert call_operation["operation_type"] == "mcp.tool.call"
    assert call_operation["provider"] == "primary"
    assert call_operation["model"] == "demo"
    assert call_operation["success"] is True


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
    _tracker.reset()

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

    assert len(recorded_operations) == 2
    call_op = recorded_operations[0]
    error_op = recorded_operations[1]
    assert call_op["operation_type"] == "mcp.tool.call"
    assert call_op["success"] is False
    assert error_op["operation_type"] == "mcp.tool.error.remote"
    assert error_op["provider"] == "primary"


@pytest.mark.asyncio
async def test_execute_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeouts should raise ToolExecutionTimeout after retries."""

    manager = ClientManager()
    service = ToolExecutionService(manager, max_attempts=2, backoff_seconds=0)
    _tracker.reset()

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
    assert len(recorded_operations) == 2
    for operation in recorded_operations:
        assert operation["operation_type"] == "mcp.tool.error.timeout"
        assert operation["success"] is False


@pytest.mark.asyncio
async def test_execute_tool_moves_to_next_server_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service should fallback to the next server and retain telemetry."""

    manager = ClientManager(servers=["primary", "backup"])
    service = ToolExecutionService(manager, max_attempts=1, backoff_seconds=0)

    results: dict[str, CallToolResult] = {
        "primary": CallToolResult(content=[], structuredContent=None, isError=True),
        "backup": CallToolResult(
            content=[], structuredContent={"ok": True}, isError=False
        ),
    }

    class DummySession:
        def __init__(self, name: str) -> None:
            self._name = name

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            read_timeout_seconds=None,
        ):  # noqa: D401,ANN001
            assert tool_name == "demo"
            assert arguments == {"value": 1}
            return results[self._name]

    @asynccontextmanager
    async def fake_open_session(client_arg: object, server_name: str):  # noqa: D401
        yield DummySession(server_name)

    monkeypatch.setattr(
        _module.ToolExecutionService, "_open_session", staticmethod(fake_open_session)
    )

    outcome = await service.execute_tool("demo", arguments={"value": 1})
    assert outcome.server_name == "backup"
    assert outcome.result.structuredContent == {"ok": True}

    op_types = [operation["operation_type"] for operation in recorded_operations]
    assert op_types.count("mcp.tool.error.remote") == 1
    assert op_types.count("mcp.tool.call") == 2
    assert any(operation["provider"] == "backup" for operation in recorded_operations)


@pytest.mark.asyncio
async def test_execute_tool_records_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected errors should raise ToolExecutionError and emit telemetry."""

    manager = ClientManager()
    service = ToolExecutionService(manager, max_attempts=1, backoff_seconds=0)

    class DummySession:
        async def call_tool(self, *_: Any, **__: Any):  # noqa: D401
            raise RuntimeError("boom")

    @asynccontextmanager
    async def fake_open_session(client_arg: object, server_name: str):  # noqa: D401
        yield DummySession()

    monkeypatch.setattr(
        _module.ToolExecutionService, "_open_session", staticmethod(fake_open_session)
    )

    with pytest.raises(ToolExecutionError) as exc:
        await service.execute_tool("demo", arguments={})
    assert exc.value.server_name == "primary"
    assert len(recorded_operations) == 1
    operation = recorded_operations[0]
    assert operation["operation_type"] == "mcp.tool.error.exception"
    assert operation["success"] is False
