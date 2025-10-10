"""Tests for tool execution service with container-supplied MCP client."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import CallToolResult, Tool

from src.config.models import MCPClientConfig, MCPServerConfig, MCPTransport
from src.services.agents.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionInvalidArgument,
    ToolExecutionResult,
    ToolExecutionService,
    ToolExecutionTimeout,
)


@pytest.fixture
def operations(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture telemetry operations emitted during tests."""

    ops: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "src.services.agents.tool_execution_service.record_ai_operation",
        lambda **payload: ops.append(payload),
    )
    return ops


def build_config(
    server_names: Iterable[str], timeout_ms: int = 1000
) -> MCPClientConfig:
    """Construct an MCPClientConfig covering the requested server identifiers."""

    servers = [
        MCPServerConfig(
            name=name,
            transport=MCPTransport.STDIO,
            command="stub",
            args=[],
            url=None,
            timeout_ms=timeout_ms,
        )
        for name in server_names
    ]
    return MCPClientConfig(enabled=True, request_timeout_ms=timeout_ms, servers=servers)


def service_with_client(
    client: Any,
    *,
    servers: Iterable[str] = ("primary",),
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
) -> ToolExecutionService:
    """Instantiate ToolExecutionService with the provided client stub."""

    config = build_config(servers)
    typed_client = cast(MultiServerMCPClient, client)
    return ToolExecutionService(
        client=typed_client,
        mcp_config=config,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )


def stub_session(
    handler: Callable[[str, dict[str, Any], float], CallToolResult],
) -> Callable[[Any, str], Any]:
    """Return an _open_session stub invoking the supplied handler."""

    @asynccontextmanager
    async def _context(_client: Any, server_name: str):
        class _Session:
            async def call_tool(
                self,
                tool_name: str,
                arguments: dict[str, Any],
                *,
                read_timeout_seconds: float,
            ) -> CallToolResult:
                return handler(server_name, arguments, read_timeout_seconds)

        yield _Session()

    return _context


@pytest.mark.asyncio
async def test_execute_tool_success(
    monkeypatch: pytest.MonkeyPatch, operations: list[dict[str, Any]]
) -> None:
    """Successful execution should return ToolExecutionResult and record telemetry."""

    call_tracker: dict[str, dict[str, Any]] = {}

    def _handler(
        server: str, payload: dict[str, Any], timeout: float
    ) -> CallToolResult:
        call_tracker[server] = {"payload": payload, "timeout": timeout}
        return CallToolResult(content=[], structuredContent={"value": 1}, isError=False)

    client = SimpleNamespace(connections={"primary": object()})
    service = service_with_client(client)
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(stub_session(_handler)),
    )

    result = await service.execute_tool("demo", arguments={"foo": "bar"})

    assert isinstance(result, ToolExecutionResult)
    assert result.server_name == "primary"
    assert result.result.structuredContent == {"value": 1}
    assert call_tracker["primary"]["payload"] == {"foo": "bar"}
    assert isinstance(call_tracker["primary"]["timeout"], float)
    assert operations[-1]["operation_type"] == "mcp.tool.call"
    assert operations[-1]["success"] is True


@pytest.mark.asyncio
async def test_execute_tool_invalid_arguments() -> None:
    """Non-mapping arguments should raise validation errors."""

    client = SimpleNamespace(connections={"primary": object()})
    service = service_with_client(client)

    with pytest.raises(ToolExecutionInvalidArgument):
        await service.execute_tool("demo", arguments=cast(Any, ["not", "mapping"]))


@pytest.mark.asyncio
async def test_execute_tool_remote_failure(
    monkeypatch: pytest.MonkeyPatch, operations: list[dict[str, Any]]
) -> None:
    """Remote errors should raise ToolExecutionFailure and emit telemetry."""

    def _handler(
        server: str, payload: dict[str, Any], timeout: float
    ) -> CallToolResult:
        return CallToolResult(content=[], structuredContent=None, isError=True)

    client = SimpleNamespace(connections={"primary": object()})
    service = service_with_client(client, max_attempts=1)
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(stub_session(_handler)),
    )

    with pytest.raises(ToolExecutionFailure):
        await service.execute_tool("demo", arguments={})

    op_types = [op["operation_type"] for op in operations]
    assert op_types.count("mcp.tool.call") == 1
    assert op_types.count("mcp.tool.error.remote") == 1


@pytest.mark.asyncio
async def test_execute_tool_timeout(
    monkeypatch: pytest.MonkeyPatch, operations: list[dict[str, Any]]
) -> None:
    """Timeouts should raise ToolExecutionTimeout after retry attempts."""

    def _handler(
        server: str, payload: dict[str, Any], timeout: float
    ) -> CallToolResult:
        raise TimeoutError("timeout")

    client = SimpleNamespace(connections={"primary": object()})
    service = service_with_client(client, max_attempts=2, backoff_seconds=0.0)
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(stub_session(_handler)),
    )

    with pytest.raises(ToolExecutionTimeout):
        await service.execute_tool("demo", arguments={})

    assert all(op["operation_type"] == "mcp.tool.error.timeout" for op in operations)


@pytest.mark.asyncio
async def test_execute_tool_moves_to_next_server(
    monkeypatch: pytest.MonkeyPatch, operations: list[dict[str, Any]]
) -> None:
    """Service should fall back to secondary servers when primaries fail."""

    responses: dict[str, CallToolResult] = {
        "primary": CallToolResult(content=[], structuredContent=None, isError=True),
        "backup": CallToolResult(
            content=[], structuredContent={"ok": True}, isError=False
        ),
    }

    def _handler(
        server: str, payload: dict[str, Any], timeout: float
    ) -> CallToolResult:
        return responses[server]

    client = SimpleNamespace(connections={"primary": object(), "backup": object()})
    service = service_with_client(client, servers=("primary", "backup"), max_attempts=1)
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(stub_session(_handler)),
    )

    result = await service.execute_tool("demo", arguments={"value": 1})

    assert result.server_name == "backup"
    assert result.result.structuredContent == {"ok": True}
    assert any(op["provider"] == "backup" for op in operations)


@pytest.mark.asyncio
async def test_execute_tool_records_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch, operations: list[dict[str, Any]]
) -> None:
    """Unexpected exceptions should raise ToolExecutionError and log telemetry."""

    def _handler(
        server: str, payload: dict[str, Any], timeout: float
    ) -> CallToolResult:
        raise RuntimeError("boom")

    client = SimpleNamespace(connections={"primary": object()})
    service = service_with_client(client, max_attempts=1)
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(stub_session(_handler)),
    )

    with pytest.raises(ToolExecutionError) as exc:
        await service.execute_tool("demo", arguments={})

    assert exc.value.server_name == "primary"
    assert operations[-1]["operation_type"] == "mcp.tool.error.exception"


@pytest.mark.asyncio
async def test_execute_tool_requires_servers() -> None:
    """Missing MCP servers should raise ToolExecutionFailure immediately."""

    config = MCPClientConfig(enabled=True, request_timeout_ms=5000, servers=[])
    client = cast(MultiServerMCPClient, SimpleNamespace(connections={}))

    async def _client_factory() -> MultiServerMCPClient:
        return client

    service = ToolExecutionService(_client_factory, config)

    with pytest.raises(ToolExecutionFailure, match="No MCP servers configured"):
        await service.execute_tool("demo", arguments={})


@pytest.mark.asyncio
async def test_detect_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_tools should enumerate tools for every configured server."""

    tools_per_server = {
        "primary": [Tool(name="search", description="Search docs", inputSchema={})],
        "backup": [Tool(name="qa", description="Answer questions", inputSchema={})],
    }

    @asynccontextmanager
    async def _context(_client: Any, server_name: str):
        class _Session:
            async def list_tools(self) -> list[Tool]:
                return tools_per_server[server_name]

        yield _Session()

    client = SimpleNamespace(connections={name: object() for name in tools_per_server})
    monkeypatch.setattr(
        ToolExecutionService,
        "_open_session",
        staticmethod(_context),
    )

    summary = await ToolExecutionService.detect_tools(cast(Any, client))

    assert summary == {"primary": ["search"], "backup": ["qa"]}
