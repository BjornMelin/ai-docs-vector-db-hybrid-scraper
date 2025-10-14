# pylint: disable=import-error

"""Execution utilities for MCP tools with retries, telemetry, and tracing."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import CallToolResult
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.config.models import MCPClientConfig
from src.services.errors import APIError
from src.services.observability.tracking import record_ai_operation


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_DEFAULT_MAX_ATTEMPTS = 2
_DEFAULT_BACKOFF_SECONDS = 0.25

_OP_TOOL_CALL = "mcp.tool.call"
_OP_TOOL_ERROR_TIMEOUT = "mcp.tool.error.timeout"
_OP_TOOL_ERROR_EXCEPTION = "mcp.tool.error.exception"
_OP_TOOL_ERROR_REMOTE = "mcp.tool.error.remote"


class ToolExecutionError(RuntimeError):
    """Base error when an MCP tool invocation fails."""

    def __init__(self, message: str, *, server_name: str | None = None) -> None:
        super().__init__(message)
        self.server_name = server_name


class ToolExecutionInvalidArgument(ToolExecutionError):  # noqa: N818
    """Raised when invalid arguments are provided to a tool call."""


class ToolExecutionTimeout(ToolExecutionError):  # noqa: N818
    """Raised when a tool call exceeds its read timeout."""


class ToolExecutionFailure(ToolExecutionError):  # noqa: N818
    """Raised when the remote tool reports an error payload."""


@dataclass(slots=True)
class ToolExecutionResult:
    """Structured representation of an MCP tool invocation."""

    tool_name: str
    server_name: str
    duration_ms: float
    result: CallToolResult

    @property
    def is_error(self) -> bool:
        """Return ``True`` when the MCP response marks an error."""
        return bool(self.result.isError)

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "duration_ms": self.duration_ms,
            "is_error": self.result.isError,
            "structured_content": self.result.structuredContent,
            "content": [item.model_dump() for item in self.result.content],
            "meta": self.result.meta,
        }


def _validate_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if not isinstance(arguments, Mapping):
        msg = "Tool arguments must be provided as a mapping"
        raise ToolExecutionInvalidArgument(msg)
    validated: dict[str, Any] = {}
    for key, value in arguments.items():
        if not isinstance(key, str):
            msg = "Tool argument keys must be strings"
            raise ToolExecutionInvalidArgument(msg)
        validated[key] = value
    return validated


class ToolExecutionService:
    """Execute MCP tools using the shared ``MultiServerMCPClient``."""

    def __init__(
        self,
        client: MultiServerMCPClient | Callable[[], Awaitable[MultiServerMCPClient]],
        mcp_config: MCPClientConfig,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        if callable(client):
            self._client_factory = client
            self._client: MultiServerMCPClient | None = None
        else:
            self._client = client
            self._client_factory = None
        self._config = mcp_config
        self._max_attempts = max(1, max_attempts)
        self._backoff_seconds = max(0.0, backoff_seconds)

    async def _resolve_client(self) -> MultiServerMCPClient:
        if self._client is not None:
            return self._client
        if self._client_factory is None:
            msg = "MCP client resolver is not configured"
            raise RuntimeError(msg)
        client = await self._client_factory()
        self._client = client
        return client

    async def execute_tool(
        self,
        tool_name: str,
        *,
        arguments: Mapping[str, Any] | None = None,
        server_name: str | None = None,
        read_timeout_ms: int | None = None,
    ) -> ToolExecutionResult:
        """Execute an MCP tool and return the structured result."""
        payload = _validate_arguments(arguments)
        client = await self._resolve_client()
        mcp_config = self._config
        if not mcp_config or not mcp_config.servers:
            msg = "No MCP servers configured for tool execution"
            raise ToolExecutionFailure(msg)
        timeout_ms = read_timeout_ms or mcp_config.request_timeout_ms

        server_candidates = (
            [server_name]
            if server_name
            else [server.name for server in mcp_config.servers]
        )

        last_error: ToolExecutionError | None = None
        for candidate in server_candidates:
            execution, error = await self._execute_on_server(
                client=client,
                server_name=candidate,
                tool_name=tool_name,
                payload=payload,
                timeout_seconds=timeout_ms / 1000.0,
            )
            if execution is not None:
                return execution
            if error is not None:
                last_error = error
                logger.debug(
                    "Tool '%s' failed on server '%s' after %s attempts",
                    tool_name,
                    candidate,
                    self._max_attempts,
                )

        if last_error is not None:
            raise last_error
        msg = f"All MCP servers failed for tool '{tool_name}'"
        raise ToolExecutionFailure(msg)

    async def _execute_on_server(
        self,
        *,
        client: MultiServerMCPClient,
        server_name: str,
        tool_name: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[ToolExecutionResult | None, ToolExecutionError | None]:
        # pylint: disable=too-many-arguments,too-many-locals
        """Execute a tool on a single server with retries and telemetry."""
        last_error: ToolExecutionError | None = None
        for attempt in range(1, self._max_attempts + 1):
            start_time = time.perf_counter()
            with tracer.start_as_current_span(
                "mcp.tool.execute",
                attributes={
                    "mcp.tool.name": tool_name,
                    "mcp.server.name": server_name,
                    "mcp.attempt": attempt,
                },
            ) as span:
                try:
                    async with self._open_session(client, server_name) as session:
                        result = await session.call_tool(
                            tool_name,
                            payload,
                            read_timeout_seconds=timeout_seconds,
                        )
                except TimeoutError as exc:
                    duration = time.perf_counter() - start_time
                    last_error = ToolExecutionTimeout(
                        f"MCP tool '{tool_name}' timed out on server '{server_name}'",
                        server_name=server_name,
                    )
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, "timeout"))
                    record_ai_operation(
                        operation_type=_OP_TOOL_ERROR_TIMEOUT,
                        provider=server_name,
                        model=tool_name,
                        duration_s=duration,
                        success=False,
                    )
                    logger.warning(
                        "Timeout invoking MCP tool '%s' on server '%s' (attempt %s)",
                        tool_name,
                        server_name,
                        attempt,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    duration = time.perf_counter() - start_time
                    last_error = ToolExecutionError(
                        f"Unexpected error executing '{tool_name}' on '{server_name}'",
                        server_name=server_name,
                    )
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, "exception"))
                    record_ai_operation(
                        operation_type=_OP_TOOL_ERROR_EXCEPTION,
                        provider=server_name,
                        model=tool_name,
                        duration_s=duration,
                        success=False,
                    )
                    logger.exception(
                        "Unexpected failure invoking MCP tool '%s' on server '%s'",
                        tool_name,
                        server_name,
                    )
                else:
                    duration = time.perf_counter() - start_time
                    duration_ms = duration * 1000.0
                    execution = ToolExecutionResult(
                        tool_name=tool_name,
                        server_name=server_name,
                        duration_ms=duration_ms,
                        result=result,
                    )
                    record_ai_operation(
                        operation_type=_OP_TOOL_CALL,
                        provider=server_name,
                        model=tool_name,
                        duration_s=duration,
                        success=not execution.is_error,
                    )
                    if execution.is_error:
                        last_error = ToolExecutionFailure(
                            f"MCP tool '{tool_name}' reported an error "
                            f"on server '{server_name}'",
                            server_name=server_name,
                        )
                        span.record_exception(last_error)
                        span.set_status(Status(StatusCode.ERROR, "tool_error"))
                        record_ai_operation(
                            operation_type=_OP_TOOL_ERROR_REMOTE,
                            provider=server_name,
                            model=tool_name,
                            duration_s=duration,
                            success=False,
                        )
                        logger.error(
                            "MCP tool '%s' reported error on server '%s'",
                            tool_name,
                            server_name,
                        )
                    else:
                        span.set_status(Status(StatusCode.OK))
                        return execution, None

            if attempt < self._max_attempts and self._backoff_seconds > 0:
                base_delay = self._backoff_seconds * attempt
                jitter = random.uniform(0, self._backoff_seconds)
                await asyncio.sleep(base_delay + jitter)

        return None, last_error

    @staticmethod
    async def detect_tools(client: MultiServerMCPClient) -> dict[str, list[str]]:
        """Return the tools exposed by each configured server."""
        summary: dict[str, list[str]] = {}
        for name in client.connections:
            async with ToolExecutionService._open_session(client, name) as session:
                tools = await session.list_tools()
                summary[name] = [tool.name for tool in tools]
        return summary

    @staticmethod
    @asynccontextmanager
    async def _open_session(
        client: MultiServerMCPClient, server_name: str
    ) -> AsyncIterator[Any]:
        """Yield an MCP session with consistent error conversion."""
        try:
            async with client.session(server_name) as session:
                yield session
        except ValueError as exc:  # server not configured
            msg = f"MCP server '{server_name}' is not configured"
            raise APIError(msg) from exc


__all__ = [
    "ToolExecutionError",
    "ToolExecutionFailure",
    "ToolExecutionInvalidArgument",
    "ToolExecutionResult",
    "ToolExecutionService",
    "ToolExecutionTimeout",
]
