"""Dynamic discovery of MCP tool capabilities with TTL caching."""

# pylint: disable=too-many-instance-attributes,too-many-return-statements

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from opentelemetry import trace

from src.services.errors import APIError
from src.services.observability.tracking import record_ai_operation


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_OP_DISCOVERY_CACHE = "mcp.discovery.cache"
_OP_DISCOVERY_REFRESH = "mcp.discovery.refresh"
_OP_DISCOVERY_CHANGE = "mcp.discovery.change"
_OP_DISCOVERY_RUN = "mcp.discovery.run"
_OP_DISCOVERY_ERROR = "mcp.discovery.error"


class ToolCapabilityType(str, Enum):
    """High-level categories for discovered tools."""

    SEARCH = "search"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    MANAGEMENT = "management"
    OTHER = "other"


@dataclass(slots=True)
class ToolCapability:
    """Description of a discovered tool capability."""

    name: str
    server: str
    description: str
    capability_type: ToolCapabilityType
    input_schema: tuple[str, ...]
    output_schema: tuple[str, ...]
    metadata: dict[str, Any]
    last_refreshed: str

    def model_dump(self) -> dict[str, Any]:
        """Return a serialisable representation."""

        return {
            "name": self.name,
            "server": self.server,
            "description": self.description,
            "capability_type": self.capability_type.value,
            "input_schema": list(self.input_schema),
            "output_schema": list(self.output_schema),
            "metadata": self.metadata,
            "last_refreshed": self.last_refreshed,
        }


class DynamicToolDiscovery:  # pylint: disable=too-many-instance-attributes
    """Discover tools from configured MCP servers with TTL-based caching."""

    def __init__(
        self,
        client: MultiServerMCPClient | Callable[[], Awaitable[MultiServerMCPClient]],
        *,
        cache_ttl_seconds: int = 60,
        telemetry_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialise the dynamic tool discovery service."""

        if callable(client):
            self._client_factory = client
            self._client: MultiServerMCPClient | None = None
        else:
            self._client = client
            self._client_factory = None
        self._cache_ttl_seconds = max(0, cache_ttl_seconds)
        self._telemetry_hook = telemetry_hook
        self._capabilities: dict[str, ToolCapability] = {}
        self._cache_expires_at = 0.0
        self._lock = asyncio.Lock()

    async def _resolve_client(self) -> MultiServerMCPClient:
        """Return the MCP client, creating it if necessary."""

        if self._client is not None:
            return self._client
        if self._client_factory is None:
            msg = "MCP client resolver is not configured"
            raise RuntimeError(msg)
        client = await self._client_factory()
        self._client = client
        return client

    async def refresh(self, *, force: bool = False) -> None:
        """Populate the capability cache when the TTL expires.

        Args:
            force: If ``True``, refresh even when the cache TTL has not elapsed.
        """

        now = time.monotonic()
        if not force and self._cache_ttl_seconds and now < self._cache_expires_at:
            record_ai_operation(
                operation_type=_OP_DISCOVERY_CACHE,
                provider="mcp.discovery",
                model="hit",
                duration_s=0.0,
            )
            return
        async with self._lock:
            now = time.monotonic()
            if not force and self._cache_ttl_seconds and now < self._cache_expires_at:
                record_ai_operation(
                    operation_type=_OP_DISCOVERY_CACHE,
                    provider="mcp.discovery",
                    model="hit",
                    duration_s=0.0,
                )
                return

            start_time = time.perf_counter()
            with tracer.start_as_current_span(
                "mcp.discovery.refresh",
                attributes={"mcp.discovery.force": force},
            ):
                capabilities = await self._discover_capabilities()
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            record_ai_operation(
                operation_type=_OP_DISCOVERY_CACHE,
                provider="mcp.discovery",
                model="miss" if not force else "force_refresh",
                duration_s=0.0,
            )
            record_ai_operation(
                operation_type=_OP_DISCOVERY_REFRESH,
                provider="mcp.discovery",
                model="refresh",
                duration_s=duration_ms / 1000.0,
            )

            previous_keys = set(self._capabilities.keys())
            new_keys = set(capabilities.keys())
            added_keys = new_keys - previous_keys
            removed_keys = previous_keys - new_keys
            for key in added_keys:
                server, _ = key.split(":", 1)
                record_ai_operation(
                    operation_type=_OP_DISCOVERY_CHANGE,
                    provider=server,
                    model="added",
                    duration_s=0.0,
                )
            for key in removed_keys:
                server, _ = key.split(":", 1)
                record_ai_operation(
                    operation_type=_OP_DISCOVERY_CHANGE,
                    provider=server,
                    model="removed",
                    duration_s=0.0,
                )

            self._capabilities = capabilities
            self._cache_expires_at = now + self._cache_ttl_seconds
            if self._telemetry_hook:
                try:
                    self._telemetry_hook(
                        {
                            "tool_count": len(capabilities),
                            "servers": sorted(
                                {cap.server for cap in capabilities.values()}
                            ),
                            "added_tools": sorted(added_keys),
                            "removed_tools": sorted(removed_keys),
                            "refresh_duration_ms": duration_ms,
                            "timestamp": datetime.now(tz=UTC).isoformat(),
                        }
                    )
                except Exception:  # pragma: no cover - telemetry must not break flow
                    logger.exception("Telemetry hook failed for DynamicToolDiscovery")

    def get_capabilities(self) -> tuple[ToolCapability, ...]:
        """Return cached capabilities as an immutable sequence."""

        return tuple(self._capabilities.values())

    def get_capability(
        self, name: str, server: str | None = None
    ) -> ToolCapability | None:
        """Return a specific capability by tool name and optional server.

        Args:
            name: Tool identifier.
            server: Optional MCP server limiting the lookup.

        Returns:
            Matching ``ToolCapability`` or ``None`` when absent.
        """

        if server is not None:
            return self._capabilities.get(self._cache_key(server, name))
        for capability in self._capabilities.values():
            if capability.name == name:
                return capability
        return None

    async def _discover_capabilities(self) -> dict[str, ToolCapability]:
        # pylint: disable=too-many-return-statements
        """Collect tool metadata from each configured MCP server.

        Returns:
            Mapping from cache key to ``ToolCapability`` records.

        Raises:
            APIError: If no MCP connections are configured.
        """

        client = await self._resolve_client()
        if not getattr(client, "connections", None):
            msg = "No MCP server connections available for discovery"
            raise APIError(msg)

        capabilities: dict[str, ToolCapability] = {}
        for server_name in client.connections:
            record_ai_operation(
                operation_type=_OP_DISCOVERY_RUN,
                provider=server_name,
                model="discover",
                duration_s=0.0,
            )
            try:
                async with self._open_session(client, server_name) as session:
                    tools = await session.list_tools()
            except Exception:  # pragma: no cover - defensive guard
                record_ai_operation(
                    operation_type=_OP_DISCOVERY_ERROR,
                    provider=server_name,
                    model="session_error",
                    duration_s=0.0,
                    success=False,
                )
                logger.exception(
                    "Tool discovery failed for MCP server '%s'", server_name
                )
                continue

            for tool in tools:
                name = getattr(tool, "name", "")
                description = getattr(tool, "description", "") or ""
                input_schema = self._extract_schema_keys(
                    getattr(tool, "inputSchema", None)
                )
                output_schema = self._extract_schema_keys(
                    getattr(tool, "outputSchema", None)
                )
                capability = ToolCapability(
                    name=name,
                    server=server_name,
                    description=description,
                    capability_type=self._infer_capability_type(name, description),
                    input_schema=input_schema,
                    output_schema=output_schema,
                    metadata={
                        "input_schema": getattr(tool, "inputSchema", None),
                        "output_schema": getattr(tool, "outputSchema", None),
                    },
                    last_refreshed=datetime.now(tz=UTC).isoformat(),
                )
                capabilities[self._cache_key(server_name, name)] = capability
        return capabilities

    @staticmethod
    def _cache_key(server: str, name: str) -> str:
        return f"{server}:{name}"

    @staticmethod
    def _extract_schema_keys(schema: Mapping[str, Any] | None) -> tuple[str, ...]:
        """Return sorted property keys from a JSON schema mapping."""

        if not schema:
            return ()
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            return tuple(sorted(str(key) for key in properties))
        return ()

    @staticmethod
    def _infer_capability_type(name: str, description: str) -> ToolCapabilityType:
        text = f"{name} {description}".lower()
        if any(keyword in text for keyword in ("search", "query", "lookup")):
            return ToolCapabilityType.SEARCH
        if any(keyword in text for keyword in ("retrieve", "fetch", "vector")):
            return ToolCapabilityType.RETRIEVAL
        if any(keyword in text for keyword in ("generate", "write", "summar")):
            return ToolCapabilityType.GENERATION
        if any(keyword in text for keyword in ("analyse", "analyze", "classify")):
            return ToolCapabilityType.ANALYSIS
        if any(keyword in text for keyword in ("synthes", "aggregate", "combine")):
            return ToolCapabilityType.SYNTHESIS
        if any(keyword in text for keyword in ("config", "manage", "admin")):
            return ToolCapabilityType.MANAGEMENT
        return ToolCapabilityType.OTHER

    @staticmethod
    @asynccontextmanager
    async def _open_session(
        client: MultiServerMCPClient, server_name: str
    ) -> AsyncIterator[Any]:
        """Open an MCP session with uniform error translation.

        Args:
            client: Multi-server MCP client responsible for the session.
            server_name: Identifier of the server to target.

        Yields:
            Active MCP session for the requested server.
        """

        try:
            async with client.session(server_name) as session:
                yield session
        except ValueError as exc:  # server not configured
            record_ai_operation(
                operation_type=_OP_DISCOVERY_ERROR,
                provider=server_name,
                model="not_configured",
                duration_s=0.0,
                success=False,
            )
            msg = f"MCP server '{server_name}' is not configured"
            raise APIError(msg) from exc


__all__ = ["DynamicToolDiscovery", "ToolCapability", "ToolCapabilityType"]
