"""Dynamic discovery of MCP tool capabilities with TTL caching."""

# pylint: disable=import-error

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from src.infrastructure.client_manager import ClientManager
from src.services.errors import APIError


logger = logging.getLogger(__name__)


class ToolCapabilityType(str, Enum):
    """Categories describing the primary function of an MCP tool."""

    SEARCH = "search"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    SYNTHESIS = "synthesis"
    ORCHESTRATION = "orchestration"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ToolMetrics:
    """Lightweight telemetry snapshot for a tool."""

    average_latency_ms: float = 0.0
    success_rate: float = 0.0
    accuracy_score: float = 0.0
    cost_per_execution: float = 0.0
    reliability_score: float = 0.0


class ToolCapability(BaseModel):
    """Structured representation of an MCP tool capability."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    capability_type: ToolCapabilityType
    description: str = ""
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    requirements: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    compatible_tools: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    metrics: ToolMetrics | None = None
    confidence_score: float = Field(default=0.6)
    last_updated: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class DynamicToolDiscovery:
    """Discover and cache MCP tools exposed by MultiServerMCPClient."""

    def __init__(
        self,
        client_manager: ClientManager,
        *,
        cache_ttl_seconds: int = 60,
        telemetry_hook: Callable[[ToolCapability], None] | None = None,
    ) -> None:
        self._client_manager = client_manager
        self._cache_ttl = timedelta(seconds=max(1, cache_ttl_seconds))
        self._telemetry_hook = telemetry_hook
        self._last_refresh: datetime | None = None
        self._capabilities: dict[str, ToolCapability] = {}

    async def refresh(self) -> None:
        """Refresh cached capabilities when the TTL expires."""

        now = datetime.now(tz=UTC)
        if self._last_refresh and now - self._last_refresh < self._cache_ttl:
            return

        await self._scan_available_tools(timestamp=now.isoformat())
        self._assess_tool_compatibility()
        self._last_refresh = datetime.now(tz=UTC)
        logger.info("DynamicToolDiscovery refreshed %d tools", len(self._capabilities))

    def get_capabilities(self) -> Sequence[ToolCapability]:
        """Return the currently cached capabilities."""

        return tuple(self._capabilities.values())

    async def _scan_available_tools(self, *, timestamp: str) -> None:
        """Populate the internal cache by querying every configured MCP server."""

        self._capabilities.clear()
        try:
            client = await cast(Any, self._client_manager).get_mcp_client()
        except APIError as exc:
            logger.warning("Skipping tool discovery; MCP client unavailable: %s", exc)
            return

        for server_name, connection in client.connections.items():
            try:
                async with cast(Any, client).session(server_name) as session:
                    tools = await session.list_tools()
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "Failed to list tools from MCP server '%s'", server_name
                )
                continue

            transport = connection.get("transport")
            for tool in tools:
                capability = self._build_capability(
                    tool=tool,
                    server_name=server_name,
                    transport=transport,
                    timestamp=timestamp,
                )
                self._capabilities[capability.name] = capability
                if self._telemetry_hook:
                    try:
                        self._telemetry_hook(capability)
                    except (
                        Exception
                    ):  # pragma: no cover - telemetry must never break discovery
                        logger.exception(
                            "Telemetry hook failed for tool '%s'", capability.name
                        )

    def _build_capability(
        self,
        *,
        tool: Any,
        server_name: str,
        transport: str | None,
        timestamp: str,
    ) -> ToolCapability:
        """Convert raw MCP tool metadata into a `ToolCapability`."""

        description = tool.description or tool.title or tool.name
        capability_type = self._infer_capability_type(tool.name, description)
        input_types = self._extract_schema_keys(tool.inputSchema, "payload")
        output_types = self._extract_schema_keys(tool.outputSchema, "content")

        return ToolCapability(
            name=tool.name,
            capability_type=capability_type,
            description=description or "",
            input_types=input_types,
            output_types=output_types,
            requirements={"server": server_name},
            constraints={"transport": transport or "unknown"},
            last_updated=timestamp,
        )

    @staticmethod
    def _infer_capability_type(
        name: str, description: str | None
    ) -> ToolCapabilityType:
        """Heuristically classify a tool based on its metadata."""

        text = f"{name} {(description or '').lower()}".lower()
        keyword_map: list[tuple[ToolCapabilityType, tuple[str, ...]]] = [
            (ToolCapabilityType.SEARCH, ("search", "retrieve", "lookup", "query")),
            (ToolCapabilityType.RETRIEVAL, ("vector", "embedding")),
            (
                ToolCapabilityType.GENERATION,
                ("generate", "answer", "compose", "summarize"),
            ),
            (ToolCapabilityType.ANALYSIS, ("analyze", "assess", "score", "inspect")),
            (ToolCapabilityType.CLASSIFICATION, ("classify", "tag", "label")),
            (ToolCapabilityType.SYNTHESIS, ("synthes", "combine")),
            (ToolCapabilityType.ORCHESTRATION, ("orchestrate", "coordinate")),
        ]
        for capability, keywords in keyword_map:
            if any(term in text for term in keywords):
                return capability
        return ToolCapabilityType.UNKNOWN

    @staticmethod
    def _extract_schema_keys(schema: Any, default_key: str) -> list[str]:
        """Return JSON schema property keys or a default placeholder."""

        if not schema or not isinstance(schema, dict):
            return [default_key]
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            return sorted(properties.keys())
        return [default_key]

    def _assess_tool_compatibility(self) -> None:
        """Populate `compatible_tools` with simple chaining heuristics."""

        for tool_name, tool in self._capabilities.items():
            compatible: set[str] = set()
            if tool.capability_type in {
                ToolCapabilityType.SEARCH,
                ToolCapabilityType.RETRIEVAL,
            }:
                compatible.update(
                    other_name
                    for other_name, other in self._capabilities.items()
                    if other.capability_type
                    in {ToolCapabilityType.GENERATION, ToolCapabilityType.ANALYSIS}
                    and other_name != tool_name
                )
            elif tool.capability_type == ToolCapabilityType.ANALYSIS:
                compatible.update(
                    other_name
                    for other_name, other in self._capabilities.items()
                    if other.capability_type
                    in {ToolCapabilityType.SEARCH, ToolCapabilityType.RETRIEVAL}
                    and other_name != tool_name
                )
            tool.compatible_tools = sorted(compatible)
