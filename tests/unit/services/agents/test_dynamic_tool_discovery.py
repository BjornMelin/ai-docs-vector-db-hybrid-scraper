"""Tests for dynamic discovery of MCP tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapabilityType,
)


@dataclass(slots=True)
class DummyTool:
    """Lightweight representation of an MCP tool entry."""

    name: str
    description: str
    inputSchema: dict[str, Any] | None = None  # noqa: N815
    outputSchema: dict[str, Any] | None = None  # noqa: N815


class DummySession:
    """Async context manager for MCP sessions returning deterministic tools."""

    def __init__(self, tools: list[DummyTool]) -> None:
        self._tools = tools

    async def __aenter__(self) -> DummySession:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    async def list_tools(self) -> list[DummyTool]:
        await asyncio.sleep(0)
        return self._tools


class DummyClient:
    """Multi-server client stub exposing session() and connections metadata."""

    def __init__(self, inventory: dict[str, list[DummyTool]]) -> None:
        self.connections = inventory

    def session(self, server_name: str) -> DummySession:  # type: ignore[override]
        return DummySession(self.connections[server_name])


def _make_discovery(
    client: DummyClient,
    *,
    cache_ttl_seconds: int = 60,
    telemetry_hook: Callable[[dict[str, Any]], None] | None = None,
) -> DynamicToolDiscovery:
    typed_client = cast(MultiServerMCPClient, client)
    return DynamicToolDiscovery(
        typed_client,
        cache_ttl_seconds=cache_ttl_seconds,
        telemetry_hook=telemetry_hook,
    )


@pytest.mark.asyncio
async def test_refresh_respects_ttl_and_force_flag() -> None:
    """Cache refresh should respect TTL and allow forced updates."""

    primary_tools = [
        DummyTool(
            name="hybrid_search",
            description="Search documents",
            inputSchema={"properties": {"query": {"type": "string"}}},
        )
    ]
    client = DummyClient({"primary": primary_tools})
    events: list[dict[str, Any]] = []
    discovery = _make_discovery(
        client,
        cache_ttl_seconds=60,
        telemetry_hook=events.append,
    )

    await discovery.refresh(force=True)
    assert len(discovery.get_capabilities()) == 1
    assert events  # telemetry should emit cache miss + refresh events

    client.connections["primary"].append(
        DummyTool(
            name="qa_generate",
            description="Generate answers",
            outputSchema={"properties": {"answer": {"type": "string"}}},
        )
    )

    await discovery.refresh()
    assert len(discovery.get_capabilities()) == 1  # cache hit, no change

    await discovery.refresh(force=True)
    names = {cap.name for cap in discovery.get_capabilities()}
    assert names == {"hybrid_search", "qa_generate"}


@pytest.mark.asyncio
async def test_capability_metadata_and_type_detection() -> None:
    """Discovery should classify tool schemas into capability types."""

    tools = [
        DummyTool(
            name="semantic_search",
            description="Perform semantic search",
            inputSchema={"properties": {"query": {"type": "string"}}},
        ),
        DummyTool(
            name="report_analysis",
            description="Analyse monthly reports",
            outputSchema={"properties": {"insights": {"type": "array"}}},
        ),
    ]
    discovery = _make_discovery(DummyClient({"primary": tools}))
    await discovery.refresh(force=True)

    search_capability = discovery.get_capability("semantic_search", server="primary")
    assert search_capability is not None
    assert search_capability.capability_type == ToolCapabilityType.SEARCH
    assert search_capability.input_schema == ("query",)

    analysis_capability = discovery.get_capability("report_analysis")
    assert analysis_capability is not None
    assert analysis_capability.capability_type == ToolCapabilityType.ANALYSIS
    assert analysis_capability.output_schema == ("insights",)


@pytest.mark.asyncio
async def test_get_capabilities_returns_copy() -> None:
    """Callers should receive an immutable snapshot of cached capabilities."""

    tools = [DummyTool(name="ops", description="Operations tool")]
    discovery = _make_discovery(DummyClient({"primary": tools}))
    await discovery.refresh(force=True)

    first_snapshot = discovery.get_capabilities()
    second_snapshot = discovery.get_capabilities()
    assert first_snapshot is not second_snapshot


@pytest.mark.asyncio
async def test_cache_hit_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache hits should emit telemetry events with correct metadata."""

    tools = [DummyTool(name="ops", description="Operations tool")]
    client = DummyClient({"primary": tools})
    telemetry: list[dict[str, Any]] = []
    discovery = _make_discovery(
        client,
        cache_ttl_seconds=120,
        telemetry_hook=telemetry.append,
    )

    await discovery.refresh(force=True)  # populate cache
    await discovery.refresh()  # should hit cache

    assert any(event.get("tool_count") == 1 for event in telemetry)
