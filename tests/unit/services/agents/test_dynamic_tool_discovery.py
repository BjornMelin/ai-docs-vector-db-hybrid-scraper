"""Tests for the dynamic tool discovery service."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/services/agents/dynamic_tool_discovery.py"
)

spec = importlib.util.spec_from_file_location(
    "dynamic_tool_discovery_under_test", MODULE_PATH
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[arg-type]

DynamicToolDiscovery = module.DynamicToolDiscovery
ToolCapabilityType = module.ToolCapabilityType


@dataclass(slots=True)
class DummyTool:
    name: str
    description: str
    inputSchema: dict[str, Any] | None = None  # noqa: N815
    outputSchema: dict[str, Any] | None = None  # noqa: N815


class DummySession:
    def __init__(self, tools: list[DummyTool]) -> None:
        self._tools = tools

    async def __aenter__(self) -> DummySession:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401, ANN001
        return None

    async def list_tools(self) -> list[DummyTool]:  # noqa: D401
        await asyncio.sleep(0)
        return self._tools


class DummyClient:
    def __init__(self, inventory: dict[str, list[DummyTool]]) -> None:
        self.connections = inventory

    def session(self, server_name: str) -> DummySession:  # type: ignore[override]
        return DummySession(self.connections[server_name])


class DummyClientManager:
    def __init__(self, client: DummyClient) -> None:
        self._client = client

    async def get_mcp_client(self) -> DummyClient:  # noqa: D401
        return self._client


@pytest.mark.asyncio
async def test_refresh_respects_ttl_and_force_flag() -> None:
    """Discovery should honour TTL caching and allow forced refreshes."""

    primary_tools = [
        DummyTool(
            name="hybrid_search",
            description="Search documents",
            inputSchema={"properties": {"query": {"type": "string"}}},
        )
    ]
    client = DummyClient({"primary": primary_tools})
    events: list[dict[str, Any]] = []
    discovery = DynamicToolDiscovery(
        DummyClientManager(client),
        cache_ttl_seconds=60,
        telemetry_hook=events.append,
    )

    await discovery.refresh(force=True)
    assert len(discovery.get_capabilities()) == 1
    assert events

    client.connections["primary"].append(
        DummyTool(
            name="qa_generate",
            description="Generate answers",
            outputSchema={"properties": {"answer": {"type": "string"}}},
        )
    )

    await discovery.refresh()
    assert len(discovery.get_capabilities()) == 1

    await discovery.refresh(force=True)
    capabilities = discovery.get_capabilities()
    assert {cap.name for cap in capabilities} == {"hybrid_search", "qa_generate"}


@pytest.mark.asyncio
async def test_capability_metadata_and_type_detection() -> None:
    """Discovery should classify tools and expose schema keys."""

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
    discovery = DynamicToolDiscovery(
        DummyClientManager(DummyClient({"primary": tools}))
    )
    await discovery.refresh(force=True)

    search_capability = discovery.get_capability("semantic_search", server="primary")
    assert search_capability is not None
    assert search_capability.capability_type == ToolCapabilityType.SEARCH
    assert search_capability.input_schema == ("query",)

    analysis_capability = discovery.get_capability("report_analysis")
    assert analysis_capability is not None
    assert analysis_capability.capability_type == ToolCapabilityType.ANALYSIS
    assert analysis_capability.output_schema == ("insights",)
