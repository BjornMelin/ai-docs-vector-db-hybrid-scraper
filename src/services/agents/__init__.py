"""Pure Pydantic-AI native agentic RAG system.

This module provides autonomous agents for intelligent query processing,
tool composition, and multi-agent coordination using native patterns only.
"""

from .agentic_orchestrator import (
    AgenticOrchestrator,
    get_orchestrator,
    orchestrate_tools,
)
from .core import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    create_agent_dependencies,
)
from .dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    discover_tools_for_task,
    get_discovery_engine,
)
from .query_orchestrator import QueryOrchestrator


__all__ = [
    "AgentState",
    "AgenticOrchestrator",
    "BaseAgent",
    "BaseAgentDependencies",
    "DynamicToolDiscovery",
    "QueryOrchestrator",
    "ToolCapability",
    "create_agent_dependencies",
    "discover_tools_for_task",
    "get_discovery_engine",
    "get_orchestrator",
    "orchestrate_tools",
]
