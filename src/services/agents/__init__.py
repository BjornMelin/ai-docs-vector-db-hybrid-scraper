"""Pydantic-AI based agentic RAG system.

This module provides autonomous agents for intelligent query processing,
tool composition, and multi-agent coordination in RAG workflows.
"""

from .core import (
    AgentState,
    BaseAgent,
    BaseAgentDependencies,
    create_agent_dependencies,
)
from .query_orchestrator import QueryOrchestrator
from .tool_composition import ToolCompositionEngine


__all__ = [
    "AgentState",
    "BaseAgent",
    "BaseAgentDependencies",
    "QueryOrchestrator",
    "ToolCompositionEngine",
    "create_agent_dependencies",
]
