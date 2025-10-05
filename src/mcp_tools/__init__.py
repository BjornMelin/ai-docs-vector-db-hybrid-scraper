"""MCP server package for registering tool modules."""

from __future__ import annotations

from .tool_registry import register_all_tools


__all__ = ["register_all_tools"]
