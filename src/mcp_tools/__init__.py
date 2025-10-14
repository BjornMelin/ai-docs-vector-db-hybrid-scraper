"""MCP server package for registering tool modules."""

from __future__ import annotations


__all__ = ["register_all_tools"]


def register_all_tools(*args, **kwargs):
    """Import and delegate to the registry function lazily."""
    from .tool_registry import register_all_tools as _register_all_tools

    return _register_all_tools(*args, **kwargs)
