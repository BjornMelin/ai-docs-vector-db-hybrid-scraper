"""Stubbed LangChain MCP session helpers for unit tests."""


class Connection:
    """Minimal stub for Connection."""

    async def __aenter__(self) -> "Connection":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Exit the async context manager."""
        return
