"""Stubbed LangChain MCP client module for unit tests."""


class MultiServerMCPClient:
    """Minimal stub for MultiServerMCPClient."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    async def connect(self) -> None:
        """No-op async connect."""

    async def disconnect(self) -> None:
        """No-op async disconnect."""
