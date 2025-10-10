"""Pytest fixtures for mcp_services unit tests."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
async def mock_client_manager():
    """Test client manager for mcp services SystemService tool modules.

    Mocks only the methods actually used by SystemService tool modules:
    - get_embedding_manager() for embeddings.py
    """

    manager = AsyncMock()
    manager.get_embedding_manager = AsyncMock()
    return manager
