"""Minimal conftest for agents tests to avoid torch conflicts."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.agents.core import AgentState, BaseAgentDependencies


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for agent testing."""
    mock_client_manager = Mock(spec=ClientManager)
    config = get_config()
    session_state = AgentState(session_id=str(uuid4()))

    return BaseAgentDependencies(
        client_manager=mock_client_manager, config=config, session_state=session_state
    )
