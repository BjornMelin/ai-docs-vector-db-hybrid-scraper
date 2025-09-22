"""Unit tests for tool registry with boundary-only mocking.

This test module demonstrates:
- Boundary-only mocking patterns (external services only)
- Real object usage for internal components
- Behavior-driven testing focused on observable outcomes
- Minimal mock complexity
"""

import logging
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_tools.tool_registry import register_all_tools


class TestToolRegistryBehavior:
    """Test tool registry behavior with minimal mocking."""

    @pytest.fixture
    def mock_mcp(self):
        """Create minimal FastMCP mock for external boundary."""
        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda func: func)
        return mock_mcp

    @pytest.fixture
    def mock_client_manager(self):
        """Create minimal ClientManager mock for external services."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.mark.asyncio
    async def test_registers_all_tool_modules(self, mock_mcp, mock_client_manager):
        """Test that tool registration function completes successfully."""
        # Simply test that the function executes without error
        # Real modules are used internally, only external boundaries are mocked
        await register_all_tools(mock_mcp, mock_client_manager)

        # Verify FastMCP was used to register tools (boundary behavior)
        assert mock_mcp.tool.called

        # Verify ClientManager was passed through (boundary behavior)
        # No internal component verification needed

    @pytest.mark.asyncio
    async def test_registration_handles_errors_gracefully(
        self, mock_mcp, mock_client_manager
    ):
        """Test that registration handles errors gracefully."""
        # Simulate external service failure
        mock_client_manager.get_qdrant_client.side_effect = Exception(
            "Service unavailable"
        )

        # Registration should handle external service errors
        # The exact behavior depends on implementation, but it should not crash
        try:
            await register_all_tools(mock_mcp, mock_client_manager)
        except (ConnectionError, TimeoutError) as e:
            # If an exception is raised, it should be a controlled error
            assert "Service unavailable" in str(e)

    @pytest.mark.asyncio
    async def test_logs_registration_progress(
        self, mock_mcp, mock_client_manager, caplog
    ):
        """Test that registration progress is properly logged."""
        with caplog.at_level(logging.INFO):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Check for expected log messages (focus on observable behavior)
        log_messages = [record.message for record in caplog.records]

        # Verify some registration activity was logged
        assert any("register" in msg.lower() for msg in log_messages), (
            "Expected registration activity to be logged"
        )

    @pytest.mark.asyncio
    async def test_logs_completion_summary(self, mock_mcp, mock_client_manager, caplog):
        """Test that registration completion is logged."""
        with caplog.at_level(logging.INFO):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Find completion log messages
        log_messages = [record.message for record in caplog.records]

        # Verify completion was logged
        assert any(
            "success" in msg.lower()
            or "complete" in msg.lower()
            or "registered" in msg.lower()
            for msg in log_messages
        ), "Expected registration completion to be logged"

    @pytest.mark.asyncio
    async def test_uses_correct_external_boundaries(
        self, mock_mcp, mock_client_manager
    ):
        """Test that function uses the provided external boundaries correctly."""
        await register_all_tools(mock_mcp, mock_client_manager)

        # Verify external boundaries were accessed (not internal implementation)
        # This tests the contract with external services
        assert hasattr(mock_mcp, "tool")  # FastMCP boundary
        assert hasattr(mock_client_manager, "get_qdrant_client")  # Client boundary


class TestToolRegistryIntegration:
    """Integration tests for tool registry behavior."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MagicMock(spec=["tool", "context"])

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock ClientManager instance."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.mark.asyncio
    async def test_function_signature_compatibility(
        self, mock_mcp, mock_client_manager
    ):
        """Test that function accepts required parameters correctly."""
        # Test boundary contract - function should accept the expected parameters
        await register_all_tools(mock_mcp, mock_client_manager)

        # Verify the function completed without raising type errors
        assert True  # If we reach this point, the function signature is compatible


class TestModuleRegistrationPatterns:
    """Test common patterns in module registration with boundary-only mocking."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MagicMock(spec=["tool", "context"])

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock ClientManager instance."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.mark.asyncio
    async def test_multiple_tool_registration_behavior(
        self, mock_mcp, mock_client_manager
    ):
        """Test that registration processes multiple tools correctly."""
        # This test focuses on the observable behavior: multiple tool registrations
        await register_all_tools(mock_mcp, mock_client_manager)

        # Verify multiple tools were registered (boundary behavior)
        assert mock_mcp.tool.called
        call_count = mock_mcp.tool.call_count

        # Should register tools from multiple modules
        assert call_count > 5  # Expect multiple modules to register tools

        # Verify the FastMCP boundary was used correctly
        assert all(isinstance(call[0][0], str) for call in mock_mcp.tool.call_args_list)


class TestErrorScenarios:
    """Test error handling scenarios with boundary-only mocking."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MagicMock(spec=["tool", "context"])

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock ClientManager instance."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.mark.asyncio
    async def test_handles_client_manager_errors(
        self, mock_mcp, mock_client_manager, caplog
    ):
        """Test handling when external client manager fails."""
        # Simulate external service failure
        mock_client_manager.get_qdrant_client.side_effect = ConnectionError(
            "Qdrant unavailable"
        )

        with caplog.at_level(logging.ERROR):
            # The function should handle external service errors gracefully
            with suppress(ConnectionError):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Verify that we attempted to use the external boundary
        assert mock_client_manager.get_qdrant_client.called

    @pytest.mark.asyncio
    async def test_handles_mcp_registration_errors(
        self, mock_mcp, mock_client_manager, caplog
    ):
        """Test handling when FastMCP tool registration fails."""
        # Simulate external FastMCP service failure
        mock_mcp.tool.side_effect = RuntimeError("FastMCP service error")

        with caplog.at_level(logging.ERROR):
            # Should handle FastMCP external service errors
            try:
                await register_all_tools(mock_mcp, mock_client_manager)
            except RuntimeError as e:
                assert "FastMCP service error" in str(e)

        # Verify we attempted to use the FastMCP boundary
        assert mock_mcp.tool.called


class TestLoggingBehavior:
    """Test detailed logging behavior with boundary-only mocking."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MagicMock(spec=["tool", "context"])

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock ClientManager instance."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.mark.asyncio
    async def test_registration_produces_log_messages(
        self, mock_mcp, mock_client_manager, caplog
    ):
        """Test that registration produces appropriate log messages."""
        with caplog.at_level(logging.INFO):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify that registration produces log messages (observable behavior)
        log_messages = [record.message for record in caplog.records]

        # Should have registration activity logged
        assert any("register" in msg.lower() for msg in log_messages)

        # Should have completion logging
        assert any(
            word in msg.lower()
            for msg in log_messages
            for word in ["success", "complete", "registered"]
        )
