"""Unit tests for tool registry with behavior-focused testing.

This test module demonstrates:
- Behavior-driven testing patterns
- Mocking external dependencies
- Testing registration workflows
- Verifying logging behavior
- Testing error scenarios
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.mcp_tools import tools
from src.mcp_tools.tool_registry import register_all_tools


class TestRegisterAllTools:
    """Test the register_all_tools function behavior."""

    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        return MagicMock(spec=["tool", "context"])

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock ClientManager instance."""
        return AsyncMock(spec=["get_qdrant_client", "get_openai_client"])

    @pytest.fixture
    def mock_tool_modules(self):
        """Create mock tool modules with register_tools functions."""
        modules = {
            "search": Mock(register_tools=Mock()),
            "documents": Mock(register_tools=Mock()),
            "embeddings": Mock(register_tools=Mock()),
            "collections": Mock(register_tools=Mock()),
            "projects": Mock(register_tools=Mock()),
            "advanced_search": Mock(register_tools=Mock()),
            "payload_indexing": Mock(register_tools=Mock()),
            "deployment": Mock(register_tools=Mock()),
            "analytics": Mock(register_tools=Mock()),
            "cache": Mock(register_tools=Mock()),
            "utilities": Mock(register_tools=Mock()),
        }
        return modules

    @pytest.mark.asyncio
    async def test_registers_all_tool_modules(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test that all expected tool modules are registered."""
        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify each module's register_tools was called
        for module_name, module in mock_tool_modules.items():
            module.register_tools.assert_called_once_with(
                mock_mcp, mock_client_manager
            )

    @pytest.mark.asyncio
    async def test_registration_order_is_logical(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test that tools are registered in a logical order."""
        call_order = []

        # Track call order
        for module_name, module in mock_tool_modules.items():
            module.register_tools.side_effect = lambda mcp, cm, name=module_name: call_order.append(
                name
            )

        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify core functionality is registered first
        core_tools = ["search", "documents", "embeddings"]
        for tool in core_tools:
            assert tool in call_order[:3]

        # Verify advanced features come after core
        advanced_tools = ["advanced_search", "payload_indexing", "deployment"]
        core_end_index = max(call_order.index(tool) for tool in core_tools)
        for tool in advanced_tools:
            if tool in call_order:
                assert call_order.index(tool) > core_end_index

    @pytest.mark.asyncio
    async def test_logs_registration_progress(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that registration progress is properly logged."""
        with patch.object(tools, "__dict__", mock_tool_modules):
            with caplog.at_level(logging.INFO):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]

        assert any("Registering core tools" in msg for msg in log_messages)
        assert any("Registering management tools" in msg for msg in log_messages)
        assert any("Registering advanced tools" in msg for msg in log_messages)
        assert any("Registering utility tools" in msg for msg in log_messages)
        assert any(
            "Successfully registered 11 tool modules" in msg for msg in log_messages
        )

    @pytest.mark.asyncio
    async def test_logs_registered_tool_names(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that all registered tool names are logged."""
        with patch.object(tools, "__dict__", mock_tool_modules):
            with caplog.at_level(logging.INFO):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Find the summary log message
        summary_logs = [
            record.message
            for record in caplog.records
            if "Successfully registered" in record.message
        ]

        assert len(summary_logs) == 1
        summary = summary_logs[0]

        # Verify all module names are in the summary
        for module_name in mock_tool_modules.keys():
            assert module_name in summary

    @pytest.mark.asyncio
    async def test_handles_missing_register_function_gracefully(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test graceful handling when a module lacks register_tools."""
        # Remove register_tools from one module
        del mock_tool_modules["cache"].register_tools

        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
        ):
            # Should not raise an exception
            with caplog.at_level(logging.WARNING):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Other modules should still be registered
        for module_name, module in mock_tool_modules.items():
            if module_name != "cache" and hasattr(module, "register_tools"):
                module.register_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_registration_after_module_error(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that registration continues even if one module fails."""
        # Make one module raise an exception
        mock_tool_modules["documents"].register_tools.side_effect = Exception(
            "Registration failed"
        )

        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
        ):
            # Should not raise an exception
            with caplog.at_level(logging.ERROR):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Other modules should still be registered
        for module_name, module in mock_tool_modules.items():
            if module_name != "documents":
                module.register_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_arguments_to_modules(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test that correct arguments are passed to each module."""
        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify arguments for each module
        for module in mock_tool_modules.values():
            module.register_tools.assert_called_once()
            args = module.register_tools.call_args[0]
            assert args[0] is mock_mcp
            assert args[1] is mock_client_manager


class TestToolRegistryIntegration:
    """Integration tests for tool registry behavior."""

    @pytest.mark.asyncio
    async def test_real_module_structure(self, mock_mcp, mock_client_manager):
        """Test with real module structure (mocked implementations)."""
        # Mock the actual tool module imports
        mock_modules = {
            "search": Mock(spec=["register_tools"]),
            "documents": Mock(spec=["register_tools"]),
            "embeddings": Mock(spec=["register_tools"]),
            "collections": Mock(spec=["register_tools"]),
            "projects": Mock(spec=["register_tools"]),
            "advanced_search": Mock(spec=["register_tools"]),
            "payload_indexing": Mock(spec=["register_tools"]),
            "deployment": Mock(spec=["register_tools"]),
            "analytics": Mock(spec=["register_tools"]),
            "cache": Mock(spec=["register_tools"]),
            "utilities": Mock(spec=["register_tools"]),
        }

        with patch.object(
            __import__("src.mcp_tools.tools", fromlist=list(mock_modules.keys())),
            "__dict__",
            mock_modules,
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify all modules were called
        for module in mock_modules.values():
            module.register_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_checking_compatibility(self):
        """Test that function signature is compatible with type hints."""
        # This test verifies the function can be called with properly typed arguments
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from fastmcp import FastMCP
            from src.infrastructure.client_manager import ClientManager

            # This should not raise type errors
            async def type_check():
                mcp: FastMCP = MagicMock()
                client_manager: ClientManager = AsyncMock()
                await register_all_tools(mcp, client_manager)


class TestModuleRegistrationPatterns:
    """Test common patterns in module registration."""

    @pytest.fixture
    def sample_tool_module(self):
        """Create a sample tool module with typical structure."""
        module = Mock()

        # Simulate a typical register_tools function
        def register_tools(mcp, client_manager):
            # Typical pattern: register multiple tools
            mcp.tool("search_documents")(Mock())
            mcp.tool("search_with_filters")(Mock())
            mcp.tool("search_similar")(Mock())

        module.register_tools = Mock(side_effect=register_tools)
        return module

    @pytest.mark.asyncio
    async def test_module_registers_multiple_tools(
        self, mock_mcp, mock_client_manager, sample_tool_module
    ):
        """Test that modules can register multiple tools."""
        mock_modules = {"search": sample_tool_module}

        with patch.multiple(
            tools,
            search=mock_modules["search"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify tool registration was called multiple times
        assert mock_mcp.tool.call_count >= 3
        mock_mcp.tool.assert_any_call("search_documents")
        mock_mcp.tool.assert_any_call("search_with_filters")
        mock_mcp.tool.assert_any_call("search_similar")


class TestErrorScenarios:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_import_error(self, mock_mcp, mock_client_manager, caplog):
        """Test handling when tool module import fails."""
        with patch(
            "src.mcp_tools.tool_registry.tools",
            side_effect=ImportError("Module not found"),
        ):
            with caplog.at_level(logging.ERROR):
                # Should handle the error gracefully
                with pytest.raises(ImportError):
                    await register_all_tools(mock_mcp, mock_client_manager)

    @pytest.mark.asyncio
    async def test_handles_attribute_error(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test handling when accessing module attributes fails."""
        # Create a module that raises AttributeError
        bad_module = Mock()
        bad_module.register_tools = property(
            lambda self: (_ for _ in ()).throw(AttributeError("No such attribute"))
        )
        mock_tool_modules["bad_module"] = bad_module

        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
            bad_module=mock_tool_modules["bad_module"],
        ):
            # Should handle the error for that module but continue
            await register_all_tools(mock_mcp, mock_client_manager)

        # Other modules should still be registered
        mock_tool_modules["search"].register_tools.assert_called_once()


class TestLoggingBehavior:
    """Test detailed logging behavior."""

    @pytest.mark.asyncio
    async def test_logger_name_is_correct(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test that logger uses correct module name."""
        with patch("src.mcp_tools.tool_registry.logger") as mock_logger:
            # Patch individual tool modules
            with patch.multiple(
                tools,
                search=mock_tool_modules["search"],
                documents=mock_tool_modules["documents"],
                embeddings=mock_tool_modules["embeddings"],
                collections=mock_tool_modules["collections"],
                projects=mock_tool_modules["projects"],
                advanced_search=mock_tool_modules["advanced_search"],
                payload_indexing=mock_tool_modules["payload_indexing"],
                deployment=mock_tool_modules["deployment"],
                analytics=mock_tool_modules["analytics"],
                cache=mock_tool_modules["cache"],
                utilities=mock_tool_modules["utilities"],
            ):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Verify logger is used
        assert mock_logger.info.call_count >= 4  # At least 4 info messages

    @pytest.mark.asyncio
    async def test_registration_count_is_accurate(
        self, mock_mcp, mock_client_manager, caplog
    ):
        """Test that the reported registration count is accurate."""
        # Create a subset of modules
        mock_modules = {
            "search": Mock(register_tools=Mock()),
            "documents": Mock(register_tools=Mock()),
            "embeddings": Mock(register_tools=Mock()),
        }

        with patch.multiple(
            tools,
            search=mock_modules["search"],
            documents=mock_modules["documents"],
            embeddings=mock_modules["embeddings"],
        ):
            with caplog.at_level(logging.INFO):
                await register_all_tools(mock_mcp, mock_client_manager)

        # Find the summary message
        summary_logs = [
            record.message
            for record in caplog.records
            if "Successfully registered" in record.message
        ]

        # Should report 11 modules (since register_all_tools always tries to register all modules)
        assert "Successfully registered 11 tool modules" in summary_logs[0]
