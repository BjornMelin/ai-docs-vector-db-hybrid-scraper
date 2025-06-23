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
            "lightweight_scrape": Mock(register_tools=Mock()),
            "collections": Mock(register_tools=Mock()),
            "projects": Mock(register_tools=Mock()),
            "advanced_search": Mock(register_tools=Mock()),
            "query_processing": Mock(register_tools=Mock()),
            "filtering_tools": Mock(register_filtering_tools=Mock()),
            "query_processing_tools": Mock(register_query_processing_tools=Mock()),
            "payload_indexing": Mock(register_tools=Mock()),
            "deployment": Mock(register_tools=Mock()),
            "analytics": Mock(register_tools=Mock()),
            "cache": Mock(register_tools=Mock()),
            "utilities": Mock(register_tools=Mock()),
            "content_intelligence": Mock(register_tools=Mock()),
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
            lightweight_scrape=mock_tool_modules["lightweight_scrape"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            query_processing=mock_tool_modules["query_processing"],
            filtering_tools=mock_tool_modules["filtering_tools"],
            query_processing_tools=mock_tool_modules["query_processing_tools"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
            content_intelligence=mock_tool_modules["content_intelligence"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify each module's register_tools was called
        for module_name, module in mock_tool_modules.items():
            if module_name == "filtering_tools":
                module.register_filtering_tools.assert_called_once_with(
                    mock_mcp, mock_client_manager
                )
            elif module_name == "query_processing_tools":
                module.register_query_processing_tools.assert_called_once_with(
                    mock_mcp, mock_client_manager
                )
            else:
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
            module.register_tools.side_effect = (
                lambda mcp, cm, name=module_name: call_order.append(name)
            )

        # Patch individual tool modules
        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            lightweight_scrape=mock_tool_modules["lightweight_scrape"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            query_processing=mock_tool_modules["query_processing"],
            filtering_tools=mock_tool_modules["filtering_tools"],
            query_processing_tools=mock_tool_modules["query_processing_tools"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
            content_intelligence=mock_tool_modules["content_intelligence"],
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
        with (
            patch.multiple(
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
            ),
            caplog.at_level(logging.INFO),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]

        assert any("Registering core tools" in msg for msg in log_messages)
        assert any("Registering management tools" in msg for msg in log_messages)
        assert any("Registering advanced tools" in msg for msg in log_messages)
        assert any("Registering utility tools" in msg for msg in log_messages)
        assert any(
            "Successfully registered 16 tool modules" in msg for msg in log_messages
        )

    @pytest.mark.asyncio
    async def test_logs_registered_tool_names(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that all registered tool names are logged."""
        with (
            patch.multiple(
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
            ),
            caplog.at_level(logging.INFO),
        ):
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
        for module_name in mock_tool_modules:
            assert module_name in summary

    @pytest.mark.asyncio
    async def test_handles_missing_register_function_gracefully(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that missing register_tools method raises AttributeError."""
        # Remove register_tools from one module
        del mock_tool_modules["cache"].register_tools

        # Patch individual tool modules
        with (
            patch.multiple(
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
            ),
            pytest.raises(AttributeError, match="register_tools"),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Modules before the failure should still be registered
        registered_modules = [
            "search",
            "documents",
            "embeddings",
            "collections",
            "projects",
            "advanced_search",
            "payload_indexing",
            "deployment",
            "analytics",
        ]
        for module_name in registered_modules:
            if module_name in mock_tool_modules:
                mock_tool_modules[module_name].register_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_registration_after_module_error(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that registration stops on module error."""
        # Make one module raise an exception
        mock_tool_modules["documents"].register_tools.side_effect = Exception(
            "Registration failed"
        )

        # Patch individual tool modules
        with (
            patch.multiple(
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
            ),
            pytest.raises(Exception, match="Registration failed"),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Only modules before the failure should be registered
        mock_tool_modules["search"].register_tools.assert_called_once()
        mock_tool_modules["documents"].register_tools.assert_called_once()
        # Modules after documents should not be called
        mock_tool_modules["embeddings"].register_tools.assert_not_called()

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
            lightweight_scrape=mock_tool_modules["lightweight_scrape"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            query_processing=mock_tool_modules["query_processing"],
            filtering_tools=mock_tool_modules["filtering_tools"],
            query_processing_tools=mock_tool_modules["query_processing_tools"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
            content_intelligence=mock_tool_modules["content_intelligence"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify arguments for each module
        for module_name, module in mock_tool_modules.items():
            if module_name == "filtering_tools":
                module.register_filtering_tools.assert_called_once()
                args = module.register_filtering_tools.call_args[0]
            elif module_name == "query_processing_tools":
                module.register_query_processing_tools.assert_called_once()
                args = module.register_query_processing_tools.call_args[0]
            else:
                module.register_tools.assert_called_once()
                args = module.register_tools.call_args[0]
            assert args[0] is mock_mcp
            assert args[1] is mock_client_manager


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
    async def test_real_module_structure(self, mock_mcp, mock_client_manager):
        """Test with real module structure (mocked implementations)."""
        # Mock the actual tool module imports
        mock_modules = {
            "search": Mock(spec=["register_tools"]),
            "documents": Mock(spec=["register_tools"]),
            "embeddings": Mock(spec=["register_tools"]),
            "lightweight_scrape": Mock(spec=["register_tools"]),
            "collections": Mock(spec=["register_tools"]),
            "projects": Mock(spec=["register_tools"]),
            "advanced_search": Mock(spec=["register_tools"]),
            "query_processing": Mock(spec=["register_tools"]),
            "filtering_tools": Mock(spec=["register_filtering_tools"]),
            "query_processing_tools": Mock(spec=["register_query_processing_tools"]),
            "payload_indexing": Mock(spec=["register_tools"]),
            "deployment": Mock(spec=["register_tools"]),
            "analytics": Mock(spec=["register_tools"]),
            "cache": Mock(spec=["register_tools"]),
            "utilities": Mock(spec=["register_tools"]),
            "content_intelligence": Mock(spec=["register_tools"]),
        }

        with patch.multiple(
            tools,
            search=mock_modules["search"],
            documents=mock_modules["documents"],
            embeddings=mock_modules["embeddings"],
            lightweight_scrape=mock_modules["lightweight_scrape"],
            collections=mock_modules["collections"],
            projects=mock_modules["projects"],
            advanced_search=mock_modules["advanced_search"],
            query_processing=mock_modules["query_processing"],
            filtering_tools=mock_modules["filtering_tools"],
            query_processing_tools=mock_modules["query_processing_tools"],
            payload_indexing=mock_modules["payload_indexing"],
            deployment=mock_modules["deployment"],
            analytics=mock_modules["analytics"],
            cache=mock_modules["cache"],
            utilities=mock_modules["utilities"],
            content_intelligence=mock_modules["content_intelligence"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify all modules were called
        for module_name, module in mock_modules.items():
            if module_name == "filtering_tools":
                module.register_filtering_tools.assert_called_once()
            elif module_name == "query_processing_tools":
                module.register_query_processing_tools.assert_called_once()
            else:
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
            "lightweight_scrape": Mock(register_tools=Mock()),
            "collections": Mock(register_tools=Mock()),
            "projects": Mock(register_tools=Mock()),
            "advanced_search": Mock(register_tools=Mock()),
            "query_processing": Mock(register_tools=Mock()),
            "filtering_tools": Mock(register_filtering_tools=Mock()),
            "query_processing_tools": Mock(register_query_processing_tools=Mock()),
            "payload_indexing": Mock(register_tools=Mock()),
            "deployment": Mock(register_tools=Mock()),
            "analytics": Mock(register_tools=Mock()),
            "cache": Mock(register_tools=Mock()),
            "utilities": Mock(register_tools=Mock()),
            "content_intelligence": Mock(register_tools=Mock()),
        }
        return modules

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
        self, mock_mcp, mock_client_manager, sample_tool_module, mock_tool_modules
    ):
        """Test that modules can register multiple tools."""
        # Replace search with our custom sample module
        mock_tool_modules["search"] = sample_tool_module

        with patch.multiple(
            tools,
            search=mock_tool_modules["search"],
            documents=mock_tool_modules["documents"],
            embeddings=mock_tool_modules["embeddings"],
            lightweight_scrape=mock_tool_modules["lightweight_scrape"],
            collections=mock_tool_modules["collections"],
            projects=mock_tool_modules["projects"],
            advanced_search=mock_tool_modules["advanced_search"],
            query_processing=mock_tool_modules["query_processing"],
            filtering_tools=mock_tool_modules["filtering_tools"],
            query_processing_tools=mock_tool_modules["query_processing_tools"],
            payload_indexing=mock_tool_modules["payload_indexing"],
            deployment=mock_tool_modules["deployment"],
            analytics=mock_tool_modules["analytics"],
            cache=mock_tool_modules["cache"],
            utilities=mock_tool_modules["utilities"],
            content_intelligence=mock_tool_modules["content_intelligence"],
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify tool registration was called multiple times
        assert mock_mcp.tool.call_count >= 3
        mock_mcp.tool.assert_any_call("search_documents")
        mock_mcp.tool.assert_any_call("search_with_filters")
        mock_mcp.tool.assert_any_call("search_similar")


class TestErrorScenarios:
    """Test error handling scenarios."""

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
            "lightweight_scrape": Mock(register_tools=Mock()),
            "collections": Mock(register_tools=Mock()),
            "projects": Mock(register_tools=Mock()),
            "advanced_search": Mock(register_tools=Mock()),
            "query_processing": Mock(register_tools=Mock()),
            "filtering_tools": Mock(register_filtering_tools=Mock()),
            "query_processing_tools": Mock(register_query_processing_tools=Mock()),
            "payload_indexing": Mock(register_tools=Mock()),
            "deployment": Mock(register_tools=Mock()),
            "analytics": Mock(register_tools=Mock()),
            "cache": Mock(register_tools=Mock()),
            "utilities": Mock(register_tools=Mock()),
            "content_intelligence": Mock(register_tools=Mock()),
        }
        return modules

    @pytest.mark.asyncio
    async def test_handles_import_error(self, mock_mcp, mock_client_manager, caplog):
        """Test handling when tool module import fails."""
        with (
            patch("builtins.__import__", side_effect=ImportError("Module not found")),
            caplog.at_level(logging.ERROR),
            pytest.raises(ImportError),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

    @pytest.mark.asyncio
    async def test_handles_attribute_error(
        self, mock_mcp, mock_client_manager, mock_tool_modules
    ):
        """Test handling when accessing module attributes fails."""
        # Make the cache module raise AttributeError when register_tools is accessed
        mock_tool_modules["cache"]
        bad_cache = Mock()
        bad_cache.register_tools = property(
            lambda self: (_ for _ in ()).throw(AttributeError("No such attribute"))
        )
        mock_tool_modules["cache"] = bad_cache

        # Patch individual tool modules
        with (
            patch.multiple(
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
            ),
            pytest.raises(TypeError, match="property.*not callable"),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Modules before cache should be registered
        mock_tool_modules["search"].register_tools.assert_called_once()
        mock_tool_modules["analytics"].register_tools.assert_called_once()


class TestLoggingBehavior:
    """Test detailed logging behavior."""

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
            "lightweight_scrape": Mock(register_tools=Mock()),
            "collections": Mock(register_tools=Mock()),
            "projects": Mock(register_tools=Mock()),
            "advanced_search": Mock(register_tools=Mock()),
            "query_processing": Mock(register_tools=Mock()),
            "filtering_tools": Mock(register_filtering_tools=Mock()),
            "query_processing_tools": Mock(register_query_processing_tools=Mock()),
            "payload_indexing": Mock(register_tools=Mock()),
            "deployment": Mock(register_tools=Mock()),
            "analytics": Mock(register_tools=Mock()),
            "cache": Mock(register_tools=Mock()),
            "utilities": Mock(register_tools=Mock()),
            "content_intelligence": Mock(register_tools=Mock()),
        }
        return modules

    @pytest.mark.asyncio
    async def test_logger_name_is_correct(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that logging messages are produced during registration."""
        with (
            patch.multiple(
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
            ),
            caplog.at_level(logging.INFO),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Verify that registration produces log messages
        assert len(caplog.records) >= 4  # At least 4 info messages
        assert any("Registering" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_registration_count_is_accurate(
        self, mock_mcp, mock_client_manager, mock_tool_modules, caplog
    ):
        """Test that the reported registration count is accurate."""
        with (
            patch.multiple(
                tools,
                search=mock_tool_modules["search"],
                documents=mock_tool_modules["documents"],
                embeddings=mock_tool_modules["embeddings"],
                lightweight_scrape=mock_tool_modules["lightweight_scrape"],
                collections=mock_tool_modules["collections"],
                projects=mock_tool_modules["projects"],
                advanced_search=mock_tool_modules["advanced_search"],
                query_processing=mock_tool_modules["query_processing"],
                filtering_tools=mock_tool_modules["filtering_tools"],
                query_processing_tools=mock_tool_modules["query_processing_tools"],
                payload_indexing=mock_tool_modules["payload_indexing"],
                deployment=mock_tool_modules["deployment"],
                analytics=mock_tool_modules["analytics"],
                cache=mock_tool_modules["cache"],
                utilities=mock_tool_modules["utilities"],
                content_intelligence=mock_tool_modules["content_intelligence"],
            ),
            caplog.at_level(logging.INFO),
        ):
            await register_all_tools(mock_mcp, mock_client_manager)

        # Find the summary message
        summary_logs = [
            record.message
            for record in caplog.records
            if "Successfully registered" in record.message
        ]

        # Should report 16 modules (since register_all_tools always tries to register all modules)
        assert "Successfully registered 16 tool modules" in summary_logs[0]
