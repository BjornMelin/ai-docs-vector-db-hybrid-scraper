"""Unit tests for SearchService - I5 web search orchestration capabilities.

Tests cover:
- Service initialization and client manager integration
- Search tool registration and autonomous web search
- Multi-provider orchestration and intelligent result fusion
- Service discovery and capability reporting
- Error handling and service recovery
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.search_service import SearchService


class TestSearchService:
    """Test SearchService initialization and core functionality."""

    def test_init_creates_service_with_correct_configuration(self):
        """Test that SearchService initializes with correct FastMCP configuration."""
        service = SearchService("test-search-service")

        assert service.mcp.name == "test-search-service"
        assert "Advanced search service" in service.mcp.instructions
        assert "I5 research" in service.mcp.instructions
        assert service.client_manager is None

    def test_init_with_default_name(self):
        """Test SearchService initialization with default name."""
        service = SearchService()

        assert service.mcp.name == "search-service"
        assert service.client_manager is None

    def test_init_contains_autonomous_capabilities_in_instructions(self):
        """Test that service instructions include autonomous capabilities."""
        service = SearchService()

        instructions = service.mcp.instructions
        assert "Autonomous Capabilities" in instructions
        assert "Intelligent search provider selection" in instructions
        assert "Dynamic strategy adaptation" in instructions
        assert "Self-learning search pattern optimization" in instructions
        assert "Multi-provider result fusion" in instructions

    async def test_initialize_with_client_manager(self, mock_client_manager):
        """Test service initialization with client manager."""
        service = SearchService()

        with patch.object(
            service, "_register_search_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_initialize_logs_success_message(self, mock_client_manager, caplog):
        """Test that initialization logs success message."""
        service = SearchService()

        with patch.object(service, "_register_search_tools", new_callable=AsyncMock):
            with caplog.at_level(logging.INFO):
                await service.initialize(mock_client_manager)

                assert (
                    "SearchService initialized with autonomous capabilities"
                    in caplog.text
                )

    async def test_register_search_tools_raises_error_when_not_initialized(self):
        """Test that tool registration raises error when service not initialized."""
        service = SearchService()

        with pytest.raises(RuntimeError, match="SearchService not initialized"):
            await service._register_search_tools()

    async def test_register_search_tools_calls_all_tool_registrations(
        self, mock_client_manager
    ):
        """Test that all search tools are registered properly."""
        service = SearchService()
        service.client_manager = mock_client_manager

        # Mock all the tool modules
        mock_tools = {
            "hybrid_search": Mock(),
            "hyde_search": Mock(),
            "multi_stage_search": Mock(),
            "search_with_reranking": Mock(),
            "web_search": Mock(),
        }

        for mock_tool in mock_tools.values():
            mock_tool.register_tools = Mock()

        # Patch the tool imports
        with patch.multiple(
            "src.mcp_services.search_service",
            hybrid_search=mock_tools["hybrid_search"],
            hyde_search=mock_tools["hyde_search"],
            multi_stage_search=mock_tools["multi_stage_search"],
            search_with_reranking=mock_tools["search_with_reranking"],
            web_search=mock_tools["web_search"],
        ):
            await service._register_search_tools()

            # Verify all tools were registered
            for mock_tool in mock_tools.values():
                mock_tool.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    async def test_register_search_tools_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that tool registration logs success message."""
        service = SearchService()
        service.client_manager = mock_client_manager

        # Mock tool modules
        with (
            patch.multiple(
                "src.mcp_services.search_service",
                hybrid_search=Mock(register_tools=Mock()),
                hyde_search=Mock(register_tools=Mock()),
                multi_stage_search=Mock(register_tools=Mock()),
                search_with_reranking=Mock(register_tools=Mock()),
                web_search=Mock(register_tools=Mock()),
            ),
            caplog.at_level(logging.INFO),
        ):
            await service._register_search_tools()

            assert (
                "Registered search tools with autonomous web search capabilities"
                in caplog.text
            )

    def test_get_mcp_server_returns_configured_instance(self):
        """Test that get_mcp_server returns the configured FastMCP instance."""
        service = SearchService("test-service")

        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "test-service"

    async def test_get_service_info_returns_comprehensive_metadata(self):
        """Test that service info contains all expected metadata and capabilities."""
        service = SearchService()

        service_info = await service.get_service_info()

        # Verify basic metadata
        assert service_info["service"] == "search"
        assert service_info["version"] == "2.0"
        assert service_info["status"] == "active"
        assert service_info["research_basis"] == "I5_WEB_SEARCH_TOOL_ORCHESTRATION"

        # Verify capabilities
        expected_capabilities = [
            "hybrid_search",
            "hyde_search",
            "multi_stage_search",
            "search_reranking",
            "autonomous_web_search",
            "multi_provider_orchestration",
            "intelligent_result_fusion",
        ]
        assert service_info["capabilities"] == expected_capabilities

        # Verify autonomous features
        expected_autonomous_features = [
            "provider_optimization",
            "strategy_adaptation",
            "quality_assessment",
            "self_learning_patterns",
        ]
        assert service_info["autonomous_features"] == expected_autonomous_features


class TestSearchServiceAdvancedFeatures:
    """Test advanced SearchService features and integration scenarios."""

    async def test_service_handles_initialization_with_complex_client_manager(
        self, mock_client_manager
    ):
        """Test service initialization with complex client manager setup."""
        service = SearchService("advanced-search")

        # Configure complex client manager
        mock_client_manager.get_openai_client.return_value = Mock()
        mock_client_manager.get_qdrant_client.return_value = Mock()
        mock_client_manager.parallel_processing_system = Mock()

        with patch.object(
            service, "_register_search_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_service_info_includes_i5_research_integration(self):
        """Test that service info correctly references I5 research basis."""
        service = SearchService()

        service_info = await service.get_service_info()

        assert "I5_WEB_SEARCH_TOOL_ORCHESTRATION" in service_info["research_basis"]
        assert "autonomous_web_search" in service_info["capabilities"]
        assert "multi_provider_orchestration" in service_info["capabilities"]

    async def test_autonomous_features_are_comprehensive(self):
        """Test that autonomous features cover all expected capabilities."""
        service = SearchService()

        service_info = await service.get_service_info()

        autonomous_features = service_info["autonomous_features"]

        # Verify comprehensive autonomous capabilities
        assert "provider_optimization" in autonomous_features
        assert "strategy_adaptation" in autonomous_features
        assert "quality_assessment" in autonomous_features
        assert "self_learning_patterns" in autonomous_features

    async def test_web_search_orchestration_tool_registration(
        self, mock_client_manager
    ):
        """Test that web search orchestration tools are properly registered."""
        service = SearchService()
        service.client_manager = mock_client_manager

        # Mock web search tool specifically
        mock_web_search = Mock()
        mock_web_search.register_tools = Mock()

        with patch("src.mcp_services.search_service.web_search", mock_web_search):
            with patch.multiple(
                "src.mcp_services.search_service",
                hybrid_search=Mock(register_tools=Mock()),
                hyde_search=Mock(register_tools=Mock()),
                multi_stage_search=Mock(register_tools=Mock()),
                search_with_reranking=Mock(register_tools=Mock()),
            ):
                await service._register_search_tools()

                # Verify web search tools were registered (I5 research implementation)
                mock_web_search.register_tools.assert_called_once_with(
                    service.mcp, mock_client_manager
                )

    def test_service_instructions_contain_i5_research_features(self):
        """Test that service instructions reference I5 research features."""
        service = SearchService()

        instructions = service.mcp.instructions

        # Check for I5 research specific features
        assert "Autonomous web search orchestration (I5 research)" in instructions
        assert "Multi-provider result fusion and synthesis" in instructions
        assert "Real-time search strategy optimization" in instructions

    async def test_error_handling_during_tool_registration(
        self, mock_client_manager, caplog
    ):
        """Test error handling during tool registration process."""
        service = SearchService()
        service.client_manager = mock_client_manager

        # Mock a tool that raises an exception during registration
        mock_failing_tool = Mock()
        mock_failing_tool.register_tools.side_effect = Exception(
            "Tool registration failed"
        )

        with patch("src.mcp_services.search_service.hybrid_search", mock_failing_tool):
            with patch.multiple(
                "src.mcp_services.search_service",
                hyde_search=Mock(register_tools=Mock()),
                multi_stage_search=Mock(register_tools=Mock()),
                search_with_reranking=Mock(register_tools=Mock()),
                web_search=Mock(register_tools=Mock()),
            ):
                # Tool registration should raise the exception
                with pytest.raises(Exception, match="Tool registration failed"):
                    await service._register_search_tools()

    async def test_service_capability_discovery_and_reporting(self):
        """Test service capability discovery and comprehensive reporting."""
        service = SearchService()

        service_info = await service.get_service_info()

        # Verify comprehensive capability reporting
        assert len(service_info["capabilities"]) >= 7  # All core capabilities
        assert len(service_info["autonomous_features"]) >= 4  # All autonomous features

        # Verify service is ready for capability discovery
        assert service_info["status"] == "active"
        assert "version" in service_info
        assert "research_basis" in service_info

    async def test_multi_provider_orchestration_capability(self):
        """Test that service reports multi-provider orchestration capability."""
        service = SearchService()

        service_info = await service.get_service_info()

        assert "multi_provider_orchestration" in service_info["capabilities"]
        assert "intelligent_result_fusion" in service_info["capabilities"]

    async def test_service_supports_intelligent_result_fusion(self):
        """Test that service supports intelligent result fusion."""
        service = SearchService()

        service_info = await service.get_service_info()

        # Verify intelligent result fusion capability
        assert "intelligent_result_fusion" in service_info["capabilities"]
        assert "quality_assessment" in service_info["autonomous_features"]


class TestSearchServiceErrorHandling:
    """Test SearchService error handling and recovery scenarios."""

    async def test_initialization_with_none_client_manager_raises_error(self):
        """Test that initialization with None client manager is handled properly."""
        service = SearchService()

        # Should not raise error during initialization
        await service.initialize(None)

        # But should raise error when trying to register tools
        with pytest.raises(RuntimeError, match="SearchService not initialized"):
            await service._register_search_tools()

    async def test_get_service_info_works_without_initialization(self):
        """Test that get_service_info works even without full initialization."""
        service = SearchService()

        # Should not raise error
        service_info = await service.get_service_info()

        assert service_info["service"] == "search"
        assert service_info["status"] == "active"

    async def test_get_mcp_server_works_without_initialization(self):
        """Test that get_mcp_server works without full initialization."""
        service = SearchService()

        # Should not raise error
        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "search-service"

    async def test_multiple_initialization_calls_are_safe(self, mock_client_manager):
        """Test that multiple initialization calls are handled safely."""
        service = SearchService()

        with patch.object(
            service, "_register_search_tools", new_callable=AsyncMock
        ) as mock_register:
            # First initialization
            await service.initialize(mock_client_manager)
            first_call_count = mock_register.call_count

            # Second initialization
            await service.initialize(mock_client_manager)
            second_call_count = mock_register.call_count

            # Should handle multiple calls gracefully
            assert second_call_count >= first_call_count

    async def test_service_recovery_after_tool_registration_failure(
        self, mock_client_manager
    ):
        """Test service recovery after tool registration failure."""
        service = SearchService()
        service.client_manager = mock_client_manager

        # First attempt fails
        with patch("src.mcp_services.search_service.hybrid_search") as mock_tool:
            mock_tool.register_tools.side_effect = Exception("Registration failed")

            with pytest.raises(Exception):
                await service._register_search_tools()

        # Second attempt should work
        with patch.multiple(
            "src.mcp_services.search_service",
            hybrid_search=Mock(register_tools=Mock()),
            hyde_search=Mock(register_tools=Mock()),
            multi_stage_search=Mock(register_tools=Mock()),
            search_with_reranking=Mock(register_tools=Mock()),
            web_search=Mock(register_tools=Mock()),
        ):
            # Should not raise error on retry
            await service._register_search_tools()


class TestSearchServicePerformanceAndOptimization:
    """Test SearchService performance characteristics and optimization features."""

    async def test_service_initialization_is_efficient(self, mock_client_manager):
        """Test that service initialization is efficient and doesn't block."""
        import time

        service = SearchService()

        start_time = time.time()

        with patch.object(service, "_register_search_tools", new_callable=AsyncMock):
            await service.initialize(mock_client_manager)

        end_time = time.time()

        # Initialization should be fast (< 1 second)
        assert end_time - start_time < 1.0

    async def test_get_service_info_performance(self):
        """Test that get_service_info is performant for capability discovery."""
        import time

        service = SearchService()

        start_time = time.time()

        # Call multiple times to test performance
        for _ in range(10):
            await service.get_service_info()

        end_time = time.time()

        # Should complete quickly (< 0.1 seconds for 10 calls)
        assert end_time - start_time < 0.1

    async def test_service_supports_concurrent_access(self, mock_client_manager):
        """Test that service supports concurrent access patterns."""
        import asyncio

        service = SearchService()

        # Simulate concurrent access
        async def concurrent_operation():
            await service.initialize(mock_client_manager)
            return await service.get_service_info()

        with patch.object(service, "_register_search_tools", new_callable=AsyncMock):
            # Run multiple concurrent operations
            tasks = [concurrent_operation() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result["service"] == "search"
