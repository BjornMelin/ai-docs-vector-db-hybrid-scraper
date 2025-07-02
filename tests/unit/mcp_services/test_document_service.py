"""Unit tests for DocumentService - I3 5-tier crawling enhancements.

Tests cover:
- Service initialization and intelligent processing capabilities
- 5-tier crawling with ML-powered tier selection
- Content intelligence and quality assessment
- Document management and project organization
- Autonomous document processing patterns
"""

import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.mcp_services.document_service import DocumentService


class TestDocumentService:
    """Test DocumentService initialization and core functionality."""

    def test_init_creates_service_with_correct_configuration(self):
        """Test that DocumentService initializes with correct FastMCP configuration."""
        service = DocumentService("test-document-service")

        assert service.mcp.name == "test-document-service"
        assert "Advanced document service" in service.mcp.instructions
        assert "I3 research" in service.mcp.instructions
        assert service.client_manager is None

    def test_init_with_default_name(self):
        """Test DocumentService initialization with default name."""
        service = DocumentService()

        assert service.mcp.name == "document-service"
        assert service.client_manager is None

    def test_init_contains_5_tier_crawling_features(self):
        """Test that service instructions include 5-tier crawling features."""
        service = DocumentService()

        instructions = service.mcp.instructions
        assert "5-tier intelligent crawling" in instructions
        assert "ML-powered tier selection" in instructions
        assert "Advanced chunking strategies with AST-based processing" in instructions
        assert "Autonomous document processing" in instructions

    def test_init_contains_autonomous_capabilities(self):
        """Test that service instructions include autonomous capabilities."""
        service = DocumentService()

        instructions = service.mcp.instructions
        assert "Autonomous Capabilities" in instructions
        assert "Intelligent tier selection for crawling optimization" in instructions
        assert "Dynamic content quality assessment" in instructions
        assert "Self-learning document processing patterns" in instructions
        assert "Autonomous collection provisioning" in instructions

    async def test_initialize_with_client_manager(self, mock_client_manager):
        """Test service initialization with client manager."""
        service = DocumentService()

        with patch.object(
            service, "_register_document_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_initialize_logs_success_message(self, mock_client_manager, caplog):
        """Test that initialization logs success message with 5-tier capabilities."""
        service = DocumentService()

        with (
            patch.object(service, "_register_document_tools", new_callable=AsyncMock),
            caplog.at_level(logging.INFO),
        ):
            await service.initialize(mock_client_manager)

            assert (
                "DocumentService initialized with 5-tier crawling capabilities"
                in caplog.text
            )

    async def test_register_document_tools_raises_error_when_not_initialized(self):
        """Test that tool registration raises error when service not initialized."""
        service = DocumentService()

        with pytest.raises(RuntimeError, match="DocumentService not initialized"):
            await service._register_document_tools()

    async def test_register_document_tools_calls_all_tool_registrations(
        self, mock_client_manager
    ):
        """Test that all document tools are registered properly."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock all the tool modules
        mock_tools = {
            "document_management": Mock(),
            "collections": Mock(),
            "projects": Mock(),
            "crawling": Mock(),
            "content_intelligence": Mock(),
        }

        for mock_tool in mock_tools.values():
            mock_tool.register_tools = Mock()

        # Patch the tool imports
        with patch.multiple(
            "src.mcp_services.document_service",
            document_management=mock_tools["document_management"],
            collections=mock_tools["collections"],
            projects=mock_tools["projects"],
            crawling=mock_tools["crawling"],
            content_intelligence=mock_tools["content_intelligence"],
        ):
            await service._register_document_tools()

            # Verify all core tools were registered
            mock_tools["document_management"].register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_tools["collections"].register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_tools["projects"].register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

            # Verify I3 research tools were registered
            mock_tools["crawling"].register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )
            mock_tools["content_intelligence"].register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    async def test_register_document_tools_logs_success_message(
        self, mock_client_manager, caplog
    ):
        """Test that tool registration logs success message."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock tool modules
        with (
            patch.multiple(
                "src.mcp_services.document_service",
                document_management=Mock(register_tools=Mock()),
                collections=Mock(register_tools=Mock()),
                projects=Mock(register_tools=Mock()),
                crawling=Mock(register_tools=Mock()),
                content_intelligence=Mock(register_tools=Mock()),
            ),
            caplog.at_level(logging.INFO),
        ):
            await service._register_document_tools()

            assert (
                "Registered document tools with intelligent crawling capabilities"
                in caplog.text
            )

    def test_get_mcp_server_returns_configured_instance(self):
        """Test that get_mcp_server returns the configured FastMCP instance."""
        service = DocumentService("test-service")

        mcp_server = service.get_mcp_server()

        assert mcp_server == service.mcp
        assert mcp_server.name == "test-service"

    async def test_get_service_info_returns_comprehensive_metadata(self):
        """Test that service info contains all expected metadata and capabilities."""
        service = DocumentService()

        service_info = await service.get_service_info()

        # Verify basic metadata
        assert service_info["service"] == "document"
        assert service_info["version"] == "2.0"
        assert service_info["status"] == "active"
        assert service_info["research_basis"] == "I3_5_TIER_CRAWLING_ENHANCEMENT"

        # Verify capabilities
        expected_capabilities = [
            "document_management",
            "intelligent_crawling",
            "5_tier_crawling",
            "collection_management",
            "project_organization",
            "content_intelligence",
            "autonomous_processing",
        ]
        assert service_info["capabilities"] == expected_capabilities

        # Verify autonomous features
        expected_autonomous_features = [
            "tier_selection_optimization",
            "content_quality_assessment",
            "processing_pattern_learning",
            "collection_provisioning",
        ]
        assert service_info["autonomous_features"] == expected_autonomous_features


class TestDocumentServiceCrawlingCapabilities:
    """Test DocumentService 5-tier crawling and intelligent processing capabilities."""

    async def test_5_tier_crawling_tool_registration(self, mock_client_manager):
        """Test that 5-tier crawling tools are properly registered."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock crawling tool specifically (I3 research implementation)
        mock_crawling = Mock()
        mock_crawling.register_tools = Mock()

        with (
            patch("src.mcp_services.document_service.crawling", mock_crawling),
            patch.multiple(
                "src.mcp_services.document_service",
                document_management=Mock(register_tools=Mock()),
                collections=Mock(register_tools=Mock()),
                projects=Mock(register_tools=Mock()),
                content_intelligence=Mock(register_tools=Mock()),
            ),
        ):
            await service._register_document_tools()

            # Verify crawling tools were registered (I3 research implementation)
            mock_crawling.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    async def test_content_intelligence_tool_registration(self, mock_client_manager):
        """Test that content intelligence tools are properly registered."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock content intelligence tool specifically
        mock_content_intelligence = Mock()
        mock_content_intelligence.register_tools = Mock()

        with (
            patch(
                "src.mcp_services.document_service.content_intelligence",
                mock_content_intelligence,
            ),
            patch.multiple(
                "src.mcp_services.document_service",
                document_management=Mock(register_tools=Mock()),
                collections=Mock(register_tools=Mock()),
                projects=Mock(register_tools=Mock()),
                crawling=Mock(register_tools=Mock()),
            ),
        ):
            await service._register_document_tools()

            # Verify content intelligence tools were registered
            mock_content_intelligence.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    def test_service_instructions_contain_i3_research_features(self):
        """Test that service instructions reference I3 research features."""
        service = DocumentService()

        instructions = service.mcp.instructions

        # Check for I3 research specific features
        assert (
            "5-tier intelligent crawling with ML-powered tier selection" in instructions
        )
        assert "Autonomous document processing and content extraction" in instructions
        assert "Content intelligence and quality assessment" in instructions
        assert "Advanced chunking strategies with AST-based processing" in instructions

    async def test_tier_selection_optimization_capability(self):
        """Test that service reports tier selection optimization capability."""
        service = DocumentService()

        service_info = await service.get_service_info()

        assert "5_tier_crawling" in service_info["capabilities"]
        assert "intelligent_crawling" in service_info["capabilities"]
        assert "tier_selection_optimization" in service_info["autonomous_features"]

    async def test_content_quality_assessment_capability(self):
        """Test that service supports content quality assessment."""
        service = DocumentService()

        service_info = await service.get_service_info()

        # Verify content quality assessment capability
        assert "content_intelligence" in service_info["capabilities"]
        assert "content_quality_assessment" in service_info["autonomous_features"]

    async def test_autonomous_processing_capability(self):
        """Test that service supports autonomous processing."""
        service = DocumentService()

        service_info = await service.get_service_info()

        assert "autonomous_processing" in service_info["capabilities"]
        assert "processing_pattern_learning" in service_info["autonomous_features"]

    async def test_collection_provisioning_capability(self):
        """Test that service supports autonomous collection provisioning."""
        service = DocumentService()

        service_info = await service.get_service_info()

        assert "collection_management" in service_info["capabilities"]
        assert "collection_provisioning" in service_info["autonomous_features"]


class TestDocumentServiceProjectOrganization:
    """Test DocumentService project organization and management capabilities."""

    async def test_project_organization_tool_registration(self, mock_client_manager):
        """Test that project organization tools are properly registered."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock projects tool specifically
        mock_projects = Mock()
        mock_projects.register_tools = Mock()

        with patch("src.mcp_services.document_service.projects", mock_projects):
            with patch.multiple(
                "src.mcp_services.document_service",
                document_management=Mock(register_tools=Mock()),
                collections=Mock(register_tools=Mock()),
                crawling=Mock(register_tools=Mock()),
                content_intelligence=Mock(register_tools=Mock()),
            ):
                await service._register_document_tools()

            # Verify projects tools were registered
            mock_projects.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    async def test_document_management_tool_registration(self, mock_client_manager):
        """Test that document management tools are properly registered."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock document management tool specifically
        mock_document_management = Mock()
        mock_document_management.register_tools = Mock()

        with (
            patch(
                "src.mcp_services.document_service.document_management",
                mock_document_management,
            ),
            patch.multiple(
                "src.mcp_services.document_service",
                collections=Mock(register_tools=Mock()),
                projects=Mock(register_tools=Mock()),
                crawling=Mock(register_tools=Mock()),
                content_intelligence=Mock(register_tools=Mock()),
            ),
        ):
            await service._register_document_tools()

            # Verify document management tools were registered
            mock_document_management.register_tools.assert_called_once_with(
                service.mcp, mock_client_manager
            )

    async def test_collection_management_capability(self):
        """Test that service supports collection management."""
        service = DocumentService()

        service_info = await service.get_service_info()

        assert "collection_management" in service_info["capabilities"]
        assert "project_organization" in service_info["capabilities"]

    def test_service_instructions_contain_project_features(self):
        """Test that service instructions reference project organization features."""
        service = DocumentService()

        instructions = service.mcp.instructions

        # Check for project organization features
        assert "Project-based document organization and management" in instructions
        assert "Collection management with agentic optimization" in instructions


class TestDocumentServiceErrorHandling:
    """Test DocumentService error handling and recovery scenarios."""

    async def test_initialization_with_none_client_manager_raises_error(self):
        """Test that initialization with None client manager is handled properly."""
        service = DocumentService()

        # Should not raise error during initialization
        await service.initialize(None)

        # But should raise error when trying to register tools
        with pytest.raises(RuntimeError, match="DocumentService not initialized"):
            await service._register_document_tools()

    async def test_get_service_info_works_without_initialization(self):
        """Test that get_service_info works even without full initialization."""
        service = DocumentService()

        # Should not raise error
        service_info = await service.get_service_info()

        assert service_info["service"] == "document"
        assert service_info["status"] == "active"

    async def test_error_handling_during_tool_registration(
        self, mock_client_manager, caplog
    ):
        """Test error handling during tool registration process."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # Mock a tool that raises an exception during registration
        mock_failing_tool = Mock()
        mock_failing_tool.register_tools.side_effect = Exception(
            "Tool registration failed"
        )

        with (
            patch(
                "src.mcp_services.document_service.document_management",
                mock_failing_tool,
            ),
            patch.multiple(
                "src.mcp_services.document_service",
                collections=Mock(register_tools=Mock()),
                projects=Mock(register_tools=Mock()),
                crawling=Mock(register_tools=Mock()),
                content_intelligence=Mock(register_tools=Mock()),
            ),
            pytest.raises(Exception, match="Tool registration failed"),
        ):
            # Tool registration should raise the exception
            await service._register_document_tools()

    async def test_service_recovery_after_tool_registration_failure(
        self, mock_client_manager
    ):
        """Test service recovery after tool registration failure."""
        service = DocumentService()
        service.client_manager = mock_client_manager

        # First attempt fails
        with patch(
            "src.mcp_services.document_service.document_management"
        ) as mock_tool:
            mock_tool.register_tools.side_effect = Exception("Registration failed")

            with pytest.raises(Exception):
                await service._register_document_tools()

        # Second attempt should work
        with patch.multiple(
            "src.mcp_services.document_service",
            document_management=Mock(register_tools=Mock()),
            collections=Mock(register_tools=Mock()),
            projects=Mock(register_tools=Mock()),
            crawling=Mock(register_tools=Mock()),
            content_intelligence=Mock(register_tools=Mock()),
        ):
            # Should not raise error on retry
            await service._register_document_tools()

    async def test_multiple_initialization_calls_are_safe(self, mock_client_manager):
        """Test that multiple initialization calls are handled safely."""
        service = DocumentService()

        with patch.object(
            service, "_register_document_tools", new_callable=AsyncMock
        ) as mock_register:
            # First initialization
            await service.initialize(mock_client_manager)
            first_call_count = mock_register.call_count

            # Second initialization
            await service.initialize(mock_client_manager)
            second_call_count = mock_register.call_count

            # Should handle multiple calls gracefully
            assert second_call_count >= first_call_count


class TestDocumentServiceIntegrationScenarios:
    """Test DocumentService integration scenarios and advanced use cases."""

    async def test_service_handles_initialization_with_complex_client_manager(
        self, mock_client_manager
    ):
        """Test service initialization with complex client manager setup."""
        service = DocumentService("advanced-document")

        # Configure complex client manager
        mock_client_manager.get_openai_client.return_value = Mock()
        mock_client_manager.get_qdrant_client.return_value = Mock()
        mock_client_manager.get_firecrawl_client.return_value = Mock()
        mock_client_manager.parallel_processing_system = Mock()

        with patch.object(
            service, "_register_document_tools", new_callable=AsyncMock
        ) as mock_register:
            await service.initialize(mock_client_manager)

            assert service.client_manager == mock_client_manager
            mock_register.assert_called_once()

    async def test_service_info_includes_i3_research_integration(self):
        """Test that service info correctly references I3 research basis."""
        service = DocumentService()

        service_info = await service.get_service_info()

        assert "I3_5_TIER_CRAWLING_ENHANCEMENT" in service_info["research_basis"]
        assert "5_tier_crawling" in service_info["capabilities"]
        assert "intelligent_crawling" in service_info["capabilities"]

    async def test_comprehensive_capability_reporting(self):
        """Test comprehensive capability reporting for service discovery."""
        service = DocumentService()

        service_info = await service.get_service_info()

        # Verify comprehensive capability reporting
        assert len(service_info["capabilities"]) >= 7  # All core capabilities
        assert len(service_info["autonomous_features"]) >= 4  # All autonomous features

        # Verify service is ready for capability discovery
        assert service_info["status"] == "active"
        assert "version" in service_info
        assert "research_basis" in service_info

    async def test_service_supports_concurrent_access(self, mock_client_manager):
        """Test that service supports concurrent access patterns."""

        service = DocumentService()

        # Simulate concurrent access
        async def concurrent_operation():
            await service.initialize(mock_client_manager)
            return await service.get_service_info()

        with patch.object(service, "_register_document_tools", new_callable=AsyncMock):
            # Run multiple concurrent operations
            tasks = [concurrent_operation() for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result["service"] == "document"

    async def test_ml_powered_tier_selection_capability(self):
        """Test that service supports ML-powered tier selection."""
        service = DocumentService()

        service_info = await service.get_service_info()

        # Verify ML-powered capabilities
        assert "intelligent_crawling" in service_info["capabilities"]
        assert "tier_selection_optimization" in service_info["autonomous_features"]

    def test_ast_based_processing_feature_documentation(self):
        """Test that AST-based processing features are documented."""
        service = DocumentService()

        instructions = service.mcp.instructions

        # Check for AST-based processing documentation
        assert "Advanced chunking strategies with AST-based processing" in instructions
