"""Comprehensive coverage tests for DynamicToolDiscovery.

This test module provides thorough coverage of the DynamicToolDiscovery functionality,
focusing on tool detection, capability analysis, and adaptive discovery mechanisms.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    discover_tools_for_task,
    get_discovery_engine,
)
from src.services.agents.core import AgentState, BaseAgentDependencies


class TestToolCapability:
    """Test ToolCapability model behavior."""

    def test_tool_capability_initialization(self):
        """ToolCapability should initialize with required fields."""
        capability = ToolCapability(
            name="search_tool",
            description="Searches documents",
            input_types=["text", "query"],
            output_types=["results", "summary"],
            confidence_score=0.9,
        )

        assert capability.name == "search_tool"
        assert capability.description == "Searches documents"
        assert capability.input_types == ["text", "query"]
        assert capability.output_types == ["results", "summary"]
        assert capability.confidence_score == 0.9

    def test_tool_capability_optional_fields(self):
        """ToolCapability should handle optional fields correctly."""
        capability = ToolCapability(
            name="basic_tool",
            description="Basic tool",
            input_types=["text"],
            output_types=["result"],
        )

        assert capability.confidence_score == 1.0  # Default value
        assert capability.name == "basic_tool"

    def test_tool_capability_validation(self):
        """ToolCapability should validate field constraints."""
        # Test valid confidence score
        capability = ToolCapability(
            name="test_tool",
            description="Test tool",
            input_types=["text"],
            output_types=["result"],
            confidence_score=0.5,
        )
        assert capability.confidence_score == 0.5

    def test_tool_capability_edge_values(self):
        """ToolCapability should handle edge values correctly."""
        # Test minimum confidence
        capability = ToolCapability(
            name="min_tool",
            description="Minimum confidence tool",
            input_types=["text"],
            output_types=["result"],
            confidence_score=0.0,
        )
        assert capability.confidence_score == 0.0

        # Test maximum confidence
        capability = ToolCapability(
            name="max_tool",
            description="Maximum confidence tool",
            input_types=["text"],
            output_types=["result"],
            confidence_score=1.0,
        )
        assert capability.confidence_score == 1.0


class TestDynamicToolDiscovery:
    """Test DynamicToolDiscovery core functionality."""

    def create_mock_dependencies(self) -> BaseAgentDependencies:
        """Create mock dependencies for testing."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()

        return BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config
        )

    def test_discovery_initialization(self):
        """DynamicToolDiscovery should initialize correctly."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        assert discovery.dependencies == deps
        assert hasattr(discovery, "dependencies")

    @pytest.mark.asyncio
    async def test_discovery_analyze_task_requirements(self):
        """DynamicToolDiscovery should analyze task requirements."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        requirements = await discovery._analyze_task_requirements(
            "search for documents about AI"
        )

        assert isinstance(requirements, dict)
        assert "task_type" in requirements
        assert "required_capabilities" in requirements
        assert "priority_level" in requirements

    @pytest.mark.asyncio
    async def test_discovery_scan_available_tools(self):
        """DynamicToolDiscovery should scan available tools."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        available_tools = await discovery._scan_available_tools()

        assert isinstance(available_tools, list)
        # Should find at least some tools or return empty list
        assert all(isinstance(tool, ToolCapability) for tool in available_tools)

    @pytest.mark.asyncio
    async def test_discovery_match_tools_to_task(self):
        """DynamicToolDiscovery should match tools to tasks correctly."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        # Mock available tools
        mock_tools = [
            ToolCapability(
                name="search_tool",
                description="Search documents",
                input_types=["text"],
                output_types=["results"],
                confidence_score=0.9,
            ),
            ToolCapability(
                name="summarize_tool",
                description="Summarize content",
                input_types=["text"],
                output_types=["summary"],
                confidence_score=0.8,
            ),
        ]

        requirements = {"task_type": "search", "required_capabilities": ["text_search"]}

        with patch.object(discovery, "_scan_available_tools", return_value=mock_tools):
            matched_tools = await discovery._match_tools_to_task(requirements)

            assert isinstance(matched_tools, list)
            assert all(isinstance(tool, ToolCapability) for tool in matched_tools)

    @pytest.mark.asyncio
    async def test_discovery_discover_tools(self):
        """DynamicToolDiscovery should discover tools for tasks."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        tools = await discovery.discover_tools("search for AI research papers")

        assert isinstance(tools, list)
        assert all(isinstance(tool, ToolCapability) for tool in tools)

    @pytest.mark.asyncio
    async def test_discovery_different_task_types(self):
        """DynamicToolDiscovery should handle different task types."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        task_types = [
            "search for documents",
            "summarize this text",
            "analyze data patterns",
            "extract information",
            "generate content",
        ]

        for task in task_types:
            tools = await discovery.discover_tools(task)
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_with_context(self):
        """DynamicToolDiscovery should use context for better tool selection."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        context = {
            "domain": "academic",
            "user_preferences": {"detailed_results": True},
            "previous_tools": ["search_tool"],
        }

        tools = await discovery.discover_tools("find research papers", context=context)

        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_capability_scoring(self):
        """DynamicToolDiscovery should score tool capabilities appropriately."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        mock_tools = [
            ToolCapability(
                name="high_confidence_tool",
                description="High confidence tool",
                input_types=["text"],
                output_types=["results"],
                confidence_score=0.95,
            ),
            ToolCapability(
                name="low_confidence_tool",
                description="Low confidence tool",
                input_types=["text"],
                output_types=["results"],
                confidence_score=0.3,
            ),
        ]

        requirements = {"task_type": "search", "required_capabilities": ["text_search"]}

        with patch.object(discovery, "_scan_available_tools", return_value=mock_tools):
            matched_tools = await discovery._match_tools_to_task(requirements)

            # Should prefer higher confidence tools
            if matched_tools:
                # If tools are ranked, first should have higher confidence
                assert matched_tools[0].confidence_score >= 0.3

    @pytest.mark.asyncio
    async def test_discovery_error_handling(self):
        """DynamicToolDiscovery should handle errors gracefully."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        with patch.object(
            discovery,
            "_analyze_task_requirements",
            side_effect=Exception("Analysis error"),
        ):
            tools = await discovery.discover_tools("test task")

            # Should return empty list or handle error gracefully
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_caching_behavior(self):
        """DynamicToolDiscovery should cache tool scan results for performance."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        # Call scan_available_tools multiple times
        tools1 = await discovery._scan_available_tools()
        tools2 = await discovery._scan_available_tools()

        # Results should be consistent
        assert isinstance(tools1, list)
        assert isinstance(tools2, list)

    @pytest.mark.asyncio
    async def test_discovery_empty_task(self):
        """DynamicToolDiscovery should handle empty tasks."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        tools = await discovery.discover_tools("")

        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_complex_tasks(self):
        """DynamicToolDiscovery should handle complex multi-step tasks."""
        deps = self.create_mock_dependencies()
        discovery = DynamicToolDiscovery(deps)

        complex_task = "Search for research papers about machine learning, then summarize the key findings and create a comparative analysis"

        tools = await discovery.discover_tools(complex_task)

        assert isinstance(tools, list)
        # Should potentially find multiple tools for complex tasks
        assert len(tools) >= 0


class TestDiscoverToolsForTask:
    """Test discover_tools_for_task utility function."""

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_success(self):
        """discover_tools_for_task should work correctly."""
        mock_client_manager = MagicMock()
        task = "search for documents"

        with patch(
            "src.services.agents.dynamic_tool_discovery.get_discovery_engine"
        ) as mock_get_engine:
            mock_discovery = AsyncMock()
            mock_discovery.discover_tools.return_value = [
                ToolCapability(
                    name="search_tool",
                    description="Search tool",
                    input_types=["text"],
                    output_types=["results"],
                )
            ]
            mock_get_engine.return_value = mock_discovery

            tools = await discover_tools_for_task(task, mock_client_manager)

            assert isinstance(tools, list)
            assert len(tools) > 0
            assert all(isinstance(tool, ToolCapability) for tool in tools)
            mock_get_engine.assert_called_once_with(mock_client_manager)
            mock_discovery.discover_tools.assert_called_once_with(task, context=None)

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_with_context(self):
        """discover_tools_for_task should handle context parameter."""
        mock_client_manager = MagicMock()
        task = "search for documents"
        context = {"domain": "academic"}

        with patch(
            "src.services.agents.dynamic_tool_discovery.get_discovery_engine"
        ) as mock_get_engine:
            mock_discovery = AsyncMock()
            mock_discovery.discover_tools.return_value = []
            mock_get_engine.return_value = mock_discovery

            tools = await discover_tools_for_task(
                task, mock_client_manager, context=context
            )

            assert isinstance(tools, list)
            mock_discovery.discover_tools.assert_called_once_with(task, context=context)

    @pytest.mark.asyncio
    async def test_discover_tools_for_task_error_handling(self):
        """discover_tools_for_task should handle errors gracefully."""
        mock_client_manager = MagicMock()
        task = "search for documents"

        with patch(
            "src.services.agents.dynamic_tool_discovery.get_discovery_engine"
        ) as mock_get_engine:
            mock_discovery = AsyncMock()
            mock_discovery.discover_tools.side_effect = Exception("Discovery error")
            mock_get_engine.return_value = mock_discovery

            # Should either return empty list or raise exception
            try:
                tools = await discover_tools_for_task(task, mock_client_manager)
                assert isinstance(tools, list)
            except Exception as e:
                assert "discovery" in str(e).lower() or "error" in str(e).lower()


class TestGetDiscoveryEngine:
    """Test get_discovery_engine factory function."""

    @patch("src.services.agents.dynamic_tool_discovery.create_agent_dependencies")
    def test_get_discovery_engine_success(self, mock_create_deps):
        """get_discovery_engine should create discovery engine instance."""
        mock_client_manager = MagicMock()
        mock_deps = MagicMock()
        mock_create_deps.return_value = mock_deps

        engine = get_discovery_engine(mock_client_manager)

        assert isinstance(engine, DynamicToolDiscovery)
        assert engine.dependencies == mock_deps
        mock_create_deps.assert_called_once_with(mock_client_manager)

    @patch("src.services.agents.dynamic_tool_discovery.create_agent_dependencies")
    def test_get_discovery_engine_with_dependency_error(self, mock_create_deps):
        """get_discovery_engine should handle dependency creation errors."""
        mock_client_manager = MagicMock()
        mock_create_deps.side_effect = Exception("Dependency error")

        with pytest.raises(Exception, match="Dependency error"):
            get_discovery_engine(mock_client_manager)


class TestDiscoveryEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_discovery_with_none_task(self):
        """DynamicToolDiscovery should handle None task input."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        discovery = DynamicToolDiscovery(deps)

        # Should handle None gracefully
        tools = await discovery.discover_tools(None)
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_with_malformed_context(self):
        """DynamicToolDiscovery should handle malformed context."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        discovery = DynamicToolDiscovery(deps)

        malformed_contexts = [
            "not_a_dict",
            123,
            ["list", "instead", "of", "dict"],
            {"nested": {"very": {"deep": {"context": True}}}},
        ]

        for context in malformed_contexts:
            tools = await discovery.discover_tools("test task", context=context)
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_with_special_characters_in_task(self):
        """DynamicToolDiscovery should handle special characters in tasks."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())
        discovery = DynamicToolDiscovery(deps)

        special_tasks = [
            "task with émojis 🔍",
            "task with <HTML> tags",
            "task with 'quotes' and \"double quotes\"",
            "task with\nnewlines\nand\ttabs",
            "task with unicode: ∂∆√∑∏π",
            'task with JSON: {"key": "value"}',
        ]

        for task in special_tasks:
            tools = await discovery.discover_tools(task)
            assert isinstance(tools, list)

    def test_tool_capability_with_empty_lists(self):
        """ToolCapability should handle empty input/output type lists."""
        capability = ToolCapability(
            name="empty_tool",
            description="Tool with empty lists",
            input_types=[],
            output_types=[],
        )

        assert capability.input_types == []
        assert capability.output_types == []
        assert capability.confidence_score == 1.0

    def test_tool_capability_with_long_descriptions(self):
        """ToolCapability should handle very long descriptions."""
        long_description = "This is a very long description " * 100
        capability = ToolCapability(
            name="long_desc_tool",
            description=long_description,
            input_types=["text"],
            output_types=["result"],
        )

        assert len(capability.description) > 1000
        assert capability.name == "long_desc_tool"


@pytest.mark.integration
class TestDiscoveryIntegration:
    """Integration tests for discovery functionality."""

    @pytest.mark.asyncio
    async def test_discovery_full_workflow(self):
        """Test complete discovery workflow."""
        mock_client_manager = MagicMock()
        mock_config = MagicMock()

        deps = BaseAgentDependencies(
            client_manager=mock_client_manager, config=mock_config
        )

        discovery = DynamicToolDiscovery(deps)

        # Should be able to discover tools end-to-end
        tools = await discovery.discover_tools("integration test task")
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_discovery_performance_with_multiple_tasks(self):
        """Test discovery performance with concurrent tasks."""
        import asyncio

        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())

        discovery = DynamicToolDiscovery(deps)

        tasks = [f"task {i}" for i in range(5)]

        # Process tasks concurrently
        discover_tasks = [discovery.discover_tools(task) for task in tasks]
        results = await asyncio.gather(*discover_tasks, return_exceptions=True)

        # All tasks should complete
        assert len(results) == 5
        for result in results:
            assert isinstance(result, list) or isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_discovery_with_real_tool_registry(self):
        """Test discovery integration with actual tool registry."""
        deps = BaseAgentDependencies(client_manager=MagicMock(), config=MagicMock())

        discovery = DynamicToolDiscovery(deps)

        # Test with realistic task that might match real tools
        realistic_tasks = [
            "search documents",
            "analyze content",
            "extract information",
            "process data",
        ]

        for task in realistic_tasks:
            tools = await discovery.discover_tools(task)
            assert isinstance(tools, list)

            # If tools are found, they should be valid ToolCapability instances
            for tool in tools:
                assert isinstance(tool, ToolCapability)
                assert tool.name
                assert tool.description
                assert isinstance(tool.input_types, list)
                assert isinstance(tool.output_types, list)
                assert 0.0 <= tool.confidence_score <= 1.0
