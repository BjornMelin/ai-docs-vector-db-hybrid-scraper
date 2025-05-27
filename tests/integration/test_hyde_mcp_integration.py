#!/usr/bin/env python3
"""
Integration tests for HyDE with MCP server.

Tests the integration of HyDE functionality with the unified MCP server,
including tool execution, request/response handling, and error scenarios.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp import Context

# Import MCP request/response models
from src.mcp.models.requests import HyDESearchRequest
from src.mcp.models.responses import SearchResult
from src.unified_mcp_server import hyde_search
from src.unified_mcp_server import hyde_search_advanced


class TestHyDEMCPIntegration:
    """Test cases for HyDE integration with MCP server."""

    @pytest.fixture
    def mock_context(self):
        """Mock FastMCP context for testing."""
        ctx = AsyncMock(spec=Context)
        return ctx

    @pytest.fixture
    def mock_service_manager(self):
        """Mock service manager with HyDE engine."""
        with patch("src.unified_mcp_server.service_manager") as mock_manager:
            # Mock initialization
            mock_manager.initialize = AsyncMock()

            # Mock HyDE engine
            mock_hyde_engine = AsyncMock()
            mock_hyde_engine.enhanced_search.return_value = [
                {
                    "id": "1",
                    "content": "HyDE generated result about machine learning algorithms",
                    "score": 0.95,
                    "url": "https://example.com/ml",
                    "title": "ML Guide",
                    "metadata": {"source": "documentation"},
                },
                {
                    "id": "2",
                    "content": "Deep learning tutorial with practical examples",
                    "score": 0.89,
                    "url": "https://example.com/dl",
                    "title": "DL Tutorial",
                    "metadata": {"source": "tutorial"},
                },
            ]
            mock_manager.hyde_engine = mock_hyde_engine

            # Mock embedding manager for reranking
            mock_embedding_manager = AsyncMock()
            mock_embedding_manager.rerank_results.return_value = [
                {
                    "content": "Reranked result 1",
                    "original": {"id": "1", "score": 0.95},
                },
                {
                    "content": "Reranked result 2",
                    "original": {"id": "2", "score": 0.89},
                },
            ]
            mock_manager.embedding_manager = mock_embedding_manager

            return mock_manager

    @pytest.fixture
    def mock_security_validator(self):
        """Mock security validator."""
        with patch("src.unified_mcp_server.SecurityValidator") as mock_validator:
            validator_instance = MagicMock()
            validator_instance.validate_collection_name.side_effect = lambda x: x
            validator_instance.validate_query_string.side_effect = lambda x: x
            mock_validator.from_unified_config.return_value = validator_instance
            return validator_instance

    @pytest.mark.asyncio
    async def test_hyde_search_basic_request(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test basic HyDE search request through MCP server."""
        request = HyDESearchRequest(
            query="What is machine learning?",
            collection="documentation",
            limit=5,
        )

        results = await hyde_search(request, mock_context)

        # Verify results
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, SearchResult) for result in results)

        # Verify service calls
        assert mock_service_manager.initialize.called
        assert mock_service_manager.hyde_engine.enhanced_search.called

        # Verify context logging
        assert mock_context.info.called

    @pytest.mark.asyncio
    async def test_hyde_search_with_domain(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test HyDE search with domain specification."""
        request = HyDESearchRequest(
            query="REST API authentication methods",
            collection="api_docs",
            domain="api",
            num_generations=3,
            limit=10,
        )

        results = await hyde_search(request, mock_context)

        assert isinstance(results, list)
        assert len(results) > 0

        # Verify HyDE engine was called with domain
        hyde_call_args = mock_service_manager.hyde_engine.enhanced_search.call_args
        assert hyde_call_args[1]["domain"] == "api"
        assert hyde_call_args[1]["query"] == request.query
        assert hyde_call_args[1]["collection_name"] == request.collection

    @pytest.mark.asyncio
    async def test_hyde_search_with_reranking(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test HyDE search with BGE reranking enabled."""
        request = HyDESearchRequest(
            query="database optimization techniques",
            collection="tech_docs",
            enable_reranking=True,
            limit=5,
        )

        results = await hyde_search(request, mock_context)

        assert isinstance(results, list)
        assert len(results) <= request.limit

        # Verify reranking was called
        assert mock_service_manager.embedding_manager.rerank_results.called

        # Verify debug logging for reranking
        debug_calls = [
            call
            for call in mock_context.debug.call_args_list
            if "reranking" in str(call).lower()
        ]
        assert len(debug_calls) > 0

    @pytest.mark.asyncio
    async def test_hyde_search_fallback_when_engine_unavailable(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test fallback to regular search when HyDE engine is unavailable."""
        # Mock HyDE engine as unavailable
        mock_service_manager.hyde_engine = None

        # Mock regular search function
        with patch("src.unified_mcp_server.search_documents") as mock_search:
            mock_search.return_value = [
                SearchResult(
                    id="fallback_1",
                    content="Fallback search result",
                    score=0.8,
                    url="https://example.com/fallback",
                    title="Fallback",
                )
            ]

            request = HyDESearchRequest(
                query="fallback test query",
                collection="docs",
                limit=3,
            )

            results = await hyde_search(request, mock_context)

            assert isinstance(results, list)
            assert len(results) > 0

            # Verify fallback was used
            assert mock_search.called
            assert mock_context.warning.called

    @pytest.mark.asyncio
    async def test_hyde_search_error_handling(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test error handling in HyDE search."""
        # Mock HyDE engine failure
        mock_service_manager.hyde_engine.enhanced_search.side_effect = Exception(
            "HyDE search failed"
        )

        # Mock fallback search
        with patch("src.unified_mcp_server.search_documents") as mock_search:
            mock_search.return_value = [
                SearchResult(
                    id="error_fallback_1",
                    content="Error fallback result",
                    score=0.7,
                )
            ]

            request = HyDESearchRequest(
                query="error test query",
                collection="docs",
                limit=3,
            )

            results = await hyde_search(request, mock_context)

            assert isinstance(results, list)

            # Verify error was logged and fallback was attempted
            assert mock_context.error.called
            assert mock_context.warning.called  # Fallback warning
            assert mock_search.called

    @pytest.mark.asyncio
    async def test_hyde_search_advanced_basic(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test advanced HyDE search with basic parameters."""
        results = await hyde_search_advanced(
            query="advanced search test",
            collection="documentation",
            limit=5,
            ctx=mock_context,
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "metrics" in results
        assert "hyde_config" in results
        assert "request_id" in results

        # Verify results structure
        assert isinstance(results["results"], list)
        assert isinstance(results["metrics"], dict)
        assert "search_time_ms" in results["metrics"]

    @pytest.mark.asyncio
    async def test_hyde_search_advanced_with_ab_testing(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test advanced HyDE search with A/B testing enabled."""
        # Mock regular search for A/B comparison
        mock_regular_results = [
            MagicMock(score=0.75, id="regular_1"),
            MagicMock(score=0.70, id="regular_2"),
        ]
        mock_service_manager.qdrant_service.hybrid_search.return_value = (
            mock_regular_results
        )

        results = await hyde_search_advanced(
            query="A/B testing query",
            collection="docs",
            enable_ab_testing=True,
            limit=3,
            ctx=mock_context,
        )

        assert isinstance(results, dict)
        assert "ab_test_results" in results
        assert results["ab_test_results"] is not None

        # Verify A/B test metrics
        ab_results = results["ab_test_results"]
        assert "hyde_count" in ab_results
        assert "regular_count" in ab_results
        assert "hyde_avg_score" in ab_results
        assert "regular_avg_score" in ab_results

    @pytest.mark.asyncio
    async def test_hyde_search_advanced_with_custom_parameters(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test advanced HyDE search with custom generation parameters."""
        results = await hyde_search_advanced(
            query="custom parameters test",
            collection="docs",
            domain="tutorial",
            num_generations=7,
            generation_temperature=0.8,
            enable_reranking=True,
            use_cache=False,
            limit=8,
            ctx=mock_context,
        )

        assert isinstance(results, dict)

        # Verify configuration was applied
        hyde_config = results["hyde_config"]
        assert hyde_config["domain"] == "tutorial"
        assert hyde_config["num_generations"] == 7
        assert hyde_config["temperature"] == 0.8
        assert hyde_config["use_cache"] is False

        # Verify metrics include generation parameters
        metrics = results["metrics"]
        assert "generation_parameters" in metrics
        gen_params = metrics["generation_parameters"]
        assert gen_params["num_generations"] == 7
        assert gen_params["temperature"] == 0.8
        assert gen_params["domain"] == "tutorial"

    @pytest.mark.asyncio
    async def test_hyde_search_advanced_error_handling(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test error handling in advanced HyDE search."""
        # Mock HyDE engine as unavailable
        mock_service_manager.hyde_engine = None

        with pytest.raises(ValueError, match="HyDE engine not initialized"):
            await hyde_search_advanced(
                query="error test",
                collection="docs",
                ctx=mock_context,
            )

        assert mock_context.error.called

    @pytest.mark.asyncio
    async def test_hyde_search_security_validation(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test security validation in HyDE search requests."""
        request = HyDESearchRequest(
            query="test query",
            collection="test_collection",
            limit=5,
        )

        await hyde_search(request, mock_context)

        # Verify security validation was called
        assert mock_security_validator.validate_collection_name.called
        assert mock_security_validator.validate_query_string.called

        # Verify the calls were made with correct parameters
        mock_security_validator.validate_collection_name.assert_called_with(
            "test_collection"
        )
        mock_security_validator.validate_query_string.assert_called_with("test query")

    @pytest.mark.asyncio
    async def test_hyde_search_request_tracking(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test request ID generation and tracking."""
        request = HyDESearchRequest(
            query="tracking test query",
            collection="docs",
            limit=3,
        )

        _results = await hyde_search(request, mock_context)

        # Verify request tracking in context logs
        info_calls = mock_context.info.call_args_list
        assert len(info_calls) >= 2  # Start and completion messages

        # Check that request ID is consistent across logs
        start_message = str(info_calls[0])
        completion_message = str(info_calls[-1])

        # Both should contain the same request ID pattern
        assert "Starting HyDE search" in start_message
        assert "completed" in completion_message

    @pytest.mark.asyncio
    async def test_hyde_search_result_format_conversion(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test conversion of HyDE engine results to SearchResult format."""
        # Mock HyDE engine returning different result formats
        mixed_results = [
            # Dict format
            {
                "id": "dict_1",
                "content": "Dict formatted result",
                "score": 0.9,
                "url": "https://example.com/dict",
                "title": "Dict Result",
                "metadata": {"type": "dict"},
            },
            # Mock Qdrant point object
            MagicMock(
                id="point_1",
                score=0.85,
                payload={
                    "content": "Point formatted result",
                    "url": "https://example.com/point",
                    "title": "Point Result",
                    "type": "point",
                },
            ),
        ]

        mock_service_manager.hyde_engine.enhanced_search.return_value = mixed_results

        request = HyDESearchRequest(
            query="format conversion test",
            collection="docs",
            limit=5,
            include_metadata=True,
        )

        results = await hyde_search(request, mock_context)

        # Verify all results are converted to SearchResult objects
        assert all(isinstance(result, SearchResult) for result in results)
        assert len(results) == 2

        # Verify dict conversion
        dict_result = next((r for r in results if r.id == "dict_1"), None)
        assert dict_result is not None
        assert dict_result.content == "Dict formatted result"
        assert dict_result.score == 0.9
        assert dict_result.metadata is not None

        # Verify point conversion
        point_result = next((r for r in results if r.id == "point_1"), None)
        assert point_result is not None
        assert point_result.content == "Point formatted result"
        assert point_result.score == 0.85

    @pytest.mark.asyncio
    async def test_hyde_search_concurrent_requests(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test handling of concurrent HyDE search requests."""
        requests = [
            HyDESearchRequest(
                query=f"concurrent query {i}",
                collection="docs",
                limit=3,
            )
            for i in range(5)
        ]

        # Create concurrent tasks
        tasks = [hyde_search(request, mock_context) for request in requests]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should succeed
        assert len(results) == 5
        for result in results:
            if not isinstance(result, Exception):
                assert isinstance(result, list)
                assert all(isinstance(r, SearchResult) for r in result)

    @pytest.mark.asyncio
    async def test_hyde_search_performance_metrics(
        self, mock_context, mock_service_manager, mock_security_validator
    ):
        """Test performance metrics collection in advanced HyDE search."""
        results = await hyde_search_advanced(
            query="performance test query",
            collection="docs",
            limit=5,
            ctx=mock_context,
        )

        metrics = results["metrics"]

        # Verify all expected metrics are present
        assert "search_time_ms" in metrics
        assert "results_found" in metrics
        assert "reranking_applied" in metrics
        assert "cache_used" in metrics
        assert "generation_parameters" in metrics

        # Verify metric types
        assert isinstance(metrics["search_time_ms"], int | float)
        assert isinstance(metrics["results_found"], int)
        assert isinstance(metrics["reranking_applied"], bool)
        assert isinstance(metrics["cache_used"], bool)
        assert isinstance(metrics["generation_parameters"], dict)
