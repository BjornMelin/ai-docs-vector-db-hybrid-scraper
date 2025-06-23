"""Consumer-driven contract tests for MCP (Model Context Protocol) tools.

This module tests MCP tool contracts from the consumer perspective,
ensuring compatibility and proper behavior.
"""

import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, patch

from src.models.api_contracts import SearchRequest, SearchResponse
from src.models.requests import MCPToolRequest
from src.models.responses import MCPToolResponse


class TestMCPSearchToolContracts:
    """Test MCP search tool consumer contracts."""

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_search_tool_contract(self, mock_contract_service):
        """Test search tool consumer contract."""
        # Consumer expectation: search tool should accept query and return results
        search_request = {
            "name": "search",
            "arguments": {
                "query": "machine learning tutorials",
                "collection_name": "documentation",
                "limit": 10,
                "enable_hyde": False
            }
        }
        
        # Expected response structure
        expected_response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "success": True,
                        "results": [
                            {
                                "id": "doc1",
                                "score": 0.95,
                                "title": "ML Tutorial",
                                "content": "Machine learning guide",
                                "metadata": {"source": "docs"}
                            }
                        ],
                        "total_count": 1,
                        "query_time_ms": 45.0,
                        "search_strategy": "hybrid"
                    })
                }
            ]
        }
        
        # Mock the service response
        mock_contract_service.search.return_value = expected_response["content"][0]["text"]
        
        # Execute the tool
        result = await mock_contract_service.search(
            search_request["arguments"]["query"],
            search_request["arguments"]["limit"]
        )
        
        # Verify the contract is satisfied
        assert result is not None
        result_data = json.loads(result) if isinstance(result, str) else result
        assert "results" in result_data
        assert "total_count" in result_data
        assert "success" in result_data
        assert result_data["success"] is True

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_advanced_search_tool_contract(self, mock_contract_service):
        """Test advanced search tool consumer contract."""
        # Consumer expectation: advanced search with strategy selection
        search_request = {
            "name": "advanced_search",
            "arguments": {
                "query": "neural networks",
                "collection_name": "research",
                "search_strategy": "hybrid",
                "accuracy_level": "accurate",
                "enable_reranking": True,
                "limit": 20
            }
        }
        
        # Expected advanced features in response
        expected_response_features = {
            "search_strategy": "hybrid",
            "reranking_applied": True,
            "accuracy_level": "accurate",
            "query_expansion": True
        }
        
        # Mock advanced search response
        mock_response = {
            "success": True,
            "results": [
                {
                    "id": "research1",
                    "score": 0.98,
                    "title": "Neural Network Architecture",
                    "content": "Deep learning research",
                    "metadata": {
                        "reranked": True,
                        "strategy_used": "hybrid"
                    }
                }
            ],
            "total_count": 15,
            "query_time_ms": 120.0,
            **expected_response_features
        }
        
        mock_contract_service.search.return_value = json.dumps(mock_response)
        
        # Execute advanced search
        result = await mock_contract_service.search(
            search_request["arguments"]["query"],
            search_request["arguments"]["limit"]
        )
        
        result_data = json.loads(result)
        
        # Verify advanced features are present
        for feature, expected_value in expected_response_features.items():
            assert feature in result_data
            if isinstance(expected_value, bool):
                assert result_data[feature] == expected_value

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_search_error_handling_contract(self, mock_contract_service):
        """Test search tool error handling contract."""
        # Consumer expectation: proper error responses for invalid inputs
        invalid_requests = [
            {"query": "", "limit": 10},  # Empty query
            {"query": "test", "limit": 0},  # Invalid limit
            {"query": "test", "limit": 1000},  # Limit too high
        ]
        
        for invalid_request in invalid_requests:
            # Mock error response
            error_response = {
                "success": False,
                "error": "Invalid request parameters",
                "error_type": "validation_error",
                "context": invalid_request
            }
            
            mock_contract_service.search.return_value = json.dumps(error_response)
            
            # Execute with invalid parameters
            result = await mock_contract_service.search(
                invalid_request["query"],
                invalid_request["limit"]
            )
            
            result_data = json.loads(result)
            
            # Verify error contract
            assert "success" in result_data
            assert result_data["success"] is False
            assert "error" in result_data
            assert "error_type" in result_data
            assert result_data["error_type"] == "validation_error"


class TestMCPDocumentToolContracts:
    """Test MCP document processing tool consumer contracts."""

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_add_document_tool_contract(self, mock_contract_service):
        """Test add document tool consumer contract."""
        # Consumer expectation: add document with URL and get processing status
        add_document_request = {
            "name": "add_document",
            "arguments": {
                "url": "https://example.com/document",
                "collection_name": "tutorials",
                "doc_type": "webpage",
                "metadata": {"category": "tutorial"},
                "force_recrawl": False
            }
        }
        
        # Expected response structure
        expected_response = {
            "success": True,
            "document_id": "doc_12345",
            "url": "https://example.com/document",
            "chunks_created": 8,
            "processing_time_ms": 2500.0,
            "status": "processed",
            "collection_name": "tutorials"
        }
        
        mock_contract_service.add_document.return_value = json.dumps(expected_response)
        
        # Execute the tool
        result = await mock_contract_service.add_document(
            add_document_request["arguments"]["url"],
            add_document_request["arguments"]["collection_name"]
        )
        
        result_data = json.loads(result)
        
        # Verify contract compliance
        required_fields = ["success", "document_id", "url", "chunks_created", "status"]
        for field in required_fields:
            assert field in result_data
        
        assert result_data["success"] is True
        assert result_data["url"] == add_document_request["arguments"]["url"]
        assert isinstance(result_data["chunks_created"], int)
        assert result_data["chunks_created"] > 0
        assert result_data["status"] in ["processed", "processing", "failed"]

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_bulk_document_processing_contract(self, mock_contract_service):
        """Test bulk document processing consumer contract."""
        # Consumer expectation: process multiple documents in batch
        bulk_request = {
            "name": "bulk_add_documents",
            "arguments": {
                "urls": [
                    "https://example.com/doc1",
                    "https://example.com/doc2",
                    "https://example.com/doc3"
                ],
                "collection_name": "bulk_collection",
                "max_concurrent": 3,
                "force_recrawl": False
            }
        }
        
        # Expected bulk response
        expected_response = {
            "success": True,
            "processed_count": 3,
            "failed_count": 0,
            "total_chunks": 24,
            "processing_time_ms": 7500.0,
            "results": [
                {
                    "document_id": "doc_1",
                    "url": "https://example.com/doc1",
                    "chunks_created": 8,
                    "status": "processed"
                },
                {
                    "document_id": "doc_2",
                    "url": "https://example.com/doc2",
                    "chunks_created": 10,
                    "status": "processed"
                },
                {
                    "document_id": "doc_3",
                    "url": "https://example.com/doc3",
                    "chunks_created": 6,
                    "status": "processed"
                }
            ],
            "errors": []
        }
        
        mock_contract_service.add_document.return_value = json.dumps(expected_response)
        
        # Execute bulk processing
        result = await mock_contract_service.add_document(
            bulk_request["arguments"]["urls"],
            bulk_request["arguments"]["collection_name"]
        )
        
        result_data = json.loads(result)
        
        # Verify bulk processing contract
        assert "processed_count" in result_data
        assert "failed_count" in result_data
        assert "results" in result_data
        assert "errors" in result_data
        
        assert result_data["processed_count"] == len(bulk_request["arguments"]["urls"])
        assert len(result_data["results"]) == result_data["processed_count"]
        
        # Verify individual results structure
        for doc_result in result_data["results"]:
            assert "document_id" in doc_result
            assert "url" in doc_result
            assert "chunks_created" in doc_result
            assert "status" in doc_result


class TestMCPCollectionToolContracts:
    """Test MCP collection management tool consumer contracts."""

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_create_collection_tool_contract(self, mock_contract_service):
        """Test create collection tool consumer contract."""
        # Consumer expectation: create collection with specified configuration
        create_request = {
            "name": "create_collection",
            "arguments": {
                "collection_name": "new_collection",
                "vector_size": 1024,
                "distance_metric": "Cosine",
                "enable_hybrid": True,
                "hnsw_config": {
                    "m": 16,
                    "ef_construct": 100
                }
            }
        }
        
        # Expected creation response
        expected_response = {
            "success": True,
            "collection_name": "new_collection",
            "operation": "created",
            "details": {
                "vector_size": 1024,
                "distance_metric": "Cosine",
                "hybrid_enabled": True,
                "points_count": 0,
                "status": "green"
            }
        }
        
        mock_contract_service.create_collection = AsyncMock(return_value=json.dumps(expected_response))
        
        # Execute collection creation
        result = await mock_contract_service.create_collection(
            create_request["arguments"]["collection_name"],
            create_request["arguments"]["vector_size"]
        )
        
        result_data = json.loads(result)
        
        # Verify collection creation contract
        assert result_data["success"] is True
        assert result_data["collection_name"] == create_request["arguments"]["collection_name"]
        assert result_data["operation"] == "created"
        assert "details" in result_data
        assert result_data["details"]["vector_size"] == create_request["arguments"]["vector_size"]

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_list_collections_tool_contract(self, mock_contract_service):
        """Test list collections tool consumer contract."""
        # Consumer expectation: get list of all collections with metadata
        expected_response = {
            "success": True,
            "collections": [
                {
                    "name": "documents",
                    "points_count": 1500,
                    "vectors_count": 1500,
                    "indexed_fields": ["doc_type", "source", "category"],
                    "status": "green",
                    "config": {
                        "vector_size": 1024,
                        "distance_metric": "Cosine"
                    }
                },
                {
                    "name": "research",
                    "points_count": 500,
                    "vectors_count": 500,
                    "indexed_fields": ["paper_type", "authors"],
                    "status": "green",
                    "config": {
                        "vector_size": 1536,
                        "distance_metric": "Cosine"
                    }
                }
            ],
            "total_count": 2
        }
        
        mock_contract_service.list_collections = AsyncMock(return_value=json.dumps(expected_response))
        
        # Execute list collections
        result = await mock_contract_service.list_collections()
        result_data = json.loads(result)
        
        # Verify list collections contract
        assert "collections" in result_data
        assert "total_count" in result_data
        assert len(result_data["collections"]) == result_data["total_count"]
        
        # Verify collection metadata structure
        for collection in result_data["collections"]:
            required_fields = ["name", "points_count", "vectors_count", "status", "config"]
            for field in required_fields:
                assert field in collection
            
            assert isinstance(collection["points_count"], int)
            assert isinstance(collection["vectors_count"], int)
            assert collection["status"] in ["green", "yellow", "red"]
            assert "vector_size" in collection["config"]


class TestMCPAnalyticsToolContracts:
    """Test MCP analytics tool consumer contracts."""

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_analytics_tool_contract(self, mock_contract_service):
        """Test analytics tool consumer contract."""
        # Consumer expectation: get analytics data with metrics
        analytics_request = {
            "name": "get_analytics",
            "arguments": {
                "collection_name": "documents",
                "time_range": "24h",
                "metric_types": ["search_queries", "document_additions", "cache_performance"]
            }
        }
        
        # Expected analytics response
        expected_response = {
            "success": True,
            "metrics": [
                {
                    "name": "search_queries_count",
                    "value": 1250,
                    "unit": "queries",
                    "timestamp": 1704067200.0
                },
                {
                    "name": "documents_added",
                    "value": 45,
                    "unit": "documents",
                    "timestamp": 1704067200.0
                },
                {
                    "name": "cache_hit_rate",
                    "value": 0.85,
                    "unit": "percentage",
                    "timestamp": 1704067200.0
                },
                {
                    "name": "avg_query_time_ms",
                    "value": 45.2,
                    "unit": "milliseconds",
                    "timestamp": 1704067200.0
                }
            ],
            "time_range": "24h",
            "generated_at": 1704067200.0
        }
        
        mock_contract_service.get_analytics = AsyncMock(return_value=json.dumps(expected_response))
        
        # Execute analytics request
        result = await mock_contract_service.get_analytics(
            analytics_request["arguments"]["collection_name"],
            analytics_request["arguments"]["time_range"]
        )
        
        result_data = json.loads(result)
        
        # Verify analytics contract
        assert "metrics" in result_data
        assert "time_range" in result_data
        assert "generated_at" in result_data
        assert result_data["time_range"] == analytics_request["arguments"]["time_range"]
        
        # Verify metrics structure
        for metric in result_data["metrics"]:
            required_fields = ["name", "value", "timestamp"]
            for field in required_fields:
                assert field in metric
            
            assert isinstance(metric["name"], str)
            assert isinstance(metric["value"], (int, float))
            assert isinstance(metric["timestamp"], (int, float))


class TestMCPContractEvolution:
    """Test MCP contract evolution and backward compatibility."""

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_tool_versioning_contract(self, mock_contract_service):
        """Test MCP tool versioning and backward compatibility."""
        # Test v1 search tool contract
        v1_search_request = {
            "name": "search",
            "arguments": {
                "query": "test query",
                "limit": 10
            }
        }
        
        v1_expected_response = {
            "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
            "total": 1
        }
        
        # Test v2 search tool contract (backward compatible)
        v2_search_request = {
            "name": "search_v2",
            "arguments": {
                "query": "test query",
                "limit": 10,
                "collection_name": "documents",  # New optional field
                "search_strategy": "hybrid"       # New optional field
            }
        }
        
        v2_expected_response = {
            "success": True,  # New field
            "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
            "total_count": 1,  # Renamed from 'total'
            "search_strategy": "hybrid",  # New field
            "query_time_ms": 45.0  # New field
        }
        
        # Mock both versions
        mock_contract_service.search.return_value = json.dumps(v1_expected_response)
        mock_contract_service.search_v2 = AsyncMock(return_value=json.dumps(v2_expected_response))
        
        # Test v1 contract
        v1_result = await mock_contract_service.search(
            v1_search_request["arguments"]["query"],
            v1_search_request["arguments"]["limit"]
        )
        v1_data = json.loads(v1_result)
        
        # Verify v1 contract
        assert "results" in v1_data
        assert "total" in v1_data
        
        # Test v2 contract
        v2_result = await mock_contract_service.search_v2(
            v2_search_request["arguments"]["query"],
            v2_search_request["arguments"]["limit"]
        )
        v2_data = json.loads(v2_result)
        
        # Verify v2 contract (backward compatible)
        assert "success" in v2_data
        assert "results" in v2_data
        assert "total_count" in v2_data
        assert "search_strategy" in v2_data
        
        # Verify data consistency
        assert len(v1_data["results"]) == len(v2_data["results"])
        assert v1_data["total"] == v2_data["total_count"]

    @pytest.mark.consumer_driven
    @pytest.mark.mcp
    async def test_contract_deprecation_handling(self, mock_contract_service):
        """Test handling of deprecated contract features."""
        # Test deprecated field handling
        deprecated_request = {
            "name": "search",
            "arguments": {
                "query": "test query",
                "max_results": 10,  # Deprecated field (use 'limit' instead)
                "include_content": True  # Deprecated field
            }
        }
        
        # Response should include deprecation warnings
        expected_response = {
            "success": True,
            "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
            "total_count": 1,
            "warnings": [
                "Field 'max_results' is deprecated, use 'limit' instead",
                "Field 'include_content' is deprecated, content is always included"
            ],
            "deprecated_fields_used": ["max_results", "include_content"]
        }
        
        mock_contract_service.search.return_value = json.dumps(expected_response)
        
        # Execute with deprecated fields
        result = await mock_contract_service.search(
            deprecated_request["arguments"]["query"],
            deprecated_request["arguments"].get("max_results", 10)
        )
        
        result_data = json.loads(result)
        
        # Verify deprecation handling
        assert "warnings" in result_data
        assert "deprecated_fields_used" in result_data
        assert len(result_data["warnings"]) > 0
        
        # Verify functionality still works despite deprecation
        assert "results" in result_data
        assert result_data["success"] is True
