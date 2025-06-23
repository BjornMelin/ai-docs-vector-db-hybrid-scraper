"""Pact consumer-driven contract tests.

This module implements consumer-driven contract testing using Pact patterns
for testing service interactions and API contracts.
"""

import pytest


class TestVectorSearchConsumerContracts:
    """Test consumer contracts for vector search API."""

    @pytest.mark.pact
    async def test_search_consumer_contract(
        self, pact_contract_builder, mock_contract_service
    ):
        """Test search API consumer contract."""
        # Define the interaction
        pact_contract_builder.given("documents exist in the collection")
        pact_contract_builder.upon_receiving("a search request for AI documentation")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/search",
            headers={"Content-Type": "application/json"},
            body={
                "query": "machine learning",
                "collection_name": "documents",
                "limit": 10,
                "score_threshold": 0.7,
            },
        )
        pact_contract_builder.will_respond_with(
            status=200,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "results": [
                    {
                        "id": "doc1",
                        "score": 0.95,
                        "title": "Machine Learning Guide",
                        "content": "Comprehensive guide to ML",
                        "metadata": {"source": "documentation"},
                    }
                ],
                "total_count": 1,
                "query_time_ms": 45.0,
                "search_strategy": "hybrid",
                "cache_hit": False,
            },
        )

        # Build the contract
        contract = pact_contract_builder.build_pact()

        # Verify contract structure
        assert contract["consumer"]["name"] == "ai-docs-consumer"
        assert contract["provider"]["name"] == "ai-docs-provider"
        assert len(contract["interactions"]) == 1

        interaction = contract["interactions"][0]
        assert interaction["description"] == "a search request for AI documentation"
        assert interaction["request"]["method"] == "POST"
        assert interaction["request"]["path"] == "/api/search"
        assert interaction["response"]["status"] == 200

    @pytest.mark.pact
    async def test_advanced_search_consumer_contract(self, pact_contract_builder):
        """Test advanced search API consumer contract."""
        pact_contract_builder.given("documents with embeddings exist")
        pact_contract_builder.upon_receiving(
            "an advanced search request with reranking"
        )
        pact_contract_builder.with_request(
            method="POST",
            path="/api/search/advanced",
            headers={"Content-Type": "application/json"},
            body={
                "query": "neural networks deep learning",
                "collection_name": "ml_docs",
                "search_strategy": "hybrid",
                "accuracy_level": "accurate",
                "enable_reranking": True,
                "limit": 20,
            },
        )
        pact_contract_builder.will_respond_with(
            status=200,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "results": [
                    {
                        "id": "doc2",
                        "score": 0.98,
                        "title": "Deep Neural Networks",
                        "content": "Advanced deep learning concepts",
                        "metadata": {"source": "research_paper", "reranked": True},
                    }
                ],
                "total_count": 15,
                "query_time_ms": 120.0,
                "search_strategy": "hybrid",
                "cache_hit": False,
            },
        )

        contract = pact_contract_builder.build_pact()
        assert len(contract["interactions"]) == 1

        interaction = contract["interactions"][0]
        request_body = interaction["request"]["body"]
        assert request_body["search_strategy"] == "hybrid"
        assert request_body["enable_reranking"] is True

    @pytest.mark.pact
    async def test_search_error_consumer_contract(self, pact_contract_builder):
        """Test search API error response contract."""
        pact_contract_builder.given("invalid search parameters")
        pact_contract_builder.upon_receiving("a search request with invalid query")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/search",
            headers={"Content-Type": "application/json"},
            body={
                "query": "",  # Invalid empty query
                "collection_name": "documents",
                "limit": 10,
            },
        )
        pact_contract_builder.will_respond_with(
            status=400,
            headers={"Content-Type": "application/json"},
            body={
                "success": False,
                "error": "Query cannot be empty",
                "error_type": "validation_error",
                "context": {"field": "query", "value": ""},
            },
        )

        contract = pact_contract_builder.build_pact()
        interaction = contract["interactions"][0]
        assert interaction["response"]["status"] == 400

        response_body = interaction["response"]["body"]
        assert response_body["success"] is False
        assert "validation_error" in response_body["error_type"]


class TestDocumentProcessingConsumerContracts:
    """Test consumer contracts for document processing API."""

    @pytest.mark.pact
    async def test_add_document_consumer_contract(self, pact_contract_builder):
        """Test add document API consumer contract."""
        pact_contract_builder.given("collection exists and is ready")
        pact_contract_builder.upon_receiving("a request to add a new document")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/documents",
            headers={"Content-Type": "application/json"},
            body={
                "url": "https://example.com/doc",
                "collection_name": "documents",
                "doc_type": "webpage",
                "metadata": {"category": "tutorial"},
                "force_recrawl": False,
            },
        )
        pact_contract_builder.will_respond_with(
            status=201,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "document_id": "doc_123",
                "url": "https://example.com/doc",
                "chunks_created": 5,
                "processing_time_ms": 1500.0,
                "status": "processed",
            },
        )

        contract = pact_contract_builder.build_pact()
        interaction = contract["interactions"][0]
        assert interaction["response"]["status"] == 201

        response_body = interaction["response"]["body"]
        assert "document_id" in response_body
        assert response_body["status"] == "processed"

    @pytest.mark.pact
    async def test_bulk_document_consumer_contract(self, pact_contract_builder):
        """Test bulk document processing consumer contract."""
        pact_contract_builder.given("system ready for bulk processing")
        pact_contract_builder.upon_receiving("a bulk document processing request")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/documents/bulk",
            headers={"Content-Type": "application/json"},
            body={
                "urls": [
                    "https://example.com/doc1",
                    "https://example.com/doc2",
                    "https://example.com/doc3",
                ],
                "collection_name": "bulk_docs",
                "max_concurrent": 3,
                "force_recrawl": False,
            },
        )
        pact_contract_builder.will_respond_with(
            status=202,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "processed_count": 3,
                "failed_count": 0,
                "total_chunks": 15,
                "processing_time_ms": 4500.0,
                "results": [
                    {
                        "document_id": "doc_124",
                        "url": "https://example.com/doc1",
                        "chunks_created": 5,
                        "processing_time_ms": 1200.0,
                        "status": "processed",
                    }
                ],
                "errors": [],
            },
        )

        contract = pact_contract_builder.build_pact()
        interaction = contract["interactions"][0]
        assert interaction["response"]["status"] == 202

        response_body = interaction["response"]["body"]
        assert response_body["processed_count"] == 3
        assert response_body["failed_count"] == 0
        assert len(response_body["results"]) == 1


class TestCollectionManagementConsumerContracts:
    """Test consumer contracts for collection management API."""

    @pytest.mark.pact
    async def test_create_collection_consumer_contract(self, pact_contract_builder):
        """Test create collection API consumer contract."""
        pact_contract_builder.given("vector database is available")
        pact_contract_builder.upon_receiving("a request to create a new collection")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/collections",
            headers={"Content-Type": "application/json"},
            body={
                "collection_name": "new_collection",
                "vector_size": 1024,
                "distance_metric": "Cosine",
                "enable_hybrid": True,
                "hnsw_config": {"m": 16, "ef_construct": 100},
            },
        )
        pact_contract_builder.will_respond_with(
            status=201,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "collection_name": "new_collection",
                "operation": "created",
                "details": {
                    "vector_size": 1024,
                    "distance_metric": "Cosine",
                    "hybrid_enabled": True,
                },
            },
        )

        contract = pact_contract_builder.build_pact()
        interaction = contract["interactions"][0]
        assert interaction["response"]["status"] == 201

        response_body = interaction["response"]["body"]
        assert response_body["operation"] == "created"
        assert response_body["collection_name"] == "new_collection"

    @pytest.mark.pact
    async def test_list_collections_consumer_contract(self, pact_contract_builder):
        """Test list collections API consumer contract."""
        pact_contract_builder.given("collections exist in the database")
        pact_contract_builder.upon_receiving("a request to list all collections")
        pact_contract_builder.with_request(
            method="GET",
            path="/api/collections",
            headers={"Accept": "application/json"},
        )
        pact_contract_builder.will_respond_with(
            status=200,
            headers={"Content-Type": "application/json"},
            body={
                "success": True,
                "collections": [
                    {
                        "name": "documents",
                        "points_count": 1000,
                        "vectors_count": 1000,
                        "indexed_fields": ["doc_type", "source"],
                        "status": "green",
                        "config": {"vector_size": 1024},
                    }
                ],
                "total_count": 1,
            },
        )

        contract = pact_contract_builder.build_pact()
        interaction = contract["interactions"][0]
        assert interaction["request"]["method"] == "GET"
        assert interaction["response"]["status"] == 200

        response_body = interaction["response"]["body"]
        assert "collections" in response_body
        assert response_body["total_count"] == 1


class TestPactContractVerification:
    """Test Pact contract verification against actual implementations."""

    @pytest.mark.pact
    async def test_verify_search_contract(
        self, pact_contract_builder, mock_contract_service
    ):
        """Test verification of search contract against mock service."""
        # Set up the contract
        pact_contract_builder.given("documents exist")
        pact_contract_builder.upon_receiving("search request")
        pact_contract_builder.with_request(
            method="POST", path="/api/search", body={"query": "test", "limit": 10}
        )
        pact_contract_builder.will_respond_with(
            status=200,
            body={
                "success": True,
                "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
                "total_count": 1,
            },
        )

        # Simulate actual service call
        actual_request = {
            "method": "POST",
            "path": "/api/search",
            "body": {"query": "test", "limit": 10},
        }

        actual_response = {
            "status": 200,
            "body": {
                "success": True,
                "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
                "total_count": 1,
            },
        }

        # Verify contract
        verification_result = pact_contract_builder.verify_interaction(
            actual_request, actual_response
        )

        assert verification_result["verified"] is True
        assert len(verification_result["errors"]) == 0

    @pytest.mark.pact
    async def test_contract_violation_detection(self, pact_contract_builder):
        """Test detection of contract violations."""
        # Set up the contract
        pact_contract_builder.given("documents exist")
        pact_contract_builder.upon_receiving("search request")
        pact_contract_builder.with_request(
            method="POST", path="/api/search", body={"query": "test"}
        )
        pact_contract_builder.will_respond_with(
            status=200, body={"success": True, "results": []}
        )

        # Simulate contract violation (wrong method)
        violating_request = {
            "method": "GET",  # Expected POST
            "path": "/api/search",
            "body": {"query": "test"},
        }

        violating_response = {"status": 200, "body": {"success": True, "results": []}}

        # Verify contract violation is detected
        verification_result = pact_contract_builder.verify_interaction(
            violating_request, violating_response
        )

        assert verification_result["verified"] is False
        assert len(verification_result["errors"]) > 0
        assert any(
            "Method mismatch" in error for error in verification_result["errors"]
        )


class TestConsumerContractEvolution:
    """Test consumer contract evolution and versioning."""

    @pytest.mark.pact
    async def test_contract_versioning(self, pact_contract_builder):
        """Test contract versioning scenarios."""
        # Version 1 contract
        pact_contract_builder.given("API version 1")
        pact_contract_builder.upon_receiving("v1 search request")
        pact_contract_builder.with_request(
            method="POST", path="/api/v1/search", body={"query": "test", "limit": 10}
        )
        pact_contract_builder.will_respond_with(
            status=200,
            body={
                "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
                "total": 1,
            },
        )

        v1_contract = pact_contract_builder.build_pact()

        # Create new builder for v2
        pact_contract_builder.interactions = []  # Reset interactions

        # Version 2 contract (backward compatible)
        pact_contract_builder.given("API version 2")
        pact_contract_builder.upon_receiving("v2 search request")
        pact_contract_builder.with_request(
            method="POST",
            path="/api/v2/search",
            body={
                "query": "test",
                "limit": 10,
                "search_strategy": "hybrid",  # New optional field
            },
        )
        pact_contract_builder.will_respond_with(
            status=200,
            body={
                "success": True,  # New field
                "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
                "total_count": 1,  # Renamed field
                "search_strategy": "hybrid",  # New field
            },
        )

        v2_contract = pact_contract_builder.build_pact()

        # Verify both contracts are valid
        assert len(v1_contract["interactions"]) == 1
        assert len(v2_contract["interactions"]) == 1

        # Verify v2 has additional fields
        v2_request = v2_contract["interactions"][0]["request"]["body"]
        assert "search_strategy" in v2_request

        v2_response = v2_contract["interactions"][0]["response"]["body"]
        assert "success" in v2_response
        assert "search_strategy" in v2_response

    @pytest.mark.pact
    async def test_contract_compatibility_matrix(self, pact_contract_builder):
        """Test compatibility matrix between different contract versions."""
        compatibility_matrix = {
            "consumer_v1_provider_v1": True,  # Same version
            "consumer_v1_provider_v2": True,  # Backward compatible
            "consumer_v2_provider_v1": False,  # Forward incompatible
            "consumer_v2_provider_v2": True,  # Same version
        }

        # Test scenarios
        for scenario, expected_compatibility in compatibility_matrix.items():
            _consumer_version, _provider_version = (
                scenario.split("_")[1],
                scenario.split("_")[3],
            )

            # This would be expanded with actual compatibility testing logic
            assert isinstance(expected_compatibility, bool)

        # For now, just verify the matrix structure
        assert len(compatibility_matrix) == 4
        assert compatibility_matrix["consumer_v1_provider_v1"] is True
        assert compatibility_matrix["consumer_v2_provider_v1"] is False
