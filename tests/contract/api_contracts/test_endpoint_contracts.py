"""API endpoint contract validation tests.

This module tests API endpoint contracts for request/response validation,
backward compatibility, and breaking change detection.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


class TestAPIEndpointContracts:
    """Test API endpoint contract validation."""

    @pytest.fixture
    def mock_api_app(self):
        """Create mock API app for testing."""

        async def search_endpoint(request):
            """Mock search endpoint."""
            if request.method == "POST":
                body = await request.json()
                query = body.get("query")

                if not query:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "error": "Query is required",
                            "error_type": "validation_error",
                        },
                    )

                return JSONResponse(
                    {
                        "success": True,
                        "results": [
                            {
                                "id": "doc1",
                                "score": 0.95,
                                "title": "Test Document",
                                "content": "Test content",
                                "metadata": {"source": "test"},
                            }
                        ],
                        "total_count": 1,
                        "query_time_ms": 50.0,
                        "search_strategy": "hybrid",
                        "cache_hit": False,
                        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
                    }
                )

            return JSONResponse(
                status_code=405, content={"error": "Method not allowed"}
            )

        async def documents_endpoint(request):
            """Mock documents endpoint."""
            if request.method == "POST":
                body = await request.json()
                url = body.get("url")

                if not url:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "error": "URL is required",
                            "error_type": "validation_error",
                        },
                    )

                return JSONResponse(
                    status_code=201,
                    content={
                        "success": True,
                        "document_id": "doc_123",
                        "url": url,
                        "chunks_created": 5,
                        "processing_time_ms": 1500.0,
                        "status": "processed",
                        "timestamp": datetime.now(tz=timezone.utc).timestamp(),
                    },
                )

            return JSONResponse(
                status_code=405, content={"error": "Method not allowed"}
            )

        async def health_endpoint(request):
            """Mock health endpoint."""
            return JSONResponse(
                {
                    "success": True,
                    "status": "healthy",
                    "services": {"qdrant": "healthy", "redis": "healthy"},
                    "uptime_seconds": 3600.0,
                    "version": "1.0.0",
                    "timestamp": datetime.now(tz=timezone.utc).timestamp(),
                }
            )

        routes = [
            Route("/api/search", search_endpoint, methods=["POST"]),
            Route("/api/documents", documents_endpoint, methods=["POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ]

        return Starlette(routes=routes)

    @pytest.fixture
    def client(self, mock_api_app):
        """Create test client."""
        return TestClient(mock_api_app)

    @pytest.mark.api_contract
    def test_search_endpoint_contract(
        self, client, api_contract_validator, contract_test_data
    ):
        """Test search endpoint contract validation."""
        # Register contract
        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        # Test valid request
        request_data = {"query": "machine learning", "limit": 10}

        # Validate request against contract
        request_validation = api_contract_validator.validate_request(
            "/api/search", "POST", json=request_data
        )
        assert request_validation["valid"], (
            f"Request validation failed: {request_validation['errors']}"
        )

        # Make actual request
        response = client.post("/api/search", json=request_data)
        assert response.status_code == 200

        # Validate response against contract
        response_validation = api_contract_validator.validate_response(
            "/api/search", "POST", response.status_code, response.json()
        )
        assert response_validation["valid"], (
            f"Response validation failed: {response_validation['errors']}"
        )

    @pytest.mark.api_contract
    def test_search_endpoint_error_contract(
        self, client, api_contract_validator, contract_test_data
    ):
        """Test search endpoint error response contract."""
        # Register contract
        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        # Test invalid request (empty query)
        request_data = {"query": ""}

        # Make request that should fail
        response = client.post("/api/search", json=request_data)
        assert response.status_code == 400

        # Validate error response
        response_validation = api_contract_validator.validate_response(
            "/api/search", "POST", response.status_code, response.json()
        )
        assert response_validation["valid"], (
            f"Error response validation failed: {response_validation['errors']}"
        )

        # Check error structure
        error_data = response.json()
        assert "error" in error_data
        assert error_data["success"] is False

    @pytest.mark.api_contract
    def test_documents_endpoint_contract(
        self, client, api_contract_validator, contract_test_data
    ):
        """Test documents endpoint contract validation."""
        # Register contract
        api_contract_validator.register_contract(
            "/api/documents", contract_test_data["api_contracts"]["/api/documents"]
        )

        # Test valid request
        request_data = {
            "url": "https://example.com/document",
            "collection": "test_docs",
            "metadata": {"type": "tutorial"},
        }

        # Validate request
        request_validation = api_contract_validator.validate_request(
            "/api/documents", "POST", json=request_data
        )
        assert request_validation["valid"], (
            f"Request validation failed: {request_validation['errors']}"
        )

        # Make request
        response = client.post("/api/documents", json=request_data)
        assert response.status_code == 201

        # Validate response
        response_validation = api_contract_validator.validate_response(
            "/api/documents", "POST", response.status_code, response.json()
        )
        assert response_validation["valid"], (
            f"Response validation failed: {response_validation['errors']}"
        )

    @pytest.mark.api_contract
    def test_health_endpoint_contract(self, client):
        """Test health endpoint contract."""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()

        # Validate health response structure
        required_fields = ["success", "status", "services", "timestamp"]
        for field in required_fields:
            assert field in health_data, f"Missing required field: {field}"

        assert health_data["success"] is True
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(health_data["services"], dict)
        assert len(health_data["services"]) > 0


class TestContractVersioning:
    """Test API contract versioning and compatibility."""

    @pytest.mark.api_contract
    def test_api_version_compatibility(self, api_contract_validator):
        """Test API version compatibility."""
        # Version 1 contract
        v1_contract = {
            "GET": {
                "parameters": [{"name": "q", "type": "string", "required": True}],
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "results": {"type": "array"},
                                "total": {"type": "integer"},
                            },
                            "required": ["results", "total"],
                        }
                    }
                },
            }
        }

        # Version 2 contract (backward compatible)
        v2_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {
                        "name": "strategy",
                        "type": "string",
                        "required": False,
                    },  # New optional param
                ],
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},  # New field
                                "results": {"type": "array"},
                                "total_count": {
                                    "type": "integer"
                                },  # Renamed from 'total'
                                "search_strategy": {"type": "string"},  # New field
                            },
                            "required": ["success", "results", "total_count"],
                        }
                    }
                },
            }
        }

        # Register v1 contract
        api_contract_validator.register_contract("/api/v1/search", v1_contract)

        # Test v1 request
        v1_request = {"q": "test query"}
        v1_validation = api_contract_validator.validate_request(
            "/api/v1/search", "GET", params=v1_request
        )
        assert v1_validation["valid"]

        # Register v2 contract
        api_contract_validator.register_contract("/api/v2/search", v2_contract)

        # Test v2 request (backward compatible)
        v2_request = {"q": "test query", "strategy": "hybrid"}
        v2_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=v2_request
        )
        assert v2_validation["valid"]

        # Test v1 request on v2 endpoint (should still work)
        v1_on_v2_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=v1_request
        )
        assert v1_on_v2_validation["valid"]

    @pytest.mark.api_contract
    def test_breaking_change_detection(self, api_contract_validator):
        """Test detection of breaking changes in API contracts."""
        # Original contract
        original_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {"results": {"type": "array"}},
                            "required": ["results"],
                        }
                    }
                },
            }
        }

        # Breaking change contract (new required field)
        breaking_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                            "api_key": {"type": "string"},  # New field
                        },
                        "required": ["query", "api_key"],  # Made api_key required
                    },
                },
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {"results": {"type": "array"}},
                            "required": ["results"],
                        }
                    }
                },
            }
        }

        # Register original contract
        api_contract_validator.register_contract("/api/search", original_contract)

        # Request that was valid in original
        request_data = {"query": "test"}

        # Should be valid with original contract
        original_validation = api_contract_validator.validate_request(
            "/api/search", "POST", json=request_data
        )
        assert original_validation["valid"]

        # Register breaking contract
        api_contract_validator.register_contract("/api/search", breaking_contract)

        # Same request should now be invalid
        breaking_validation = api_contract_validator.validate_request(
            "/api/search", "POST", json=request_data
        )
        assert not breaking_validation["valid"]
        assert any("api_key" in error for error in breaking_validation["errors"])


class TestCrossServiceContracts:
    """Test contracts between different services."""

    @pytest.mark.api_contract
    def test_embedding_service_contract(self, api_contract_validator):
        """Test embedding service contract."""
        embedding_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}},
                            "model": {"type": "string"},
                            "batch_size": {"type": "integer"},
                        },
                        "required": ["texts"],
                    },
                },
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "embeddings": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                },
                                "model_used": {"type": "string"},
                                "processing_time_ms": {"type": "number"},
                            },
                            "required": ["embeddings", "model_used"],
                        }
                    }
                },
            }
        }

        api_contract_validator.register_contract("/api/embeddings", embedding_contract)

        # Test valid embedding request
        request_data = {
            "texts": ["hello world", "test document"],
            "model": "text-embedding-3-small",
            "batch_size": 100,
        }

        validation = api_contract_validator.validate_request(
            "/api/embeddings", "POST", json=request_data
        )
        assert validation["valid"]

        # Test response validation
        response_data = {
            "embeddings": [
                [0.1, 0.2, 0.3],  # First embedding
                [0.4, 0.5, 0.6],  # Second embedding
            ],
            "model_used": "text-embedding-3-small",
            "processing_time_ms": 150.0,
        }

        response_validation = api_contract_validator.validate_response(
            "/api/embeddings", "POST", 200, response_data
        )
        assert response_validation["valid"]

    @pytest.mark.api_contract
    def test_vector_db_service_contract(self, api_contract_validator):
        """Test vector database service contract."""
        vector_db_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "collection_name": {"type": "string"},
                            "vectors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "vector": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                        "payload": {"type": "object"},
                                    },
                                    "required": ["id", "vector"],
                                },
                            },
                        },
                        "required": ["collection_name", "vectors"],
                    },
                },
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "operation_id": {"type": "string"},
                                "status": {"type": "string"},
                                "points_inserted": {"type": "integer"},
                            },
                            "required": ["operation_id", "status"],
                        }
                    }
                },
            }
        }

        api_contract_validator.register_contract(
            "/api/vectors/upsert", vector_db_contract
        )

        # Test valid vector upsert request
        request_data = {
            "collection_name": "documents",
            "vectors": [
                {
                    "id": "doc1",
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "payload": {"title": "Test Document"},
                }
            ],
        }

        validation = api_contract_validator.validate_request(
            "/api/vectors/upsert", "POST", json=request_data
        )
        assert validation["valid"]


class TestContractDocumentation:
    """Test contract documentation generation and validation."""

    @pytest.mark.api_contract
    def test_generate_contract_documentation(
        self, openapi_contract_manager, contract_test_data
    ):
        """Test generation of contract documentation."""
        # Load OpenAPI spec
        spec = contract_test_data["openapi_spec"]
        openapi_contract_manager.load_spec("api_docs", spec)

        # Generate endpoints documentation
        endpoints = openapi_contract_manager.extract_endpoints("api_docs")

        # Verify documentation structure
        assert len(endpoints) > 0

        for endpoint in endpoints:
            assert "path" in endpoint
            assert "method" in endpoint
            assert "responses" in endpoint

            # Check that responses have proper structure
            responses = endpoint["responses"]
            assert len(responses) > 0

            # Should have at least one successful response
            success_responses = [
                status for status in responses if status.startswith("2")
            ]
            assert len(success_responses) > 0

    @pytest.mark.api_contract
    def test_contract_test_generation(
        self, openapi_contract_manager, contract_test_data
    ):
        """Test automatic contract test generation."""
        # Load OpenAPI spec
        spec = contract_test_data["openapi_spec"]
        openapi_contract_manager.load_spec("test_gen", spec)

        # Generate test cases
        test_cases = openapi_contract_manager.generate_contract_tests("test_gen")

        # Verify test case structure
        assert len(test_cases) > 0

        positive_tests = [tc for tc in test_cases if tc["type"] == "positive"]
        negative_tests = [tc for tc in test_cases if tc["type"] == "negative"]

        # Should have both positive and negative test cases
        assert len(positive_tests) > 0
        assert len(negative_tests) > 0

        # Verify test case content
        for test_case in test_cases:
            assert "endpoint" in test_case
            assert "method" in test_case
            assert "description" in test_case
            assert "test_data" in test_case

            if test_case["type"] == "negative":
                assert "expected_status" in test_case
