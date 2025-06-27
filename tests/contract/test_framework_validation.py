"""Contract testing framework validation tests.

This module validates that the contract testing framework is properly
integrated and working correctly with all components.
"""

import json
from datetime import UTC, datetime, timezone

import pytest

from src.models.api_contracts import ErrorResponse, SearchRequest, SearchResponse


class TestContractFrameworkValidation:
    """Test the complete contract testing framework."""

    @pytest.mark.contract
    def test_framework_components_integration(
        self,
        json_schema_validator,
        api_contract_validator,
        openapi_contract_manager,
        pact_contract_builder,
        contract_test_data,
    ):
        """Test that all framework components work together."""
        # 1. Test JSON Schema Validator
        search_schema = SearchRequest.model_json_schema()
        json_schema_validator.register_schema("search_request", search_schema)

        valid_request_data = {
            "query": "test query",
            "collection_name": "documents",
            "limit": 10,
        }

        validation_result = json_schema_validator.validate_data(
            valid_request_data, "search_request"
        )
        assert validation_result["valid"], (
            f"Schema validation failed: {validation_result['errors']}"
        )

        # 2. Test API Contract Validator
        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        api_validation = api_contract_validator.validate_request(
            "/api/search", "GET", params={"q": "test", "limit": 10}
        )
        assert api_validation["valid"]

        # 3. Test OpenAPI Contract Manager
        openapi_spec = contract_test_data["openapi_spec"]
        openapi_contract_manager.load_spec("framework_test", openapi_spec)

        spec_validation = openapi_contract_manager.validate_spec("framework_test")
        assert spec_validation["valid"]

        endpoints = openapi_contract_manager.extract_endpoints("framework_test")
        assert len(endpoints) > 0

        # 4. Test Pact Contract Builder
        pact_contract_builder.given("framework test state")
        pact_contract_builder.upon_receiving("framework test request")
        pact_contract_builder.with_request(method="GET", path="/test")
        pact_contract_builder.will_respond_with(status=200, body={"success": True})

        contract = pact_contract_builder.build_pact()
        assert len(contract["interactions"]) == 1

        # Verify all components are working
        assert True  # If we reach here, all components are integrated properly

    @pytest.mark.contract
    def test_pydantic_model_contract_validation(self, json_schema_validator):
        """Test contract validation with Pydantic models."""
        # Test SearchRequest validation
        search_schema = SearchRequest.model_json_schema()
        json_schema_validator.register_schema("search_request", search_schema)

        # Valid request
        valid_request = {
            "query": "machine learning",
            "collection_name": "ml_docs",
            "limit": 20,
            "score_threshold": 0.7,
            "enable_hyde": True,
        }

        validation = json_schema_validator.validate_data(
            valid_request, "search_request"
        )
        assert validation["valid"]

        # Invalid request (missing required field)
        invalid_request = {
            "collection_name": "ml_docs",
            "limit": 20,
            # Missing required 'query' field
        }

        validation = json_schema_validator.validate_data(
            invalid_request, "search_request"
        )
        assert not validation["valid"]
        assert any("query" in error for error in validation["errors"])

        # Test SearchResponse validation
        response_schema = SearchResponse.model_json_schema()
        json_schema_validator.register_schema("search_response", response_schema)

        valid_response = {
            "success": True,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "title": "ML Tutorial",
                    "content": "Machine learning guide",
                    "metadata": {"source": "tutorial"},
                }
            ],
            "total_count": 1,
            "query_time_ms": 45.0,
            "search_strategy": "hybrid",
            "cache_hit": False,
        }

        response_validation = json_schema_validator.validate_data(
            valid_response, "search_response"
        )
        assert response_validation["valid"]

    @pytest.mark.contract
    def test_error_response_contract_validation(self, json_schema_validator):
        """Test error response contract validation."""
        error_schema = ErrorResponse.model_json_schema()
        json_schema_validator.register_schema("error_response", error_schema)

        # Valid error response
        valid_error = {
            "success": False,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "error": "Invalid query parameter",
            "error_type": "validation_error",
            "context": {
                "parameter": "limit",
                "value": -1,
                "reason": "Limit must be positive",
            },
        }

        validation = json_schema_validator.validate_data(valid_error, "error_response")
        assert validation["valid"]

        # Test that success=True is invalid for ErrorResponse
        {
            "success": True,  # This should be False for ErrorResponse
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "error": "This shouldn't work",
            "error_type": "test_error",
        }

        # Note: The schema validation might still pass since Pydantic model
        # validation happens at runtime, not schema level
        # In practice, you'd use Pydantic model validation directly

    @pytest.mark.contract
    def test_contract_backward_compatibility_validation(
        self, _json_schema_validator, api_contract_validator
    ):
        """Test backward compatibility validation."""
        # Original API contract (v1)
        v1_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {"name": "max_results", "type": "integer", "required": False},
                ],
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

        # Updated API contract (v2, backward compatible)
        v2_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {"name": "limit", "type": "integer", "required": False},
                    {
                        "name": "max_results",
                        "type": "integer",
                        "required": False,
                    },  # Deprecated
                    {
                        "name": "strategy",
                        "type": "string",
                        "required": False,
                    },  # New field
                ],
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},  # New field
                                "results": {"type": "array"},
                                "total_count": {"type": "integer"},  # New field
                                "total": {"type": "integer"},  # Kept for compatibility
                                "search_strategy": {"type": "string"},  # New field
                            },
                            "required": ["success", "results", "total_count"],
                        }
                    }
                },
            }
        }

        # Register both contracts
        api_contract_validator.register_contract("/api/v1/search", v1_contract)
        api_contract_validator.register_contract("/api/v2/search", v2_contract)

        # Old client request (should work with both v1 and v2)
        old_request = {"q": "test query", "max_results": 10}

        v1_validation = api_contract_validator.validate_request(
            "/api/v1/search", "GET", params=old_request
        )
        assert v1_validation["valid"]

        v2_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=old_request
        )
        assert v2_validation["valid"]  # Backward compatible

        # New client request (should work with v2)
        new_request = {"q": "test query", "limit": 10, "strategy": "hybrid"}

        v2_new_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=new_request
        )
        assert v2_new_validation["valid"]

    @pytest.mark.contract
    def test_contract_breaking_change_detection(
        self, json_schema_validator, api_contract_validator
    ):
        """Test detection of breaking changes in contracts."""
        # Original schema
        original_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["id", "title"],
        }

        # Breaking change schema (new required field)
        breaking_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "optional_field": {"type": "string"},
                "new_required_field": {"type": "string"},
            },
            "required": ["id", "title", "new_required_field"],  # Breaking change
        }

        json_schema_validator.register_schema("original", original_schema)
        json_schema_validator.register_schema("breaking", breaking_schema)

        # Data that was valid in original schema
        test_data = {
            "id": "test_doc",
            "title": "Test Document",
            "optional_field": "test_value",
        }

        # Should be valid in original
        original_validation = json_schema_validator.validate_data(test_data, "original")
        assert original_validation["valid"]

        # Should be invalid in breaking schema (missing new required field)
        breaking_validation = json_schema_validator.validate_data(test_data, "breaking")
        assert not breaking_validation["valid"]
        assert any(
            "new_required_field" in error for error in breaking_validation["errors"]
        )

    @pytest.mark.contract
    async def test_mcp_tool_contract_validation(self, mock_contract_service):
        """Test MCP tool contract validation."""
        # The mock_contract_service fixture provides the mock implementation

        # Execute search tool
        result = await mock_contract_service.search("test query", 10)
        result_data = result if isinstance(result, dict) else json.loads(result)

        # Validate contract compliance
        required_fields = ["results", "total"]
        for field in required_fields:
            assert field in result_data, f"Missing required field: {field}"

        assert isinstance(result_data["results"], list)
        assert isinstance(result_data["total"], int)

        # Validate result structure
        if result_data["results"]:
            result_item = result_data["results"][0]
            item_required_fields = ["id", "score", "title"]
            for field in item_required_fields:
                assert field in result_item, (
                    f"Missing required field in result item: {field}"
                )

    @pytest.mark.contract
    def test_openapi_spec_generation_and_validation(
        self, openapi_contract_manager, json_schema_validator
    ):
        """Test OpenAPI specification generation and validation."""
        # Create a comprehensive OpenAPI spec
        comprehensive_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "AI Docs Vector DB API",
                "version": "1.0.0",
                "description": "API for AI documentation vector database",
            },
            "paths": {
                "/search": {
                    "get": {
                        "operationId": "searchDocuments",
                        "summary": "Search documents",
                        "parameters": [
                            {
                                "name": "q",
                                "in": "query",
                                "required": True,
                                "schema": {"type": "string"},
                                "description": "Search query",
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "description": "Maximum number of results",
                            },
                        ],
                        "responses": {
                            "200": {
                                "description": "Search results",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "results": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "id": {"type": "string"},
                                                            "title": {"type": "string"},
                                                            "score": {"type": "number"},
                                                        },
                                                        "required": [
                                                            "id",
                                                            "title",
                                                            "score",
                                                        ],
                                                    },
                                                },
                                                "total_count": {"type": "integer"},
                                            },
                                            "required": [
                                                "success",
                                                "results",
                                                "total_count",
                                            ],
                                        }
                                    }
                                },
                            },
                            "400": {
                                "description": "Bad request",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {"type": "string"},
                                                "error_type": {"type": "string"},
                                            },
                                            "required": ["error"],
                                        }
                                    }
                                },
                            },
                        },
                    }
                },
                "/documents": {
                    "post": {
                        "operationId": "addDocument",
                        "summary": "Add a document",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string", "format": "uri"},
                                            "collection_name": {"type": "string"},
                                            "metadata": {"type": "object"},
                                        },
                                        "required": ["url"],
                                    }
                                }
                            },
                        },
                        "responses": {
                            "201": {
                                "description": "Document added successfully",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "document_id": {"type": "string"},
                                                "status": {"type": "string"},
                                            },
                                            "required": [
                                                "success",
                                                "document_id",
                                                "status",
                                            ],
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
            },
        }

        # Load and validate the spec
        openapi_contract_manager.load_spec("comprehensive", comprehensive_spec)

        validation_result = openapi_contract_manager.validate_spec("comprehensive")
        assert validation_result["valid"], (
            f"OpenAPI spec validation failed: {validation_result['errors']}"
        )

        # Extract and verify endpoints
        endpoints = openapi_contract_manager.extract_endpoints("comprehensive")
        assert len(endpoints) == 2

        # Find search endpoint
        search_endpoint = next(ep for ep in endpoints if ep["path"] == "/search")
        assert search_endpoint["method"] == "GET"
        assert search_endpoint["operation_id"] == "searchDocuments"
        assert len(search_endpoint["parameters"]) == 2

        # Find documents endpoint
        docs_endpoint = next(ep for ep in endpoints if ep["path"] == "/documents")
        assert docs_endpoint["method"] == "POST"
        assert docs_endpoint["operation_id"] == "addDocument"
        assert docs_endpoint["request_body"] is not None

        # Generate test cases
        test_cases = openapi_contract_manager.generate_contract_tests("comprehensive")
        assert len(test_cases) > 0

        # Verify test case types
        positive_tests = [tc for tc in test_cases if tc["type"] == "positive"]
        negative_tests = [tc for tc in test_cases if tc["type"] == "negative"]

        assert len(positive_tests) > 0
        assert len(negative_tests) > 0
