"""OpenAPI specification validation tests.

This module tests OpenAPI specification compliance and validation
for all FastAPI and MCP endpoints.
"""

import json
from typing import Dict

import pytest
import schemathesis
from openapi_spec_validator import validate_spec
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError
from starlette.testclient import TestClient

from src.config import get_config
from src.services.fastapi.production_server import create_production_server


class TestOpenAPISpecValidation:
    """Test OpenAPI specification validation."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        config = get_config()
        server = create_production_server(config)
        return server.create_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def openapi_spec(self, client):
        """Get OpenAPI specification from the app."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        return response.json()

    @pytest.mark.openapi
    def test_openapi_spec_is_valid(self, openapi_spec):
        """Test that OpenAPI specification is valid."""
        try:
            validate_spec(openapi_spec)
        except OpenAPIValidationError as e:
            pytest.fail(f"OpenAPI specification is invalid: {e}")

    @pytest.mark.openapi
    def test_openapi_spec_structure(self, openapi_spec):
        """Test OpenAPI specification structure."""
        # Required fields
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec

        # OpenAPI version
        assert openapi_spec["openapi"].startswith("3.")

        # Info object
        info = openapi_spec["info"]
        assert "title" in info
        assert "version" in info

        # Paths should not be empty for production API
        assert len(openapi_spec["paths"]) > 0

    @pytest.mark.openapi
    def test_openapi_spec_has_required_endpoints(self, openapi_spec):
        """Test that OpenAPI spec includes required endpoints."""
        paths = openapi_spec["paths"]

        # Health check endpoint should be present
        assert "/health" in paths

        # Health endpoint should support GET
        assert "get" in paths["/health"]

    @pytest.mark.openapi
    def test_openapi_response_schemas(self, openapi_spec):
        """Test that all responses have proper schemas."""
        paths = openapi_spec["paths"]

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    responses = operation.get("responses", {})
                    assert len(responses) > 0, (
                        f"No responses defined for {method.upper()} {path}"
                    )

                    # Check that 200 responses have content schemas
                    if "200" in responses:
                        response_200 = responses["200"]
                        if "content" in response_200:
                            content = response_200["content"]
                            for media_type, media_schema in content.items():
                                if media_type == "application/json":
                                    assert "schema" in media_schema, (
                                        f"Missing schema for {method.upper()} {path} 200 response"
                                    )

    @pytest.mark.openapi
    def test_openapi_request_schemas(self, openapi_spec):
        """Test that request bodies have proper schemas."""
        paths = openapi_spec["paths"]

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ["post", "put", "patch"]:
                    request_body = operation.get("requestBody")
                    if request_body:
                        content = request_body.get("content", {})
                        for media_type, media_schema in content.items():
                            if media_type == "application/json":
                                assert "schema" in media_schema, (
                                    f"Missing request schema for {method.upper()} {path}"
                                )


class TestSchemathesisValidation:
    """Test using Schemathesis for property-based API testing."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        config = get_config()
        server = create_production_server(config)
        return server.create_app()

    @pytest.fixture
    def schema(self, app):
        """Create Schemathesis schema from FastAPI app."""
        return schemathesis.from_asgi("/openapi.json", app)

    @pytest.mark.openapi
    @pytest.mark.slow
    def test_api_contract_fuzzing(self, schema):
        """Test API endpoints using property-based fuzzing."""

        @schema.parametrize()
        @schemathesis.settings(max_examples=10)  # Limit for CI performance
        def test_endpoint(case):
            """Test individual endpoint cases."""
            response = case.call_asgi()

            # Basic contract validation
            case.validate_response(response)

            # Additional assertions
            assert response.status_code < 500, f"Server error: {response.status_code}"

            # Validate response content type for successful responses
            if 200 <= response.status_code < 300:
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("application/json"):
                    try:
                        json.loads(response.content)
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON response: {response.content}")

        # Run the test
        test_endpoint()


class TestMCPContractValidation:
    """Test MCP protocol contract validation."""

    @pytest.mark.contract
    @pytest.mark.mcp
    def test_mcp_tool_schemas(self, mock_contract_service):
        """Test that MCP tools have valid schemas."""
        # This would test the MCP tool definitions
        # For now, we'll test basic structure
        assert hasattr(mock_contract_service, "search")
        assert hasattr(mock_contract_service, "add_document")

    @pytest.mark.contract
    @pytest.mark.mcp
    def test_mcp_request_response_contracts(
        self, json_schema_validator, contract_test_data
    ):
        """Test MCP request/response contracts."""
        # Register schemas
        json_schema_validator.register_schema(
            "search_result", contract_test_data["json_schemas"]["search_result"]
        )
        json_schema_validator.register_schema(
            "document_input", contract_test_data["json_schemas"]["document_input"]
        )

        # Test valid search result
        valid_result = {
            "id": "doc1",
            "title": "Test Document",
            "score": 0.95,
            "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"},
        }

        validation_result = json_schema_validator.validate_data(
            valid_result, "search_result"
        )
        assert validation_result["valid"], (
            f"Validation errors: {validation_result['errors']}"
        )

        # Test invalid search result (missing required field)
        invalid_result = {"title": "Test Document", "score": 0.95}

        validation_result = json_schema_validator.validate_data(
            invalid_result, "search_result"
        )
        assert not validation_result["valid"]
        assert any("id" in error for error in validation_result["errors"])


class TestAPIContractEvolution:
    """Test API contract evolution and backward compatibility."""

    @pytest.mark.contract
    def test_backward_compatibility(self, openapi_contract_manager, contract_test_data):
        """Test backward compatibility of API contracts."""
        # Load current spec
        current_spec = contract_test_data["openapi_spec"]
        openapi_contract_manager.load_spec("current", current_spec)

        # Validate current spec
        validation_result = openapi_contract_manager.validate_spec("current")
        assert validation_result["valid"], (
            f"Current spec invalid: {validation_result['errors']}"
        )

        # Extract endpoints
        endpoints = openapi_contract_manager.extract_endpoints("current")
        assert len(endpoints) > 0

        # Verify search endpoint exists
        search_endpoints = [ep for ep in endpoints if "/search" in ep["path"]]
        assert len(search_endpoints) > 0, "Search endpoint missing"

    @pytest.mark.contract
    def test_contract_versioning(self, api_contract_validator, contract_test_data):
        """Test API contract versioning."""
        # Register v1 contract
        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        # Test valid v1 request
        validation_result = api_contract_validator.validate_request(
            "/api/search", "GET", params={"q": "test query", "limit": 10}
        )
        assert validation_result["valid"], (
            f"Request validation failed: {validation_result['errors']}"
        )

        # Test v1 response
        response_data = {
            "results": [{"id": "doc1", "title": "Test", "score": 0.95}],
            "total": 1,
        }

        validation_result = api_contract_validator.validate_response(
            "/api/search", "GET", 200, response_data
        )
        assert validation_result["valid"], (
            f"Response validation failed: {validation_result['errors']}"
        )


class TestContractBreakingChanges:
    """Test detection of contract breaking changes."""

    @pytest.mark.contract
    def test_detect_breaking_changes(self, json_schema_validator):
        """Test detection of breaking changes in schemas."""
        # Original schema
        original_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["id", "title"],
        }

        # Non-breaking change (add optional field)
        non_breaking_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
                "description": {"type": "string"},  # Optional
            },
            "required": ["id", "title"],
        }

        # Breaking change (add required field)
        breaking_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
                "required_new_field": {"type": "string"},
            },
            "required": ["id", "title", "required_new_field"],
        }

        # Test data that was valid in original
        test_data = {"id": "doc1", "title": "Test Document", "score": 0.95}

        # Should be valid in original
        result = json_schema_validator.validate_against_schema(
            test_data, original_schema
        )
        assert result["valid"]

        # Should still be valid in non-breaking change
        result = json_schema_validator.validate_against_schema(
            test_data, non_breaking_schema
        )
        assert result["valid"]

        # Should be invalid in breaking change
        result = json_schema_validator.validate_against_schema(
            test_data, breaking_schema
        )
        assert not result["valid"]
        assert any("required_new_field" in error for error in result["errors"])

    @pytest.mark.contract
    def test_api_parameter_changes(self, api_contract_validator):
        """Test detection of breaking parameter changes."""
        # Original contract with optional parameter
        original_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {"name": "limit", "type": "integer", "required": False},
                ],
                "responses": {"200": {"schema": {"type": "object"}}},
            }
        }

        # Breaking change contract (make optional parameter required)
        breaking_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {
                        "name": "limit",
                        "type": "integer",
                        "required": True,
                    },  # Now required
                ],
                "responses": {"200": {"schema": {"type": "object"}}},
            }
        }

        api_contract_validator.register_contract("/api/test", original_contract)

        # Request that was valid with original contract
        request_params = {"q": "test query"}  # Missing limit

        # Should be valid with original
        result = api_contract_validator.validate_request(
            "/api/test", "GET", params=request_params
        )
        assert result["valid"]

        # Update to breaking contract
        api_contract_validator.register_contract("/api/test", breaking_contract)

        # Should now be invalid
        result = api_contract_validator.validate_request(
            "/api/test", "GET", params=request_params
        )
        assert not result["valid"]
        assert any("limit" in error for error in result["errors"])
