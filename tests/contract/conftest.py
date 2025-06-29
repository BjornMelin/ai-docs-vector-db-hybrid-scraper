"""Contract testing fixtures and configuration.

This module provides pytest fixtures for comprehensive contract testing including
API contract validation, schema validation, consumer-driven contract testing,
OpenAPI specification testing, and Pact integration.
"""

from typing import Any
from unittest.mock import AsyncMock

import jsonschema
import pytest


class JSONSchemaValidator:
    def __init__(self):
        self.schemas = {}

    def register_schema(self, name: str, schema: dict[str, Any]):
        """Register a JSON schema for validation."""
        self.schemas[name] = schema

    def validate_data(self, data: Any, schema_name: str) -> dict[str, Any]:
        """Validate data against a registered schema."""
        if schema_name not in self.schemas:
            return {
                "valid": False,
                "errors": [f"Schema '{schema_name}' not found"],
            }

        schema = self.schemas[schema_name]
        try:
            jsonschema.validate(data, schema)
            return {"valid": True, "errors": []}
        except jsonschema.ValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "path": list(e.absolute_path),
                "schema_path": list(e.schema_path),
            }
        except jsonschema.SchemaError as e:
            return {
                "valid": False,
                "errors": [f"Invalid schema: {e!s}"],
            }

    def validate_against_schema(
        self, data: Any, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate data against a provided schema."""
        try:
            jsonschema.validate(data, schema)
            return {"valid": True, "errors": []}
        except jsonschema.ValidationError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "path": list(e.absolute_path),
                "schema_path": list(e.schema_path),
                "failed_value": e.instance,
            }
        except jsonschema.SchemaError as e:
            return {
                "valid": False,
                "errors": [f"Invalid schema: {e!s}"],
            }

    def generate_test_data(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate test data based on schema."""
        test_cases = []

        # Generate valid test case
        valid_data = self._generate_valid_data(schema)
        test_cases.append(
            {
                "type": "valid",
                "data": valid_data,
                "expected_valid": True,
            }
        )

        # Generate invalid test cases
        invalid_cases = self._generate_invalid_data(schema)
        test_cases.extend(invalid_cases)

        return test_cases

    def _generate_valid_data(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate valid data for schema."""
        data = {}
        properties = schema.get("properties", {})

        for prop_name, prop_schema in properties.items():
            if prop_schema.get("type") == "string":
                if "enum" in prop_schema:
                    data[prop_name] = prop_schema["enum"][0]
                else:
                    data[prop_name] = "test_value"
            elif prop_schema.get("type") == "integer":
                minimum = prop_schema.get("minimum", 0)
                maximum = prop_schema.get("maximum", 100)
                data[prop_name] = minimum + (maximum - minimum) // 2
            elif prop_schema.get("type") == "number":
                data[prop_name] = 50.5
            elif prop_schema.get("type") == "boolean":
                data[prop_name] = True
            elif prop_schema.get("type") == "array":
                data[prop_name] = []
            elif prop_schema.get("type") == "object":
                data[prop_name] = {}

        return data

    def _generate_invalid_data(self, schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate invalid test cases for schema."""
        invalid_cases = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Missing required field case
        if required:
            incomplete_data = self._generate_valid_data(schema)
            del incomplete_data[required[0]]
            invalid_cases.append(
                {
                    "type": "missing_required",
                    "data": incomplete_data,
                    "expected_valid": False,
                    "expected_error": f"'{required[0]}' is a required property",
                }
            )

        # Type mismatch cases
        for prop_name, prop_schema in properties.items():
            if prop_schema.get("type") == "string":
                invalid_data = self._generate_valid_data(schema)
                invalid_data[prop_name] = 12345  # Wrong type
                invalid_cases.append(
                    {
                        "type": "type_mismatch",
                        "data": invalid_data,
                        "expected_valid": False,
                        "property": prop_name,
                    }
                )

        return invalid_cases


@pytest.fixture(scope="session")
def contract_test_config():
    """Provide contract testing configuration."""
    return {
        "api_contracts": {
            "base_url": "http://localhost:8000",
            "timeout": 30,
            "retry_attempts": 3,
            "versions": ["v1", "v2"],
        },
        "schema_validation": {
            "strict_mode": True,
            "allow_additional_properties": False,
            "validate_formats": True,
        },
        "pact": {
            "consumer_name": "ai-docs-consumer",
            "provider_name": "ai-docs-provider",
            "pact_dir": "./pacts",
            "broker_url": None,  # Set to enable Pact Broker
        },
        "openapi": {
            "spec_version": "3.0.3",
            "validate_responses": True,
            "validate_requests": True,
            "strict_validation": True,
        },
        "consumer_driven": {
            "contract_formats": ["pact", "openapi", "json_schema"],
            "auto_generate_contracts": True,
            "contract_versioning": True,
        },
    }


@pytest.fixture
def _json_schema_validator():
    """JSON schema validation utilities."""
    return JSONSchemaValidator()


@pytest.fixture
def api_contract_validator():
    """API contract validation utilities."""

    class APIContractValidator:
        def __init__(self):
            self.contracts = {}
            self.base_url = "http://localhost:8000"

        def register_contract(self, endpoint: str, contract: dict[str, Any]):
            """Register an API contract for validation."""
            self.contracts[endpoint] = contract

        def validate_request(
            self, endpoint: str, method: str, **_kwargs
        ) -> dict[str, Any]:
            """Validate a request against its contract."""
            if endpoint not in self.contracts:
                return {
                    "valid": False,
                    "errors": [f"No contract found for endpoint: {endpoint}"],
                }

            contract = self.contracts[endpoint]
            method_contract = contract.get(method.upper())

            if not method_contract:
                return {
                    "valid": False,
                    "errors": [f"Method {method} not allowed for endpoint {endpoint}"],
                }

            errors = []

            # Validate request parameters
            if "parameters" in method_contract:
                param_errors = self._validate_parameters(
                    _kwargs.get("params", {}), method_contract["parameters"]
                )
                errors.extend(param_errors)

            # Validate request body
            if "requestBody" in method_contract:
                body_errors = self._validate_request_body(
                    _kwargs.get("json", _kwargs.get("data")),
                    method_contract["requestBody"],
                )
                errors.extend(body_errors)

            # Validate headers
            if "headers" in method_contract:
                header_errors = self._validate_headers(
                    _kwargs.get("headers", {}), method_contract["headers"]
                )
                errors.extend(header_errors)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
            }

        def validate_response(
            self,
            endpoint: str,
            method: str,
            status_code: int,
            response_data: Any,
            headers: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            """Validate a response against its contract."""
            if endpoint not in self.contracts:
                return {
                    "valid": False,
                    "errors": [f"No contract found for endpoint: {endpoint}"],
                }

            contract = self.contracts[endpoint]
            method_contract = contract.get(method.upper())

            if not method_contract:
                return {
                    "valid": False,
                    "errors": [
                        f"Method {method} not in contract for endpoint {endpoint}"
                    ],
                }

            responses = method_contract.get("responses", {})
            response_contract = responses.get(str(status_code))

            if not response_contract:
                # Check for default response
                response_contract = responses.get("default")
                if not response_contract:
                    return {
                        "valid": False,
                        "errors": [f"No contract for status code {status_code}"],
                    }

            errors = []

            # Validate response schema
            if "schema" in response_contract:
                validator = JSONSchemaValidator()
                schema_result = validator.validate_against_schema(
                    response_data, response_contract["schema"]
                )
                if not schema_result["valid"]:
                    errors.extend(schema_result["errors"])

            # Validate response headers
            if "headers" in response_contract and headers:
                header_errors = self._validate_response_headers(
                    headers, response_contract["headers"]
                )
                errors.extend(header_errors)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "status_code": status_code,
            }

        def _validate_parameters(
            self, params: dict[str, Any], param_contracts: list[dict[str, Any]]
        ) -> list[str]:
            """Validate request parameters."""
            errors = []

            for param_contract in param_contracts:
                param_name = param_contract["name"]
                required = param_contract.get("required", False)
                param_type = param_contract.get("type", "string")

                if required and param_name not in params:
                    errors.append(f"Required parameter '{param_name}' is missing")
                elif param_name in params:
                    param_value = params[param_name]

                    # Type validation
                    if param_type == "integer" and not isinstance(param_value, int):
                        try:
                            int(param_value)
                        except (ValueError, TypeError):
                            errors.append(
                                f"Parameter '{param_name}' must be an integer"
                            )
                    elif param_type == "number" and not isinstance(
                        param_value, int | float
                    ):
                        try:
                            float(param_value)
                        except (ValueError, TypeError):
                            errors.append(f"Parameter '{param_name}' must be a number")

            return errors

        def _validate_request_body(
            self, body: Any, body_contract: dict[str, Any]
        ) -> list[str]:
            """Validate request body."""
            errors = []

            if (
                "required" in body_contract
                and body_contract["required"]
                and body is None
            ):
                errors.append("Request body is required but not provided")

            if body is not None and "schema" in body_contract:
                validator = JSONSchemaValidator()
                result = validator.validate_against_schema(
                    body, body_contract["schema"]
                )
                if not result["valid"]:
                    errors.extend(result["errors"])

            return errors

        def _validate_headers(
            self, headers: dict[str, str], header_contracts: dict[str, Any]
        ) -> list[str]:
            """Validate request headers."""
            errors = []

            for header_name, header_contract in header_contracts.items():
                required = header_contract.get("required", False)

                if required and header_name.lower() not in [h.lower() for h in headers]:
                    errors.append(f"Required header '{header_name}' is missing")

            return errors

        def _validate_response_headers(
            self, headers: dict[str, str], header_contracts: dict[str, Any]
        ) -> list[str]:
            """Validate response headers."""
            errors = []

            for header_name, header_contract in header_contracts.items():
                required = header_contract.get("required", False)

                if required and header_name.lower() not in [h.lower() for h in headers]:
                    errors.append(
                        f"Required response header '{header_name}' is missing"
                    )

            return errors

    return APIContractValidator()


@pytest.fixture
def openapi_contract_manager():
    """OpenAPI specification contract management."""

    class OpenAPIContractManager:
        def __init__(self):
            self.specs = {}

        def load_spec(self, name: str, spec: dict[str, Any]):
            """Load an OpenAPI specification."""
            self.specs[name] = spec

        def validate_spec(self, spec_name: str) -> dict[str, Any]:
            """Validate OpenAPI specification format."""
            if spec_name not in self.specs:
                return {
                    "valid": False,
                    "errors": [f"Specification '{spec_name}' not found"],
                }

            spec = self.specs[spec_name]
            errors = []

            # Check required fields
            required_fields = ["openapi", "info", "paths"]
            for field in required_fields:
                if field not in spec:
                    errors.append(f"Required field '{field}' is missing")

            # Validate OpenAPI version
            if "openapi" in spec:
                version = spec["openapi"]
                if not version.startswith("3."):
                    errors.append(f"Unsupported OpenAPI version: {version}")

            # Validate info object
            if "info" in spec:
                info = spec["info"]
                if "title" not in info:
                    errors.append("info.title is required")
                if "version" not in info:
                    errors.append("info.version is required")

            # Validate paths
            if "paths" in spec:
                path_errors = self._validate_paths(spec["paths"])
                errors.extend(path_errors)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "spec_name": spec_name,
            }

        def extract_endpoints(self, spec_name: str) -> list[dict[str, Any]]:
            """Extract all endpoints from OpenAPI specification."""
            if spec_name not in self.specs:
                return []

            spec = self.specs[spec_name]
            endpoints = []

            for path, path_item in spec.get("paths", {}).items():
                for method, operation in path_item.items():
                    if method.lower() in [
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    ]:
                        endpoints.append(
                            {
                                "path": path,
                                "method": method.upper(),
                                "operation_id": operation.get("operationId"),
                                "summary": operation.get("summary"),
                                "description": operation.get("description"),
                                "parameters": operation.get("parameters", []),
                                "request_body": operation.get("requestBody"),
                                "responses": operation.get("responses", {}),
                                "tags": operation.get("tags", []),
                            }
                        )

            return endpoints

        def generate_contract_tests(self, spec_name: str) -> list[dict[str, Any]]:
            """Generate contract test cases from OpenAPI specification."""
            endpoints = self.extract_endpoints(spec_name)
            test_cases = []

            for endpoint in endpoints:
                # Generate positive test cases
                test_cases.append(
                    {
                        "type": "positive",
                        "endpoint": endpoint["path"],
                        "method": endpoint["method"],
                        "description": f"Test {endpoint['method']} {endpoint['path']} with valid data",
                        "test_data": self._generate_valid_test_data(endpoint),
                    }
                )

                # Generate negative test cases
                negative_cases = self._generate_negative_test_cases(endpoint)
                test_cases.extend(negative_cases)

            return test_cases

        def _validate_paths(self, paths: dict[str, Any]) -> list[str]:
            """Validate paths object in OpenAPI spec."""
            errors = []

            for path, path_item in paths.items():
                if not path.startswith("/"):
                    errors.append(f"Path '{path}' must start with '/'")

                # Validate operations
                for method, operation in path_item.items():
                    if method.lower() in [
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    ]:
                        if not isinstance(operation, dict):
                            errors.append(
                                f"Operation {method} in path {path} must be an object"
                            )
                        elif "responses" not in operation:
                            errors.append(
                                f"Operation {method} in path {path} must have responses"
                            )

            return errors

        def _generate_valid_test_data(self, endpoint: dict[str, Any]) -> dict[str, Any]:
            """Generate valid test data for an endpoint."""
            test_data = {
                "params": {},
                "headers": {},
                "json": None,
            }

            # Generate parameter data
            for param in endpoint.get("parameters", []):
                param_name = param["name"]
                param_type = param.get("type", "string")

                if param_type == "string":
                    test_data["params"][param_name] = "test_value"
                elif param_type == "integer":
                    test_data["params"][param_name] = 123
                elif param_type == "boolean":
                    test_data["params"][param_name] = True

            # Generate request body data
            request_body = endpoint.get("request_body")
            if request_body and "content" in request_body:
                content = request_body["content"]
                if "application/json" in content:
                    schema = content["application/json"].get("schema", {})
                    # Generate simple valid test data based on schema
                    test_data["json"] = self._generate_simple_test_data(schema)

            return test_data

        def _generate_negative_test_cases(
            self, endpoint: dict[str, Any]
        ) -> list[dict[str, Any]]:
            """Generate negative test cases for an endpoint."""
            negative_cases = []

            # Missing required parameters
            required_params = [
                param
                for param in endpoint.get("parameters", [])
                if param.get("required", False)
            ]

            if required_params:
                for param in required_params:
                    negative_cases.append(
                        {
                            "type": "negative",
                            "endpoint": endpoint["path"],
                            "method": endpoint["method"],
                            "description": f"Test missing required parameter: {param['name']}",
                            "test_data": {"params": {}},  # Empty params
                            "expected_status": 400,
                        }
                    )

            # Invalid parameter types
            for param in endpoint.get("parameters", []):
                if param.get("type") == "integer":
                    negative_cases.append(
                        {
                            "type": "negative",
                            "endpoint": endpoint["path"],
                            "method": endpoint["method"],
                            "description": f"Test invalid type for parameter: {param['name']}",
                            "test_data": {"params": {param["name"]: "not_an_integer"}},
                            "expected_status": 400,
                        }
                    )

            return negative_cases

        def _generate_simple_test_data(self, schema: dict[str, Any]) -> dict[str, Any]:
            """Generate simple test data based on JSON schema."""
            if not schema:
                return {"test": "data"}

            schema_type = schema.get("type", "object")

            if schema_type == "object":
                result = {}
                properties = schema.get("properties", {})
                for prop_name, prop_schema in properties.items():
                    result[prop_name] = self._generate_simple_test_data(prop_schema)
                return result
            if schema_type == "array":
                item_schema = schema.get("items", {"type": "string"})
                return [self._generate_simple_test_data(item_schema)]
            if schema_type == "string":
                return "test_string"
            if schema_type == "integer":
                return 42
            if schema_type == "number":
                return 3.14
            if schema_type == "boolean":
                return True
            return "default_value"

    return OpenAPIContractManager()


@pytest.fixture
def pact_contract_builder():
    """Pact contract building utilities."""

    class PactContractBuilder:
        def __init__(self):
            self.interactions = []
            self.consumer = "ai-docs-consumer"
            self.provider = "ai-docs-provider"

        def given(self, provider_state: str):
            """Set provider state for the interaction."""
            if not self.interactions:
                self.interactions.append({})
            self.interactions[-1]["provider_state"] = provider_state
            return self

        def upon_receiving(self, description: str):
            """Set description for the interaction."""
            if not self.interactions:
                self.interactions.append({})
            self.interactions[-1]["description"] = description
            return self

        def with_request(self, method: str, path: str, **_kwargs):
            """Define the expected request."""
            if not self.interactions:
                self.interactions.append({})

            request = {
                "method": method.upper(),
                "path": path,
            }

            if "headers" in _kwargs:
                request["headers"] = _kwargs["headers"]
            if "query" in _kwargs:
                request["query"] = _kwargs["query"]
            if "body" in _kwargs:
                request["body"] = _kwargs["body"]

            self.interactions[-1]["request"] = request
            return self

        def will_respond_with(self, status: int, **_kwargs):
            """Define the expected response."""
            if not self.interactions:
                self.interactions.append({})

            response = {"status": status}

            if "headers" in _kwargs:
                response["headers"] = _kwargs["headers"]
            if "body" in _kwargs:
                response["body"] = _kwargs["body"]

            self.interactions[-1]["response"] = response
            return self

        def build_pact(self) -> dict[str, Any]:
            """Build the complete Pact contract."""
            return {
                "consumer": {"name": self.consumer},
                "provider": {"name": self.provider},
                "interactions": self.interactions,
                "metadata": {
                    "pactSpecification": {"version": "2.0.0"},
                    "client": {"name": "pytest-pact", "version": "1.0.0"},
                },
            }

        def verify_interaction(
            self, actual_request: dict[str, Any], actual_response: dict[str, Any]
        ) -> dict[str, Any]:
            """Verify an interaction against the contract."""
            # This is a simplified verification - in practice, you'd use the Pact library
            errors = []

            for interaction in self.interactions:
                request_match = self._match_request(
                    actual_request, interaction.get("request", {})
                )
                response_match = self._match_response(
                    actual_response, interaction.get("response", {})
                )

                if request_match["matches"] and response_match["matches"]:
                    return {
                        "verified": True,
                        "interaction": interaction["description"],
                        "errors": [],
                    }
                errors.extend(request_match.get("errors", []))
                errors.extend(response_match.get("errors", []))

            return {
                "verified": False,
                "errors": errors,
            }

        def _match_request(
            self, actual: dict[str, Any], expected: dict[str, Any]
        ) -> dict[str, Any]:
            """Match actual request against expected request."""
            errors = []

            # Check method
            if "method" in expected and actual.get("method") != expected["method"]:
                errors.append(
                    f"Method mismatch: expected {expected['method']}, got {actual.get('method')}"
                )

            # Check path
            if "path" in expected and actual.get("path") != expected["path"]:
                errors.append(
                    f"Path mismatch: expected {expected['path']}, got {actual.get('path')}"
                )

            return {
                "matches": len(errors) == 0,
                "errors": errors,
            }

        def _match_response(
            self, actual: dict[str, Any], expected: dict[str, Any]
        ) -> dict[str, Any]:
            """Match actual response against expected response."""
            errors = []

            # Check status
            if "status" in expected and actual.get("status") != expected["status"]:
                errors.append(
                    f"Status mismatch: expected {expected['status']}, got {actual.get('status')}"
                )

            return {
                "matches": len(errors) == 0,
                "errors": errors,
            }

    return PactContractBuilder()


@pytest.fixture
def contract_test_data():
    """Provide test data for contract testing."""
    return {
        "api_contracts": {
            "/api/search": {
                "GET": {
                    "parameters": [
                        {"name": "q", "type": "string", "required": True},
                        {"name": "limit", "type": "integer", "required": False},
                    ],
                    "responses": {
                        "200": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "results": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "title": {"type": "string"},
                                                "score": {"type": "number"},
                                            },
                                            "required": ["id", "title", "score"],
                                        },
                                    },
                                    "_total": {"type": "integer"},
                                },
                                "required": ["results", "_total"],
                            }
                        },
                        "400": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                                "required": ["error"],
                            }
                        },
                    },
                }
            },
            "/api/documents": {
                "POST": {
                    "requestBody": {
                        "required": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "format": "uri"},
                                "collection": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                            "required": ["url"],
                        },
                    },
                    "responses": {
                        "201": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["id", "status"],
                            }
                        },
                    },
                }
            },
        },
        "json_schemas": {
            "search_result": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                        },
                    },
                },
                "required": ["id", "title", "score"],
            },
            "document_input": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "collection": {"type": "string", "minLength": 1},
                    "chunk_strategy": {
                        "type": "string",
                        "enum": ["simple", "enhanced", "semantic"],
                    },
                },
                "required": ["url"],
            },
        },
        "openapi_spec": {
            "openapi": "3.0.3",
            "info": {"title": "AI Docs API", "version": "1.0.0"},
            "paths": {
                "/search": {
                    "get": {
                        "operationId": "searchDocuments",
                        "parameters": [
                            {
                                "name": "q",
                                "in": "query",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Search results",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "results": {"type": "array"}
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
        },
    }


# Mock services for contract testing
@pytest.fixture
def _mock_contract_service():
    """Mock service for contract testing."""
    service = AsyncMock()

    async def mock_search(_query: str, _limit: int = 10):
        """Mock search endpoint."""
        return {
            "results": [
                {"id": "doc1", "title": "Test Document 1", "score": 0.95},
                {"id": "doc2", "title": "Test Document 2", "score": 0.87},
            ],
            "_total": 2,
        }

    async def mock_add_document(_url: str, _collection: str = "default"):
        """Mock add document endpoint."""
        return {"id": "doc123", "status": "created"}

    service.search = AsyncMock(side_effect=mock_search)
    service.add_document = AsyncMock(side_effect=mock_add_document)

    return service


# Pytest markers for contract test categorization
def pytest_configure(config):
    """Configure contract testing markers."""
    config.addinivalue_line("markers", "contract: mark test as contract test")
    config.addinivalue_line("markers", "api_contract: mark test as API contract test")
    config.addinivalue_line(
        "markers", "schema_validation: mark test as schema validation test"
    )
    config.addinivalue_line("markers", "pact: mark test as Pact contract test")
    config.addinivalue_line("markers", "openapi: mark test as OpenAPI contract test")
    config.addinivalue_line(
        "markers", "consumer_driven: mark test as consumer-driven contract test"
    )
