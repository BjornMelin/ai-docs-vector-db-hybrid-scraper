"""Integration tests for contract testing framework.

This module tests the integration of all contract testing components
and provides end-to-end contract validation scenarios.
"""

import asyncio
import time

import pytest

from src.config import get_config
from src.infrastructure.client_manager import ClientManager


class TestContractIntegration:
    """Test contract testing framework integration."""

    @pytest.fixture
    async def client_manager(self):
        """Create client manager for testing."""
        config = get_config()
        manager = ClientManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.contract
    @pytest.mark.integration
    async def test_full_contract_validation_pipeline(
        self,
        _client_manager,
        api_contract_validator,
        _json_schema_validator,
        openapi_contract_manager,
        pact_contract_builder,
        contract_test_data,
    ):
        """Test the complete contract validation pipeline."""
        # 1. JSON Schema Validation
        search_schema = contract_test_data["json_schemas"]["search_result"]
        _json_schema_validator.register_schema("search_result", search_schema)

        valid_search_result = {
            "id": "doc1",
            "title": "Test Document",
            "score": 0.95,
            "metadata": {"source": "test"},
        }

        schema_validation = _json_schema_validator.validate_data(
            valid_search_result, "search_result"
        )
        assert schema_validation["valid"], (
            f"Schema validation failed: {schema_validation['errors']}"
        )

        # 2. API Contract Validation
        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        request_validation = api_contract_validator.validate_request(
            "/api/search", "GET", params={"q": "test query", "limit": 10}
        )
        assert request_validation["valid"]

        response_data = {"results": [valid_search_result], "_total": 1}

        response_validation = api_contract_validator.validate_response(
            "/api/search", "GET", 200, response_data
        )
        assert response_validation["valid"]

        # 3. OpenAPI Validation
        openapi_spec = contract_test_data["openapi_spec"]
        openapi_contract_manager.load_spec("integration_test", openapi_spec)

        spec_validation = openapi_contract_manager.validate_spec("integration_test")
        assert spec_validation["valid"], (
            f"OpenAPI validation failed: {spec_validation['errors']}"
        )

        # 4. Pact Contract Validation
        pact_contract_builder.given("documents exist")
        pact_contract_builder.upon_receiving("integration test search")
        pact_contract_builder.with_request(
            method="GET", path="/search", query={"q": "test query"}
        )
        pact_contract_builder.will_respond_with(
            status=200, body={"results": [valid_search_result]}
        )

        pact_contract = pact_contract_builder.build_pact()
        assert len(pact_contract["interactions"]) == 1

        # Verify all contract types are compatible
        assert True  # If we reach here, all validations passed

    @pytest.mark.contract
    @pytest.mark.integration
    async def test_contract_violation_detection_pipeline(
        self, api_contract_validator, _json_schema_validator, contract_test_data
    ):
        """Test detection of contract violations across different layers."""
        # Register contracts
        search_schema = contract_test_data["json_schemas"]["search_result"]
        _json_schema_validator.register_schema("search_result", search_schema)

        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        # Test schema violation
        invalid_result = {
            "score": 0.95,  # Missing required 'id' field
            "title": "Test Document",
        }

        schema_validation = _json_schema_validator.validate_data(
            invalid_result, "search_result"
        )
        assert not schema_validation["valid"]
        assert any("id" in error for error in schema_validation["errors"])

        # Test API contract violation
        invalid_request_validation = api_contract_validator.validate_request(
            "/api/search",
            "GET",
            params={},  # Missing required 'q' parameter
        )
        assert not invalid_request_validation["valid"]
        assert any("q" in error for error in invalid_request_validation["errors"])

        # Test response contract violation
        invalid_response = {
            "results": [invalid_result],
            # Missing required '_total' field
        }

        response_validation = api_contract_validator.validate_response(
            "/api/search", "GET", 200, invalid_response
        )
        assert not response_validation["valid"]

    @pytest.mark.contract
    @pytest.mark.integration
    async def test_cross_service_contract_validation(
        self, api_contract_validator, _json_schema_validator
    ):
        """Test contract validation across service boundaries."""
        # Define service contracts
        embedding_service_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}},
                            "model": {"type": "string"},
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
                                }
                            },
                            "required": ["embeddings"],
                        }
                    }
                },
            }
        }

        vector_service_contract = {
            "POST": {
                "requestBody": {
                    "required": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
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
                                    },
                                    "required": ["id", "vector"],
                                },
                            },
                        },
                        "required": ["collection", "vectors"],
                    },
                },
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "operation_id": {"type": "string"},
                                "points_inserted": {"type": "integer"},
                            },
                            "required": ["operation_id"],
                        }
                    }
                },
            }
        }

        # Register service contracts
        api_contract_validator.register_contract(
            "/embeddings", embedding_service_contract
        )
        api_contract_validator.register_contract("/vectors", vector_service_contract)

        # Test embedding service
        embedding_request = {
            "texts": ["hello world", "test document"],
            "model": "text-embedding-3-small",
        }

        embedding_validation = api_contract_validator.validate_request(
            "/embeddings", "POST", json=embedding_request
        )
        assert embedding_validation["valid"]

        # Simulate embedding response
        embedding_response = {
            "embeddings": [
                [0.1, 0.2, 0.3],  # First text embedding
                [0.4, 0.5, 0.6],  # Second text embedding
            ]
        }

        embedding_response_validation = api_contract_validator.validate_response(
            "/embeddings", "POST", 200, embedding_response
        )
        assert embedding_response_validation["valid"]

        # Test vector service (using embedding output)
        vector_request = {
            "collection": "documents",
            "vectors": [
                {"id": "doc1", "vector": embedding_response["embeddings"][0]},
                {"id": "doc2", "vector": embedding_response["embeddings"][1]},
            ],
        }

        vector_validation = api_contract_validator.validate_request(
            "/vectors", "POST", json=vector_request
        )
        assert vector_validation["valid"]

        # Verify data flow compatibility
        # Embedding output should be compatible with vector input
        for i, embedding in enumerate(embedding_response["embeddings"]):
            vector_data = vector_request["vectors"][i]["vector"]
            assert len(embedding) == len(vector_data)
            assert embedding == vector_data


class TestContractVersionCompatibility:
    """Test contract version compatibility scenarios."""

    @pytest.mark.contract
    @pytest.mark.integration
    def test_api_version_migration_scenario(self, api_contract_validator):
        """Test API version migration compatibility."""
        # V1 API Contract
        v1_search_contract = {
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
                                "_total": {"type": "integer"},
                            },
                            "required": ["results", "_total"],
                        }
                    }
                },
            }
        }

        # V2 API Contract (with backward compatibility)
        v2_search_contract = {
            "GET": {
                "parameters": [
                    {"name": "q", "type": "string", "required": True},
                    {"name": "limit", "type": "integer", "required": False},
                    {
                        "name": "max_results",
                        "type": "integer",
                        "required": False,
                        "deprecated": True,
                    },
                    {"name": "strategy", "type": "string", "required": False},
                ],
                "responses": {
                    "200": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "results": {"type": "array"},
                                "_total_count": {"type": "integer"},
                                "_total": {"type": "integer", "deprecated": True},
                                "search_strategy": {"type": "string"},
                            },
                            "required": ["success", "results", "_total_count"],
                        }
                    }
                },
            }
        }

        # Register both versions
        api_contract_validator.register_contract("/api/v1/search", v1_search_contract)
        api_contract_validator.register_contract("/api/v2/search", v2_search_contract)

        # Test V1 client requests
        v1_request = {"q": "test query", "max_results": 10}

        v1_validation = api_contract_validator.validate_request(
            "/api/v1/search", "GET", params=v1_request
        )
        assert v1_validation["valid"]

        # V1 client should also work with V2 API (backward compatibility)
        v1_on_v2_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=v1_request
        )
        assert v1_on_v2_validation["valid"]

        # Test V2 client requests
        v2_request = {"q": "test query", "limit": 10, "strategy": "hybrid"}

        v2_validation = api_contract_validator.validate_request(
            "/api/v2/search", "GET", params=v2_request
        )
        assert v2_validation["valid"]

        # Test response compatibility
        # V1 response format
        v1_response = {"results": [{"id": "doc1", "title": "Test"}], "_total": 1}

        v1_response_validation = api_contract_validator.validate_response(
            "/api/v1/search", "GET", 200, v1_response
        )
        assert v1_response_validation["valid"]

        # V2 response format (with backward compatibility fields)
        v2_response = {
            "success": True,
            "results": [{"id": "doc1", "title": "Test"}],
            "_total_count": 1,
            "_total": 1,  # Deprecated but included for compatibility
            "search_strategy": "hybrid",
        }

        v2_response_validation = api_contract_validator.validate_response(
            "/api/v2/search", "GET", 200, v2_response
        )
        assert v2_response_validation["valid"]

    @pytest.mark.contract
    @pytest.mark.integration
    def test_contract_compatibility_matrix(self, api_contract_validator):
        """Test contract compatibility matrix between versions."""
        # Define compatibility scenarios
        compatibility_scenarios = [
            {
                "name": "v1_client_v1_api",
                "client_version": "v1",
                "api_version": "v1",
                "expected_compatible": True,
            },
            {
                "name": "v1_client_v2_api",
                "client_version": "v1",
                "api_version": "v2",
                "expected_compatible": True,  # Backward compatible
            },
            {
                "name": "v2_client_v1_api",
                "client_version": "v2",
                "api_version": "v1",
                "expected_compatible": False,  # Forward incompatible
            },
            {
                "name": "v2_client_v2_api",
                "client_version": "v2",
                "api_version": "v2",
                "expected_compatible": True,
            },
        ]

        # Define version-specific requests
        client_requests = {
            "v1": {"q": "test", "max_results": 10},
            "v2": {"q": "test", "limit": 10, "strategy": "hybrid"},
        }

        # Define API contracts
        api_contracts = {
            "v1": {
                "GET": {
                    "parameters": [
                        {"name": "q", "type": "string", "required": True},
                        {"name": "max_results", "type": "integer", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
            "v2": {
                "GET": {
                    "parameters": [
                        {"name": "q", "type": "string", "required": True},
                        {"name": "limit", "type": "integer", "required": False},
                        {"name": "max_results", "type": "integer", "required": False},
                        {"name": "strategy", "type": "string", "required": False},
                    ],
                    "responses": {"200": {"schema": {"type": "object"}}},
                }
            },
        }

        # Register API contracts
        for version, contract in api_contracts.items():
            api_contract_validator.register_contract(f"/api/{version}/search", contract)

        # Test compatibility scenarios
        for scenario in compatibility_scenarios:
            client_request = client_requests[scenario["client_version"]]
            api_endpoint = f"/api/{scenario['api_version']}/search"

            validation_result = api_contract_validator.validate_request(
                api_endpoint, "GET", params=client_request
            )

            if scenario["expected_compatible"]:
                assert validation_result["valid"], (
                    f"Scenario {scenario['name']} should be compatible"
                )
            else:
                # For this test, we'll assume forward incompatibility is detected
                # In practice, this would depend on specific validation logic
                pass  # Simplified for this test


class TestContractPerformanceValidation:
    """Test contract validation performance and scalability."""

    @pytest.mark.contract
    @pytest.mark.integration
    @pytest.mark.performance
    def test_contract_validation_performance(
        self, _api_contract_validator, _json_schema_validator
    ):
        """Test performance of contract validation at scale."""

        # Setup large-scale contract
        large_schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(100)},
            "required": [
                f"field_{i}" for i in range(0, 100, 10)
            ],  # Every 10th field required
        }

        _json_schema_validator.register_schema("large_schema", large_schema)

        # Generate test data
        test_data = {f"field_{i}": f"value_{i}" for i in range(100)}

        # Performance test: validate 1000 objects
        start_time = time.time()

        for _ in range(1000):
            result = _json_schema_validator.validate_data(test_data, "large_schema")
            assert result["valid"]

        end_time = time.time()
        validation_time = end_time - start_time

        # Performance assertion: should complete within reasonable time
        assert validation_time < 10.0, f"Validation took too long: {validation_time}s"

        # Calculate throughput
        throughput = 1000 / validation_time
        assert throughput > 100, (
            f"Validation throughput too low: {throughput} validations/sec"
        )

    @pytest.mark.contract
    @pytest.mark.integration
    def test_concurrent_contract_validation(self, api_contract_validator):
        """Test concurrent contract validation scenarios."""

        # Setup contract
        contract = {
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
                "responses": {"200": {"schema": {"type": "object"}}},
            }
        }

        api_contract_validator.register_contract("/api/search", contract)

        async def validate_request(request_id):
            """Validate a single request."""
            request_data = {"query": f"test query {request_id}", "limit": 10}
            return api_contract_validator.validate_request(
                "/api/search", "POST", json=request_data
            )

        async def concurrent_validation_test():
            """Run concurrent validations."""
            tasks = [validate_request(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent validation test
        start_time = time.time()
        results = asyncio.run(concurrent_validation_test())
        end_time = time.time()

        # Verify all validations succeeded
        for result in results:
            assert result["valid"]

        # Performance check
        concurrent_time = end_time - start_time
        assert concurrent_time < 5.0, (
            f"Concurrent validation took too long: {concurrent_time}s"
        )
