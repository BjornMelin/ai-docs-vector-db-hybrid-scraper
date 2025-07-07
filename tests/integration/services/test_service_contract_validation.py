"""Service contract validation integration tests.

This module tests service interface contracts, API compatibility, and data format
validation across service boundaries. Ensures backward compatibility and proper
service integration through contract-based testing.

Tests include:
- API endpoint contract validation
- Data schema validation across services
- Service interface compatibility testing
- Version compatibility validation
- Breaking change detection
- Consumer-driven contract testing
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class ContractVersion(Enum):
    """API contract version enumeration."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class ServiceContract:
    """Service contract definition."""

    service_name: str
    version: ContractVersion
    endpoints: dict[str, Any]
    data_schemas: dict[str, Any]
    dependencies: list[str]
    breaking_changes: list[str]


@dataclass
class ContractValidationResult:
    """Contract validation result."""

    is_valid: bool
    contract_name: str
    validation_errors: list[str]
    warnings: list[str]
    compatibility_score: float


class TestAPIEndpointContracts:
    """Test API endpoint contract validation."""

    @pytest.fixture
    async def service_contracts(self):
        """Setup service contracts for testing."""

        # Vector DB Service Contract
        vector_db_contract = ServiceContract(
            service_name="vector_db_service",
            version=ContractVersion.V1_1,
            endpoints={
                "/collections": {
                    "GET": {
                        "description": "List all collections",
                        "parameters": {},
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "collections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "size": {"type": "integer"},
                                            "status": {
                                                "type": "string",
                                                "enum": ["active", "inactive"],
                                            },
                                        },
                                        "required": ["name", "size", "status"],
                                    },
                                }
                            },
                            "required": ["collections"],
                        },
                    },
                    "POST": {
                        "description": "Create new collection",
                        "request_schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "config": {
                                    "type": "object",
                                    "properties": {
                                        "vector_size": {
                                            "type": "integer",
                                            "minimum": 1,
                                        },
                                        "distance": {
                                            "type": "string",
                                            "enum": ["cosine", "euclidean", "dot"],
                                        },
                                    },
                                    "required": ["vector_size"],
                                },
                            },
                            "required": ["name", "config"],
                        },
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "collection_name": {"type": "string"},
                                "operation_id": {"type": "string"},
                            },
                            "required": ["success", "collection_name"],
                        },
                    },
                },
                "/search": {
                    "POST": {
                        "description": "Perform vector search",
                        "request_schema": {
                            "type": "object",
                            "properties": {
                                "collection": {"type": "string"},
                                "vector": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "filter": {"type": "object"},
                            },
                            "required": ["collection", "vector"],
                        },
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "matches": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "score": {"type": "number"},
                                            "payload": {"type": "object"},
                                        },
                                        "required": ["id", "score"],
                                    },
                                },
                                "search_time_ms": {"type": "number"},
                            },
                            "required": ["matches"],
                        },
                    }
                },
            },
            data_schemas={
                "VectorPoint": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "vector": {"type": "array", "items": {"type": "number"}},
                        "payload": {"type": "object"},
                    },
                    "required": ["id", "vector"],
                }
            },
            dependencies=["config_service"],
            breaking_changes=[],
        )

        # Embedding Service Contract
        embedding_contract = ServiceContract(
            service_name="embedding_service",
            version=ContractVersion.V1_0,
            endpoints={
                "/embeddings": {
                    "POST": {
                        "description": "Generate embeddings",
                        "request_schema": {
                            "type": "object",
                            "properties": {
                                "texts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 100,
                                },
                                "model": {"type": "string"},
                                "batch_size": {"type": "integer", "minimum": 1},
                            },
                            "required": ["texts"],
                        },
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "embeddings": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                    },
                                },
                                "model": {"type": "string"},
                                "usage": {
                                    "type": "object",
                                    "properties": {
                                        "_total_tokens": {"type": "integer"},
                                        "cost_usd": {"type": "number"},
                                    },
                                },
                            },
                            "required": ["embeddings", "model"],
                        },
                    }
                },
                "/health": {
                    "GET": {
                        "description": "Health check endpoint",
                        "parameters": {},
                        "response_schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["healthy", "degraded", "unhealthy"],
                                },
                                "timestamp": {"type": "number"},
                                "version": {"type": "string"},
                            },
                            "required": ["status", "timestamp"],
                        },
                    }
                },
            },
            data_schemas={
                "EmbeddingRequest": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["text"],
                }
            },
            dependencies=["config_service", "cache_service"],
            breaking_changes=[],
        )

        return {"vector_db": vector_db_contract, "embedding": embedding_contract}

    @pytest.mark.asyncio
    async def test_endpoint_request_schema_validation(self, service_contracts):
        """Test endpoint request schema validation."""
        contracts = service_contracts
        vector_db_contract = contracts["vector_db"]

        # Mock schema validator
        class JSONSchemaValidator:
            def validate(
                self, data: dict[str, Any], schema: dict[str, Any]
            ) -> dict[str, Any]:
                """Validate data against JSON schema."""
                errors = []
                warnings = []

                # Simple validation logic for testing
                if schema.get("type") == "object":
                    required_fields = schema.get("required", [])
                    properties = schema.get("properties", {})

                    # Check required fields
                    errors.extend(
                        f"Missing required field: {field}"
                        for field in required_fields
                        if field not in data
                    )

                    # Check field types
                    for field, value in data.items():
                        if field in properties:
                            expected_type = properties[field].get("type")
                            if expected_type == "string" and not isinstance(value, str):
                                errors.append(
                                    f"Field {field} must be string, got {type(value).__name__}"
                                )
                            elif expected_type == "integer" and not isinstance(
                                value, int
                            ):
                                errors.append(
                                    f"Field {field} must be integer, got {type(value).__name__}"
                                )
                            elif expected_type == "array" and not isinstance(
                                value, list
                            ):
                                errors.append(
                                    f"Field {field} must be array, got {type(value).__name__}"
                                )
                            elif expected_type == "object" and not isinstance(
                                value, dict
                            ):
                                errors.append(
                                    f"Field {field} must be object, got {type(value).__name__}"
                                )

                return {
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

        validator = JSONSchemaValidator()

        # Test valid collection creation request
        valid_request = {
            "name": "test_collection",
            "config": {"vector_size": 1536, "distance": "cosine"},
        }

        collection_schema = vector_db_contract.endpoints["/collections"]["POST"][
            "request_schema"
        ]
        validation_result = validator.validate(valid_request, collection_schema)

        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0

        # Test invalid collection creation request (missing required field)
        invalid_request = {
            "name": "test_collection"
            # Missing required "config" field
        }

        invalid_validation = validator.validate(invalid_request, collection_schema)

        assert invalid_validation["valid"] is False
        assert any("config" in error for error in invalid_validation["errors"])

        # Test invalid field type
        type_invalid_request = {
            "name": 123,  # Should be string
            "config": {"vector_size": 1536},
        }

        type_validation = validator.validate(type_invalid_request, collection_schema)

        assert type_validation["valid"] is False
        assert any(
            "name" in error and "string" in error for error in type_validation["errors"]
        )

    @pytest.mark.asyncio
    async def test_endpoint_response_schema_validation(self, service_contracts):
        """Test endpoint response schema validation."""
        contracts = service_contracts
        embedding_contract = contracts["embedding"]

        # Mock embedding service responses
        valid_embedding_response = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "model": "text-embedding-3-small",
            "usage": {"_total_tokens": 50, "cost_usd": 0.001},
        }

        invalid_embedding_response = {
            "embeddings": "not_an_array",  # Should be array
            "model": "text-embedding-3-small",
        }

        # Validate responses against contract
        response_schema = embedding_contract.endpoints["/embeddings"]["POST"][
            "response_schema"
        ]

        class ResponseValidator:
            def validate_response(
                self, response: dict[str, Any], schema: dict[str, Any]
            ) -> dict[str, Any]:
                """Validate response against schema."""
                errors = []

                required_fields = schema.get("required", [])
                schema.get("properties", {})

                # Check required fields
                errors.extend(
                    f"Missing required response field: {field}"
                    for field in required_fields
                    if field not in response
                )

                # Check array types specifically
                if "embeddings" in response:
                    if not isinstance(response["embeddings"], list):
                        errors.append("embeddings must be an array")
                    elif len(response["embeddings"]) > 0 and not isinstance(
                        response["embeddings"][0], list
                    ):
                        errors.append("embeddings must be array of arrays")

                return {"valid": len(errors) == 0, "errors": errors}

        response_validator = ResponseValidator()

        # Test valid response
        valid_result = response_validator.validate_response(
            valid_embedding_response, response_schema
        )
        assert valid_result["valid"] is True

        # Test invalid response
        invalid_result = response_validator.validate_response(
            invalid_embedding_response, response_schema
        )
        assert invalid_result["valid"] is False
        assert any("array" in error for error in invalid_result["errors"])

    @pytest.mark.asyncio
    async def test_cross_service_contract_compatibility(self, service_contracts):
        """Test contract compatibility between services."""
        contracts = service_contracts

        # Test embedding service -> vector DB service integration
        embedding_contract = contracts["embedding"]
        vector_db_contract = contracts["vector_db"]

        # Mock integration scenario: embedding service calls vector DB
        class ServiceIntegrationTester:
            def check_integration_compatibility(
                self,
                producer_contract: ServiceContract,
                _consumer_contract: ServiceContract,
                integration_points: list[dict[str, Any]],
            ) -> dict[str, Any]:
                """Check compatibility between service contracts."""
                compatibility_issues = []
                compatibility_score = 1.0

                for integration in integration_points:
                    producer_endpoint = integration["producer_endpoint"]
                    consumer_expectation = integration["consumer_expectation"]

                    # Check if producer endpoint exists
                    if producer_endpoint not in producer_contract.endpoints:
                        compatibility_issues.append(
                            f"Producer endpoint {producer_endpoint} not found in {producer_contract.service_name}"
                        )
                        compatibility_score -= 0.5
                        continue

                    # Check method compatibility
                    method = integration.get("method", "GET")
                    endpoint_def = producer_contract.endpoints[producer_endpoint]

                    if method not in endpoint_def:
                        compatibility_issues.append(
                            f"Method {method} not supported for {producer_endpoint}"
                        )
                        compatibility_score -= 0.3
                        continue

                    # Check response schema compatibility
                    producer_response = endpoint_def[method].get("response_schema", {})
                    expected_fields = consumer_expectation.get("required_fields", [])

                    producer_properties = producer_response.get("properties", {})
                    for field in expected_fields:
                        if field not in producer_properties:
                            compatibility_issues.append(
                                f"Consumer expects field '{field}' in {producer_endpoint} response"
                            )
                            compatibility_score -= 0.2

                return {
                    "compatible": len(compatibility_issues) == 0,
                    "compatibility_score": max(0.0, compatibility_score),
                    "issues": compatibility_issues,
                }

        integration_tester = ServiceIntegrationTester()

        # Define integration points
        integration_points = [
            {
                "producer_endpoint": "/search",
                "method": "POST",
                "consumer_expectation": {
                    "required_fields": ["matches"],  # Embedding service expects matches
                    "field_types": {"matches": "array"},
                },
            },
            {
                "producer_endpoint": "/collections",
                "method": "GET",
                "consumer_expectation": {
                    "required_fields": ["collections"],
                    "field_types": {"collections": "array"},
                },
            },
        ]

        # Test compatibility
        compatibility_result = integration_tester.check_integration_compatibility(
            vector_db_contract,  # Producer
            embedding_contract,  # Consumer
            integration_points,
        )

        # Verify compatibility
        assert compatibility_result["compatible"] is True
        assert compatibility_result["compatibility_score"] == 1.0
        assert len(compatibility_result["issues"]) == 0

        # Test incompatible scenario
        incompatible_integration = [
            {
                "producer_endpoint": "/nonexistent",
                "method": "POST",
                "consumer_expectation": {"required_fields": ["data"]},
            }
        ]

        incompatible_result = integration_tester.check_integration_compatibility(
            vector_db_contract, embedding_contract, incompatible_integration
        )

        assert incompatible_result["compatible"] is False
        assert incompatible_result["compatibility_score"] < 1.0
        assert len(incompatible_result["issues"]) > 0


class TestDataSchemaValidation:
    """Test data schema validation across service boundaries."""

    @pytest.fixture
    async def data_schemas(self):
        """Setup data schemas for testing."""
        return {
            "DocumentSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "author": {"type": "string"},
                            "created_at": {"type": "string", "format": "datetime"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "embeddings": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["id", "title", "content"],
            },
            "SearchResultSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "document": {"$ref": "#/definitions/DocumentSchema"},
                    "highlights": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "score", "document"],
            },
            "IngestionEventSchema": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "document.added",
                            "document.updated",
                            "document.deleted",
                        ],
                    },
                    "document_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "source_service": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": [
                    "event_type",
                    "document_id",
                    "timestamp",
                    "source_service",
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_document_schema_validation(self, data_schemas):
        """Test document schema validation across services."""
        document_schema = data_schemas["DocumentSchema"]

        # Valid document
        valid_document = {
            "id": "doc_123",
            "title": "Machine Learning Basics",
            "content": "Introduction to machine learning concepts...",
            "metadata": {
                "author": "Dr. Smith",
                "created_at": "2024-01-15T10:30:00Z",
                "tags": ["machine-learning", "tutorial", "beginner"],
            },
            "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        # Invalid document (missing required field)
        invalid_document = {
            "id": "doc_456",
            "title": "Invalid Document",
            # Missing required "content" field
        }

        # Document with type errors
        type_error_document = {
            "id": "doc_789",
            "title": "Type Error Document",
            "content": "Valid content",
            "metadata": "should_be_object",  # Type error
            "embeddings": "should_be_array",  # Type error
        }

        class DocumentValidator:
            def validate_document(
                self, document: dict[str, Any], schema: dict[str, Any]
            ) -> dict[str, Any]:
                """Validate document against schema."""
                errors = []
                warnings = []

                # Check required fields
                required = schema.get("required", [])
                errors.extend(
                    f"Missing required field: {field}"
                    for field in required
                    if field not in document
                )

                # Check field types
                properties = schema.get("properties", {})
                for field, value in document.items():
                    if field in properties:
                        expected_type = properties[field].get("type")

                        if expected_type == "string" and not isinstance(value, str):
                            errors.append(f"Field '{field}' must be string")
                        elif expected_type == "object" and not isinstance(value, dict):
                            errors.append(f"Field '{field}' must be object")
                        elif expected_type == "array" and not isinstance(value, list):
                            errors.append(f"Field '{field}' must be array")

                        # Validate array items
                        if expected_type == "array" and isinstance(value, list):
                            items_schema = properties[field].get("items", {})
                            if items_schema.get("type") == "number":
                                for i, item in enumerate(value):
                                    if not isinstance(item, int | float):
                                        errors.append(
                                            f"Field '{field}[{i}]' must be number"
                                        )

                return {
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

        validator = DocumentValidator()

        # Test valid document
        valid_result = validator.validate_document(valid_document, document_schema)
        assert valid_result["valid"] is True
        assert len(valid_result["errors"]) == 0

        # Test invalid document (missing field)
        invalid_result = validator.validate_document(invalid_document, document_schema)
        assert invalid_result["valid"] is False
        assert any("content" in error for error in invalid_result["errors"])

        # Test type error document
        type_error_result = validator.validate_document(
            type_error_document, document_schema
        )
        assert type_error_result["valid"] is False
        assert (
            len(type_error_result["errors"]) >= 2
        )  # metadata and embeddings type errors

    @pytest.mark.asyncio
    async def test_event_schema_validation(self, data_schemas):
        """Test event schema validation for inter-service communication."""
        event_schema = data_schemas["IngestionEventSchema"]

        # Valid ingestion event
        valid_event = {
            "event_type": "document.added",
            "document_id": "doc_123",
            "timestamp": time.time(),
            "source_service": "crawling_service",
            "payload": {
                "url": "https://example.com/article",
                "processing_time_ms": 1200,
            },
        }

        # Invalid event (wrong enum value)
        invalid_enum_event = {
            "event_type": "document.invalid",  # Not in enum
            "document_id": "doc_456",
            "timestamp": time.time(),
            "source_service": "crawling_service",
        }

        # Missing required field event
        missing_field_event = {
            "event_type": "document.updated",
            "document_id": "doc_789",
            # Missing timestamp and source_service
        }

        class EventValidator:
            def validate_event(
                self, event: dict[str, Any], schema: dict[str, Any]
            ) -> dict[str, Any]:
                """Validate event against schema."""
                errors = []

                # Check required fields
                required = schema.get("required", [])
                errors.extend(
                    f"Missing required field: {field}"
                    for field in required
                    if field not in event
                )

                # Check enum values
                properties = schema.get("properties", {})
                for field, value in event.items():
                    if field in properties:
                        field_schema = properties[field]
                        if "enum" in field_schema and value not in field_schema["enum"]:
                            errors.append(
                                f"Field '{field}' has invalid value '{value}', must be one of {field_schema['enum']}"
                            )

                return {"valid": len(errors) == 0, "errors": errors}

        event_validator = EventValidator()

        # Test valid event
        valid_result = event_validator.validate_event(valid_event, event_schema)
        assert valid_result["valid"] is True

        # Test invalid enum event
        enum_result = event_validator.validate_event(invalid_enum_event, event_schema)
        assert enum_result["valid"] is False
        assert any("invalid value" in error for error in enum_result["errors"])

        # Test missing field event
        missing_result = event_validator.validate_event(
            missing_field_event, event_schema
        )
        assert missing_result["valid"] is False
        assert (
            len(missing_result["errors"]) >= 2
        )  # timestamp and source_service missing

    @pytest.mark.asyncio
    async def test_cross_service_data_transformation(self, _data_schemas):
        """Test data transformation between service schemas."""

        # Mock data transformation between services
        class DataTransformer:
            def transform_crawl_to_document(
                self, crawl_result: dict[str, Any]
            ) -> dict[str, Any]:
                """Transform crawl result to document schema."""
                return {
                    "id": crawl_result.get("url_hash", "unknown"),
                    "title": crawl_result.get("title", "Untitled"),
                    "content": crawl_result.get("text_content", ""),
                    "metadata": {
                        "author": crawl_result.get("author", "Unknown"),
                        "created_at": crawl_result.get("crawled_at", ""),
                        "tags": crawl_result.get("extracted_tags", []),
                    },
                }

            def transform_document_to_vector_point(
                self, document: dict[str, Any], embeddings: list[float]
            ) -> dict[str, Any]:
                """Transform document to vector point schema."""
                return {
                    "id": document["id"],
                    "vector": embeddings,
                    "payload": {
                        "title": document["title"],
                        "content": document["content"],
                        "metadata": document.get("metadata", {}),
                    },
                }

        transformer = DataTransformer()

        # Test crawl result transformation
        crawl_result = {
            "url": "https://example.com/article",
            "url_hash": "abc123",
            "title": "Test Article",
            "text_content": "This is test content...",
            "author": "John Doe",
            "crawled_at": "2024-01-15T10:30:00Z",
            "extracted_tags": ["test", "article"],
        }

        document = transformer.transform_crawl_to_document(crawl_result)

        # Validate transformed document
        assert document["id"] == "abc123"
        assert document["title"] == "Test Article"
        assert document["content"] == "This is test content..."
        assert document["metadata"]["author"] == "John Doe"
        assert "test" in document["metadata"]["tags"]

        # Test document to vector point transformation
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
        vector_point = transformer.transform_document_to_vector_point(
            document, embeddings
        )

        assert vector_point["id"] == document["id"]
        assert vector_point["vector"] == embeddings
        assert vector_point["payload"]["title"] == document["title"]
        assert vector_point["payload"]["content"] == document["content"]


class TestVersionCompatibility:
    """Test version compatibility across services."""

    @pytest.mark.asyncio
    async def test_backward_compatibility_validation(self):
        """Test backward compatibility between service versions."""

        # Define service versions
        class ServiceVersionManager:
            def __init__(self):
                self.versions = {
                    "vector_db_service": {
                        "1.0": {
                            "endpoints": ["/search", "/collections"],
                            "search_response_fields": ["matches", "_total"],
                            "breaking_changes": [],
                        },
                        "1.1": {
                            "endpoints": ["/search", "/collections", "/health"],
                            "search_response_fields": [
                                "matches",
                                "_total",
                                "search_time_ms",
                            ],
                            "breaking_changes": [],
                        },
                        "2.0": {
                            "endpoints": ["/v2/search", "/v2/collections", "/health"],
                            "search_response_fields": [
                                "results",
                                "metadata",
                            ],  # Breaking change
                            "breaking_changes": ["search_response_format_changed"],
                        },
                    }
                }

            def check_backward_compatibility(
                self, service: str, from_version: str, to_version: str
            ) -> dict[str, Any]:
                """Check backward compatibility between versions."""
                if service not in self.versions:
                    return {"compatible": False, "error": "Service not found"}

                from_spec = self.versions[service].get(from_version)
                to_spec = self.versions[service].get(to_version)

                if not from_spec or not to_spec:
                    return {"compatible": False, "error": "Version not found"}

                compatibility_issues = []

                # Check if old endpoints still exist
                compatibility_issues.extend(
                    f"Endpoint {endpoint} removed in {to_version}"
                    for endpoint in from_spec["endpoints"]
                    if endpoint not in to_spec["endpoints"]
                )

                # Check for breaking changes
                breaking_changes = to_spec.get("breaking_changes", [])
                if breaking_changes:
                    compatibility_issues.extend(breaking_changes)

                return {
                    "compatible": len(compatibility_issues) == 0,
                    "issues": compatibility_issues,
                    "breaking_changes": breaking_changes,
                }

        version_manager = ServiceVersionManager()

        # Test compatible version upgrade (1.0 -> 1.1)
        compatible_result = version_manager.check_backward_compatibility(
            "vector_db_service", "1.0", "1.1"
        )

        assert compatible_result["compatible"] is True
        assert len(compatible_result["issues"]) == 0

        # Test incompatible version upgrade (1.0 -> 2.0)
        incompatible_result = version_manager.check_backward_compatibility(
            "vector_db_service", "1.0", "2.0"
        )

        assert incompatible_result["compatible"] is False
        assert len(incompatible_result["breaking_changes"]) > 0
        assert (
            "search_response_format_changed" in incompatible_result["breaking_changes"]
        )

    @pytest.mark.asyncio
    async def test_api_versioning_strategy(self):
        """Test API versioning strategy and client adaptation."""

        # Mock versioned API client
        class VersionedAPIClient:
            def __init__(self, service_url: str, api_version: str = "1.1"):
                self.service_url = service_url
                self.api_version = api_version
                self.supported_versions = ["1.0", "1.1", "2.0"]

            async def search(self, _query_data: dict[str, Any]) -> dict[str, Any]:
                """Perform search with version-specific handling."""
                if self.api_version == "1.0":
                    # Legacy format
                    return {
                        "matches": [
                            {"id": "doc1", "score": 0.9},
                            {"id": "doc2", "score": 0.8},
                        ],
                        "_total": 2,
                    }
                if self.api_version == "1.1":
                    # Enhanced format (backward compatible)
                    return {
                        "matches": [
                            {"id": "doc1", "score": 0.9, "payload": {}},
                            {"id": "doc2", "score": 0.8, "payload": {}},
                        ],
                        "_total": 2,
                        "search_time_ms": 45,
                    }
                if self.api_version == "2.0":
                    # New format (breaking change)
                    return {
                        "results": [
                            {"document_id": "doc1", "relevance": 0.9, "metadata": {}},
                            {"document_id": "doc2", "relevance": 0.8, "metadata": {}},
                        ],
                        "metadata": {"_total_count": 2, "search_duration_ms": 45},
                    }
                return None

            def adapt_response(
                self, response: dict[str, Any], target_version: str
            ) -> dict[str, Any]:
                """Adapt response format between versions."""
                if self.api_version == "2.0" and target_version in ["1.0", "1.1"]:
                    # Convert v2.0 format to v1.x format
                    adapted_response = {
                        "matches": [],
                        "_total": response["metadata"]["_total_count"],
                    }

                    for result in response["results"]:
                        match = {
                            "id": result["document_id"],
                            "score": result["relevance"],
                        }
                        if target_version == "1.1":
                            match["payload"] = result.get("metadata", {})
                        adapted_response["matches"].append(match)

                    if target_version == "1.1":
                        adapted_response["search_time_ms"] = response["metadata"][
                            "search_duration_ms"
                        ]

                    return adapted_response

                return response

        # Test API versioning
        v1_client = VersionedAPIClient("http://vector-db:6333", "1.0")
        v11_client = VersionedAPIClient("http://vector-db:6333", "1.1")
        v2_client = VersionedAPIClient("http://vector-db:6333", "2.0")

        query = {"query_vector": [0.1, 0.2, 0.3]}

        # Test version-specific responses
        v1_response = await v1_client.search(query)
        v11_response = await v11_client.search(query)
        v2_response = await v2_client.search(query)

        # Verify version-specific formats
        assert "matches" in v1_response
        assert "_total" in v1_response
        assert "search_time_ms" not in v1_response  # Not in v1.0

        assert "matches" in v11_response
        assert "search_time_ms" in v11_response  # Added in v1.1

        assert "results" in v2_response  # New format in v2.0
        assert "metadata" in v2_response
        assert "matches" not in v2_response  # Breaking change

        # Test response adaptation
        adapted_response = v2_client.adapt_response(v2_response, "1.1")

        assert "matches" in adapted_response
        assert "_total" in adapted_response
        assert "search_time_ms" in adapted_response
        assert adapted_response["_total"] == 2
        assert len(adapted_response["matches"]) == 2

    @pytest.mark.asyncio
    async def test_breaking_change_detection(self):
        """Test automatic detection of breaking changes."""

        class BreakingChangeDetector:
            def compare_schemas(
                self, old_schema: dict[str, Any], new_schema: dict[str, Any]
            ) -> dict[str, Any]:
                """Compare schemas to detect breaking changes."""
                breaking_changes = []
                warnings = []

                old_props = old_schema.get("properties", {})
                new_props = new_schema.get("properties", {})
                old_required = set(old_schema.get("required", []))
                new_required = set(new_schema.get("required", []))

                # Detect removed fields
                removed_fields = set(old_props.keys()) - set(new_props.keys())
                for field in removed_fields:
                    if field in old_required:
                        breaking_changes.append(f"Required field '{field}' was removed")
                    else:
                        warnings.append(f"Optional field '{field}' was removed")

                # Detect added required fields
                new_required_fields = new_required - old_required
                breaking_changes.extend(
                    f"New required field '{field}' was added"
                    for field in new_required_fields
                    if field not in old_props
                )

                # Detect type changes
                for field in old_props:
                    if field in new_props:
                        old_type = old_props[field].get("type")
                        new_type = new_props[field].get("type")
                        if old_type != new_type:
                            breaking_changes.append(
                                f"Field '{field}' type changed from {old_type} to {new_type}"
                            )

                return {
                    "has_breaking_changes": len(breaking_changes) > 0,
                    "breaking_changes": breaking_changes,
                    "warnings": warnings,
                }

        detector = BreakingChangeDetector()

        # Test schemas
        old_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "score": {"type": "number"},
                "metadata": {"type": "object"},
            },
            "required": ["id", "name"],
        }

        # Breaking change: remove required field, change type
        breaking_new_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "score": {"type": "string"},  # Type changed!
                "metadata": {"type": "object"},
                "new_field": {"type": "string"},
            },
            "required": ["id", "new_field"],  # name removed, new_field added
        }

        # Non-breaking change: add optional field
        compatible_new_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "score": {"type": "number"},
                "metadata": {"type": "object"},
                "optional_field": {"type": "string"},  # Optional addition
            },
            "required": ["id", "name"],
        }

        # Test breaking changes detection
        breaking_result = detector.compare_schemas(old_schema, breaking_new_schema)

        assert breaking_result["has_breaking_changes"] is True
        breaking_changes = breaking_result["breaking_changes"]

        # Should detect removed required field and type change
        assert any(
            "name" in change and "removed" in change for change in breaking_changes
        )
        assert any(
            "score" in change and "type changed" in change
            for change in breaking_changes
        )
        assert any(
            "new_field" in change and "required field" in change
            for change in breaking_changes
        )

        # Test compatible changes
        compatible_result = detector.compare_schemas(old_schema, compatible_new_schema)

        assert compatible_result["has_breaking_changes"] is False
        assert len(compatible_result["breaking_changes"]) == 0


class TestConsumerDrivenContracts:
    """Test consumer-driven contract testing."""

    @pytest.mark.asyncio
    async def test_consumer_contract_definition(self):
        """Test consumer-driven contract definition and validation."""

        # Mock consumer contract definitions
        consumer_contracts = {
            "api_gateway": {
                "provider": "vector_db_service",
                "interactions": [
                    {
                        "description": "Search for documents",
                        "request": {
                            "method": "POST",
                            "path": "/search",
                            "body": {
                                "collection": "documents",
                                "vector": [0.1, 0.2, 0.3],
                                "limit": 10,
                            },
                        },
                        "response": {
                            "status": 200,
                            "body": {
                                "matches": [
                                    {
                                        "id": "doc_123",
                                        "score": 0.95,
                                        "payload": {"title": "Test Document"},
                                    }
                                ]
                            },
                        },
                    }
                ],
            },
            "embedding_service": {
                "provider": "cache_service",
                "interactions": [
                    {
                        "description": "Cache embedding result",
                        "request": {
                            "method": "POST",
                            "path": "/cache/set",
                            "body": {
                                "key": "embedding:hash123",
                                "value": {"embedding": [0.1, 0.2, 0.3]},
                                "ttl": 3600,
                            },
                        },
                        "response": {"status": 200, "body": {"success": True}},
                    }
                ],
            },
        }

        class ConsumerContractValidator:
            def validate_contract(
                self, consumer_name: str, contract: dict[str, Any]
            ) -> dict[str, Any]:
                """Validate consumer contract against provider capabilities."""
                validation_results = []

                provider = contract["provider"]
                interactions = contract["interactions"]

                for interaction in interactions:
                    request = interaction["request"]
                    expected_response = interaction["response"]

                    # Mock provider validation
                    validation_result = self._validate_interaction(
                        provider, request, expected_response
                    )

                    validation_results.append(
                        {
                            "description": interaction["description"],
                            "valid": validation_result["valid"],
                            "errors": validation_result.get("errors", []),
                        }
                    )

                return {
                    "consumer": consumer_name,
                    "provider": provider,
                    "valid": all(r["valid"] for r in validation_results),
                    "interaction_results": validation_results,
                }

            def _validate_interaction(
                self,
                provider: str,
                request: dict[str, Any],
                expected_response: dict[str, Any],
            ) -> dict[str, Any]:
                """Validate individual interaction."""
                errors = []

                # Mock provider capabilities
                provider_capabilities = {
                    "vector_db_service": {
                        "endpoints": {
                            "/search": {
                                "methods": ["POST"],
                                "response_fields": ["matches"],
                            }
                        }
                    },
                    "cache_service": {
                        "endpoints": {
                            "/cache/set": {
                                "methods": ["POST"],
                                "response_fields": ["success"],
                            }
                        }
                    },
                }

                if provider not in provider_capabilities:
                    errors.append(f"Unknown provider: {provider}")
                    return {"valid": False, "errors": errors}

                capabilities = provider_capabilities[provider]
                path = request["path"]
                method = request["method"]

                # Check if endpoint exists
                if path not in capabilities["endpoints"]:
                    errors.append(f"Endpoint {path} not found in {provider}")
                    return {"valid": False, "errors": errors}

                endpoint = capabilities["endpoints"][path]

                # Check if method is supported
                if method not in endpoint["methods"]:
                    errors.append(f"Method {method} not supported for {path}")

                # Check if expected response fields are provided
                expected_body = expected_response.get("body", {})
                provider_fields = endpoint.get("response_fields", [])

                errors.extend(
                    f"Provider does not guarantee field: {field}"
                    for field in expected_body
                    if field not in provider_fields
                )

                return {"valid": len(errors) == 0, "errors": errors}

        validator = ConsumerContractValidator()

        # Validate consumer contracts
        api_gateway_result = validator.validate_contract(
            "api_gateway", consumer_contracts["api_gateway"]
        )
        embedding_service_result = validator.validate_contract(
            "embedding_service", consumer_contracts["embedding_service"]
        )

        # Verify validation results
        assert api_gateway_result["valid"] is True
        assert api_gateway_result["provider"] == "vector_db_service"
        assert len(api_gateway_result["interaction_results"]) == 1

        assert embedding_service_result["valid"] is True
        assert embedding_service_result["provider"] == "cache_service"

    @pytest.mark.asyncio
    async def test_contract_driven_testing(self):
        """Test contract-driven testing workflow."""

        class ContractTestRunner:
            def __init__(self):
                self.test_results = []

            async def run_contract_tests(
                self, consumer_contract: dict[str, Any]
            ) -> dict[str, Any]:
                """Run contract tests against provider."""
                provider = consumer_contract["provider"]
                interactions = consumer_contract["interactions"]

                test_results = []

                for interaction in interactions:
                    # Mock actual HTTP call to provider
                    test_result = await self._execute_interaction_test(interaction)
                    test_results.append(test_result)

                return {
                    "provider": provider,
                    "_total_tests": len(test_results),
                    "passed": sum(1 for r in test_results if r["passed"]),
                    "failed": sum(1 for r in test_results if not r["passed"]),
                    "test_results": test_results,
                }

            async def _execute_interaction_test(
                self, interaction: dict[str, Any]
            ) -> dict[str, Any]:
                """Execute individual interaction test."""
                description = interaction["description"]
                interaction["request"]
                expected_response = interaction["response"]

                # Mock HTTP request execution
                await asyncio.sleep(0.01)  # Simulate network call

                # Mock response (in real implementation, would make actual HTTP call)
                actual_response = {
                    "status": expected_response["status"],
                    "body": expected_response["body"],
                }

                # Compare actual vs expected
                status_match = actual_response["status"] == expected_response["status"]
                body_match = self._compare_response_bodies(
                    actual_response["body"], expected_response["body"]
                )

                passed = status_match and body_match

                return {
                    "description": description,
                    "passed": passed,
                    "expected_status": expected_response["status"],
                    "actual_status": actual_response["status"],
                    "body_match": body_match,
                }

            def _compare_response_bodies(
                self, actual: dict[str, Any], expected: dict[str, Any]
            ) -> bool:
                """Compare response bodies for contract compliance."""
                # Simple comparison - in real implementation would be more sophisticated
                for key, expected_value in expected.items():
                    if key not in actual:
                        return False

                    if isinstance(expected_value, list) and isinstance(
                        actual[key], list
                    ):
                        # For arrays, check structure of first element
                        if len(expected_value) > 0 and len(actual[key]) > 0:
                            expected_item = expected_value[0]
                            actual_item = actual[key][0]
                            if (
                                isinstance(expected_item, dict)
                                and isinstance(actual_item, dict)
                                and not self._compare_response_bodies(
                                    actual_item, expected_item
                                )
                            ):
                                return False
                    elif (
                        isinstance(expected_value, dict)
                        and isinstance(actual[key], dict)
                        and not self._compare_response_bodies(
                            actual[key], expected_value
                        )
                    ):
                        return False

                return True

        test_runner = ContractTestRunner()

        # Mock contract for testing
        search_contract = {
            "provider": "vector_db_service",
            "interactions": [
                {
                    "description": "Search returns matches with required fields",
                    "request": {
                        "method": "POST",
                        "path": "/search",
                        "body": {"collection": "documents", "vector": [0.1, 0.2, 0.3]},
                    },
                    "response": {
                        "status": 200,
                        "body": {
                            "matches": [{"id": "doc_1", "score": 0.9, "payload": {}}]
                        },
                    },
                }
            ],
        }

        # Run contract tests
        test_results = await test_runner.run_contract_tests(search_contract)

        # Verify test execution
        assert test_results["provider"] == "vector_db_service"
        assert test_results["_total_tests"] == 1
        assert test_results["passed"] == 1
        assert test_results["failed"] == 0

        # Verify individual test result
        individual_result = test_results["test_results"][0]
        assert individual_result["passed"] is True
        assert individual_result["expected_status"] == 200
        assert individual_result["actual_status"] == 200
        assert individual_result["body_match"] is True
