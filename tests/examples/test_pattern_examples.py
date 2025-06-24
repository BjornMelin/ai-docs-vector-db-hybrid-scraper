"""Example tests demonstrating standardized 2025 patterns.

This module provides examples of how to implement tests following the
standardized patterns defined in the TEST_PATTERNS_STYLE_GUIDE.md.
These examples serve as templates for consistent test implementation.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import standardized helpers
from tests.utils.assertion_helpers import (
    assert_async_operation_completes,
    assert_error_response_standardized,
    assert_mock_called_with_pattern,
    assert_performance_within_threshold,
    assert_successful_response,
    assert_valid_document_chunk,
)
from tests.utils.test_factories import (
    ChunkFactory,
    DocumentFactory,
    ResponseFactory,
    VectorFactory,
    quick_success_response,
)


class TestStandardizedPatterns:
    """Example test class demonstrating standardized patterns.

    This class shows how to structure test classes with proper:
    - Type annotations
    - Docstrings
    - Async patterns
    - Fixture usage
    - Assertion patterns
    """

    @pytest.mark.asyncio
    async def test_async_operation_standard_pattern(
        self, mock_service: AsyncMock, sample_document_data: dict[str, Any]
    ) -> None:
        """Test async operation using standardized patterns.

        This test demonstrates:
        - Proper async test structure
        - Type annotations
        - Standardized assertions
        - Mock configuration

        Args:
            mock_service: Mocked async service
            sample_document_data: Test document data
        """
        # Arrange - Configure mocks and test data
        expected_result = quick_success_response({"processed": True})
        mock_service.process_document.return_value = expected_result

        # Act - Execute the operation under test
        result = await some_async_service_function(sample_document_data)

        # Assert - Verify results using standardized helpers
        assert_successful_response(result, expected_data={"processed": True})
        mock_service.process_document.assert_called_once_with(sample_document_data)

    @pytest.mark.parametrize(
        "input_data,expected_output",
        [
            pytest.param(
                {"query": "test query", "limit": 5},
                {"results": [], "total": 0},
                id="empty_results",
            ),
            pytest.param(
                {"query": "python", "limit": 10},
                {"results": ["doc1", "doc2"], "total": 2},
                id="with_results",
            ),
        ],
    )
    async def test_parametrized_search_standard_pattern(
        self,
        input_data: dict[str, Any],
        expected_output: dict[str, Any],
        mock_search_service: AsyncMock,
    ) -> None:
        """Test search with various inputs using parametrization.

        This test demonstrates:
        - Proper parametrization with IDs
        - Type annotations for parameters
        - Clear test data organization

        Args:
            input_data: Search input parameters
            expected_output: Expected search results
            mock_search_service: Mocked search service
        """
        # Arrange
        mock_search_service.search.return_value = expected_output

        # Act
        result = await search_function(**input_data)

        # Assert
        assert result == expected_output
        mock_search_service.search.assert_called_once_with(**input_data)

    def test_sync_operation_with_factory_pattern(
        self, mock_document_processor: MagicMock
    ) -> None:
        """Test synchronous operation using factory pattern for data.

        This test demonstrates:
        - Using test data factories
        - Sync test patterns
        - Proper mock validation

        Args:
            mock_document_processor: Mocked document processor
        """
        # Arrange - Use factory for consistent test data
        test_document = DocumentFactory.create_document(
            title="Factory Test Document", content="Content created by factory"
        )
        expected_chunks = [
            ChunkFactory.create_chunk(
                content="Content created by factory", title="Factory Test Document"
            )
        ]
        mock_document_processor.chunk_document.return_value = expected_chunks

        # Act
        result = process_document_sync(test_document)

        # Assert - Validate each chunk
        assert len(result) == 1
        assert_valid_document_chunk(result[0])
        assert result[0]["title"] == "Factory Test Document"

    @pytest.mark.performance
    async def test_performance_requirement_standard_pattern(
        self, mock_vector_service: AsyncMock
    ) -> None:
        """Test performance requirements using standardized timing.

        This test demonstrates:
        - Performance testing patterns
        - Timing measurement
        - Performance assertions

        Args:
            mock_vector_service: Mocked vector service
        """
        # Arrange
        test_vectors = [VectorFactory.create_vector() for _ in range(100)]
        mock_vector_service.batch_process.return_value = quick_success_response()

        # Act - Measure execution time
        start_time = time.perf_counter()
        result = await batch_process_vectors(test_vectors)
        execution_time = time.perf_counter() - start_time

        # Assert - Check performance and correctness
        assert_successful_response(result)
        assert_performance_within_threshold(
            execution_time,
            max_time_seconds=1.0,
            operation_name="Batch vector processing",
        )

    @pytest.mark.asyncio
    async def test_error_handling_standard_pattern(
        self, mock_failing_service: AsyncMock
    ) -> None:
        """Test error handling using standardized patterns.

        This test demonstrates:
        - Error condition testing
        - Exception handling
        - Error response validation

        Args:
            mock_failing_service: Service configured to fail
        """
        # Arrange - Configure service to fail
        mock_failing_service.process.side_effect = ConnectionError("Network failed")

        # Act & Assert - Test exception handling
        with pytest.raises(ServiceError) as exc_info:
            await failing_service_operation()

        assert "Network failed" in str(exc_info.value)
        assert exc_info.value.error_code == "CONNECTION_ERROR"

    async def test_mock_call_pattern_validation(
        self, mock_api_client: AsyncMock
    ) -> None:
        """Test mock call patterns using standardized validation.

        This test demonstrates:
        - Mock call pattern validation
        - Complex call verification
        - Argument validation

        Args:
            mock_api_client: Mocked API client
        """
        # Arrange
        test_data = {"key": "value"}
        mock_api_client.post.return_value = quick_success_response()

        # Act
        await api_service_call(test_data)

        # Assert - Validate call patterns
        expected_calls = [
            {"args": ("/api/endpoint",), "kwargs": {"json": test_data, "timeout": 30}}
        ]
        assert_mock_called_with_pattern(mock_api_client.post, expected_calls)

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self) -> None:
        """Test async operation timeout handling.

        This test demonstrates:
        - Async timeout testing
        - Operation completion validation
        - Timeout assertion patterns
        """

        async def slow_operation() -> str:
            """Simulate slow async operation."""
            await asyncio.sleep(0.1)
            return "completed"

        # Test that operation completes within timeout
        result = await assert_async_operation_completes(
            slow_operation, timeout_seconds=1.0, operation_name="Slow operation test"
        )

        assert result == "completed"

    @pytest.mark.integration
    @pytest.mark.database
    async def test_database_integration_pattern(
        self, test_database_session: Any, sample_document_data: dict[str, Any]
    ) -> None:
        """Test database integration using standardized patterns.

        This test demonstrates:
        - Integration test patterns
        - Database session usage
        - Transaction handling

        Args:
            test_database_session: Test database session
            sample_document_data: Test document data
        """
        # Arrange
        document = DocumentFactory.create_document(**sample_document_data)

        # Act - Database operation
        saved_document = await save_document_to_db(document, test_database_session)

        # Assert - Verify database state
        assert saved_document["id"] is not None
        assert saved_document["title"] == document["title"]

        # Verify document can be retrieved
        retrieved = await get_document_from_db(
            saved_document["id"], test_database_session
        )
        assert retrieved is not None
        assert retrieved["title"] == document["title"]

    def test_builder_pattern_usage(self) -> None:
        """Test using builder pattern for complex test data.

        This test demonstrates:
        - Builder pattern usage
        - Complex data construction
        - Fluent interface patterns
        """
        from tests.utils.test_factories import TestDataBuilder

        # Arrange - Build complex test data
        complex_data = (
            TestDataBuilder()
            .with_id("test-123")
            .with_url("https://example.com/complex")
            .with_title("Complex Test Document")
            .with_metadata({"complexity": "high", "tags": ["test", "example"]})
            .with_status("processed")
            .build()
        )

        # Act
        result = process_complex_data(complex_data)

        # Assert
        assert result["processed"] is True
        assert result["metadata"]["complexity"] == "high"

    @pytest.mark.security
    async def test_security_validation_pattern(
        self, mock_auth_service: AsyncMock, security_test_data: dict[str, Any]
    ) -> None:
        """Test security validation using standardized patterns.

        This test demonstrates:
        - Security test patterns
        - Input validation testing
        - Authentication testing

        Args:
            mock_auth_service: Mocked authentication service
            security_test_data: Security test data
        """
        # Arrange
        malicious_input = security_test_data["malicious_inputs"][0]
        mock_auth_service.validate_input.return_value = {
            "valid": False,
            "reason": "malicious",
        }

        # Act
        result = await validate_user_input(malicious_input)

        # Assert
        assert_error_response_standardized(
            result,
            expected_error_code="INVALID_INPUT",
            expected_message_fragment="malicious",
        )

    @pytest.mark.accessibility
    def test_accessibility_compliance_pattern(
        self, mock_accessibility_scanner: MagicMock
    ) -> None:
        """Test accessibility compliance using standardized patterns.

        This test demonstrates:
        - Accessibility test patterns
        - Compliance validation
        - Report analysis

        Args:
            mock_accessibility_scanner: Mocked accessibility scanner
        """
        from tests.utils.assertion_helpers import assert_accessibility_compliant

        # Arrange
        accessibility_report = {
            "violations": [
                {"impact": "minor", "description": "Missing alt text"},
                {"impact": "moderate", "description": "Low contrast"},
            ],
            "passes": ["Good heading structure", "Keyboard navigation"],
        }
        mock_accessibility_scanner.scan.return_value = accessibility_report

        # Act
        result = scan_for_accessibility_issues("https://example.com")

        # Assert
        assert_accessibility_compliant(
            result, max_violations=2, severity_threshold="minor"
        )


# Fixture examples for standardized patterns


@pytest.fixture
async def mock_service() -> AsyncMock:
    """Mock async service with standardized configuration.

    Returns:
        Configured AsyncMock for service testing
    """
    service = AsyncMock()
    service.process_document.return_value = quick_success_response()
    service.health_check.return_value = {"status": "healthy"}
    return service


@pytest.fixture
def mock_search_service() -> AsyncMock:
    """Mock search service with standardized configuration.

    Returns:
        Configured AsyncMock for search testing
    """
    service = AsyncMock()
    service.search.return_value = {"results": [], "total": 0}
    service.index_document.return_value = quick_success_response()
    return service


@pytest.fixture
def mock_document_processor() -> MagicMock:
    """Mock document processor with standardized configuration.

    Returns:
        Configured MagicMock for document processing testing
    """
    processor = MagicMock()
    processor.chunk_document.return_value = []
    processor.extract_metadata.return_value = {}
    return processor


@pytest.fixture
async def mock_vector_service() -> AsyncMock:
    """Mock vector service with standardized configuration.

    Returns:
        Configured AsyncMock for vector operations testing
    """
    service = AsyncMock()
    service.generate_embedding.return_value = VectorFactory.create_vector()
    service.batch_process.return_value = quick_success_response()
    return service


@pytest.fixture
def mock_failing_service() -> AsyncMock:
    """Mock service configured to fail for error testing.

    Returns:
        AsyncMock configured to simulate failures
    """
    service = AsyncMock()
    service.process.side_effect = Exception("Service failure")
    return service


@pytest.fixture
async def mock_api_client() -> AsyncMock:
    """Mock API client with standardized configuration.

    Returns:
        Configured AsyncMock for API testing
    """
    client = AsyncMock()
    client.post.return_value = quick_success_response()
    client.get.return_value = quick_success_response()
    return client


@pytest.fixture
def sample_document_data() -> dict[str, Any]:
    """Sample document data following standardized structure.

    Returns:
        Dictionary with sample document data
    """
    return {
        "title": "Sample Document",
        "url": "https://example.com/doc",
        "content": "This is sample content for testing.",
    }


# Example functions being tested (would normally be imported)


async def some_async_service_function(data: dict[str, Any]) -> dict[str, Any]:
    """Example async function for testing."""
    return quick_success_response({"processed": True})


async def search_function(**kwargs) -> dict[str, Any]:
    """Example search function for testing."""
    return {"results": [], "total": 0}


def process_document_sync(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Example sync document processing function."""
    return [ChunkFactory.create_chunk(content=document["content"])]


async def batch_process_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    """Example batch vector processing function."""
    return quick_success_response()


async def failing_service_operation() -> None:
    """Example function that raises ServiceError."""
    raise ServiceError("Operation failed", error_code="CONNECTION_ERROR")


async def api_service_call(data: dict[str, Any]) -> dict[str, Any]:
    """Example API service call function."""
    return quick_success_response()


async def save_document_to_db(document: dict[str, Any], session: Any) -> dict[str, Any]:
    """Example database save function."""
    document["id"] = "saved-123"
    return document


async def get_document_from_db(doc_id: str, session: Any) -> dict[str, Any]:
    """Example database retrieval function."""
    return {"id": doc_id, "title": "Retrieved Document"}


def process_complex_data(data: dict[str, Any]) -> dict[str, Any]:
    """Example complex data processing function."""
    return {"processed": True, "metadata": data.get("metadata", {})}


async def validate_user_input(user_input: str) -> dict[str, Any]:
    """Example input validation function."""
    if "malicious" in user_input.lower():
        return ResponseFactory.create_error_response(
            error_code="INVALID_INPUT", message="Input contains malicious content"
        )
    return quick_success_response()


def scan_for_accessibility_issues(url: str) -> dict[str, Any]:
    """Example accessibility scanning function."""
    return {"violations": [], "passes": ["All checks passed"]}


# Custom exception for testing
class ServiceError(Exception):
    """Custom service error for testing."""

    def __init__(self, message: str, error_code: str = "GENERAL_ERROR"):
        super().__init__(message)
        self.error_code = error_code
