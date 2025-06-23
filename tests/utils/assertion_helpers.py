"""Standardized assertion helpers for consistent test patterns.

This module provides reusable assertion functions that implement common
validation patterns across all test categories. These helpers ensure
consistent error messages and validation logic throughout the test suite.
"""

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any
from typing import TypeVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

T = TypeVar("T")


def assert_valid_response(
    response: dict[str, Any],
    expected_keys: list[str] | None = None,
    status_key: str = "status",
    expected_status: str = "success",
) -> None:
    """Assert that a response is valid and contains expected structure.

    Args:
        response: The response dictionary to validate
        expected_keys: Optional list of keys that must be present
        status_key: Key name for status field (default: "status")
        expected_status: Expected status value (default: "success")
    """
    assert isinstance(response, dict), (
        f"Response must be a dictionary, got {type(response)}"
    )

    # Check status if status_key exists
    if status_key in response:
        assert response[status_key] == expected_status, (
            f"Expected status '{expected_status}', got '{response[status_key]}'"
        )

    # Check required keys
    if expected_keys:
        missing_keys = [key for key in expected_keys if key not in response]
        assert not missing_keys, f"Missing required keys: {missing_keys}"


def assert_error_response(
    response: dict[str, Any],
    expected_error_type: str | None = None,
    error_key: str = "error",
    message_key: str = "message",
) -> None:
    """Assert that a response represents an error with proper structure.

    Args:
        response: The response dictionary to validate
        expected_error_type: Optional expected error type
        error_key: Key name for error field (default: "error")
        message_key: Key name for message field (default: "message")
    """
    assert isinstance(response, dict), (
        f"Response must be a dictionary, got {type(response)}"
    )
    assert error_key in response, f"Error response must contain '{error_key}' field"

    if expected_error_type:
        assert response[error_key] == expected_error_type, (
            f"Expected error type '{expected_error_type}', got '{response[error_key]}'"
        )

    # Error responses should have a message
    assert message_key in response, f"Error response must contain '{message_key}' field"
    assert isinstance(response[message_key], str), "Error message must be a string"
    assert len(response[message_key]) > 0, "Error message cannot be empty"


def assert_pagination_response(
    response: dict[str, Any],
    items_key: str = "items",
    total_key: str = "total",
    page_key: str = "page",
    per_page_key: str = "per_page",
) -> None:
    """Assert that a response contains proper pagination structure.

    Args:
        response: The response dictionary to validate
        items_key: Key name for items array (default: "items")
        total_key: Key name for total count (default: "total")
        page_key: Key name for current page (default: "page")
        per_page_key: Key name for items per page (default: "per_page")
    """
    assert isinstance(response, dict), (
        f"Response must be a dictionary, got {type(response)}"
    )

    # Check required pagination fields
    required_fields = [items_key, total_key, page_key, per_page_key]
    missing_fields = [field for field in required_fields if field not in response]
    assert not missing_fields, f"Missing pagination fields: {missing_fields}"

    # Validate field types and values
    assert isinstance(response[items_key], list), f"'{items_key}' must be a list"
    assert isinstance(response[total_key], int), f"'{total_key}' must be an integer"
    assert isinstance(response[page_key], int), f"'{page_key}' must be an integer"
    assert isinstance(response[per_page_key], int), (
        f"'{per_page_key}' must be an integer"
    )

    # Validate logical constraints
    assert response[total_key] >= 0, "Total count cannot be negative"
    assert response[page_key] >= 1, "Page number must be at least 1"
    assert response[per_page_key] >= 1, "Per page count must be at least 1"
    assert len(response[items_key]) <= response[per_page_key], (
        "Items count cannot exceed per_page limit"
    )


class AssertionHelpers:
    """Collection of advanced assertion utilities for testing."""

    @staticmethod
    def assert_document_structure(document: dict[str, Any]) -> None:
        """Assert that a document has the expected structure.

        Args:
            document: Document dictionary to validate
        """
        required_fields = ["id", "content", "metadata"]
        missing_fields = [field for field in required_fields if field not in document]
        assert not missing_fields, f"Document missing required fields: {missing_fields}"

        assert isinstance(document["id"], str), "Document ID must be a string"
        assert len(document["id"]) > 0, "Document ID cannot be empty"
        assert isinstance(document["content"], str), "Document content must be a string"
        assert isinstance(document["metadata"], dict), (
            "Document metadata must be a dictionary"
        )

    @staticmethod
    def assert_search_result(result: dict[str, Any], min_score: float = 0.0) -> None:
        """Assert that a search result has proper structure and score.

        Args:
            result: Search result dictionary to validate
            min_score: Minimum acceptable score (default: 0.0)
        """
        required_fields = ["id", "score", "content"]
        missing_fields = [field for field in required_fields if field not in result]
        assert not missing_fields, (
            f"Search result missing required fields: {missing_fields}"
        )

        assert isinstance(result["score"], int | float), "Score must be a number"
        assert result["score"] >= min_score, (
            f"Score {result['score']} below minimum {min_score}"
        )
        assert result["score"] <= 1.0, f"Score {result['score']} above maximum 1.0"

    @staticmethod
    def assert_embedding_vector(vector: list[float], expected_dimension: int) -> None:
        """Assert that an embedding vector has correct structure.

        Args:
            vector: Embedding vector to validate
            expected_dimension: Expected vector dimension
        """
        assert isinstance(vector, list), "Embedding vector must be a list"
        assert len(vector) == expected_dimension, (
            f"Vector dimension {len(vector)} != expected {expected_dimension}"
        )
        assert all(isinstance(x, int | float) for x in vector), (
            "All vector components must be numbers"
        )

    @staticmethod
    def assert_api_response_time(
        response_time: float, max_time: float = 1.0, operation: str = "API call"
    ) -> None:
        """Assert that an API response time is within acceptable limits.

        Args:
            response_time: Response time in seconds
            max_time: Maximum acceptable time (default: 1.0 second)
            operation: Name of the operation for error messages
        """
        assert isinstance(response_time, int | float), "Response time must be a number"
        assert response_time >= 0, "Response time cannot be negative"
        assert response_time <= max_time, (
            f"{operation} took {response_time:.3f}s, exceeds limit of {max_time}s"
        )

    @staticmethod
    def assert_configuration_valid(
        config: dict[str, Any], required_sections: list[str]
    ) -> None:
        """Assert that a configuration dictionary is valid.

        Args:
            config: Configuration dictionary to validate
            required_sections: List of required configuration sections
        """
        assert isinstance(config, dict), "Configuration must be a dictionary"

        missing_sections = [
            section for section in required_sections if section not in config
        ]
        assert not missing_sections, (
            f"Configuration missing required sections: {missing_sections}"
        )

        # Validate each section is also a dictionary
        for section in required_sections:
            assert isinstance(config[section], dict), (
                f"Configuration section '{section}' must be a dictionary"
            )

    @staticmethod
    def assert_json_serializable(obj: Any, context: str = "object") -> None:
        """Assert that an object is JSON serializable.

        Args:
            obj: Object to test for JSON serializability
            context: Context description for error messages
        """
        try:
            json.dumps(obj)
        except (TypeError, ValueError) as e:
            pytest.fail(f"{context} is not JSON serializable: {e}")

    @staticmethod
    def assert_timestamp_recent(
        timestamp: str | datetime | float, max_age_seconds: float = 60.0
    ) -> None:
        """Assert that a timestamp is recent (within specified age).

        Args:
            timestamp: Timestamp to validate (ISO string, datetime, or Unix timestamp)
            max_age_seconds: Maximum age in seconds (default: 60.0)
        """
        now = datetime.utcnow()

        if isinstance(timestamp, str):
            # Parse ISO format timestamp
            try:
                timestamp_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {timestamp}")
        elif isinstance(timestamp, datetime):
            timestamp_dt = timestamp
        elif isinstance(timestamp, int | float):
            # Unix timestamp
            timestamp_dt = datetime.utcfromtimestamp(timestamp)
        else:
            pytest.fail(f"Unsupported timestamp type: {type(timestamp)}")

        age_seconds = (now - timestamp_dt).total_seconds()
        assert age_seconds <= max_age_seconds, (
            f"Timestamp is {age_seconds:.1f}s old, exceeds limit of {max_age_seconds}s"
        )
        assert age_seconds >= -1.0, (
            "Timestamp cannot be in the future (allowing 1s clock skew)"
        )


# New standardized assertion helpers


def assert_successful_response(
    response: dict[str, Any],
    expected_data: Any = None,
    required_fields: list[str] | None = None,
) -> None:
    """Assert that response indicates success with optional data validation.

    Args:
        response: Response dictionary to validate
        expected_data: Expected data content (optional)
        required_fields: List of required fields in response

    Raises:
        AssertionError: If response doesn't indicate success or lacks required fields
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert response.get("success") is True, (
        f"Response should indicate success, got: {response}"
    )
    assert "error" not in response or response["error"] is None, (
        f"Response should not contain errors, got: {response.get('error')}"
    )

    if expected_data is not None:
        assert response.get("data") == expected_data, (
            f"Response data mismatch. Expected: {expected_data}, Got: {response.get('data')}"
        )

    if required_fields:
        for field in required_fields:
            assert field in response, f"Response missing required field: {field}"


def assert_error_response_standardized(
    response: dict[str, Any],
    expected_error_code: str | None = None,
    expected_message_fragment: str | None = None,
    should_have_details: bool = False,
) -> None:
    """Assert that response indicates an error with specific characteristics.

    Args:
        response: Response dictionary to validate
        expected_error_code: Expected error code
        expected_message_fragment: Fragment that should appear in error message
        should_have_details: Whether error should include additional details

    Raises:
        AssertionError: If response doesn't match expected error pattern
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert response.get("success") is False, "Response should indicate failure"
    assert "error" in response and response["error"] is not None, (
        "Response should contain error information"
    )

    error = response["error"]

    if expected_error_code:
        assert error.get("code") == expected_error_code, (
            f"Error code mismatch. Expected: {expected_error_code}, Got: {error.get('code')}"
        )

    if expected_message_fragment:
        message = error.get("message", "")
        assert expected_message_fragment in message, (
            f"Error message should contain '{expected_message_fragment}', got: {message}"
        )

    if should_have_details:
        assert "details" in error, "Error should include additional details"


def assert_valid_document_chunk(
    chunk: dict[str, Any], required_fields: list[str] | None = None
) -> None:
    """Assert that document chunk has all required fields and valid values.

    Args:
        chunk: Document chunk to validate
        required_fields: Additional required fields beyond defaults

    Raises:
        AssertionError: If chunk is invalid
    """
    default_required_fields = ["content", "title", "url", "chunk_index"]
    all_required_fields = default_required_fields + (required_fields or [])

    assert isinstance(chunk, dict), "Chunk must be a dictionary"

    for field in all_required_fields:
        assert field in chunk, f"Chunk missing required field: {field}"

    # Validate specific field types and values
    assert isinstance(chunk["chunk_index"], int), "chunk_index must be an integer"
    assert chunk["chunk_index"] >= 0, "chunk_index must be non-negative"
    assert isinstance(chunk["content"], str), "content must be a string"
    assert len(chunk["content"]) > 0, "content must not be empty"
    assert isinstance(chunk["title"], str), "title must be a string"
    assert isinstance(chunk["url"], str), "url must be a string"


def assert_valid_vector_point(
    point: dict[str, Any],
    expected_vector_dim: int | None = None,
    required_payload_fields: list[str] | None = None,
) -> None:
    """Assert that vector point has valid structure and content.

    Args:
        point: Vector point to validate
        expected_vector_dim: Expected vector dimensionality
        required_payload_fields: Required fields in payload

    Raises:
        AssertionError: If point is invalid
    """
    assert isinstance(point, dict), "Point must be a dictionary"
    assert "id" in point, "Point must have an id"
    assert "vector" in point, "Point must have a vector"
    assert "payload" in point, "Point must have a payload"

    # Validate vector
    vector = point["vector"]
    assert isinstance(vector, list), "Vector must be a list"
    assert all(isinstance(x, int | float) for x in vector), (
        "Vector must contain numeric values"
    )

    if expected_vector_dim:
        assert len(vector) == expected_vector_dim, (
            f"Vector dimension mismatch. Expected: {expected_vector_dim}, Got: {len(vector)}"
        )

    # Validate payload
    payload = point["payload"]
    assert isinstance(payload, dict), "Payload must be a dictionary"

    if required_payload_fields:
        for field in required_payload_fields:
            assert field in payload, f"Payload missing required field: {field}"


def assert_performance_within_threshold(
    execution_time: float, max_time_seconds: float, operation_name: str = "Operation"
) -> None:
    """Assert that operation completed within performance threshold.

    Args:
        execution_time: Actual execution time in seconds
        max_time_seconds: Maximum allowed time in seconds
        operation_name: Name of operation for error messages

    Raises:
        AssertionError: If execution time exceeds threshold
    """
    assert execution_time <= max_time_seconds, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"exceeding threshold of {max_time_seconds}s"
    )


def assert_memory_usage_within_limit(
    memory_usage_mb: float, max_memory_mb: float, operation_name: str = "Operation"
) -> None:
    """Assert that operation memory usage is within limits.

    Args:
        memory_usage_mb: Actual memory usage in MB
        max_memory_mb: Maximum allowed memory in MB
        operation_name: Name of operation for error messages

    Raises:
        AssertionError: If memory usage exceeds limit
    """
    assert memory_usage_mb <= max_memory_mb, (
        f"{operation_name} used {memory_usage_mb:.2f}MB, "
        f"exceeding limit of {max_memory_mb}MB"
    )


def assert_mock_called_with_pattern(
    mock_obj: MagicMock | AsyncMock,
    expected_calls: list[dict[str, Any]],
    exact_match: bool = False,
) -> None:
    """Assert that mock was called with expected pattern.

    Args:
        mock_obj: Mock object to validate
        expected_calls: List of expected call patterns
        exact_match: Whether to require exact call sequence match

    Raises:
        AssertionError: If mock calls don't match expected pattern
    """
    actual_calls = mock_obj.call_args_list

    if exact_match:
        assert len(actual_calls) == len(expected_calls), (
            f"Call count mismatch. Expected: {len(expected_calls)}, "
            f"Got: {len(actual_calls)}"
        )

    for i, expected_call in enumerate(expected_calls):
        if i >= len(actual_calls):
            raise AssertionError(f"Missing expected call {i + 1}: {expected_call}")

        actual_call = actual_calls[i]

        # Validate positional arguments
        if "args" in expected_call:
            actual_args = actual_call[0] if actual_call else ()
            expected_args = expected_call["args"]
            assert actual_args == expected_args, (
                f"Call {i + 1} args mismatch. Expected: {expected_args}, "
                f"Got: {actual_args}"
            )

        # Validate keyword arguments
        if "kwargs" in expected_call:
            actual_kwargs = actual_call[1] if len(actual_call) > 1 else {}
            expected_kwargs = expected_call["kwargs"]
            for key, value in expected_kwargs.items():
                assert key in actual_kwargs, f"Call {i + 1} missing kwarg: {key}"
                assert actual_kwargs[key] == value, (
                    f"Call {i + 1} kwarg {key} mismatch. Expected: {value}, "
                    f"Got: {actual_kwargs[key]}"
                )


async def assert_async_operation_completes(
    async_operation: Callable[[], Any],
    timeout_seconds: float = 5.0,
    operation_name: str = "Async operation",
) -> Any:
    """Assert that async operation completes within timeout.

    Args:
        async_operation: Async function to execute
        timeout_seconds: Maximum time to wait
        operation_name: Name for error messages

    Returns:
        Result of the async operation

    Raises:
        AssertionError: If operation times out or fails
    """
    try:
        result = await asyncio.wait_for(async_operation(), timeout=timeout_seconds)
        return result
    except TimeoutError:
        raise AssertionError(f"{operation_name} timed out after {timeout_seconds}s")
    except Exception as e:
        raise AssertionError(f"{operation_name} failed with error: {e}")


def assert_collection_has_size(
    collection: list | dict | str,
    expected_size: int,
    collection_name: str = "Collection",
) -> None:
    """Assert that collection has expected size.

    Args:
        collection: Collection to check
        expected_size: Expected size
        collection_name: Name for error messages

    Raises:
        AssertionError: If size doesn't match
    """
    actual_size = len(collection)
    assert actual_size == expected_size, (
        f"{collection_name} size mismatch. Expected: {expected_size}, "
        f"Got: {actual_size}"
    )


def assert_all_items_have_type(
    collection: list[Any], expected_type: type[T], collection_name: str = "Collection"
) -> None:
    """Assert that all items in collection have expected type.

    Args:
        collection: Collection to check
        expected_type: Expected type for all items
        collection_name: Name for error messages

    Raises:
        AssertionError: If any item has wrong type
    """
    for i, item in enumerate(collection):
        assert isinstance(item, expected_type), (
            f"{collection_name} item {i} has wrong type. "
            f"Expected: {expected_type.__name__}, Got: {type(item).__name__}"
        )


def assert_security_headers_present(
    headers: dict[str, str], required_headers: list[str] | None = None
) -> None:
    """Assert that security headers are present in HTTP response.

    Args:
        headers: HTTP headers dictionary
        required_headers: List of required security headers

    Raises:
        AssertionError: If required security headers are missing
    """
    default_required = [
        "x-content-type-options",
        "x-frame-options",
        "x-xss-protection",
        "strict-transport-security",
    ]
    all_required = required_headers or default_required

    # Convert to lowercase for case-insensitive comparison
    lower_headers = {k.lower(): v for k, v in headers.items()}

    for header in all_required:
        header_lower = header.lower()
        assert header_lower in lower_headers, (
            f"Missing required security header: {header}"
        )


def assert_api_rate_limit_respected(
    requests_per_second: float, max_allowed_rps: float, tolerance: float = 0.1
) -> None:
    """Assert that API requests respect rate limiting.

    Args:
        requests_per_second: Actual request rate
        max_allowed_rps: Maximum allowed request rate
        tolerance: Tolerance factor for rate checking

    Raises:
        AssertionError: If rate limit is exceeded
    """
    allowed_with_tolerance = max_allowed_rps * (1 + tolerance)
    assert requests_per_second <= allowed_with_tolerance, (
        f"Rate limit exceeded. Actual: {requests_per_second:.2f} RPS, "
        f"Max allowed: {max_allowed_rps:.2f} RPS (with {tolerance * 100}% tolerance)"
    )


def assert_accessibility_compliant(
    accessibility_report: dict[str, Any],
    max_violations: int = 0,
    severity_threshold: str = "minor",
) -> None:
    """Assert that accessibility report shows compliance.

    Args:
        accessibility_report: Accessibility scanning report
        max_violations: Maximum allowed violations
        severity_threshold: Minimum severity level to consider

    Raises:
        AssertionError: If accessibility violations exceed threshold
    """
    violations = accessibility_report.get("violations", [])

    # Filter by severity
    severity_levels = {"minor": 1, "moderate": 2, "serious": 3, "critical": 4}
    threshold_level = severity_levels.get(severity_threshold, 1)

    significant_violations = [
        v
        for v in violations
        if severity_levels.get(v.get("impact", "minor"), 1) >= threshold_level
    ]

    violation_count = len(significant_violations)
    assert violation_count <= max_violations, (
        f"Accessibility violations exceed threshold. "
        f"Found {violation_count} violations of severity {severity_threshold} or higher, "
        f"maximum allowed: {max_violations}"
    )


def assert_contract_compliance(
    actual_response: dict[str, Any],
    expected_schema: dict[str, Any],
    strict_mode: bool = True,
) -> None:
    """Assert that API response complies with contract schema.

    Args:
        actual_response: Actual API response
        expected_schema: Expected response schema
        strict_mode: Whether to enforce strict schema compliance

    Raises:
        AssertionError: If response doesn't match schema
    """
    try:
        import jsonschema

        jsonschema.validate(actual_response, expected_schema)
    except ImportError:
        # Fallback validation if jsonschema not available
        if strict_mode:
            required_fields = expected_schema.get("required", [])
            for field in required_fields:
                assert field in actual_response, (
                    f"Response missing required field: {field}"
                )
    except jsonschema.ValidationError as e:
        raise AssertionError(f"Response doesn't match contract schema: {e.message}")


# Performance timing decorator
def time_execution(func: Callable) -> Callable:
    """Decorator to time function execution for performance assertions."""
    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            return result, execution_time

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            return result, execution_time

        return sync_wrapper


# Context manager for resource assertion
class assert_resource_cleanup:
    """Context manager to assert proper resource cleanup."""

    def __init__(
        self, resource_tracker: Callable[[], int], resource_name: str = "resource"
    ):
        """Initialize resource cleanup assertion.

        Args:
            resource_tracker: Function that returns current resource count
            resource_name: Name of resource for error messages
        """
        self.resource_tracker = resource_tracker
        self.resource_name = resource_name
        self.initial_count = 0

    def __enter__(self):
        self.initial_count = self.resource_tracker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_count = self.resource_tracker()
        assert final_count == self.initial_count, (
            f"{self.resource_name} leak detected. "
            f"Initial: {self.initial_count}, Final: {final_count}"
        )
