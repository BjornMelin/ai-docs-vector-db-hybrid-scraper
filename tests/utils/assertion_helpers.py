"""Standardized assertion helpers for consistent test patterns.

This module provides reusable assertion functions that follow consistent
patterns for testing different aspects of the system.
"""

import asyncio
from typing import Any

import pytest


def assert_successful_response(
    response: dict[str, Any],
    expected_data: dict[str, Any] | None = None,
    expected_status: int = 200,
) -> None:
    """Assert that a response indicates success.

    Args:
        response: Response dictionary to validate
        expected_data: Expected data in response (optional)
        expected_status: Expected status code (default: 200)
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert (
        "status_code" not in response or response.get("status_code") == expected_status
    ), f"Expected status {expected_status}, got {response.get('status_code')}"

    if expected_data:
        for key, value in expected_data.items():
            assert key in response, f"Expected key '{key}' not found in response"
            assert response[key] == value, (
                f"Expected {key}={value}, got {response[key]}"
            )


def assert_error_response_standardized(
    response: dict[str, Any],
    expected_error_code: str,
    expected_message_fragment: str | None = None,
    expected_status: int = 400,
) -> None:
    """Assert that an error response follows standardized format.

    Args:
        response: Error response dictionary
        expected_error_code: Expected error code
        expected_message_fragment: Fragment that should be in error message
        expected_status: Expected HTTP status code
    """
    assert isinstance(response, dict), "Error response must be a dictionary"
    assert response.get("status_code", 400) == expected_status, (
        f"Expected error status {expected_status}, got {response.get('status_code')}"
    )

    # Check for standardized error fields
    assert "error" in response, "Error response must contain 'error' field"
    error_info = response["error"]

    if isinstance(error_info, dict):
        assert error_info.get("code") == expected_error_code, (
            f"Expected error code '{expected_error_code}', got "
            f"'{error_info.get('code')}'"
        )

        if expected_message_fragment:
            message = error_info.get("message", "")
            assert expected_message_fragment.lower() in message.lower(), (
                f"Expected message fragment '{expected_message_fragment}' not found "
                f"in '{message}'"
            )
    else:
        # Fallback for simple string error messages
        assert expected_error_code.lower() in str(error_info).lower(), (
            f"Expected error code '{expected_error_code}' not found in error message"
        )


def assert_valid_document_chunk(
    chunk: dict[str, Any],
    required_fields: list[str] | None = None,
) -> None:
    """Assert that a document chunk has valid structure.

    Args:
        chunk: Document chunk dictionary
        required_fields: Additional required fields beyond defaults
    """
    assert isinstance(chunk, dict), "Document chunk must be a dictionary"

    default_required = ["content", "title"]
    all_required = default_required + (required_fields or [])

    for field in all_required:
        assert field in chunk, f"Required field '{field}' missing from document chunk"
        assert chunk[field], f"Field '{field}' cannot be empty"

    # Content should be reasonable length
    content = chunk["content"]
    assert isinstance(content, str), "Content must be a string"
    assert 10 <= len(content) <= 10000, f"Content length {len(content)} is unreasonable"


def assert_performance_within_threshold(
    execution_time: float,
    max_time_seconds: float,
    operation_name: str = "operation",
) -> None:
    """Assert that execution time is within acceptable performance limits.

    Args:
        execution_time: Actual execution time in seconds
        max_time_seconds: Maximum acceptable time
        operation_name: Name of operation for error messages
    """
    assert execution_time >= 0, "Execution time cannot be negative"
    assert execution_time <= max_time_seconds, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"exceeding threshold of {max_time_seconds:.3f}s"
    )


def assert_mock_called_with_pattern(
    mock_call: Any,
    expected_call_patterns: list[dict[str, Any]],
) -> None:
    """Assert that a mock was called with expected patterns.

    Args:
        mock_call: Mock call object to validate
        expected_call_patterns: List of expected call patterns
    """
    actual_calls = mock_call.call_args_list

    for i, expected_pattern in enumerate(expected_call_patterns):
        assert i < len(actual_calls), (
            f"Expected at least {i + 1} calls, got {len(actual_calls)}"
        )

        actual_args, actual_kwargs = actual_calls[i]

        # Check positional arguments
        if "args" in expected_pattern:
            expected_args = expected_pattern["args"]
            assert actual_args == expected_args, (
                f"Call {i} args mismatch: expected {expected_args}, got {actual_args}"
            )

        # Check keyword arguments
        if "_kwargs" in expected_pattern:
            expected_kwargs = expected_pattern["_kwargs"]
            for key, value in expected_kwargs.items():
                assert key in actual_kwargs, (
                    f"Expected kwarg '{key}' not found in call {i}"
                )
                assert actual_kwargs[key] == value, (
                    f"Call {i} kwarg '{key}' mismatch: expected {value}, got "
                    f"{actual_kwargs[key]}"
                )


def assert_accessibility_compliant(
    accessibility_report: dict[str, Any],
    max_violations: int = 0,
    severity_threshold: str = "moderate",
) -> None:
    """Assert that accessibility scan results are compliant.

    Args:
        accessibility_report: Accessibility scan results
        max_violations: Maximum allowed violations
        severity_threshold: Minimum severity to count as violation
    """
    assert isinstance(accessibility_report, dict), (
        "Accessibility report must be a dictionary"
    )

    violations = accessibility_report.get("violations", [])
    assert isinstance(violations, list), "Violations must be a list"

    severity_levels = {"minor": 1, "moderate": 2, "serious": 3, "critical": 4}
    threshold_level = severity_levels.get(severity_threshold, 2)

    significant_violations = [
        v
        for v in violations
        if severity_levels.get(v.get("impact", "minor"), 1) >= threshold_level
    ]

    assert len(significant_violations) <= max_violations, (
        f"Found {len(significant_violations)} significant accessibility violations "
        f"(threshold: {severity_threshold}), maximum allowed: {max_violations}"
    )


async def assert_async_operation_completes(
    operation: callable,
    timeout_seconds: float = 5.0,
    operation_name: str = "async operation",
) -> Any:
    """Assert that an async operation completes within timeout.

    Args:
        operation: Async callable to test
        timeout_seconds: Maximum time to wait
        operation_name: Name for error messages

    Returns:
        Result of the operation
    """
    try:
        return await asyncio.wait_for(operation(), timeout=timeout_seconds)
    except TimeoutError:
        pytest.fail(
            f"{operation_name} did not complete within {timeout_seconds} seconds"
        )


def assert_valid_embedding_vector(
    vector: list[float],
    expected_dimensions: int | None = None,
) -> None:
    """Assert that an embedding vector is valid.

    Args:
        vector: Embedding vector to validate
        expected_dimensions: Expected vector dimensions (optional)
    """
    assert isinstance(vector, list), "Embedding vector must be a list"
    assert len(vector) > 0, "Embedding vector cannot be empty"

    # Check that all elements are numbers
    for i, value in enumerate(vector):
        assert isinstance(value, (int, float)), (
            f"Vector element {i} is not a number: {value}"
        )

    # Check reasonable value range (-1 to 1 for normalized embeddings)
    for i, value in enumerate(vector):
        assert -2 <= value <= 2, f"Vector element {i} has unreasonable value: {value}"

    if expected_dimensions:
        assert len(vector) == expected_dimensions, (
            f"Vector has {len(vector)} dimensions, expected {expected_dimensions}"
        )


def assert_search_results_relevant(
    results: list[dict[str, Any]],
    query: str,
    min_relevance_score: float = 0.1,
) -> None:
    """Assert that search results are relevant to the query.

    Args:
        results: Search results to validate
        query: Original search query
        min_relevance_score: Minimum acceptable relevance score
    """
    assert isinstance(results, list), "Search results must be a list"
    assert len(results) > 0, "Search results cannot be empty"

    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} must be a dictionary"
        assert "score" in result, f"Result {i} missing 'score' field"
        assert "content" in result, f"Result {i} missing 'content' field"

        score = result["score"]
        assert isinstance(score, (int, float)), f"Result {i} score must be numeric"
        assert score >= min_relevance_score, (
            f"Result {i} relevance score {score} below minimum {min_relevance_score}"
        )

        # Basic relevance check - result should contain query terms
        content = result["content"].lower()
        query_terms = query.lower().split()
        relevant_terms = sum(1 for term in query_terms if term in content)
        assert relevant_terms > 0, f"Result {i} appears irrelevant to query '{query}'"
