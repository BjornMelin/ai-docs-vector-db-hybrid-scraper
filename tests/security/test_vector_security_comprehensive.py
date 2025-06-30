#!/usr/bin/env python3
"""
Comprehensive security test suite for vector validation fixes.

This module provides extensive testing for DoS protection, data integrity,
input sanitization, error handling, and edge cases across all vector fields
in the vector search models.

Coverage includes:
- Vector dimension DoS attacks
- Data integrity violations (NaN/Inf/invalid types)
- Input sanitization boundary testing
- Error message security validation
- Performance impact assessment
- Edge case validation
"""

import math
import sys
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.models.vector_search import (
    AdvancedHybridSearchRequest,
    BasicSearchRequest,
    DimensionError,
    SearchStage,
    SecureSearchResult,
    SecureVectorModel,
)


class TestVectorSecurityValidation:
    """Test suite for vector security validation across all models."""

    @pytest.fixture
    def valid_vector(self) -> list[float]:
        """Provide a valid test vector."""
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.fixture
    def model_constructors(self) -> dict[str, Callable]:
        """Provide constructors for all models with vector fields."""
        return {
            "SecureVectorModel": lambda v: SecureVectorModel(values=v),
            "SearchStage": lambda v: SearchStage(
                stage_name="test", query_vector=SecureVectorModel(values=v)
            ),
            "BasicSearchRequest": lambda v: BasicSearchRequest(
                query_vector=SecureVectorModel(values=v)
            ),
            "AdvancedHybridSearchRequest": lambda v: AdvancedHybridSearchRequest(
                query_vector=SecureVectorModel(values=v)
            ),
            # Skip SecureSearchResult due to payload dependency issues
        }

    def test_valid_vectors_accepted(
        self, valid_vector: list[float], model_constructors: dict[str, Callable]
    ) -> None:
        """Test that valid vectors are accepted by all models."""
        for model_name, constructor in model_constructors.items():
            model_instance = constructor(valid_vector)

            if model_name == "SecureVectorModel":
                vector_field = model_instance.values
            else:
                # For other models, get the nested vector values
                query_vector = getattr(model_instance, "query_vector", None)
                if query_vector and hasattr(query_vector, "values"):
                    vector_field = query_vector.values
                else:
                    vector_field = None

            assert vector_field == valid_vector, (
                f"{model_name} should accept valid vector"
            )

    def test_maximum_allowed_dimensions(
        self, model_constructors: dict[str, Callable]
    ) -> None:
        """Test that maximum allowed dimensions (4096) are accepted."""
        max_vector = [1.0] * 4096

        for model_name, constructor in model_constructors.items():
            model_instance = constructor(max_vector)

            if model_name == "SecureVectorModel":
                vector_field = model_instance.values
            else:
                # For other models, get the nested vector values
                query_vector = getattr(model_instance, "query_vector", None)
                if query_vector and hasattr(query_vector, "values"):
                    vector_field = query_vector.values
                else:
                    vector_field = None

            assert len(vector_field) == 4096, (
                f"{model_name} should accept 4096 dimensions"
            )

    def test_empty_vector_rejection(
        self, model_constructors: dict[str, Callable]
    ) -> None:
        """Test that empty vectors are rejected for DoS protection."""
        empty_vector = []

        for model_name, constructor in model_constructors.items():
            with pytest.raises(
                (ValueError, DimensionError), match=r"Vector cannot be empty.*security: DoS prevention"
            ):
                constructor(empty_vector)

    @pytest.mark.parametrize("oversized_length", [4097, 5000, 10000, 50000, 100000])
    def test_oversized_vector_rejection(
        self, oversized_length: int, model_constructors: dict[str, Callable]
    ) -> None:
        """Test that oversized vectors are rejected for DoS protection."""
        oversized_vector = [1.0] * oversized_length

        for model_name, constructor in model_constructors.items():
            with pytest.raises(
                (ValueError, DimensionError),
                match=r"Vector dimensions exceed maximum allowed.*security: DoS prevention",
            ):
                constructor(oversized_vector)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_infinite_and_nan_rejection(
        self, invalid_value: float, model_constructors: dict[str, Callable]
    ) -> None:
        """Test that NaN and infinite values are rejected for data integrity."""
        invalid_vector = [1.0, invalid_value, 3.0]

        for model_name, constructor in model_constructors.items():
            with pytest.raises(
                ValueError,
                match=r"Vector element.*contains invalid value.*security: data integrity",
            ):
                constructor(invalid_vector)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "string",
            None,
            {"attack": "payload"},
            [1, 2, 3],  # nested list
            object(),
        ],
    )
    def test_non_numeric_type_rejection(
        self, invalid_type: Any, model_constructors: dict[str, Callable]
    ) -> None:
        """Test that non-numeric types are rejected for type safety."""
        invalid_vector = [1.0, invalid_type, 3.0]

        for model_name, constructor in model_constructors.items():
            with pytest.raises(
                ValueError,
                match=r"Vector element.*must be numeric.*security: type validation",
            ):
                constructor(invalid_vector)

    def test_optional_vector_field_handling(self) -> None:
        """Test that optional vector fields (SearchResult.vector) handle None correctly."""
        # Test with None vector (should work)
        result_none = SearchResult(id="test", score=0.5, vector=None)
        assert result_none.vector is None

        # Test with valid vector (should work)
        valid_vector = [1.0, 2.0, 3.0]
        result_valid = SearchResult(id="test", score=0.5, vector=valid_vector)
        assert result_valid.vector == valid_vector

        # Test with invalid vector (should fail)
        invalid_vector = [float("nan")]
        with pytest.raises(ValueError, match=r"security: data integrity"):
            SearchResult(id="test", score=0.5, vector=invalid_vector)


class TestVectorSecurityFuzzing:
    """Fuzzing tests for vector security using hypothesis."""

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=4096
        )
    )
    def test_valid_vectors_property_based(self, vector: list[float]) -> None:
        """Property-based test that all valid finite vectors are accepted."""
        try:
            stage = SearchStage(
                stage_name="property-test", query_vector=SecureVectorModel(values=vector)
            )
            assert len(stage.query_vector.values) == len(vector)
            assert all(math.isfinite(v) for v in stage.query_vector.values)
        except ValueError:
            pytest.fail(f"Valid vector should be accepted: {vector[:5]}...")

    @given(st.lists(st.floats(), min_size=4097, max_size=10000))
    def test_oversized_vectors_property_based(self, vector: list[float]) -> None:
        """Property-based test that oversized vectors are rejected."""
        with pytest.raises((ValueError, DimensionError), match=r"security: DoS prevention"):
            SearchStage(
                stage_name="oversized-property-test", query_vector=SecureVectorModel(values=vector)
            )

    @given(
        st.lists(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(),
                st.none(),
                st.dictionaries(st.text(), st.text()),
            ),
            min_size=1,
            max_size=100,
        )
    )
    def test_invalid_vector_types_property_based(self, vector: list[Any]) -> None:
        """Property-based test that invalid vector types are rejected."""
        # Check if vector contains any invalid types
        has_invalid = any(
            not isinstance(v, (int, float))
            or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            for v in vector
        )

        if has_invalid:
            with pytest.raises((ValueError, TypeError)):
                SearchStage(
                    stage_name="invalid-property-test",
                    query_vector=SecureVectorModel(values=vector),
                )
        else:
            # Should succeed if all values are valid
            stage = SearchStage(
                stage_name="valid-property-test", query_vector=SecureVectorModel(values=vector)
            )
            assert len(stage.query_vector.values) == len(vector)


class TestVectorSecurityPerformanceImpact:
    """Test performance impact of security validations."""

    def test_validation_performance_small_vectors(self) -> None:
        """Test that validation doesn't significantly impact small vector performance."""
        vector = [1.0] * 100
        iterations = 1000

        start_time = time.perf_counter()
        for i in range(iterations):
            SearchStage(
                stage_name=f"perf-small-{i}", query_vector=SecureVectorModel(values=vector)
            )
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        assert avg_time_ms < 1.0, (
            f"Validation too slow for small vectors: {avg_time_ms:.3f}ms"
        )

    def test_validation_performance_large_vectors(self) -> None:
        """Test that validation performance scales reasonably with vector size."""
        vector = [1.0] * 4096  # Maximum allowed size
        iterations = 100

        start_time = time.perf_counter()
        for i in range(iterations):
            SearchStage(
                stage_name=f"perf-large-{i}", query_vector=SecureVectorModel(values=vector)
            )
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        assert avg_time_ms < 10.0, (
            f"Validation too slow for large vectors: {avg_time_ms:.3f}ms"
        )

    def test_early_rejection_performance(self) -> None:
        """Test that early rejection for oversized vectors is fast."""
        oversized_vector = [1.0] * 10000
        iterations = 1000

        start_time = time.perf_counter()
        for i in range(iterations):
            with pytest.raises((ValueError, DimensionError)):
                SearchStage(
                    stage_name=f"early-reject-{i}",
                    query_vector=SecureVectorModel(values=oversized_vector),
                )
        end_time = time.perf_counter()

        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        assert avg_time_ms < 0.5, f"Early rejection too slow: {avg_time_ms:.3f}ms"


class TestVectorSecurityErrorHandling:
    """Test error handling and message security."""

    def test_error_messages_no_data_leakage(self) -> None:
        """Test that error messages don't leak sensitive data."""
        malicious_vector = ["secret_data", "password123", "api_key"]

        with pytest.raises(ValueError) as exc_info:
            SearchStage(
                stage_name="malicious-data-test",
                query_vector=SecureVectorModel(values=malicious_vector),
            )

        error_message = str(exc_info.value)
        # Ensure error message doesn't contain the actual malicious data
        assert "secret_data" not in error_message
        assert "password123" not in error_message
        assert "api_key" not in error_message

        # But should contain security-relevant information
        assert "security:" in error_message
        assert "type validation" in error_message

    def test_consistent_error_format(self) -> None:
        """Test that security error messages follow consistent format."""
        test_cases = [
            ([], "security: DoS prevention"),
            ([1.0] * 5000, "security: DoS prevention"),
            ([1.0, float("nan")], "security: data integrity"),
            ([1.0, "string"], "security: type validation"),
        ]

        for invalid_vector, expected_security_tag in test_cases:
            with pytest.raises((ValueError, DimensionError)) as exc_info:
                SearchStage(
                    stage_name="consistent-error-test",
                    query_vector=SecureVectorModel(values=invalid_vector),
                )

            error_message = str(exc_info.value)
            assert expected_security_tag in error_message, (
                f"Missing security tag: {expected_security_tag}"
            )

    def test_error_handling_robustness(self) -> None:
        """Test error handling doesn't crash with extreme inputs."""
        extreme_cases = [
            # Very large numbers
            [sys.maxsize, -sys.maxsize],
            # Very small numbers
            [sys.float_info.min, -sys.float_info.min],
            # Mixed types that could cause type confusion
            [True, False, 1, 0],
            # Empty nested structures
            [[], {}, set()],
        ]

        for extreme_vector in extreme_cases:
            with pytest.raises((ValueError, TypeError)) as exc_info:
                SearchStage(
                    stage_name="extreme-robustness-test",
                    query_vector=SecureVectorModel(values=extreme_vector),
                )

            # Ensure we get a proper error, not a crash
            assert isinstance(exc_info.value, (ValueError, TypeError))
            error_message = str(exc_info.value)
            assert len(error_message) > 0


class TestVectorSecurityBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_boundary_dimension_limits(self) -> None:
        """Test vectors at boundary dimension limits."""
        # Test exactly at limits
        boundary_cases = [
            ([1.0], "minimum valid dimension"),
            ([1.0] * 4096, "maximum valid dimension"),
        ]

        for vector, description in boundary_cases:
            stage = SearchStage(
                stage_name="boundary-test", query_vector=SecureVectorModel(values=vector)
            )
            assert len(stage.query_vector.values) == len(vector), f"Failed for {description}"

    def test_boundary_numeric_limits(self) -> None:
        """Test vectors with boundary numeric values."""
        boundary_values = [
            1e-308,  # Very small positive
            -1e-308,  # Very small negative
            1e308,  # Very large positive
            -1e308,  # Very large negative
            0.0,  # Zero
            -0.0,  # Negative zero
        ]

        for value in boundary_values:
            vector = [value]
            stage = SearchStage(
                stage_name="numeric-boundary-test", query_vector=SecureVectorModel(values=vector)
            )
            assert len(stage.query_vector.values) == 1
            assert math.isfinite(stage.query_vector.values[0])

    def test_mixed_valid_invalid_vectors(self) -> None:
        """Test vectors mixing valid and invalid elements."""
        mixed_cases = [
            # Valid start, invalid middle
            [1.0, 2.0, float("nan"), 4.0],
            # Valid start, invalid end
            [1.0, 2.0, 3.0, float("inf")],
            # Invalid start, valid end
            [float("-inf"), 2.0, 3.0, 4.0],
        ]

        for mixed_vector in mixed_cases:
            with pytest.raises(ValueError, match=r"security: data integrity"):
                SearchStage(
                    stage_name="mixed-vector-test",
                    query_vector=SecureVectorModel(values=mixed_vector),
                )


class TestVectorSecurityConcurrency:
    """Test vector security under concurrent access."""

    def test_validation_thread_safety(self) -> None:
        """Test that validation is thread-safe (no race conditions)."""
        import queue
        import threading

        results = queue.Queue()
        vector = [1.0, 2.0, 3.0]

        def create_model():
            try:
                stage = SearchStage(
                    stage_name="thread-safety-test",
                    query_vector=SecureVectorModel(values=vector),
                )
                results.put(("success", stage.query_vector.values))
            except Exception as e:
                results.put(("error", str(e)))

        # Create multiple threads
        threads = []
        for _ in range(50):
            thread = threading.Thread(target=create_model)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            status, data = results.get()
            if status == "success":
                success_count += 1
                assert data == vector
            else:
                pytest.fail(f"Unexpected error in thread: {data}")

        assert success_count == 50, "All threads should succeed"

    def test_concurrent_invalid_vector_handling(self) -> None:
        """Test concurrent handling of invalid vectors."""
        import queue
        import threading

        results = queue.Queue()
        invalid_vector = [float("nan")]

        def try_invalid_model():
            try:
                SearchStage(
                    stage_name="concurrent-invalid-test",
                    query_vector=SecureVectorModel(values=invalid_vector),
                )
                results.put(("unexpected_success", None))
            except ValueError as e:
                if "security: data integrity" in str(e):
                    results.put(("expected_error", str(e)))
                else:
                    results.put(("wrong_error", str(e)))
            except Exception as e:
                results.put(("unexpected_error", str(e)))

        # Create multiple threads
        threads = []
        for _ in range(30):
            thread = threading.Thread(target=try_invalid_model)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        expected_errors = 0
        while not results.empty():
            status, data = results.get()
            if status == "expected_error":
                expected_errors += 1
            else:
                pytest.fail(f"Unexpected result: {status} - {data}")

        assert expected_errors == 30, "All threads should get expected security error"


@pytest.mark.integration
class TestVectorSecurityIntegration:
    """Integration tests for vector security across the entire system."""

    def test_end_to_end_security_validation(self) -> None:
        """Test security validation in realistic usage scenarios."""
        # Simulate a realistic search workflow
        valid_query_vector = [0.1] * 1536  # Typical OpenAI embedding size

        # Create search request using secure vector model
        secure_vector = SecureVectorModel(values=valid_query_vector)
        search_request = BasicSearchRequest(query_vector=secure_vector)

        assert len(search_request.query_vector.values) == 1536

        # Test that the same validation applies to all vector fields
        hybrid_request = AdvancedHybridSearchRequest(query_vector=secure_vector)

        assert len(hybrid_request.query_vector.values) == 1536

    def test_security_validation_consistency(self) -> None:
        """Test that security validation is consistent across all model types."""
        test_vectors = [
            ([1.0] * 5000, (ValueError, DimensionError), "DoS prevention"),  # Oversized
            ([float("nan")], ValueError, "data integrity"),  # Invalid value
            (["string"], ValueError, "type validation"),  # Wrong type
        ]

        model_classes = [
            ("SecureVectorModel", lambda v: SecureVectorModel(values=v)),
            ("SearchStage", lambda v: SearchStage(
                stage_name="consistency-test", query_vector=SecureVectorModel(values=v)
            )),
            ("BasicSearchRequest", lambda v: BasicSearchRequest(
                query_vector=SecureVectorModel(values=v)
            )),
            ("AdvancedHybridSearchRequest", lambda v: AdvancedHybridSearchRequest(
                query_vector=SecureVectorModel(values=v)
            )),
        ]

        for test_vector, expected_exception, expected_keyword in test_vectors:
            for model_name, model_constructor in model_classes:
                with pytest.raises(expected_exception) as exc_info:
                    model_constructor(test_vector)

                assert expected_keyword in str(exc_info.value), (
                    f"{model_name} should include '{expected_keyword}' in error"
                )


if __name__ == "__main__":
    # Run the security test suite
    pytest.main([__file__, "-v", "--tb=short"])
