"""Unit tests for content intelligence utility helpers."""

from __future__ import annotations

import math

import pytest

from ._dependency_stubs import load_content_intelligence_module


cosine_similarity = load_content_intelligence_module("utils").cosine_similarity


class TestCosineSimilarity:
    """Validate cosine similarity helper edge cases and expected behaviour."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have perfect similarity.

        Returns:
            None: This test asserts the cosine similarity equals one.
        """
        vector = [1.0, 2.0, 3.0]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should return zero similarity.

        Returns:
            None: This test asserts the cosine similarity equals zero.
        """
        first = [1.0, 0.0, 0.0]
        second = [0.0, 1.0, 0.0]
        assert cosine_similarity(first, second) == pytest.approx(0.0)

    def test_handles_invalid_input(self) -> None:
        """Non-numeric or zero-magnitude inputs return neutral similarity.

        Returns:
            None: This test asserts invalid inputs produce neutral similarity.
        """
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
        assert cosine_similarity([math.nan, 1.0], [1.0, 2.0]) == 0.0

    def test_allows_mismatched_vector_lengths(self) -> None:
        """Gracefully handle vectors with different lengths using minimum overlap.

        Returns:
            None: This test asserts similarity is computed using overlapping entries.
        """
        assert cosine_similarity([1.0, 0.0, 2.0], [1.0, 2.0]) == pytest.approx(0.2)

    def test_casts_numeric_strings(self) -> None:
        """String representations of numbers are accepted after conversion.

        Returns:
            None: This test asserts string inputs yield expected similarity.
        """
        assert cosine_similarity(["1", "0"], [1, 0]) == pytest.approx(1.0)
