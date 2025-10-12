"""Deterministic helpers for embedding-focused tests."""

from __future__ import annotations

import random


__all__ = ["EmbeddingTestUtils", "generate_embedding_vectors"]


class EmbeddingTestUtils:
    """Expose deterministic embedding generators for tests."""

    @staticmethod
    def generate_test_embeddings(
        count: int,
        dim: int,
        seed: int | None = None,
    ) -> list[list[float]]:
        """Return ``count`` embeddings with ``dim`` dimensions each.

        Args:
            count: Number of embeddings to create.
            dim: Size of each embedding vector.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated embeddings as nested lists of floats.

        Raises:
            ValueError: If ``count`` or ``dim`` is non-positive.
        """

        if count <= 0:
            msg = "count must be a positive integer"
            raise ValueError(msg)
        if dim <= 0:
            msg = "dim must be a positive integer"
            raise ValueError(msg)

        rng = random.Random(seed)
        return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(count)]


def generate_embedding_vectors(
    count: int,
    dim: int,
    *,
    seed: int | None = None,
) -> list[list[float]]:
    """Functional wrapper around the deterministic embedding generator."""

    return EmbeddingTestUtils.generate_test_embeddings(count=count, dim=dim, seed=seed)
