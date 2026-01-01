"""Centralized test doubles for the test suite."""

from tests.doubles.services import (
    FakeCache,
    FakeCrawlManager,
    FakeEmbeddingManager,
    FakeMCPServer,
    FakeProjectStorage,
    FakeServiceContainer,
    FakeVectorStoreService,
)


__all__ = [
    "FakeCache",
    "FakeCrawlManager",
    "FakeEmbeddingManager",
    "FakeMCPServer",
    "FakeProjectStorage",
    "FakeServiceContainer",
    "FakeVectorStoreService",
]
