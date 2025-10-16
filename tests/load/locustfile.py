"""Locust load-testing scenarios exercising the FastAPI search API."""

from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task


_DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "rag" / "golden_set.jsonl"
)


@dataclass(frozen=True)
class SearchExample:
    """Golden dataset entry used for load testing."""

    query: str
    collection: str
    limit: int
    filters: dict[str, Any]


def _load_examples(path: Path) -> list[SearchExample]:
    """Load golden dataset queries from disk."""
    if not path.exists():
        return [
            SearchExample(
                query="how to configure qdrant replication",
                collection="documentation",
                limit=5,
                filters={},
            )
        ]

    examples: list[SearchExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            examples.append(
                SearchExample(
                    query=str(payload.get("query", "hybrid search")),
                    collection=str(payload.get("collection", "documentation")),
                    limit=int(payload.get("limit", 5)),
                    filters=dict(payload.get("filters", {})),
                )
            )
    return examples


_DATASET_ENV = os.getenv("AI_DOCS_BENCHMARK_DATASET")
_DATASET_PATH = Path(_DATASET_ENV).expanduser() if _DATASET_ENV else _DEFAULT_DATASET
_EXAMPLES = _load_examples(_DATASET_PATH) or _load_examples(_DEFAULT_DATASET)


class SearchUser(HttpUser):
    """Simulate RAG search traffic hitting the FastAPI service."""

    wait_time = between(0.1, 1.0)

    def on_start(self) -> None:
        """Prime the dataset iterator and configure default headers."""
        self._examples = itertools.cycle(random.sample(_EXAMPLES, len(_EXAMPLES)))
        self.client.headers.update({"Content-Type": "application/json"})

    @task(4)
    def search(self) -> None:
        """Execute a search request using an example query."""
        example = next(self._examples)
        response = self.client.post(
            "/api/v1/search",
            json={
                "query": example.query,
                "collection": example.collection,
                "limit": example.limit,
                "filters": example.filters or None,
            },
            name="POST /api/v1/search",
        )
        response.raise_for_status()

    @task(1)
    def list_collections(self) -> None:
        """Enumerate vector collections for observability checks."""
        response = self.client.get("/api/v1/collections", name="GET /collections")
        response.raise_for_status()
