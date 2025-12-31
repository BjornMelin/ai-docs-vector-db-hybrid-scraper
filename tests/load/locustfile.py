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


_FALLBACK_EXAMPLES = [
    SearchExample(
        query="how to configure qdrant replication",
        collection="documentation",
        limit=5,
        filters={},
    )
]


def _load_examples(path: Path) -> list[SearchExample]:
    """Load golden dataset queries from disk."""
    if not path.exists():
        return list(_FALLBACK_EXAMPLES)

    examples: list[SearchExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                limit_raw = payload.get("limit", 5)
                limit_val = int(limit_raw) if limit_raw is not None else 5
            except (ValueError, TypeError):
                limit_val = 5

            filters_raw = payload.get("filters")
            filters_val = filters_raw if isinstance(filters_raw, dict) else {}

            examples.append(
                SearchExample(
                    query=str(payload.get("query", "hybrid search")),
                    collection=str(payload.get("collection", "documentation")),
                    limit=limit_val,
                    filters=filters_val,
                )
            )
    return examples


_DATASET_ENV = (os.getenv("AI_DOCS_BENCHMARK_DATASET") or "").strip()
_DATASET_PATH = Path(_DATASET_ENV).expanduser() if _DATASET_ENV else None
if _DATASET_PATH and _DATASET_PATH.exists():
    _EXAMPLES = _load_examples(_DATASET_PATH) or _load_examples(_DEFAULT_DATASET)
else:
    _EXAMPLES = _load_examples(_DEFAULT_DATASET)

if not _EXAMPLES:
    _EXAMPLES = list(_FALLBACK_EXAMPLES)


class SearchUser(HttpUser):
    """Simulate RAG search traffic hitting the FastAPI service."""

    wait_time = between(0.1, 1.0)

    def on_start(self) -> None:
        """Prime the dataset iterator and configure default headers."""
        if _EXAMPLES:
            self._examples = itertools.cycle(random.sample(_EXAMPLES, len(_EXAMPLES)))
        else:
            # Fallback to a single default example when no data is available
            self._examples = itertools.cycle(
                [
                    SearchExample(
                        query="hybrid search",
                        collection="documentation",
                        limit=5,
                        filters={},
                    )
                ]
            )
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
        response = self.client.get(
            "/api/v1/collections", name="GET /api/v1/collections"
        )
        response.raise_for_status()
