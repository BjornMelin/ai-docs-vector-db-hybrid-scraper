"""Smoke-level tests for the Qdrant seeding utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from scripts.eval import seed_qdrant
from scripts.eval.seed_qdrant import _load_corpus, _seed_collection


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    """Write a minimal corpus file for seeding."""
    data_path = tmp_path / "corpus.json"
    rows = [
        {
            "id": "doc-1",
            "text": "LangGraph orchestrates retrieval and generation stages.",
            "doc_path": "docs/langgraph.md",
            "metadata": {"collection": "golden_eval"},
        }
    ]
    data_path.write_text(json.dumps(rows), encoding="utf-8")
    return data_path


def test_load_corpus_round_trips(corpus_path: Path) -> None:
    """_load_corpus should deserialize JSON corpus files."""
    records = _load_corpus(corpus_path)
    assert records[0]["id"] == "doc-1"
    assert records[0]["doc_path"] == "docs/langgraph.md"


def test_seed_collection_invokes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """_seed_collection should recreate and upsert as expected."""
    corpus = [
        {
            "id": "doc-1",
            "text": "LangGraph orchestrates retrieval and generation stages.",
            "doc_path": "docs/langgraph.md",
            "metadata": {"collection": "golden_eval"},
        }
    ]

    class _FakeEmbedder:
        def __init__(self, *_, **__):
            pass

        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _text in texts]

    class _FakeClient:
        def __init__(self) -> None:
            self.recreate_called = False
            self.upsert_points = None

        def recreate_collection(self, *_, **__) -> None:
            self.recreate_called = True

        def upsert(self, *, collection_name: str, points) -> None:
            self.upsert_points = (collection_name, list(points))

    monkeypatch.setattr(seed_qdrant, "TextEmbedding", _FakeEmbedder)

    client = _FakeClient()
    _seed_collection(
        cast(seed_qdrant.QdrantClient, client),
        corpus,
        collection_name="golden_eval",
        model_name="stub-model",
        recreate=True,
    )

    assert client.recreate_called is True
    assert client.upsert_points is not None
    collection_name, points = client.upsert_points
    assert collection_name == "golden_eval"
    assert len(points) == 1
