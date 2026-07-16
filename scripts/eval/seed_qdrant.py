"""Seed Qdrant collection with deterministic documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as http_models

from src.services.vector_db.payload_schema import normalize_document


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    """Return the corpus records stored in JSON format."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _seed_collection(
    client: QdrantClient,
    corpus: list[dict[str, Any]],
    *,
    collection_name: str,
    model_name: str,
    recreate: bool,
) -> None:
    """Create or refresh the collection using FastEmbed embeddings."""
    # Extract texts and generate embeddings
    texts = [record["text"] for record in corpus]
    embedder = TextEmbedding(model_name=model_name)
    embeddings = list(embedder.embed(texts))
    if not embeddings:
        raise RuntimeError("Corpus is empty; aborting seed.")

    # Create collection if requested
    vector_size = len(embeddings[0])
    if recreate:
        client.recreate_collection(
            collection_name,
            vectors_config=http_models.VectorParams(
                size=vector_size,
                distance=http_models.Distance.COSINE,
            ),
        )

    # Build point structures for upsert
    points = []
    for record, vector in zip(corpus, embeddings, strict=True):
        document = normalize_document(
            Document(
                page_content=record["text"],
                metadata={
                    "doc_id": str(record["id"]),
                    "tenant": collection_name,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "source": record["doc_path"],
                    "uri_or_path": record["doc_path"],
                    "category": record["metadata"].get("category"),
                },
            ),
            id_hint=str(record["id"]),
        )
        vector_list = [float(value) for value in vector]
        points.append(
            http_models.PointStruct(
                id=str(document.id),
                vector=vector_list,
                payload={
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                },
            )
        )

    # Insert points into collection
    client.upsert(collection_name=collection_name, points=points)


def main() -> None:
    """CLI entrypoint for seeding the golden_eval collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host to seed",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port",
    )
    parser.add_argument(
        "--collection",
        default="golden_eval",
        help="Collection name to create or refresh",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="FastEmbed model name",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/golden_corpus.json"),
        help="Path to the corpus JSON file",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before inserting documents",
    )
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus)
    client = QdrantClient(host=args.host, port=args.port)
    _seed_collection(
        client,
        corpus,
        collection_name=args.collection,
        model_name=args.model,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
