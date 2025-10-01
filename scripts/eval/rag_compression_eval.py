"""Evaluate deterministic contextual compression against a labelled dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import get_config
from src.services.query_processing.rag import DeterministicContextCompressor
from src.services.vector_db.service import VectorStoreService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a JSON file containing evaluation samples.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional override for the collection name used during evaluation.",
    )
    return parser.parse_args()


async def _load_vector_service(collection_override: str | None) -> VectorStoreService:
    config = get_config()
    if collection_override:
        config.qdrant.collection_name = collection_override
    service = VectorStoreService(config=config)
    await service.initialize()
    return service


def _build_documents(raw_documents: list[dict[str, Any]]) -> list[Document]:
    documents: list[Document] = []
    for raw in raw_documents:
        content = str(raw.get("content") or raw.get("text") or "")
        metadata = raw.get("metadata")
        metadata_copy = dict(metadata) if isinstance(metadata, dict) else {}
        documents.append(Document(page_content=content, metadata=metadata_copy))
    return documents


async def _evaluate(  # pylint: disable=too-many-locals
    dataset_path: Path, collection_override: str | None
) -> None:
    vector_service = await _load_vector_service(collection_override)
    compressor = DeterministicContextCompressor.from_rag_settings(
        vector_service, get_config().rag
    )

    if not compressor.enabled:
        print(
            "Compression is disabled in the active configuration; nothing to evaluate."
        )
        return

    with dataset_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    total_samples = 0
    recall_hits = 0
    recall_total = 0
    aggregate_tokens_before = 0
    aggregate_tokens_after = 0
    aggregate_documents_compressed = 0

    for entry in dataset:
        query = str(entry.get("query", "")).strip()
        raw_documents = entry.get("documents") or []
        if not query or not raw_documents:
            continue

        documents = _build_documents(raw_documents)
        compressed_docs, stats = await compressor.compress(query, documents)
        total_samples += 1
        aggregate_tokens_before += stats.tokens_before
        aggregate_tokens_after += stats.tokens_after
        aggregate_documents_compressed += stats.documents_compressed

        relevant_phrases = entry.get("relevant_phrases")
        if isinstance(relevant_phrases, list) and relevant_phrases:
            recall_total += len(relevant_phrases)
            compressed_text = " \n".join(doc.page_content for doc in compressed_docs)
            recall_hits += sum(
                1 for phrase in relevant_phrases if phrase and phrase in compressed_text
            )

    await vector_service.cleanup()

    if total_samples == 0:
        print("No valid samples found in the dataset.")
        return

    reduction = 0.0
    if aggregate_tokens_before:
        reduction = 1.0 - (aggregate_tokens_after / aggregate_tokens_before)

    recall = None
    if recall_total:
        recall = recall_hits / recall_total

    print("=== Compression Evaluation Summary ===")
    print(f"Samples evaluated: {total_samples}")
    print(f"Documents compressed: {aggregate_documents_compressed}")
    print(f"Tokens before: {aggregate_tokens_before}")
    print(f"Tokens after:  {aggregate_tokens_after}")
    print(f"Token reduction: {reduction:.2%}")
    if recall is not None:
        print(f"Recall proxy (phrase retention): {recall:.2%}")
    else:
        print("Recall proxy: n/a (no relevant phrases supplied)")


def main() -> None:
    args = _parse_args()
    asyncio.run(_evaluate(args.input, args.collection))


if __name__ == "__main__":
    main()
