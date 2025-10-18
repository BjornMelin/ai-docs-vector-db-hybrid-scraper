"""Evaluate LangChain contextual compression pipeline."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_core.documents import Document


try:
    from langchain_community.document_transformers import EmbeddingsRedundantFilter
    from langchain_community.embeddings import (
        FastEmbedEmbeddings,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "FastEmbedEmbeddings requires the 'langchain-community' package."
    ) from exc

from src.config import get_settings
from src.infrastructure.bootstrap import container_session, ensure_container
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


async def _load_vector_service(
    collection_override: str | None,
    *,
    settings: Any | None = None,
) -> VectorStoreService:
    config = settings or get_settings()
    container = await ensure_container(settings=config)
    service = container.vector_store_service()
    if service is None:
        raise RuntimeError("Vector store service unavailable")
    if collection_override:
        service.collection_name = collection_override
    if hasattr(service, "is_initialized") and not service.is_initialized():
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


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


# pylint: disable=too-many-locals,too-many-statements


async def _evaluate(  # pylint: disable=too-many-locals
    dataset_path: Path, collection_override: str | None
) -> None:
    """Execute contextual compression evaluation against dataset.

    Args:
        dataset_path: Path to evaluation dataset.
        collection_override: Optional collection name override.
    """
    config = get_settings()
    exit_stack = AsyncExitStack()
    vector_service: VectorStoreService | None = None

    try:
        # Initialize vector service
        load_service = _load_vector_service
        if Mock is not None and isinstance(load_service, Mock):
            vector_service = await load_service(collection_override)
        else:
            await exit_stack.enter_async_context(
                container_session(settings=config, force_reload=True)
            )
            vector_service = await load_service(collection_override, settings=config)

        # Check compression configuration
        rag_config = config.rag
        if not rag_config.compression_enabled:
            print(
                "Compression is disabled in the active configuration; "
                "nothing to evaluate."
            )
            return

        # Setup compression pipeline
        fastembed_config = getattr(vector_service.config, "fastembed", None)
        model_name = getattr(fastembed_config, "dense_model", None)
        embeddings = FastEmbedEmbeddings(
            model_name=cast(str, model_name or "BAAI/bge-small-en-v1.5")
        )
        compressor = DocumentCompressorPipeline(
            transformers=[
                EmbeddingsRedundantFilter(embeddings=embeddings),
                EmbeddingsFilter(
                    embeddings=embeddings,
                    similarity_threshold=rag_config.compression_similarity_threshold,
                ),
            ]
        )

        # Load evaluation dataset
        with dataset_path.open(encoding="utf-8") as handle:
            dataset = json.load(handle)

        # Initialize metrics tracking
        total_samples = 0
        recall_hits = 0
        recall_total = 0
        aggregate_tokens_before = 0
        aggregate_tokens_after = 0
        aggregate_documents_compressed = 0

        # Process each evaluation sample
        for entry in dataset:
            query = str(entry.get("query", "")).strip()
            raw_documents = entry.get("documents") or []
            if not query or not raw_documents:
                continue

            documents = _build_documents(raw_documents)
            compressed_docs = compressor.compress_documents(documents, query=query)
            total_samples += 1

            # Calculate token metrics
            tokens_before = sum(_estimate_tokens(doc.page_content) for doc in documents)
            tokens_after = sum(
                _estimate_tokens(doc.page_content) for doc in compressed_docs
            )
            aggregate_tokens_before += tokens_before
            aggregate_tokens_after += tokens_after
            aggregate_documents_compressed += max(
                0, len(documents) - len(compressed_docs)
            )

            # Calculate recall metrics
            relevant_phrases = entry.get("relevant_phrases")
            if isinstance(relevant_phrases, list) and relevant_phrases:
                recall_total += len(relevant_phrases)
                compressed_text = " \n".join(
                    doc.page_content for doc in compressed_docs
                )
                recall_hits += sum(
                    1
                    for phrase in relevant_phrases
                    if phrase and phrase in compressed_text
                )

    finally:
        # Cleanup resources
        if vector_service is not None:
            await vector_service.cleanup()
        await exit_stack.aclose()

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
    print(f"Documents removed: {aggregate_documents_compressed}")
    print(f"Tokens before: {aggregate_tokens_before}")
    print(f"Tokens after:  {aggregate_tokens_after}")
    print(f"Token reduction: {reduction:.2%}")
    if recall is not None:
        print(f"Recall proxy (phrase retention): {recall:.2%}")
    else:
        print("Recall proxy: n/a (no relevant phrases supplied)")


def main() -> None:
    """Run the compression evaluation CLI entry point."""
    args = _parse_args()
    asyncio.run(_evaluate(args.input, args.collection))


if __name__ == "__main__":
    main()
