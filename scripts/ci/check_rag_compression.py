"""CI quality gate for LangChain-powered contextual compression."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class _PrecomputedEmbeddings(Embeddings):
    """Embeddings interface backed by dataset-provided vectors."""

    def __init__(
        self,
        *,
        query_embedding: list[float],
        sentence_embeddings: dict[str, list[float]],
    ) -> None:
        self._query_embedding = query_embedding
        self._sentence_embeddings = sentence_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            try:
                embedding = self._sentence_embeddings[text]
            except KeyError as exc:  # pragma: no cover - dataset mismatch
                msg = f"Missing embedding for sentence: {text!r}"
                raise TypeError(msg) from exc
            embeddings.append(list(embedding))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        _ = text  # maintain compatibility with Embeddings signature
        return list(self._query_embedding)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Dataset must be a list of samples")
    return data


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


# pylint: disable=too-many-locals


async def _run(
    dataset_path: Path, min_reduction: float, min_recall: float | None
) -> bool:
    """Execute compression quality gate evaluation against dataset.

    Args:
        dataset_path: Path to evaluation dataset.
        min_reduction: Minimum required token reduction ratio.
        min_recall: Optional minimum recall threshold.

    Returns:
        True if quality thresholds met.
    """
    samples = _load_dataset(dataset_path)

    total_tokens_before = 0
    total_tokens_after = 0
    recall_total = 0
    recall_hits = 0

    for sample in samples:
        query_embedding = sample.get("query_embedding")
        sentence_embeddings = sample.get("sentence_embeddings")
        if not isinstance(query_embedding, list) or not isinstance(
            sentence_embeddings, dict
        ):
            raise TypeError("Sample missing embeddings")

        query = str(sample.get("query", "")).strip()
        if not query:
            raise ValueError("Sample missing 'query'")

        documents_raw = sample.get("documents")
        if not isinstance(documents_raw, list) or not documents_raw:
            raise ValueError("Sample missing 'documents'")

        documents = [
            Document(
                page_content=str(item.get("content", "")),
                metadata=item.get("metadata")
                if isinstance(item.get("metadata"), dict)
                else {},
            )
            for item in documents_raw
        ]

        embeddings = _PrecomputedEmbeddings(
            query_embedding=query_embedding,
            sentence_embeddings=sentence_embeddings,
        )
        pipeline = DocumentCompressorPipeline(
            transformers=[
                EmbeddingsRedundantFilter(embeddings=embeddings),
                EmbeddingsFilter(
                    embeddings=embeddings,
                    similarity_threshold=0.75,
                ),
            ]
        )

        compressed_docs = pipeline.compress_documents(documents, query=query)

        total_tokens_before += sum(
            _estimate_tokens(doc.page_content) for doc in documents
        )
        total_tokens_after += sum(
            _estimate_tokens(doc.page_content) for doc in compressed_docs
        )

        relevant_phrases = sample.get("relevant_phrases")
        if isinstance(relevant_phrases, list) and relevant_phrases:
            compressed_text = " \n".join(doc.page_content for doc in compressed_docs)
            for phrase in relevant_phrases:
                if phrase:
                    recall_total += 1
                    if phrase in compressed_text:
                        recall_hits += 1

    if total_tokens_before == 0:
        raise ValueError("Dataset produced zero baseline tokens")

    reduction = 1.0 - (total_tokens_after / total_tokens_before)
    if reduction < min_reduction:
        print(
            f"Compression reduction {reduction:.2%} below minimum {min_reduction:.2%}",
            file=sys.stderr,
        )
        return False

    if recall_total and min_recall is not None:
        recall = recall_hits / recall_total
        if recall < min_recall:
            print(
                f"Compression recall {recall:.2%} below minimum {min_recall:.2%}",
                file=sys.stderr,
            )
            return False

    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the compression evaluation dataset",
    )
    parser.add_argument(
        "--min-reduction",
        type=float,
        default=0.3,
        help="Minimum acceptable token reduction ratio (default: 0.3)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="Minimum acceptable recall proxy (default: 0.8)",
    )
    return parser.parse_args()


def main() -> int:
    """Execute CLI entry point for the compression quality gate."""
    args = _parse_args()
    try:
        success = asyncio.run(
            _run(
                dataset_path=args.dataset,
                min_reduction=args.min_reduction,
                min_recall=args.min_recall,
            )
        )
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
    ) as exc:  # pragma: no cover - CLI safeguard
        print(f"Compression quality gate failed: {exc}", file=sys.stderr)
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
