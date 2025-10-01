"""Deterministic contextual compression for RAG pipelines."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document

from src.services.errors import EmbeddingServiceError


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True, frozen=True)
class CompressionConfig:
    """Configuration values controlling deterministic compression."""

    enabled: bool = True
    similarity_threshold: float = 0.75
    mmr_lambda: float = 0.65
    token_ratio: float = 0.6
    absolute_max_tokens: int = 400
    min_sentences: int = 1
    max_sentences: int = 8


@dataclass(slots=True)
class CompressionStats:
    """Aggregate statistics produced by the compression pass."""

    documents_processed: int = 0
    documents_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def reduction_ratio(self) -> float:
        """Return the overall token reduction ratio."""

        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialisable view of the stats."""

        return {
            "documents_processed": self.documents_processed,
            "documents_compressed": self.documents_compressed,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction_ratio": self.reduction_ratio,
        }


class DeterministicContextCompressor:
    """Apply deterministic sentence-level compression using embeddings."""

    def __init__(self, vector_service: Any, config: CompressionConfig) -> None:
        self._vector_service = vector_service
        self._config = config
        self._last_stats: CompressionStats | None = None

    @classmethod
    def from_rag_settings(
        cls,
        vector_service: Any,
        rag_config: Any,
    ) -> DeterministicContextCompressor:
        """Create a compressor from the runtime RAG configuration."""

        def _get(attr: str, default: Any) -> Any:
            return getattr(rag_config, attr, default)

        config = CompressionConfig(
            enabled=_get("compression_enabled", True),
            similarity_threshold=_get("compression_similarity_threshold", 0.75),
            mmr_lambda=_get("compression_mmr_lambda", 0.65),
            token_ratio=_get("compression_token_ratio", 0.6),
            absolute_max_tokens=_get("compression_absolute_max_tokens", 400),
            min_sentences=_get("compression_min_sentences", 1),
            max_sentences=_get("compression_max_sentences", 8),
        )
        return cls(vector_service, config)

    @property
    def last_stats(self) -> CompressionStats | None:
        """Return statistics collected during the last compression run."""

        return self._last_stats

    @property
    def enabled(self) -> bool:
        """Indicate whether compression is active."""

        return self._config.enabled

    async def compress(  # pylint: disable=too-many-locals
        self, query: str, documents: Sequence[Document]
    ) -> tuple[list[Document], CompressionStats]:
        """Compress the supplied documents for the given query."""

        stats = CompressionStats()
        if not self._config.enabled or not documents:
            self._last_stats = stats
            return list(documents), stats

        try:
            query_embedding = np.asarray(
                await self._vector_service.embed_query(query), dtype=float
            )
        except EmbeddingServiceError:
            # Bubble the error upstream, but do not mutate documents partially.
            raise
        except Exception as exc:  # pragma: no cover - defensive safety net
            raise EmbeddingServiceError(
                f"Failed to embed query for compression: {exc}"
            ) from exc

        compressed_documents: list[Document] = []
        for document in documents:
            stats.documents_processed += 1
            content = document.page_content or ""
            sentences = self._split_sentences(content)
            if not sentences:
                compressed_documents.append(document)
                continue

            sentence_tokens = [
                self._estimate_tokens(sentence) for sentence in sentences
            ]
            stats.tokens_before += sum(sentence_tokens)

            # Skip compression if the document is already short.
            if len(sentences) <= self._config.min_sentences:
                stats.tokens_after += sum(sentence_tokens)
                compressed_documents.append(document)
                continue

            try:
                sentence_embeddings = await self._vector_service.embed_documents(
                    sentences
                )
            except EmbeddingServiceError:
                raise
            except Exception as exc:  # pragma: no cover - defensive safety net
                raise EmbeddingServiceError(
                    f"Failed to embed sentences for compression: {exc}"
                ) from exc

            indices = self._select_sentence_indices(
                query_embedding,
                np.asarray(sentence_embeddings, dtype=float),
                sentence_tokens,
            )

            if not indices:
                indices = list(range(min(self._config.min_sentences, len(sentences))))

            indices.sort()
            selected_sentences = [sentences[index] for index in indices]
            compressed_text = self._join_sentences(selected_sentences)
            tokens_after = sum(sentence_tokens[index] for index in indices)
            stats.tokens_after += tokens_after

            metadata = dict(document.metadata or {})
            compression_metadata = {
                "applied": compressed_text.strip() != content.strip(),
                "sentences_before": len(sentences),
                "sentences_after": len(indices),
                "tokens_before": sum(sentence_tokens),
                "tokens_after": tokens_after,
            }
            metadata.setdefault("_compression", compression_metadata)

            if compression_metadata["applied"]:
                stats.documents_compressed += 1

            compressed_documents.append(
                Document(page_content=compressed_text, metadata=metadata)
            )

        self._last_stats = stats
        return compressed_documents, stats

    def _select_sentence_indices(  # pylint: disable=too-many-locals
        self,
        query_embedding: np.ndarray,
        sentence_embeddings: np.ndarray,
        token_counts: Sequence[int],
    ) -> list[int]:
        """Select sentence indices using maximal marginal relevance."""

        total_sentences = sentence_embeddings.shape[0]
        if total_sentences == 0:
            return []

        target_count = max(
            self._config.min_sentences,
            min(
                self._config.max_sentences,
                max(1, int(math.ceil(total_sentences * self._config.token_ratio))),
            ),
        )

        relevance_scores = [
            self._cosine_similarity(query_embedding, sentence_embeddings[idx])
            for idx in range(total_sentences)
        ]

        remaining = list(range(total_sentences))
        selected: list[int] = []

        while remaining and len(selected) < target_count:
            best_idx = None
            best_score = -float("inf")
            for idx in remaining:
                relevance = relevance_scores[idx]
                if (
                    len(selected) >= self._config.min_sentences
                    and relevance < self._config.similarity_threshold
                ):
                    continue
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self._cosine_similarity(
                            sentence_embeddings[idx], sentence_embeddings[sel]
                        )
                        for sel in selected
                    )
                score = (
                    self._config.mmr_lambda * relevance
                    - (1.0 - self._config.mmr_lambda) * redundancy
                )
                if score > best_score or (
                    math.isclose(score, best_score)
                    and (best_idx is None or idx < best_idx)
                ):
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        if len(selected) < self._config.min_sentences and remaining:
            missing = self._config.min_sentences - len(selected)
            candidates = sorted(
                remaining,
                key=lambda idx: (-relevance_scores[idx], idx),
            )
            selected.extend(candidates[:missing])

        selected.sort()
        tokens_total = sum(token_counts[index] for index in selected)
        while (
            selected
            and tokens_total > self._config.absolute_max_tokens
            and len(selected) > self._config.min_sentences
        ):
            removed = selected.pop()
            tokens_total = sum(token_counts[index] for index in selected)
            if removed in remaining:
                remaining.remove(removed)

        return selected

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences while preserving order."""

        if not text:
            return []
        sentences = _SENTENCE_SPLIT_PATTERN.split(text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def _estimate_tokens(sentence: str) -> int:
        """Estimate token count using whitespace-separated words."""

        words = sentence.split()
        return max(1, len(words))

    @staticmethod
    def _join_sentences(sentences: Iterable[str]) -> str:
        """Join selected sentences back into a single string."""

        return " ".join(sentence.strip() for sentence in sentences if sentence.strip())

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""

        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)
