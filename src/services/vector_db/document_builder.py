"""Utilities for translating LangChain documents into ingestion payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from .types import TextDocument


# pylint: disable=too-many-instance-attributes  # aggregation of canonical fields
@dataclass(slots=True)
class DocumentBuildParams:
    """Parameters required to translate chunks into text documents."""

    doc_id: str
    tenant: str
    source_url: str
    title: str
    provider: str
    quality_score: float | None = None
    default_content_type: str | None = None
    language_hint: str | None = None
    base_metadata: Mapping[str, Any] | None = None
    enriched_content: Any | None = None


def _extract_chunk_hash(metadata: dict[str, Any]) -> str | None:
    chunk_hash = metadata.pop("chunk_id", None)
    if isinstance(chunk_hash, str) and chunk_hash.strip():
        return chunk_hash
    return None


def _normalize_language(metadata: dict[str, Any], fallback: str | None) -> None:
    explicit = metadata.pop("language", None) or metadata.get("lang")
    language = explicit or fallback
    if language:
        metadata["lang"] = language
    elif "lang" in metadata:
        metadata.pop("lang")


def build_text_documents(
    chunks: Sequence[Document],
    params: DocumentBuildParams,
) -> list[TextDocument]:
    """Return TextDocument payloads derived from chunked documents."""

    total_chunks = len(chunks)
    timestamp = datetime.now(UTC).isoformat()
    base_payload = dict(params.base_metadata or {})
    base_payload.setdefault("provider", params.provider)
    if params.quality_score is not None:
        base_payload.setdefault("quality_score", params.quality_score)
    base_payload.setdefault("source", params.source_url)
    base_payload.setdefault("uri_or_path", params.source_url)
    base_payload.setdefault("title", params.title)
    if params.default_content_type:
        base_payload.setdefault("content_type", params.default_content_type)
    if params.language_hint:
        base_payload.setdefault("lang", params.language_hint)

    documents: list[TextDocument] = []
    for index, chunk in enumerate(chunks):
        chunk_metadata = dict(base_payload)
        chunk_metadata.update(
            {k: v for k, v in (chunk.metadata or {}).items() if v is not None}
        )

        chunk_hash = _extract_chunk_hash(chunk_metadata)

        chunk_metadata["doc_id"] = str(chunk_metadata.get("doc_id") or params.doc_id)
        chunk_metadata["tenant"] = chunk_metadata.get("tenant") or params.tenant
        chunk_metadata["source"] = chunk_metadata.get("source") or params.source_url
        chunk_metadata["uri_or_path"] = (
            chunk_metadata.get("uri_or_path") or params.source_url
        )
        chunk_metadata["title"] = chunk_metadata.get("title") or params.title
        chunk_metadata["content_type"] = (
            chunk_metadata.get("content_type")
            or chunk_metadata.pop("mime_type", None)
            or params.default_content_type
        )

        _normalize_language(chunk_metadata, params.language_hint)

        start_index = chunk_metadata.pop("start_index", None)
        if isinstance(start_index, int):
            chunk_metadata.setdefault("start_char", start_index)
            chunk_metadata.setdefault("end_char", start_index + len(chunk.page_content))

        chunk_metadata["chunk_index"] = index
        chunk_metadata["chunk_id"] = index
        chunk_metadata["total_chunks"] = total_chunks
        chunk_metadata.setdefault("created_at", timestamp)
        chunk_metadata.setdefault("updated_at", chunk_metadata["created_at"])

        if chunk_hash:
            chunk_metadata["chunk_hash"] = chunk_hash

        if params.enriched_content:
            enriched = params.enriched_content
            classification = enriched.classification
            quality = enriched.quality_score
            enrichment_meta = enriched.metadata
            chunk_metadata.update(
                {
                    "content_type": classification.primary_type.value,
                    "content_confidence": classification.confidence_scores.get(
                        classification.primary_type,
                        0.0,
                    ),
                    "quality_overall": quality.overall_score,
                    "quality_completeness": quality.completeness,
                    "quality_relevance": quality.relevance,
                    "quality_confidence": quality.confidence,
                    "ci_word_count": enrichment_meta.word_count,
                    "ci_char_count": enrichment_meta.char_count,
                    "ci_language": enrichment_meta.language,
                    "ci_semantic_tags": enrichment_meta.semantic_tags,
                    "content_intelligence_analyzed": True,
                }
            )
            if classification.secondary_types:
                chunk_metadata["secondary_content_types"] = [
                    item.value for item in classification.secondary_types
                ]
        else:
            chunk_metadata["content_intelligence_analyzed"] = False

        documents.append(
            TextDocument(
                id=f"{chunk_metadata['doc_id']}:{index}",
                content=chunk.page_content,
                metadata=chunk_metadata,
            )
        )

    return documents


__all__ = ["DocumentBuildParams", "build_text_documents"]


def build_params_from_crawl(
    crawl_result: Mapping[str, Any],
    *,
    fallback_url: str,
    tenant: str,
    doc_id: str,
    enriched_content: Any | None = None,
) -> DocumentBuildParams:
    """Create :class:`DocumentBuildParams` from a crawl result."""

    metadata_block = crawl_result.get("metadata", {})
    if not isinstance(metadata_block, Mapping):
        metadata_block = {}

    source_url = crawl_result.get("url", fallback_url)
    return DocumentBuildParams(
        doc_id=doc_id,
        tenant=tenant,
        source_url=source_url,
        title=crawl_result.get("title") or metadata_block.get("title", ""),
        provider=crawl_result.get("provider", "unknown"),
        quality_score=crawl_result.get("quality_score"),
        default_content_type=(
            crawl_result.get("content_type")
            or metadata_block.get("content_type")
            or metadata_block.get("mime_type")
        ),
        language_hint=metadata_block.get("language") or metadata_block.get("lang"),
        base_metadata=metadata_block,
        enriched_content=enriched_content,
    )


__all__.append("build_params_from_crawl")
