"""Utilities for splitting documents into LangChain ``Document`` chunks."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from bs4 import BeautifulSoup, FeatureNotFound
from langchain_core.documents import Document


try:  # pragma: no cover - optional extra for LangChain 0.3+
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - fallback for lean installs
    from langchain.text_splitter import (  # type: ignore[assignment]
        RecursiveCharacterTextSplitter,
    )

from src.config.models import ChunkingConfig, ChunkingStrategy
from src.models.validators import validate_chunk_parameters


logger = logging.getLogger(__name__)

_BASIC_SEPARATORS: Sequence[str] = ("\n\n", "\n", " ", "")
_ENHANCED_SEPARATORS: Sequence[str] = (
    "\n```",
    "\n~~~",
    "\n### ",
    "\n## ",
    "\n# ",
    "\n\n",
    "\n",
    " ",
    "",
)

_HTML_PRIMARY_TAGS: Sequence[str] = ("article", "main", "section")
_HTML_BLOCK_TAGS: Sequence[str] = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "pre",
    "code",
    "blockquote",
    "table",
)
_HTML_REGEX = re.compile(
    r"<(?P<tag>html|body|main|article|section|div|p|h[1-6])\b", re.IGNORECASE
)
_JSON_PREFIX = re.compile(r"^\s*[\[{]")
_WHITESPACE = re.compile(r"\s+")


def _coerce_to_text(content: Any) -> str:
    """Extract textual content from crawler responses."""

    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("markdown", "text", "html", "content"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if content is None:
        return ""
    return str(content)


def _normalize(text: str) -> str:
    """Collapse whitespace and strip boundaries."""

    return _WHITESPACE.sub(" ", text).strip()


def _separators_for_strategy(strategy: ChunkingStrategy) -> Sequence[str]:
    """Return separators tuned for the requested chunking strategy."""

    if strategy is ChunkingStrategy.ENHANCED:
        return _ENHANCED_SEPARATORS
    return _BASIC_SEPARATORS


class DocumentChunker:
    """High-level document chunker that preserves semantic boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        validate_chunk_parameters(config.chunk_size, config.chunk_overlap)
        self._char_splitter = self._build_character_splitter()
        self._token_splitter = self._build_token_splitter()

    def _build_character_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=list(_separators_for_strategy(self.config.strategy)),
            add_start_index=True,
        )

    def _build_token_splitter(self) -> RecursiveCharacterTextSplitter:
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.config.token_model,
                chunk_size=self.config.token_chunk_size,
                chunk_overlap=self.config.token_chunk_overlap,
                add_start_index=True,
            )
        except ValueError:  # Unknown encoding name
            logger.warning(
                "Unknown token model '%s'; falling back to character splitter",
                self.config.token_model,
            )
            splitter = self._build_character_splitter()
        return splitter

    def chunk_content(
        self,
        content: Any,
        *,
        title: str = "",
        url: str = "",
        language: str | None = None,
    ) -> list[Document]:
        """Chunk the supplied content into LangChain ``Document`` instances."""

        raw_text = _coerce_to_text(content)
        if not raw_text.strip():
            return []

        base_metadata: dict[str, Any] = {
            "title": title,
            "source": url or title or "",
            "url": url,
        }
        if language:
            base_metadata["language"] = language

        segments = self._segment_input(raw_text)
        documents: list[Document] = []
        running_offset = 0

        for segment_type, segment_text in segments:
            segment_metadata = dict(base_metadata)
            segment_metadata["segment_type"] = segment_type
            docs = self._split_segment(
                segment_text, metadata=segment_metadata, base_offset=running_offset
            )
            documents.extend(docs)
            running_offset += len(segment_text)

        for index, doc in enumerate(documents):
            doc.metadata.setdefault("chunk_index", index)
            doc.metadata.setdefault(
                "end_index", doc.metadata.get("start_index", 0) + len(doc.page_content)
            )

        return documents

    def _split_segment(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any],
        base_offset: int,
    ) -> list[Document]:
        if not text.strip():
            return []

        splitter = self._token_splitter
        if len(text) <= self.config.token_chunk_size:
            splitter = self._char_splitter

        seed_document = Document(page_content=text, metadata=dict(metadata))
        chunks = splitter.split_documents([seed_document])
        for chunk in chunks:
            start_index = chunk.metadata.get("start_index", 0) + base_offset
            chunk.metadata["start_index"] = start_index
            chunk.metadata.setdefault("length", len(chunk.page_content))
        return chunks

    def _segment_input(self, text: str) -> list[tuple[str, str]]:
        if self._looks_like_json(text):
            return self._segment_json(text)
        if self.config.enable_semantic_html_segmentation and self._looks_like_html(
            text
        ):
            html_segments = self._segment_html(text)
            if html_segments:
                return [(segment, "html") for segment in html_segments]
        return [(text, "text")]

    def _segment_json(self, text: str) -> list[tuple[str, str]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [(text, "text")]

        segments: list[str] = []

        def _append(value: Any) -> None:
            serialized = json.dumps(value, ensure_ascii=False, indent=2)
            if len(serialized) <= self.config.json_max_chars:
                segments.append(serialized)
                return
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    _append({key: nested})
                return
            if isinstance(value, list):
                for item in value:
                    _append(item)
                return
            segments.append(serialized[: self.config.json_max_chars])

        _append(payload)
        if not segments:
            return [(text, "json")]
        return [(segment, "json") for segment in segments]

    def _segment_html(self, text: str) -> list[str]:
        try:
            soup = BeautifulSoup(text, "lxml")
        except FeatureNotFound:  # pragma: no cover - minimal environments
            soup = BeautifulSoup(text, "html.parser")

        segments: list[str] = []
        for selector in _HTML_PRIMARY_TAGS:
            for element in soup.select(selector):
                extracted = element.get_text(separator="\n", strip=True)
                if extracted:
                    segments.append(self._prepare_html_segment(extracted))

        if not segments:
            for element in soup.select(",".join(_HTML_BLOCK_TAGS)):
                extracted = element.get_text(separator="\n", strip=True)
                if extracted:
                    segments.append(self._prepare_html_segment(extracted))

        if not segments:
            extracted = soup.get_text(separator="\n", strip=True)
            if extracted:
                segments.append(self._prepare_html_segment(extracted))

        return segments

    def _prepare_html_segment(self, text: str) -> str:
        if self.config.normalize_html_text:
            return _normalize(text)
        return text

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        return bool(_HTML_REGEX.search(text))

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        return bool(_JSON_PREFIX.match(text.strip()))


def chunk_content(
    content: Any,
    *,
    config: ChunkingConfig,
    metadata: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Split raw content into LangChain ``Document`` chunks."""

    chunker = DocumentChunker(config)
    title = metadata.get("title") if metadata else ""
    url = metadata.get("url") if metadata else ""
    language = metadata.get("language") if metadata else None
    documents = chunker.chunk_content(
        content,
        title=title or "",
        url=url or "",
        language=language,
    )

    if metadata:
        for doc in documents:
            enriched_meta = dict(metadata)
            enriched_meta.update(doc.metadata)
            doc.metadata = enriched_meta

    return documents


__all__ = ["DocumentChunker", "chunk_content"]
