"""High-level document chunking utilities built on LangChain splitters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TokenTextSplitter,
)

from src.config.models import ChunkingConfig


# Header definitions mirror LangChain defaults to preserve hierarchy.
_MARKDOWN_HEADERS: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

_HTML_HEADERS: list[tuple[str, str]] = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("h5", "Header 5"),
    ("h6", "Header 6"),
]

_TOKEN_KIND = "token"  # noqa: S105 - symbolic label for token-aware chunking
_KIND_TOKEN_ALIASES = {_TOKEN_KIND, "tokens", "token-aware", "token_aware"}


_EXTENSION_LANGUAGE_MAP: dict[str, Language] = {
    "c": Language.C,
    "cc": Language.CPP,
    "cpp": Language.CPP,
    "cs": Language.CSHARP,
    "cxx": Language.CPP,
    "elixir": Language.ELIXIR,
    "ex": Language.ELIXIR,
    "exs": Language.ELIXIR,
    "go": Language.GO,
    "h": Language.C,
    "hs": Language.HASKELL,
    "java": Language.JAVA,
    "js": Language.JS,
    "kt": Language.KOTLIN,
    "kts": Language.KOTLIN,
    "lua": Language.LUA,
    "md": Language.MARKDOWN,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "ps1": Language.POWERSHELL,
    "py": Language.PYTHON,
    "rb": Language.RUBY,
    "rs": Language.RUST,
    "scala": Language.SCALA,
    "sol": Language.SOL,
    "swift": Language.SWIFT,
    "ts": Language.TS,
    "tsx": Language.TS,
    "vb": Language.VISUALBASIC6,
}


_MIME_KIND_MAP: dict[str, str] = {
    "application/json": "json",
    "application/ld+json": "json",
    "application/vnd.api+json": "json",
    "application/javascript": "code",
    "text/javascript": "code",
    "text/x-python": "code",
    "text/x-c": "code",
    "text/x-c++": "code",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/plain": "text",
}


def infer_extension(uri_or_path: str | None) -> str | None:
    """Infer a lowercase file extension from a URI or filesystem path."""

    if not uri_or_path:
        return None

    parsed = urlparse(uri_or_path)
    candidate = parsed.path or uri_or_path
    if not candidate:
        return None

    sanitized = candidate.split("?")[0].split("#")[0]
    suffix = PurePosixPath(sanitized).suffix
    if not suffix:
        return None
    return suffix.lstrip(".").lower() or None


def _coerce_language(value: Any) -> Language | None:
    if isinstance(value, Language):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        for language in Language:
            if language.value == lowered:
                return language
    return None


def infer_language(
    extension: str | None, metadata: Mapping[str, Any] | None = None
) -> str | None:
    """Infer the programming language identifier for metadata enrichment."""

    metadata = metadata or {}
    language: Language | None = None

    if metadata:
        language = _coerce_language(metadata.get("language"))
        if language is None:
            explicit = metadata.get("programming_language") or metadata.get("lang")
            language = _coerce_language(explicit)

    if language is None and extension:
        language = _EXTENSION_LANGUAGE_MAP.get(extension)

    return language.value if language else None


def _infer_kind_from_candidate(metadata: Mapping[str, Any]) -> str | None:
    candidate = metadata.get("kind") or metadata.get("content_kind")
    if isinstance(candidate, str) and candidate:
        return candidate.lower()
    return None


def _infer_kind_from_mime(metadata: Mapping[str, Any]) -> str | None:
    mime_type = (
        metadata.get("mime_type")
        or metadata.get("content_type")
        or metadata.get("metadata", {}).get("content_type")
    )
    if isinstance(mime_type, str):
        lowered = mime_type.split(";")[0].strip().lower()
        return _MIME_KIND_MAP.get(lowered)
    return None


def _infer_kind_from_extension(metadata: Mapping[str, Any]) -> str | None:
    extension = metadata.get("extension") or infer_extension(
        metadata.get("uri_or_path") or metadata.get("url")
    )
    if isinstance(extension, str):
        lowered = extension.lower()
        if lowered in {"md", "markdown"}:
            return "markdown"
        if lowered in {"html", "htm", "xhtml"}:
            return "html"
        if lowered == "json":
            return "json"
        if lowered in _EXTENSION_LANGUAGE_MAP:
            return "code"
    return None


def _infer_kind_from_token_hint(metadata: Mapping[str, Any]) -> str | None:
    token_hint = metadata.get("token_aware") or metadata.get("chunking_mode")
    if isinstance(token_hint, str):
        hint_value = token_hint.strip().lower()
        if hint_value in _KIND_TOKEN_ALIASES:
            return _TOKEN_KIND
    elif token_hint is True:
        return _TOKEN_KIND
    return None


_INFER_STRATEGIES: tuple[
    Callable[[Mapping[str, Any]], str | None],
    ...,
] = (
    _infer_kind_from_candidate,
    _infer_kind_from_mime,
    _infer_kind_from_extension,
    _infer_kind_from_token_hint,
)


def infer_document_kind(
    metadata: Mapping[str, Any] | None, default: str = "text"
) -> str:
    """Infer a logical document kind from crawler metadata."""

    if metadata is None:
        return default

    for strategy in _INFER_STRATEGIES:
        inferred_kind = strategy(metadata)
        if inferred_kind:
            return inferred_kind

    return default


def _normalize_kind(kind: str | None, metadata: Mapping[str, Any] | None) -> str:
    if isinstance(kind, str) and kind:
        lowered = kind.lower()
        if lowered == "auto":
            return infer_document_kind(metadata)
        if lowered in _KIND_TOKEN_ALIASES:
            return "token"
        return lowered
    return infer_document_kind(metadata)


def _apply_recursive_refinement(
    documents: list[Document],
    cfg: ChunkingConfig,
) -> list[Document]:
    if not documents:
        return []

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=True,
    )
    return recursive.split_documents(documents)


def _split_plain_text(raw: str, cfg: ChunkingConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.create_documents([raw], metadatas=[{}])


def _split_token_aware(raw: str, cfg: ChunkingConfig) -> list[Document]:
    splitter = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name=cfg.token_model,
        chunk_size=cfg.token_chunk_size,
        chunk_overlap=cfg.token_chunk_overlap,
    )
    chunks = splitter.split_text(raw)
    return [Document(page_content=chunk, metadata={}) for chunk in chunks]


def _split_markdown(raw: str, cfg: ChunkingConfig) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MARKDOWN_HEADERS)
    documents = header_splitter.split_text(raw)
    return _apply_recursive_refinement(documents, cfg)


def _split_html(raw: str, cfg: ChunkingConfig) -> list[Document]:
    if cfg.enable_semantic_html_segmentation:
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=_HTML_HEADERS,
            max_chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            normalize_text=cfg.normalize_html_text,
        )
        documents = splitter.split_text(raw)
    else:
        if cfg.normalize_html_text:
            splitter = HTMLHeaderTextSplitter(headers_to_split_on=_HTML_HEADERS)
        else:
            splitter = HTMLSectionSplitter(headers_to_split_on=_HTML_HEADERS)
        documents = splitter.split_text(raw)
    return _apply_recursive_refinement(documents, cfg)


def _split_code(
    raw: str, cfg: ChunkingConfig, metadata: Mapping[str, Any] | None
) -> list[Document]:
    extension = metadata.get("extension") if metadata else None
    language_name = metadata.get("language") if metadata else None
    language = _coerce_language(language_name) or _EXTENSION_LANGUAGE_MAP.get(
        extension or "", None
    )
    if language is None:
        language = Language.MARKDOWN

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=True,
    )
    return splitter.create_documents([raw], metadatas=[{}])


def _split_json(raw: str, cfg: ChunkingConfig) -> list[Document]:
    try:
        json_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "Invalid JSON payload supplied to chunker"
        raise ValueError(msg) from exc

    splitter = RecursiveJsonSplitter(max_chunk_size=cfg.json_max_chars)
    chunks = splitter.split_text(json_payload)
    return [Document(page_content=chunk, metadata={}) for chunk in chunks]


def _build_base_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(meta or {})
    source = metadata.get("source") or metadata.get("url")
    uri_or_path = (
        metadata.get("uri_or_path")
        or metadata.get("uri")
        or metadata.get("path")
        or source
    )
    title = metadata.get("title") or metadata.get("name") or metadata.get("page_title")
    extension = metadata.get("extension") or infer_extension(uri_or_path)
    language = infer_language(extension, metadata)
    mime_type = metadata.get("mime_type") or metadata.get("content_type")

    base: dict[str, Any] = {}
    if source:
        base["source"] = source
    if uri_or_path:
        base["uri_or_path"] = uri_or_path
    if title:
        base["title"] = title
    if extension:
        base["extension"] = extension
    if language:
        base["language"] = language
    if mime_type:
        base["mime_type"] = mime_type
    return base


def _merge_metadata(
    documents: list[Document],
    base_metadata: dict[str, Any],
    kind: str,
) -> list[Document]:
    merged: list[Document] = []
    base_identifier = (
        base_metadata.get("uri_or_path")
        or base_metadata.get("source")
        or base_metadata.get("title")
        or "document"
    )
    for index, document in enumerate(documents):
        metadata = dict(base_metadata)
        if document.metadata:
            metadata.update(document.metadata)
        metadata["kind"] = kind
        metadata["chunk_index"] = index
        chunk_key = f"{base_identifier}:{index}".encode("utf-8", "ignore")
        metadata["chunk_id"] = hashlib.blake2s(chunk_key, digest_size=8).hexdigest()
        merged.append(Document(page_content=document.page_content, metadata=metadata))
    return merged


def chunk_to_documents(
    raw: str,
    meta: Mapping[str, Any] | None,
    kind: str,
    cfg: ChunkingConfig,
) -> list[Document]:
    """Split raw content into LangChain documents using adaptive splitters."""

    if not raw:
        return []

    base_metadata = _build_base_metadata(meta)
    normalized_kind = _normalize_kind(kind, {**base_metadata, **(meta or {})})

    if normalized_kind == "token":
        documents = _split_token_aware(raw, cfg)
    elif normalized_kind == "markdown":
        documents = _split_markdown(raw, cfg)
    elif normalized_kind == "html":
        documents = _split_html(raw, cfg)
    elif normalized_kind == "code":
        documents = _split_code(raw, cfg, base_metadata)
    elif normalized_kind == "json":
        documents = _split_json(raw, cfg)
    else:
        documents = _split_plain_text(raw, cfg)

    return _merge_metadata(documents, base_metadata, normalized_kind)


__all__ = [
    "chunk_to_documents",
    "infer_document_kind",
    "infer_extension",
    "infer_language",
]
