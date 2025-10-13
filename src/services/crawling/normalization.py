"""Helpers for normalizing crawler outputs across providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.services.browser.models import BrowserResult


_CONTENT_KEYS = ("markdown", "html", "text")


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _extract_content_block(payload: Mapping[str, Any]) -> dict[str, str]:
    content = payload.get("content")
    normalized: dict[str, str] = {}
    if isinstance(content, Mapping):
        for key in _CONTENT_KEYS:
            candidate = _coerce_str(content.get(key))
            if candidate.strip():
                normalized[key] = candidate
    elif isinstance(content, str) and content.strip():
        normalized["text"] = content
    return normalized


def _merge_content_sources(*sources: Mapping[str, Any] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        if not source:
            continue
        if isinstance(source, Mapping):
            for key in _CONTENT_KEYS:
                candidate = _coerce_str(source.get(key))
                if candidate.strip() and key not in merged:
                    merged[key] = candidate
    return merged


def normalize_crawler_output(
    result: BrowserResult | Mapping[str, Any],
    *,
    fallback_url: str,
) -> dict[str, Any]:
    """Return a dictionary-normalized crawler payload.

    Args:
        result: BrowserResult instance or mapping payload returned by a crawler.
        fallback_url: URL used when the payload omits an explicit URL.

    Returns:
        Dictionary containing canonical keys (``success``, ``url``, ``title``,
        ``metadata`` and structured ``content``). Textual representations are
        provided under ``content`` with ``markdown``/``html``/``text`` keys when
        available. The function purposefully limits itself to normalization so
        callers can apply domain-specific enrichment without duplicating the
        heavy coercion logic.
    """

    if isinstance(result, BrowserResult):
        metadata = dict(result.metadata or {})
        metadata.setdefault("provider", result.provider.value)
        normalized_content = _merge_content_sources(metadata.get("content"))
        if result.html and result.html.strip():
            normalized_content.setdefault("html", result.html)
        if result.content and result.content.strip():
            normalized_content.setdefault("text", result.content)
        payload: dict[str, Any] = {
            "success": result.success,
            "url": result.url or fallback_url,
            "title": result.title,
            "content": normalized_content,
            "raw_html": metadata.get("raw_html") or result.html,
            "metadata": metadata,
            "provider": result.provider.value,
        }
    elif isinstance(result, Mapping):
        payload = dict(result)
        payload.setdefault("success", True)
        payload.setdefault("url", fallback_url)
        payload.setdefault("title", "")
    else:  # pragma: no cover - defensive guard against unsupported payloads
        msg = "Crawler payload must be a mapping or BrowserResult"
        raise TypeError(msg)

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    metadata = dict(metadata)
    metadata.setdefault("url", payload.get("url", fallback_url))
    metadata.setdefault("title", payload.get("title", ""))
    if "provider" not in metadata and payload.get("provider"):
        metadata["provider"] = payload["provider"]

    normalized_content = _extract_content_block(payload)
    fallback_content = _merge_content_sources(metadata, payload.get("content"))
    normalized_content.update({k: v for k, v in fallback_content.items() if v.strip()})
    payload["content"] = normalized_content

    if payload.get("raw_html") and isinstance(payload["raw_html"], str):
        payload["raw_html"] = payload["raw_html"].strip()
    elif normalized_content.get("html"):
        payload["raw_html"] = normalized_content["html"]

    payload["metadata"] = metadata
    return payload


def resolve_chunk_inputs(
    payload: Mapping[str, Any],
    *,
    fallback_url: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Return raw content, metadata payload, and kind hint for chunking."""

    def _coerce(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value
        return None

    content_block = payload.get("content")
    raw_content: str | None = None
    kind_hint: str | None = None

    if isinstance(content_block, Mapping):
        for key in ("markdown", "html", "text"):
            candidate = _coerce(content_block.get(key))
            if candidate:
                raw_content = candidate
                kind_hint = key
                break
    elif isinstance(content_block, str):
        candidate = _coerce(content_block)
        if candidate:
            raw_content = candidate

    if raw_content is None:
        candidate = _coerce(payload.get("raw_html"))
        if candidate:
            raw_content = candidate
            kind_hint = "html"

    if raw_content is None:
        msg = "No chunkable content returned by crawler"
        raise ValueError(msg)

    metadata_block = payload.get("metadata")
    if not isinstance(metadata_block, Mapping):
        metadata_block = {}
    metadata_payload: dict[str, Any] = {
        "source": payload.get("url", fallback_url),
        "uri_or_path": payload.get("url", fallback_url),
        "title": payload.get("title") or metadata_block.get("title", ""),
        "mime_type": payload.get("content_type") or metadata_block.get("content_type"),
        "metadata": dict(metadata_block),
    }

    language = metadata_block.get("language") or metadata_block.get("lang")
    if language:
        metadata_payload["language"] = language
    kind = metadata_block.get("kind") or metadata_block.get("content_kind")
    if kind:
        metadata_payload["kind"] = kind

    return raw_content, metadata_payload, kind_hint


__all__ = ["normalize_crawler_output", "resolve_chunk_inputs"]
