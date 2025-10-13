"""Unit tests for the versioned API routers."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_vector_service_dependency
from src.api.routers.v1 import documents as documents_router, search as search_router
from src.models.search import SearchRecord


@pytest.fixture()
def app_with_overrides() -> FastAPI:
    """Create a FastAPI app mounting the v1 routers with stub dependencies."""

    app = FastAPI()
    app.include_router(search_router.router, prefix="/api/v1")
    app.include_router(documents_router.router, prefix="/api/v1")
    stub = _StubVectorService()

    async def _get_stub_service() -> _StubVectorService:
        return stub

    app.dependency_overrides[get_vector_service_dependency] = _get_stub_service
    app.state.stub_vector_service = stub
    return app


class _StubVectorService:
    """In-memory stub that emulates the vector service contract."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self._failures: dict[str, Exception] = {}

    def set_failure(self, operation: str, exc: Exception) -> None:
        """Inject a failure for the specified operation."""

        self._failures[operation] = exc

    def clear_failures(self) -> None:
        """Remove all injected failures."""

        self._failures.clear()

    async def search_documents(
        self,
        collection: str,
        query: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,  # noqa: ARG002 - signature parity
        group_by: str | None = None,  # noqa: ARG002
        group_size: int | None = None,  # noqa: ARG002
        overfetch_multiplier: float | None = None,  # noqa: ARG002
        normalize_scores: bool | None = None,  # noqa: ARG002
    ) -> list[SearchRecord]:
        if failure := self._failures.get("search_documents"):
            raise failure

        return [
            SearchRecord(
                id="doc-1",
                content=f"{query}-{collection}",
                score=0.75,
                grouping_applied=bool(group_by),
            )
        ][:limit]

    async def search_vector(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,  # noqa: ARG002 - signature parity
    ) -> list[SearchRecord]:
        if failure := self._failures.get("search_vector"):
            raise failure

        magnitude = sum(vector)
        return [
            SearchRecord(
                id="doc-vector",
                content=f"{collection}:{magnitude}",
                score=1.0,
            )
        ][:limit]

    async def add_document(
        self,
        collection: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if failure := self._failures.get("add_document"):
            raise failure

        document_id = f"{collection}-{len(self.documents) + 1}"
        payload: dict[str, Any] = {
            "id": document_id,
            "content": content,
            "collection": collection,
        }
        if metadata:
            payload["metadata"] = metadata
        self.documents[document_id] = payload
        return document_id

    async def get_document(
        self,
        collection: str,
        document_id: str,
    ) -> dict[str, Any] | None:
        if failure := self._failures.get("get_document"):
            raise failure

        payload = self.documents.get(document_id)
        if payload is None:
            return None
        payload.setdefault("collection", collection)
        payload.setdefault("id", document_id)
        return payload

    async def delete_document(
        self,
        collection: str,
        document_id: str,
    ) -> bool:
        if failure := self._failures.get("delete_document"):
            raise failure

        return self.documents.pop(document_id, None) is not None

    async def list_documents(
        self,
        collection: str,
        *,
        limit: int,
        offset: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if failure := self._failures.get("list_documents"):
            raise failure

        docs = [
            doc
            for doc in self.documents.values()
            if doc.get("collection") == collection
        ]
        return docs[:limit], None

    async def list_collections(self) -> list[str]:
        if failure := self._failures.get("list_collections"):
            raise failure

        return ["documentation", "guides"]


def test_post_search_uses_canonical_contract(app_with_overrides: FastAPI) -> None:
    """POST /search should validate SearchRequest and return SearchResponse."""

    with TestClient(app_with_overrides) as client:
        response = client.post(
            "/api/v1/search",
            json={"query": "install", "collection": "documentation", "limit": 5},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["query"] == "install"
    assert payload["records"][0]["id"] == "doc-1"
    assert payload["records"][0]["content"] == "install-documentation"


def test_post_search_supports_query_vector(app_with_overrides: FastAPI) -> None:
    """POST /search accepts query vectors when the query is omitted."""

    with TestClient(app_with_overrides) as client:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "ignored",
                "collection": "documentation",
                "limit": 1,
                "query_vector": [0.5, 0.75],
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["records"][0]["id"] == "doc-vector"


def test_post_search_handles_service_failure(app_with_overrides: FastAPI) -> None:
    """Search endpoint surfaces 500 when the service raises unexpected errors."""

    stub: _StubVectorService = app_with_overrides.state.stub_vector_service
    stub.set_failure("search_documents", RuntimeError("boom"))
    try:
        with TestClient(app_with_overrides) as client:
            response = client.post(
                "/api/v1/search",
                json={"query": "install", "collection": "documentation", "limit": 5},
            )
        assert response.status_code == 500
        assert (
            response.json()["detail"]
            == "Search request failed due to an internal error."
        )
    finally:
        stub.clear_failures()


def test_document_lifecycle_endpoints(app_with_overrides: FastAPI) -> None:
    """Document add/get/list/delete endpoints expose canonical payloads."""

    with TestClient(app_with_overrides) as client:
        add_response = client.post(
            "/api/v1/documents",
            json={
                "content": "Hello world",
                "collection": "documentation",
                "metadata": {"source": "unit-test"},
            },
        )
        assert add_response.status_code == 200
        document_id = add_response.json()["id"]

        fetch_response = client.get(
            f"/api/v1/documents/{document_id}",
            params={"collection": "documentation"},
        )
        assert fetch_response.status_code == 200
        payload = fetch_response.json()
        assert payload["id"] == document_id
        assert payload["content"] == "Hello world"

        list_response = client.get(
            "/api/v1/documents",
            params={"collection": "documentation", "limit": 10},
        )
        assert list_response.status_code == 200
        list_payload = list_response.json()
        assert list_payload["count"] == 1
        assert list_payload["documents"][0]["id"] == document_id

        delete_response = client.delete(
            f"/api/v1/documents/{document_id}",
            params={"collection": "documentation"},
        )
        assert delete_response.status_code == 200

        missing_response = client.get(
            f"/api/v1/documents/{document_id}",
            params={"collection": "documentation"},
        )
        assert missing_response.status_code == 404


def test_add_document_handles_service_failure(app_with_overrides: FastAPI) -> None:
    """Document creation surfaces 500 when the service raises unexpected errors."""

    stub: _StubVectorService = app_with_overrides.state.stub_vector_service
    stub.set_failure("add_document", RuntimeError("boom"))
    try:
        with TestClient(app_with_overrides) as client:
            response = client.post(
                "/api/v1/documents",
                json={
                    "content": "Hello world",
                    "collection": "documentation",
                },
            )
        assert response.status_code == 500
        assert (
            response.json()["detail"] == "Failed to add document to the vector store."
        )
    finally:
        stub.clear_failures()
