"""Tests for the crawl4ai bulk embedder workflow."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner
from langchain_core.documents import Document

from src.config import Settings
from src.crawl4ai_bulk_embedder import (
    BulkEmbedder,
    ChunkGenerationError,
    ContentExtractionError,
    ProcessingState,
    ScrapingError,
    _async_main,
    main,
)
from src.services.errors import ServiceError


pytestmark = pytest.mark.filterwarnings(
    "ignore:jsonschema.exceptions.RefResolutionError is deprecated:DeprecationWarning"
)


def create_httpx_response(
    url: str, content: str, status_code: int = 200
) -> httpx.Response:
    """Build an HTTPX response object for sitemap stubs."""

    request = httpx.Request("GET", url)
    return httpx.Response(
        status_code=status_code, content=content.encode("utf-8"), request=request
    )


def stub_async_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[
        str, httpx.Response | Exception | Callable[[], httpx.Response | Exception]
    ],
) -> None:
    """Patch httpx.AsyncClient with deterministic responses."""

    class AsyncClientStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - simple stub
            self._responses = responses

        async def __aenter__(self):  # type: ignore[override]
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:  # type: ignore[override]
            return False

        async def get(self, url: str) -> httpx.Response:
            handler = self._responses.get(url)
            if handler is None:
                raise AssertionError(f"No stubbed response for URL: {url}")

            result = handler() if callable(handler) else handler
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr("httpx.AsyncClient", AsyncClientStub)


@pytest.fixture
def mock_config() -> MagicMock:
    """Return a configuration stub with embedding defaults."""

    config = MagicMock(spec=Settings)
    config.openai = MagicMock(dimensions=1536, model="text-embedding-3-small")
    config.fastembed = MagicMock(
        generate_sparse=True, dense_model="BAAI/bge-small-en-v1.5"
    )
    config.chunking = MagicMock(chunk_size=1000, chunk_overlap=200)
    return config


@pytest.fixture
def sample_urls() -> list[str]:
    """Sample URLs reused across tests."""

    return [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://docs.example.com/api",
    ]


@pytest.fixture
def sample_state_data() -> dict[str, Any]:
    """Serialized state payload used for load/save tests."""

    return {
        "urls_to_process": ["https://example.com/page3"],
        "completed_urls": ["https://example.com/page1"],
        "failed_urls": {"https://example.com/error": "404 Not Found"},
        "total_chunks_processed": 10,
        "total_embeddings_generated": 10,
        "start_time": "2024-01-01T00:00:00",
        "last_checkpoint": "2024-01-01T01:00:00",
        "collection_name": "test_collection",
    }


@pytest.fixture
def embedder_factory(
    mock_config: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., BulkEmbedder]:
    """Build BulkEmbedder instances with isolated state files."""

    counter = 0

    def factory(**overrides: Any) -> BulkEmbedder:
        nonlocal counter
        counter += 1

        container = overrides.pop("container", MagicMock())
        crawl_manager = overrides.pop("crawl_manager", MagicMock())
        embedding_manager = overrides.pop("embedding_manager", MagicMock())
        vector_service = overrides.pop("vector_service", MagicMock())
        vector_service.is_initialized.return_value = overrides.pop(
            "vector_initialized", True
        )
        container.browser_manager.return_value = crawl_manager
        container.embedding_manager.return_value = embedding_manager
        container.vector_store_service.return_value = vector_service

        get_container_value = overrides.pop("get_container_value", container)
        monkeypatch.setattr(
            "src.crawl4ai_bulk_embedder.get_container",
            lambda: get_container_value,
        )
        init_mock = AsyncMock(return_value=container)
        shutdown_mock = AsyncMock()
        monkeypatch.setattr(
            "src.crawl4ai_bulk_embedder.initialize_container",
            init_mock,
        )
        monkeypatch.setattr(
            "src.crawl4ai_bulk_embedder.shutdown_container",
            shutdown_mock,
        )

        state_path = overrides.pop(
            "state_file",
            tmp_path / f"state_{uuid.uuid4().hex}.json",
        )
        embedder = BulkEmbedder(
            config=overrides.pop("config", mock_config),
            collection_name=overrides.pop("collection_name", "test_collection"),
            state_file=state_path,
            container=container
            if overrides.pop("use_container_override", True)
            else None,
            **overrides,
        )
        embedder._test_context = SimpleNamespace(
            container=container,
            crawl_manager=crawl_manager,
            embedding_manager=embedding_manager,
            vector_service=vector_service,
            init_mock=init_mock,
            shutdown_mock=shutdown_mock,
        )
        return embedder

    return factory


class TestProcessingState:
    """ProcessingState behaviour."""

    def test_defaults(self) -> None:
        state = ProcessingState()
        assert state.urls_to_process == []
        assert state.completed_urls == []
        assert state.failed_urls == {}
        assert state.total_chunks_processed == 0
        assert state.collection_name == "bulk_embeddings"

    def test_from_mapping(self, sample_state_data: dict[str, Any]) -> None:
        state = ProcessingState.model_validate(sample_state_data)
        assert state.urls_to_process == ["https://example.com/page3"]
        assert state.completed_urls == ["https://example.com/page1"]
        assert state.failed_urls == {"https://example.com/error": "404 Not Found"}
        assert state.total_chunks_processed == 10
        assert state.collection_name == "test_collection"


class TestStatePersistence:
    """State load/save helpers."""

    def test_load_state_from_file(
        self,
        mock_config: MagicMock,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
        sample_state_data: dict[str, Any],
    ) -> None:
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_data), encoding="utf-8")

        embedder = embedder_factory(config=mock_config, state_file=state_file)

        assert embedder.state.completed_urls == ["https://example.com/page1"]
        assert embedder.state.total_chunks_processed == 10

    def test_load_state_invalid_json(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
    ) -> None:
        state_file = tmp_path / "invalid.json"
        state_file.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            embedder_factory(state_file=state_file)

    def test_save_state(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
    ) -> None:
        embedder = embedder_factory()
        embedder.state.completed_urls = ["https://example.com/test"]
        embedder.state.total_chunks_processed = 5

        embedder._save_state()  # pylint: disable=protected-access

        saved = json.loads(embedder.state_file.read_text(encoding="utf-8"))
        assert saved["completed_urls"] == ["https://example.com/test"]
        assert saved["total_chunks_processed"] == 5


class TestServiceInitialization:
    """Service wiring tests."""

    @pytest.mark.asyncio
    async def test_initialize_services(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()
        ctx = embedder._test_context

        vector_service = ctx.vector_service
        vector_service.is_initialized.return_value = False
        vector_service.initialize = AsyncMock()
        vector_service.list_collections = AsyncMock(return_value=["other"])
        vector_service.embedding_dimension = 1536
        vector_service.ensure_collection = AsyncMock()

        await embedder.initialize_services()

        vector_service.initialize.assert_awaited_once()
        vector_service.ensure_collection.assert_awaited_once()
        assert embedder.crawl_manager is ctx.crawl_manager
        assert embedder.embedding_manager is ctx.embedding_manager
        assert embedder.vector_service is vector_service


class TestUrlLoading:
    """URL ingestion utilities."""

    @pytest.mark.asyncio
    async def test_load_urls_from_text(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
    ) -> None:
        embedder = embedder_factory()
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text(
            """https://example.com/page1
https://example.com/page2
# comment
https://example.com/page3
""",
            encoding="utf-8",
        )

        urls = await embedder.load_urls_from_file(urls_file)
        assert urls == [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]

    @pytest.mark.asyncio
    async def test_load_urls_from_csv(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
    ) -> None:
        embedder = embedder_factory()
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            """url,title
https://example.com/page1,Page 1
https://example.com/page2,Page 2
""",
            encoding="utf-8",
        )

        urls = await embedder.load_urls_from_file(csv_file)
        assert urls == [
            "https://example.com/page1",
            "https://example.com/page2",
        ]

    @pytest.mark.asyncio
    async def test_load_urls_from_json(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        tmp_path: Path,
    ) -> None:
        embedder = embedder_factory()
        json_file = tmp_path / "urls.json"
        json_file.write_text(
            json.dumps(
                [
                    "https://example.com/page1",
                    {"url": "https://example.com/page2"},
                ]
            ),
            encoding="utf-8",
        )

        urls = await embedder.load_urls_from_file(json_file)
        assert urls == [
            "https://example.com/page1",
            "https://example.com/page2",
        ]

        json_file.write_text(
            json.dumps(
                {"urls": ["https://example.com/page3", "https://example.com/page4"]}
            ),
            encoding="utf-8",
        )
        urls = await embedder.load_urls_from_file(json_file)
        assert urls == [
            "https://example.com/page3",
            "https://example.com/page4",
        ]

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        embedder = embedder_factory()
        sitemap_url = "https://example.com/sitemap.xml"
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
</urlset>"""

        stub_async_client(
            monkeypatch,
            {sitemap_url: create_httpx_response(sitemap_url, sitemap_xml)},
        )

        urls = await embedder.load_urls_from_sitemap(sitemap_url)
        assert urls == [
            "https://example.com/page1",
            "https://example.com/page2",
        ]

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap_index(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        embedder = embedder_factory()
        index_url = "https://example.com/index.xml"
        child_url = "https://example.com/sitemap1.xml"
        index_xml = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap1.xml</loc></sitemap>
</sitemapindex>"""
        child_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
</urlset>"""

        stub_async_client(
            monkeypatch,
            {
                index_url: create_httpx_response(index_url, index_xml),
                child_url: create_httpx_response(child_url, child_xml),
            },
        )

        urls = await embedder.load_urls_from_sitemap(index_url)
        assert urls == ["https://example.com/page1"]

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap_http_error(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        embedder = embedder_factory()
        sitemap_url = "https://example.com/missing.xml"
        request = httpx.Request("GET", sitemap_url)
        error = httpx.HTTPStatusError(
            "404 Not Found",
            request=request,
            response=httpx.Response(status_code=404, request=request),
        )

        stub_async_client(monkeypatch, {sitemap_url: lambda: error})

        with pytest.raises(httpx.HTTPStatusError):
            await embedder.load_urls_from_sitemap(sitemap_url)


class TestProcessingPipeline:
    """URL processing scenarios."""

    @pytest.mark.asyncio
    async def test_process_url_success(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()
        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {
                    "markdown": "# Title\n\nBody",
                    "text": "Title Body",
                },
                "metadata": {"title": "Title", "description": "desc"},
                "provider": "crawl4ai",
            }
        )
        embedder.embedding_manager = AsyncMock()
        embedder.embedding_manager.generate_embeddings = AsyncMock(
            return_value={
                "embeddings": [[0.1] * 3, [0.2] * 3],
                "sparse_embeddings": [{"0": 0.5}, {"1": 0.6}],
            }
        )
        embedder.vector_service = AsyncMock()
        embedder.vector_service.upsert_vectors = AsyncMock()

        chunk_documents = [
            Document(page_content="Chunk 1", metadata={"start_index": 0}),
            Document(page_content="Chunk 2", metadata={"start_index": 100}),
        ]

        with patch(
            "src.crawl4ai_bulk_embedder.split_content_into_documents",
            return_value=chunk_documents,
        ):
            result = await embedder.process_url("https://example.com/test")

        assert result["success"] is True
        assert result["chunks"] == 2
        embedder.embedding_manager.generate_embeddings.assert_awaited_once()
        embedder.vector_service.upsert_vectors.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_url_failure(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()
        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={"success": False, "error": "404 Not Found"}
        )

        result = await embedder.process_url("https://example.com/404")

        assert result["success"] is False
        assert result["chunks"] == 0
        assert result["error"] == "404 Not Found"

    @pytest.mark.asyncio
    async def test_process_url_no_content(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()
        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "", "text": ""},
                "metadata": {},
                "provider": "crawl4ai",
            }
        )

        result = await embedder.process_url("https://example.com/empty")
        assert result["success"] is False
        assert result["error"] == "No content extracted"

    @pytest.mark.asyncio
    async def test_process_url_no_chunks(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()
        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {"markdown": "Short", "text": "Short"},
                "metadata": {},
                "provider": "crawl4ai",
            }
        )

        with patch(
            "src.crawl4ai_bulk_embedder.split_content_into_documents",
            return_value=[],
        ):
            result = await embedder.process_url("https://example.com/short")

        assert result["success"] is False
        assert result["error"] == "No chunks generated"


class TestBatchProcessing:
    """Batch orchestrator tests."""

    @pytest.mark.asyncio
    async def test_process_urls_batch_updates_state(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        sample_urls: list[str],
    ) -> None:
        embedder = embedder_factory()

        async def mock_process(url: str) -> dict[str, Any]:
            return {"url": url, "success": True, "chunks": 2, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state"),
        ):
            results = await embedder.process_urls_batch(
                urls=sample_urls,
                max_concurrent=2,
            )

        assert results == {
            "total": 3,
            "successful": 3,
            "failed": 0,
            "results": [
                {"url": url, "success": True, "chunks": 2, "error": None}
                for url in sample_urls
            ],
        }
        assert embedder.state.completed_urls == sample_urls
        assert embedder.state.total_chunks_processed == 6
        assert embedder.state.total_embeddings_generated == 6

    @pytest.mark.asyncio
    async def test_process_urls_batch_without_progress_saves_periodically(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()

        async def mock_process(url: str) -> dict[str, Any]:
            return {"url": url, "success": True, "chunks": 1, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state") as save_state,
        ):
            await embedder.process_urls_batch(
                urls=[f"https://example.com/{i}" for i in range(15)],
                max_concurrent=5,
                progress=None,
            )

        assert save_state.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_urls_batch_handles_exceptions(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
    ) -> None:
        embedder = embedder_factory()

        async def mock_process(url: str) -> dict[str, Any]:
            if "exception" in url:
                raise RuntimeError("Processing failed")
            return {"url": url, "success": True, "chunks": 2, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state"),
        ):
            results = await embedder.process_urls_batch(
                urls=["https://example.com/good", "https://example.com/exception"],
                max_concurrent=2,
            )

        assert results["total"] == 2
        assert results["successful"] == 1
        assert results["failed"] == 1


class TestRunFlow:
    """High-level run orchestration."""

    @pytest.mark.asyncio
    async def test_run_with_resume_skips_completed(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        sample_urls: list[str],
    ) -> None:
        embedder = embedder_factory()
        embedder.state.completed_urls = [sample_urls[0]]
        embedder.state.start_time = datetime.now(tz=UTC)

        with (
            patch.object(embedder, "initialize_services", new_callable=AsyncMock),
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as process_batch,
        ):
            process_batch.return_value = {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "results": [],
            }
            await embedder.run(urls=sample_urls, max_concurrent=5, resume=True)

        processed_urls = process_batch.call_args.kwargs["urls"]
        assert sample_urls[0] not in processed_urls
        assert len(processed_urls) == 2

    @pytest.mark.asyncio
    async def test_run_without_resume_processes_all(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        sample_urls: list[str],
    ) -> None:
        embedder = embedder_factory()
        embedder.state.completed_urls = [sample_urls[0]]
        embedder.state.start_time = datetime.now(tz=UTC)

        with (
            patch.object(embedder, "initialize_services", new_callable=AsyncMock),
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as process_batch,
        ):
            process_batch.return_value = {
                "total": 3,
                "successful": 3,
                "failed": 0,
                "results": [],
            }
            await embedder.run(urls=sample_urls, max_concurrent=5, resume=False)

        processed_urls = process_batch.call_args.kwargs["urls"]
        assert len(processed_urls) == 3

    @pytest.mark.asyncio
    async def test_run_all_urls_completed(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        sample_urls: list[str],
    ) -> None:
        embedder = embedder_factory()
        embedder.state.completed_urls = sample_urls.copy()
        embedder.state.start_time = datetime.now(tz=UTC)

        with (
            patch.object(
                embedder, "initialize_services", new_callable=AsyncMock
            ) as init_services,
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as process_batch,
        ):
            await embedder.run(urls=sample_urls, resume=True)

        init_services.assert_not_called()
        process_batch.assert_not_called()


class TestSummary:
    """Summary output."""

    def test_display_summary_reports_failures(
        self,
        embedder_factory: Callable[..., BulkEmbedder],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        embedder = embedder_factory()
        embedder.state.failed_urls = {
            "https://example.com/404": "404 Not Found",
            "https://example.com/500": "Internal Error",
        }
        embedder.state.total_chunks_processed = 10
        embedder.state.total_embeddings_generated = 10
        embedder.state.start_time = datetime.now(tz=UTC)

        results = {"total": 5, "successful": 3, "failed": 2, "results": []}
        embedder._display_summary(results)  # pylint: disable=protected-access

        captured = capsys.readouterr()
        assert "Failed URLs:" in captured.out
        assert captured.out.count("•") == len(embedder.state.failed_urls)


class TestCLI:
    """CLI entrypoint tests."""

    def test_cli_with_urls(self) -> None:
        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.get_settings") as mock_get_config,
            patch("src.crawl4ai_bulk_embedder.configure_logging"),
        ):
            mock_get_config.return_value = MagicMock()
            mock_run.side_effect = lambda coro: coro.close()
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-u", "https://example.com/page1", "-u", "https://example.com/page2"],
            )

        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_cli_with_file(self, tmp_path: Path) -> None:
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com/page1\n", encoding="utf-8")

        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.get_settings") as mock_get_config,
            patch("src.crawl4ai_bulk_embedder.configure_logging"),
        ):
            mock_get_config.return_value = MagicMock()
            mock_run.side_effect = lambda coro: coro.close()
            runner = CliRunner()
            result = runner.invoke(main, ["-f", str(urls_file)])

        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_cli_with_sitemap(self) -> None:
        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.get_settings") as mock_get_config,
            patch("src.crawl4ai_bulk_embedder.configure_logging"),
        ):
            mock_get_config.return_value = MagicMock()
            mock_run.side_effect = lambda coro: coro.close()
            runner = CliRunner()
            result = runner.invoke(main, ["-s", "https://example.com/sitemap.xml"])

        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_cli_no_input(self) -> None:
        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.get_settings") as mock_get_config,
            patch("src.crawl4ai_bulk_embedder.configure_logging"),
        ):
            mock_get_config.return_value = MagicMock()
            mock_run.side_effect = lambda coro: coro.close()
            runner = CliRunner()
            result = runner.invoke(main, [])

        assert result.exit_code == 1
        assert "Must provide URLs" in result.output

    def test_cli_with_all_options(self, tmp_path: Path) -> None:
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com\n", encoding="utf-8")
        config_file = tmp_path / "config.json"
        config_file.write_text("{}", encoding="utf-8")

        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.get_settings") as mock_get_config,
            patch("src.crawl4ai_bulk_embedder.configure_logging"),
        ):
            mock_get_config.return_value = MagicMock()
            mock_run.side_effect = lambda coro: coro.close()
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "-f",
                    str(urls_file),
                    "--collection",
                    "custom_collection",
                    "--concurrent",
                    "10",
                    "--config",
                    str(config_file),
                    "--state-file",
                    str(tmp_path / "state.json"),
                    "--no-resume",
                    "--verbose",
                ],
            )

        assert result.exit_code == 0
        mock_run.assert_called_once()


class TestAsyncMain:
    """Async main entrypoint helpers."""

    @pytest.mark.asyncio
    async def test_async_main_with_urls(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        init_mock = AsyncMock()
        shutdown_mock = AsyncMock()
        with (
            patch("src.crawl4ai_bulk_embedder.initialize_container", init_mock),
            patch("src.crawl4ai_bulk_embedder.shutdown_container", shutdown_mock),
            patch("src.crawl4ai_bulk_embedder.BulkEmbedder") as embedder_cls,
        ):
            embedder = AsyncMock()
            embedder.run = AsyncMock()
            embedder_cls.return_value = embedder

            await _async_main(
                urls=["https://example.com/1", "https://example.com/2"],
                file=None,
                sitemap=None,
                collection="test_collection",
                concurrent=5,
                config=mock_config,
                state_file=tmp_path / "state.json",
                resume=True,
            )

        embedder.run.assert_called_once()
        run_kwargs = embedder.run.call_args.kwargs
        assert run_kwargs["urls"] == ["https://example.com/1", "https://example.com/2"]
        assert run_kwargs["max_concurrent"] == 5
        assert run_kwargs["resume"] is True
        init_mock.assert_awaited_once()
        shutdown_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_main_with_file_and_sitemap(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text(
            "https://example.com/file1\nhttps://example.com/file2\n", encoding="utf-8"
        )

        init_mock = AsyncMock()
        shutdown_mock = AsyncMock()
        with (
            patch("src.crawl4ai_bulk_embedder.initialize_container", init_mock),
            patch("src.crawl4ai_bulk_embedder.shutdown_container", shutdown_mock),
            patch("src.crawl4ai_bulk_embedder.BulkEmbedder") as embedder_cls,
        ):
            embedder = AsyncMock()
            embedder.run = AsyncMock()
            embedder.load_urls_from_file = AsyncMock(
                return_value=[
                    "https://example.com/file1",
                    "https://example.com/file2",
                ]
            )
            embedder.load_urls_from_sitemap = AsyncMock(
                return_value=[
                    "https://example.com/sitemap1",
                    "https://example.com/sitemap2",
                ]
            )
            embedder_cls.return_value = embedder

            await _async_main(
                urls=[],
                file=urls_file,
                sitemap="https://example.com/sitemap.xml",
                collection="test_collection",
                concurrent=10,
                config=mock_config,
                state_file=tmp_path / "state.json",
                resume=False,
            )

        combined_urls = embedder.run.call_args.kwargs["urls"]
        assert len(combined_urls) == 4
        embedder.load_urls_from_file.assert_awaited_once_with(urls_file)
        embedder.load_urls_from_sitemap.assert_awaited_once_with(
            "https://example.com/sitemap.xml"
        )
        init_mock.assert_awaited_once()
        shutdown_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_main_deduplicates_urls(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        init_mock = AsyncMock()
        shutdown_mock = AsyncMock()
        with (
            patch("src.crawl4ai_bulk_embedder.initialize_container", init_mock),
            patch("src.crawl4ai_bulk_embedder.shutdown_container", shutdown_mock),
            patch("src.crawl4ai_bulk_embedder.BulkEmbedder") as embedder_cls,
        ):
            embedder = AsyncMock()
            embedder.run = AsyncMock()
            embedder_cls.return_value = embedder

            await _async_main(
                urls=[
                    "https://example.com/1",
                    "https://example.com/2",
                    "https://example.com/1",
                ],
                file=None,
                sitemap=None,
                collection="test_collection",
                concurrent=5,
                config=mock_config,
                state_file=tmp_path / "state.json",
                resume=True,
            )

        unique_urls = embedder.run.call_args.kwargs["urls"]
        assert sorted(unique_urls) == ["https://example.com/1", "https://example.com/2"]
        init_mock.assert_awaited_once()
        shutdown_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_main_cleanup_on_error(
        self,
        mock_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        shutdown_mock = AsyncMock()
        with (
            patch("src.crawl4ai_bulk_embedder.initialize_container", AsyncMock()),
            patch("src.crawl4ai_bulk_embedder.shutdown_container", shutdown_mock),
            patch("src.crawl4ai_bulk_embedder.BulkEmbedder") as embedder_cls,
        ):
            embedder = AsyncMock()
            embedder.run = AsyncMock(side_effect=RuntimeError("Test error"))
            embedder_cls.return_value = embedder

            with pytest.raises(RuntimeError):
                await _async_main(
                    urls=["https://example.com/1"],
                    file=None,
                    sitemap=None,
                    collection="test_collection",
                    concurrent=5,
                    config=mock_config,
                    state_file=tmp_path / "state.json",
                    resume=True,
                )

        shutdown_mock.assert_awaited_once()


def test_bulk_embedder_errors_subclass_service_error() -> None:
    """Ensure bulk embedder custom exceptions derive from ServiceError."""

    assert issubclass(ScrapingError, ServiceError)
    assert issubclass(ContentExtractionError, ServiceError)
    assert issubclass(ChunkGenerationError, ServiceError)
