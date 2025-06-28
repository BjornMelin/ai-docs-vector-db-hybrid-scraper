"""Extended tests for crawl4ai_bulk_embedder to improve coverage."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from src.config import Config
from src.crawl4ai_bulk_embedder import BulkEmbedder, ProcessingState, _async_main
from src.infrastructure.client_manager import ClientManager


class TestError(Exception):
    """Custom exception for this module."""


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock(spec=Config)

    # Mock nested attributes
    config.openai = MagicMock()
    config.openai.dimensions = 1536
    config.openai.model = "text-embedding-3-small"

    config.fastembed = MagicMock()
    config.fastembed.generate_sparse = True
    config.fastembed.model = "BAAI/bge-small-en-v1.5"

    config.chunking = MagicMock()
    config.chunking.chunk_size = 1000
    config.chunking.chunk_overlap = 200

    return config


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    return MagicMock(spec=ClientManager)


class TestBulkEmbedderExtended:
    """Extended tests for BulkEmbedder class."""

    def test_load_state_invalid_json(self, mock_config, mock_client_manager, tmp_path):
        """Test loading state from invalid JSON file."""
        state_file = tmp_path / "invalid_state.json"
        state_file.write_text("invalid json content {")

        # Should create default state when loading fails
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=state_file,
        )

        assert isinstance(embedder.state, ProcessingState)
        assert len(embedder.state.completed_urls) == 0

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap_with_index(
        self, mock_config, mock_client_manager
    ):
        """Test loading URLs from sitemap index."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Mock sitemap index response
        sitemap_index_xml = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap>
        <loc>https://example.com/sitemap1.xml</loc>
    </sitemap>
</sitemapindex>"""

        # Mock sub-sitemap response
        sub_sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/page1</loc>
    </url>
</urlset>"""

        with patch("httpx.AsyncClient") as mock_client:
            # Setup responses
            async def mock_get(url):
                response = AsyncMock()
                response.raise_for_status = AsyncMock()
                if "sitemap1.xml" in url:
                    response.content = sub_sitemap_xml.encode()
                else:
                    response.content = sitemap_index_xml.encode()
                return response

            mock_client.return_value.__aenter__.return_value.get = mock_get

            urls = await embedder.load_urls_from_sitemap(
                "https://example.com/sitemap_index.xml"
            )

            assert len(urls) == 1
            assert "https://example.com/page1" in urls

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap_http_error(
        self, mock_config, mock_client_manager
    ):
        """Test loading URLs from sitemap with HTTP error."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        with patch("httpx.AsyncClient") as mock_client:
            # Create a proper mock request for the error
            mock_request = Mock()
            mock_request.url = "https://example.com/missing.xml"

            # Create HTTPStatusError with proper response mock
            mock_response = Mock()
            mock_response.status_code = 404
            error = httpx.HTTPStatusError(
                "404 Not Found", request=mock_request, response=mock_response
            )

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=error
            )

            with pytest.raises(httpx.HTTPStatusError):
                await embedder.load_urls_from_sitemap("https://example.com/missing.xml")

    @pytest.mark.asyncio
    async def test_process_url_no_content(self, mock_config, mock_client_manager):
        """Test processing URL with no content extracted."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {
                    "markdown": "",
                    "text": "",
                },
                "metadata": {},
                "provider": "crawl4ai",
            }
        )

        result = await embedder.process_url("https://example.com/empty")

        assert result["success"] is False
        assert "No content extracted" in result["error"]

    @pytest.mark.asyncio
    async def test_process_url_no_chunks(self, mock_config, mock_client_manager):
        """Test processing URL that generates no chunks."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        embedder.crawl_manager = AsyncMock()
        embedder.embedding_manager = AsyncMock()
        embedder.qdrant_service = AsyncMock()

        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {
                    "markdown": "Very short content",
                    "text": "Very short content",
                },
                "metadata": {},
                "provider": "crawl4ai",
            }
        )

        # Mock chunking to return empty list
        mock_chunk_result = MagicMock()
        mock_chunk_result.chunks = []

        with patch("src.crawl4ai_bulk_embedder.DocumentChunker") as mock_chunker_class:
            mock_chunker = MagicMock()
            mock_chunker.chunk_text.return_value = mock_chunk_result
            mock_chunker_class.return_value = mock_chunker

            result = await embedder.process_url("https://example.com/short")

            assert result["success"] is False
            assert "No chunks generated" in result["error"]

    @pytest.mark.asyncio
    async def test_process_urls_batch_without_progress(
        self, mock_config, mock_client_manager, tmp_path
    ):
        """Test batch processing without progress tracking."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=tmp_path / "test_state.json",
        )

        # Mock process_url
        async def mock_process(url):
            return {"url": url, "success": True, "chunks": 2, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state") as mock_save,
        ):
            results = await embedder.process_urls_batch(
                urls=["https://example.com/1"] * 15,  # 15 URLs to trigger periodic save
                max_concurrent=5,
                progress=None,  # No progress tracking
            )

            assert results["_total"] == 15
            assert results["successful"] == 15
            # Should have saved state at least once (after 10 completions)
            assert mock_save.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_urls_batch_with_exceptions(
        self, mock_config, mock_client_manager, tmp_path
    ):
        """Test batch processing with exceptions."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=tmp_path / "test_state.json",
        )

        # Mock process_url to raise exception
        async def mock_process(url):
            if "exception" in url:
                msg = "Processing failed"
                raise TestError(msg)
            return {"url": url, "success": True, "chunks": 2, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state"),
        ):
            results = await embedder.process_urls_batch(
                urls=["https://example.com/good", "https://example.com/exception"],
                max_concurrent=2,
            )

            assert results["_total"] == 2
            # One should succeed, one should fail due to exception
            successful_count = sum(
                1
                for r in results["results"]
                if isinstance(r, dict) and r.get("success")
            )
            assert successful_count == 1

    @pytest.mark.asyncio
    async def test_run_all_urls_completed(self, mock_config, mock_client_manager):
        """Test run when all URLs are already completed."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        urls = ["https://example.com/1", "https://example.com/2"]
        embedder.state.completed_urls = urls.copy()

        with (
            patch.object(embedder, "initialize_services", new_callable=AsyncMock),
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as mock_process,
        ):
            await embedder.run(urls=urls, resume=True)

            # Should not process any URLs
            mock_process.assert_not_called()

    def test_display_summary_with_failed_urls(
        self, mock_config, mock_client_manager, capsys
    ):
        """Test display summary with failed URLs."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Set up state with failed URLs
        embedder.state.failed_urls = {
            "https://example.com/404": "404 Not Found",
            "https://example.com/500": "500 Internal Server Error",
        }
        embedder.state._total_chunks_processed = 10
        embedder.state._total_embeddings_generated = 10

        results = {
            "_total": 5,
            "successful": 3,
            "failed": 2,
            "results": [],
        }

        embedder._display_summary(results)

        # Check console output
        captured = capsys.readouterr()
        assert "Failed URLs:" in captured.out
        assert "404 Not Found" in captured.out
        assert "500 Internal Server Error" in captured.out


class TestAsyncMain:
    """Test the async main function."""

    @pytest.mark.asyncio
    async def test_async_main_with_urls(self, mock_config, tmp_path):
        """Test async main with direct URLs."""
        with patch("src.crawl4ai_bulk_embedder.ClientManager") as mock_cm_class:
            mock_client_manager = AsyncMock()
            mock_cm_class.return_value = mock_client_manager

            with patch(
                "src.crawl4ai_bulk_embedder.BulkEmbedder"
            ) as mock_embedder_class:
                mock_embedder = AsyncMock()
                mock_embedder.run = AsyncMock()
                mock_embedder_class.return_value = mock_embedder

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

                # Verify embedder was created and run
                mock_embedder.run.assert_called_once()
                call_args = mock_embedder.run.call_args[1]
                assert len(call_args["urls"]) == 2
                assert call_args["max_concurrent"] == 5
                assert call_args["resume"] is True

                # Verify cleanup
                mock_client_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_main_with_file_and_sitemap(self, mock_config, tmp_path):
        """Test async main with file and sitemap inputs."""
        # Create test file
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com/file1\nhttps://example.com/file2\n")

        with patch("src.crawl4ai_bulk_embedder.ClientManager") as mock_cm_class:
            mock_client_manager = AsyncMock()
            mock_cm_class.return_value = mock_client_manager

            with patch(
                "src.crawl4ai_bulk_embedder.BulkEmbedder"
            ) as mock_embedder_class:
                mock_embedder = AsyncMock()
                mock_embedder.run = AsyncMock()
                mock_embedder.load_urls_from_file = AsyncMock(
                    return_value=[
                        "https://example.com/file1",
                        "https://example.com/file2",
                    ]
                )
                mock_embedder.load_urls_from_sitemap = AsyncMock(
                    return_value=[
                        "https://example.com/sitemap1",
                        "https://example.com/sitemap2",
                    ]
                )
                mock_embedder_class.return_value = mock_embedder

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

                # Verify file and sitemap were loaded
                mock_embedder.load_urls_from_file.assert_called_once_with(urls_file)
                mock_embedder.load_urls_from_sitemap.assert_called_once_with(
                    "https://example.com/sitemap.xml"
                )

                # Verify run was called with combined URLs
                mock_embedder.run.assert_called_once()
                call_args = mock_embedder.run.call_args[1]
                assert len(call_args["urls"]) == 4  # 2 from file + 2 from sitemap

    @pytest.mark.asyncio
    async def test_async_main_with_duplicates(self, mock_config, tmp_path):
        """Test async main removes duplicate URLs."""
        with patch("src.crawl4ai_bulk_embedder.ClientManager") as mock_cm_class:
            mock_client_manager = AsyncMock()
            mock_cm_class.return_value = mock_client_manager

            with patch(
                "src.crawl4ai_bulk_embedder.BulkEmbedder"
            ) as mock_embedder_class:
                mock_embedder = AsyncMock()
                mock_embedder.run = AsyncMock()
                mock_embedder_class.return_value = mock_embedder

                # URLs with duplicates
                await _async_main(
                    urls=[
                        "https://example.com/1",
                        "https://example.com/2",
                        "https://example.com/1",  # Duplicate
                        "https://example.com/2",  # Duplicate
                    ],
                    file=None,
                    sitemap=None,
                    collection="test_collection",
                    concurrent=5,
                    config=mock_config,
                    state_file=tmp_path / "state.json",
                    resume=True,
                )

                # Verify only unique URLs are passed
                mock_embedder.run.assert_called_once()
                call_args = mock_embedder.run.call_args[1]
                assert len(call_args["urls"]) == 2  # Only unique URLs

    @pytest.mark.asyncio
    async def test_async_main_cleanup_on_error(self, mock_config, tmp_path):
        """Test async main cleans up on error."""
        with patch("src.crawl4ai_bulk_embedder.ClientManager") as mock_cm_class:
            mock_client_manager = AsyncMock()
            mock_cm_class.return_value = mock_client_manager

            with patch(
                "src.crawl4ai_bulk_embedder.BulkEmbedder"
            ) as mock_embedder_class:
                mock_embedder = AsyncMock()
                # Make run raise an exception
                mock_embedder.run = AsyncMock(side_effect=RuntimeError("Test error"))
                mock_embedder_class.return_value = mock_embedder

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

                # Verify cleanup was still called
                mock_client_manager.cleanup.assert_called_once()
