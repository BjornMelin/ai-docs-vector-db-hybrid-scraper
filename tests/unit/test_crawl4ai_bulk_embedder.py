"""Comprehensive tests for crawl4ai_bulk_embedder module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.config import Config
from src.crawl4ai_bulk_embedder import BulkEmbedder, ProcessingState, main
from src.infrastructure.client_manager import ClientManager
from src.models.document_processing import Chunk


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


@pytest.fixture
def temp_state_file(tmp_path):
    """Create temporary state file."""
    return tmp_path / "test_state.json"


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://docs.example.com/api",
    ]


@pytest.fixture
def sample_state_data():
    """Sample state data for testing."""
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


class TestProcessingState:
    """Test ProcessingState model."""

    def test_default_initialization(self):
        """Test default state initialization."""
        state = ProcessingState()
        assert state.urls_to_process == []
        assert state.completed_urls == []
        assert state.failed_urls == {}
        assert state.total_chunks_processed == 0
        assert state.total_embeddings_generated == 0
        assert state.collection_name == "bulk_embeddings"

    def test_from_dict(self, sample_state_data):
        """Test state creation from dictionary."""
        state = ProcessingState.model_validate(sample_state_data)
        assert state.urls_to_process == ["https://example.com/page3"]
        assert state.completed_urls == ["https://example.com/page1"]
        assert state.failed_urls == {"https://example.com/error": "404 Not Found"}
        assert state.total_chunks_processed == 10
        assert state.collection_name == "test_collection"


class TestBulkEmbedder:
    """Test BulkEmbedder class."""

    def test_initialization(self, mock_config, mock_client_manager, temp_state_file):
        """Test embedder initialization."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            collection_name="test_collection",
            state_file=temp_state_file,
        )

        assert embedder.config == mock_config
        assert embedder.client_manager == mock_client_manager
        assert embedder.collection_name == "test_collection"
        assert embedder.state_file == temp_state_file
        assert isinstance(embedder.state, ProcessingState)

    def test_load_state_from_file(
        self, mock_config, mock_client_manager, temp_state_file, sample_state_data
    ):
        """Test loading state from existing file."""
        # Write state file
        with temp_state_file.open("w") as f:
            json.dump(sample_state_data, f)

        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=temp_state_file,
        )

        assert embedder.state.completed_urls == ["https://example.com/page1"]
        assert embedder.state.total_chunks_processed == 10

    def test_save_state(self, mock_config, mock_client_manager, temp_state_file):
        """Test saving state to file."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=temp_state_file,
        )

        embedder.state.completed_urls = ["https://example.com/test"]
        embedder.state.total_chunks_processed = 5
        embedder._save_state()

        # Verify file was written
        assert temp_state_file.exists()

        # Load and verify
        with temp_state_file.open() as f:
            saved_data = json.load(f)

        assert saved_data["completed_urls"] == ["https://example.com/test"]
        assert saved_data["total_chunks_processed"] == 5

    @pytest.mark.asyncio
    async def test_initialize_services(self, mock_config, mock_client_manager):
        """Test service initialization."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            collection_name="test_collection",
        )

        # Mock services
        mock_crawl_manager = AsyncMock()
        mock_embedding_manager = AsyncMock()
        mock_qdrant_service = AsyncMock()

        mock_client_manager.get_crawl_manager = AsyncMock(
            return_value=mock_crawl_manager
        )
        mock_client_manager.get_embedding_manager = AsyncMock(
            return_value=mock_embedding_manager
        )
        mock_client_manager.get_qdrant_service = AsyncMock(
            return_value=mock_qdrant_service
        )

        # Mock collection doesn't exist
        mock_qdrant_service.list_collections = AsyncMock(
            return_value=["other_collection"]
        )
        mock_qdrant_service.create_collection = AsyncMock(return_value=True)

        await embedder.initialize_services()

        # Verify services initialized
        assert embedder.crawl_manager == mock_crawl_manager
        assert embedder.embedding_manager == mock_embedding_manager
        assert embedder.qdrant_service == mock_qdrant_service

        # Verify collection created
        mock_qdrant_service.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="general",
        )

    @pytest.mark.asyncio
    async def test_load_urls_from_txt_file(
        self, mock_config, mock_client_manager, tmp_path
    ):
        """Test loading URLs from text file."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Create test file
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("""https://example.com/page1
https://example.com/page2
# This is a comment
https://example.com/page3

""")

        urls = await embedder.load_urls_from_file(urls_file)
        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page3" in urls

    @pytest.mark.asyncio
    async def test_load_urls_from_csv_file(
        self, mock_config, mock_client_manager, tmp_path
    ):
        """Test loading URLs from CSV file."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Create test CSV
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text("""url,title
https://example.com/page1,Page 1
https://example.com/page2,Page 2
""")

        urls = await embedder.load_urls_from_file(csv_file)
        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    @pytest.mark.asyncio
    async def test_load_urls_from_json_file(
        self, mock_config, mock_client_manager, tmp_path
    ):
        """Test loading URLs from JSON file."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Test list format
        json_file = tmp_path / "urls.json"
        json_file.write_text(
            json.dumps(
                [
                    "https://example.com/page1",
                    {"url": "https://example.com/page2"},
                ]
            )
        )

        urls = await embedder.load_urls_from_file(json_file)
        assert len(urls) == 2

        # Test dict format
        json_file.write_text(
            json.dumps(
                {"urls": ["https://example.com/page3", "https://example.com/page4"]}
            )
        )

        urls = await embedder.load_urls_from_file(json_file)
        assert len(urls) == 2
        assert "https://example.com/page3" in urls

    @pytest.mark.asyncio
    async def test_load_urls_from_sitemap(self, mock_config, mock_client_manager):
        """Test loading URLs from sitemap."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Mock sitemap response
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://example.com/page1</loc>
    </url>
    <url>
        <loc>https://example.com/page2</loc>
    </url>
</urlset>"""

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.content = sitemap_xml.encode()
            mock_response.raise_for_status = AsyncMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            urls = await embedder.load_urls_from_sitemap(
                "https://example.com/sitemap.xml"
            )

            assert len(urls) == 2
            assert "https://example.com/page1" in urls
            assert "https://example.com/page2" in urls

    @pytest.mark.asyncio
    async def test_process_url_success(self, mock_config, mock_client_manager):
        """Test successful URL processing."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Initialize mock services
        embedder.crawl_manager = AsyncMock()
        embedder.embedding_manager = AsyncMock()
        embedder.qdrant_service = AsyncMock()

        # Mock scraping result
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": True,
                "content": {
                    "markdown": "# Test Page\n\nThis is test content.",
                    "text": "Test Page\nThis is test content.",
                },
                "metadata": {
                    "title": "Test Page",
                    "description": "A test page",
                },
                "provider": "crawl4ai",
            }
        )

        # Mock chunking

        mock_chunk_result = MagicMock()
        mock_chunk_result.chunks = [
            Chunk(
                content="Chunk 1",
                start_pos=0,
                end_pos=100,
                chunk_index=0,
                chunk_type="text",
                has_code=False,
            ),
            Chunk(
                content="Chunk 2",
                start_pos=100,
                end_pos=200,
                chunk_index=1,
                chunk_type="text",
                has_code=False,
            ),
        ]

        with patch("src.crawl4ai_bulk_embedder.DocumentChunker") as mock_chunker_class:
            mock_chunker = MagicMock()
            mock_chunker.chunk_text.return_value = mock_chunk_result
            mock_chunker_class.return_value = mock_chunker

            # Mock embeddings
            embedder.embedding_manager.generate_embeddings = AsyncMock(
                return_value={
                    "embeddings": [[0.1] * 1536, [0.2] * 1536],
                    "sparse_embeddings": [{"0": 0.5}, {"1": 0.6}],
                }
            )

            # Mock Qdrant upsert
            embedder.qdrant_service.upsert_points = AsyncMock(return_value=True)

            result = await embedder.process_url("https://example.com/test")

            assert result["success"] is True
            assert result["chunks"] == 2
            assert result["error"] is None

            # Verify calls
            embedder.crawl_manager.scrape_url.assert_called_once()
            embedder.embedding_manager.generate_embeddings.assert_called_once()
            embedder.qdrant_service.upsert_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_url_failure(self, mock_config, mock_client_manager):
        """Test URL processing failure."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        embedder.crawl_manager = AsyncMock()
        embedder.crawl_manager.scrape_url = AsyncMock(
            return_value={
                "success": False,
                "error": "404 Not Found",
            }
        )

        result = await embedder.process_url("https://example.com/404")

        assert result["success"] is False
        assert result["error"] == "404 Not Found"
        assert result["chunks"] == 0

    @pytest.mark.asyncio
    async def test_process_urls_batch(
        self, mock_config, mock_client_manager, sample_urls, tmp_path
    ):
        """Test batch URL processing."""
        # Use a fresh state file for this test
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
            state_file=tmp_path / "test_batch_state.json",
        )

        # Mock process_url
        async def mock_process(url):
            if "error" in url:
                return {"url": url, "success": False, "chunks": 0, "error": "Failed"}
            return {"url": url, "success": True, "chunks": 2, "error": None}

        with (
            patch.object(embedder, "process_url", side_effect=mock_process),
            patch.object(embedder, "_save_state"),
        ):
            results = await embedder.process_urls_batch(
                urls=[*sample_urls, "https://example.com/error"],
                max_concurrent=2,
            )

            assert results["total"] == 4
            assert results["successful"] == 3
            assert results["failed"] == 1

            # Check state updates
            assert len(embedder.state.completed_urls) == 3
            assert len(embedder.state.failed_urls) == 1
            assert embedder.state.total_chunks_processed == 6

    @pytest.mark.asyncio
    async def test_run_with_resume(self, mock_config, mock_client_manager, sample_urls):
        """Test running with resume functionality."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Set some completed URLs
        embedder.state.completed_urls = [sample_urls[0]]

        with (
            patch.object(embedder, "initialize_services", new_callable=AsyncMock),
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as mock_process,
        ):
            mock_process.return_value = {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "results": [],
            }

            await embedder.run(urls=sample_urls, max_concurrent=5, resume=True)

            # Should only process uncompleted URLs
            call_args = mock_process.call_args[1]
            processed_urls = call_args["urls"]
            assert len(processed_urls) == 2
            assert sample_urls[0] not in processed_urls

    @pytest.mark.asyncio
    async def test_run_without_resume(
        self, mock_config, mock_client_manager, sample_urls
    ):
        """Test running without resume."""
        embedder = BulkEmbedder(
            config=mock_config,
            client_manager=mock_client_manager,
        )

        # Set some completed URLs
        embedder.state.completed_urls = [sample_urls[0]]

        with (
            patch.object(embedder, "initialize_services", new_callable=AsyncMock),
            patch.object(
                embedder, "process_urls_batch", new_callable=AsyncMock
            ) as mock_process,
        ):
            mock_process.return_value = {
                "total": 3,
                "successful": 3,
                "failed": 0,
                "results": [],
            }

            await embedder.run(urls=sample_urls, max_concurrent=5, resume=False)

            # Should process all URLs
            call_args = mock_process.call_args[1]
            processed_urls = call_args["urls"]
            assert len(processed_urls) == 3


class TestCLI:
    """Test CLI functionality."""

    def test_cli_with_urls(self):
        """Test CLI with direct URLs."""
        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.ConfigLoader.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            runner = CliRunner()

            result = runner.invoke(
                main,
                [
                    "-u",
                    "https://example.com/page1",
                    "-u",
                    "https://example.com/page2",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_cli_with_file(self, tmp_path):
        """Test CLI with file input."""
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com/page1\n")

        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.ConfigLoader.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            runner = CliRunner()

            result = runner.invoke(
                main,
                [
                    "-f",
                    str(urls_file),
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_cli_with_sitemap(self):
        """Test CLI with sitemap."""
        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.ConfigLoader.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

            runner = CliRunner()

            result = runner.invoke(
                main,
                [
                    "-s",
                    "https://example.com/sitemap.xml",
                ],
            )

            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_cli_no_input(self):
        """Test CLI with no input."""
        runner = CliRunner()

        result = runner.invoke(main, [])

        assert result.exit_code == 1
        assert "Error: Must provide URLs" in result.output

    def test_cli_with_all_options(self, tmp_path):
        """Test CLI with all options."""
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://example.com\n")

        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        with (
            patch("src.crawl4ai_bulk_embedder.asyncio.run") as mock_run,
            patch("src.crawl4ai_bulk_embedder.ConfigLoader.load_config") as mock_config,
        ):
            mock_config.return_value = MagicMock()

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

            # Check arguments passed to async main
            call_args = mock_run.call_args[0][0]
            assert call_args == mock_run.call_args[0][0]  # Verify it's a coroutine
