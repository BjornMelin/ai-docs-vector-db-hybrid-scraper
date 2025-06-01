"""Unit tests for config enums module."""

from enum import Enum

from src.config.enums import ChunkingStrategy
from src.config.enums import CollectionStatus
from src.config.enums import CrawlProvider
from src.config.enums import DocumentStatus
from src.config.enums import EmbeddingModel
from src.config.enums import EmbeddingProvider
from src.config.enums import Environment
from src.config.enums import FusionAlgorithm
from src.config.enums import LogLevel
from src.config.enums import QualityTier
from src.config.enums import SearchAccuracy
from src.config.enums import SearchStrategy
from src.config.enums import VectorType


class TestEnvironment:
    """Test Environment enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.PRODUCTION.value == "production"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(Environment.DEVELOPMENT, str)
        assert Environment.DEVELOPMENT == "development"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in Environment]
        assert "development" in members
        assert "testing" in members
        assert "production" in members
        assert len(members) == 3


class TestLogLevel:
    """Test LogLevel enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(LogLevel.INFO, str)
        assert LogLevel.INFO == "INFO"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in LogLevel]
        assert len(members) == 5
        assert all(
            level in members
            for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )


class TestEmbeddingProvider:
    """Test EmbeddingProvider enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.FASTEMBED.value == "fastembed"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(EmbeddingProvider.OPENAI, str)
        assert EmbeddingProvider.OPENAI == "openai"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in EmbeddingProvider]
        assert len(members) == 2
        assert "openai" in members
        assert "fastembed" in members


class TestCrawlProvider:
    """Test CrawlProvider enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert CrawlProvider.CRAWL4AI.value == "crawl4ai"
        assert CrawlProvider.FIRECRAWL.value == "firecrawl"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(CrawlProvider.CRAWL4AI, str)
        assert CrawlProvider.CRAWL4AI == "crawl4ai"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in CrawlProvider]
        assert len(members) == 2
        assert "crawl4ai" in members
        assert "firecrawl" in members


class TestChunkingStrategy:
    """Test ChunkingStrategy enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ChunkingStrategy.BASIC.value == "basic"
        assert ChunkingStrategy.ENHANCED.value == "enhanced"
        assert ChunkingStrategy.AST.value == "ast"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(ChunkingStrategy.ENHANCED, str)
        assert ChunkingStrategy.ENHANCED == "enhanced"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in ChunkingStrategy]
        assert len(members) == 3
        assert all(strategy in members for strategy in ["basic", "enhanced", "ast"])


class TestSearchStrategy:
    """Test SearchStrategy enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert SearchStrategy.DENSE.value == "dense"
        assert SearchStrategy.SPARSE.value == "sparse"
        assert SearchStrategy.HYBRID.value == "hybrid"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(SearchStrategy.HYBRID, str)
        assert SearchStrategy.HYBRID == "hybrid"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in SearchStrategy]
        assert len(members) == 3
        assert all(strategy in members for strategy in ["dense", "sparse", "hybrid"])


class TestEmbeddingModel:
    """Test EmbeddingModel enum."""

    def test_openai_models(self):
        """Test OpenAI model values."""
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value == "text-embedding-3-small"
        assert EmbeddingModel.TEXT_EMBEDDING_3_LARGE.value == "text-embedding-3-large"

    def test_fastembed_models(self):
        """Test FastEmbed model values."""
        assert EmbeddingModel.NV_EMBED_V2.value == "nvidia/NV-Embed-v2"
        assert EmbeddingModel.BGE_SMALL_EN_V15.value == "BAAI/bge-small-en-v1.5"
        assert EmbeddingModel.BGE_LARGE_EN_V15.value == "BAAI/bge-large-en-v1.5"

    def test_sparse_models(self):
        """Test sparse model values."""
        assert EmbeddingModel.SPLADE_PP_EN_V1.value == "prithvida/Splade_PP_en_v1"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(EmbeddingModel.TEXT_EMBEDDING_3_SMALL, str)
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL == "text-embedding-3-small"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in EmbeddingModel]
        assert len(members) == 6
        # Check categories
        openai_models = [m for m in members if m.startswith("text-embedding")]
        assert len(openai_models) == 2
        nvidia_models = [m for m in members if "nvidia" in m]
        assert len(nvidia_models) == 1
        bge_models = [m for m in members if "bge" in m]
        assert len(bge_models) == 2
        splade_models = [m for m in members if "Splade" in m]
        assert len(splade_models) == 1


class TestQualityTier:
    """Test QualityTier enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert QualityTier.ECONOMY.value == "economy"
        assert QualityTier.BALANCED.value == "balanced"
        assert QualityTier.PREMIUM.value == "premium"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(QualityTier.BALANCED, str)
        assert QualityTier.BALANCED == "balanced"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in QualityTier]
        assert len(members) == 3
        assert all(tier in members for tier in ["economy", "balanced", "premium"])


class TestDocumentStatus:
    """Test DocumentStatus enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.COMPLETED.value == "completed"
        assert DocumentStatus.FAILED.value == "failed"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(DocumentStatus.PENDING, str)
        assert DocumentStatus.PENDING == "pending"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in DocumentStatus]
        assert len(members) == 4
        assert all(
            status in members
            for status in ["pending", "processing", "completed", "failed"]
        )


class TestCollectionStatus:
    """Test CollectionStatus enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert CollectionStatus.GREEN.value == "green"
        assert CollectionStatus.YELLOW.value == "yellow"
        assert CollectionStatus.RED.value == "red"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(CollectionStatus.GREEN, str)
        assert CollectionStatus.GREEN == "green"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in CollectionStatus]
        assert len(members) == 3
        assert all(status in members for status in ["green", "yellow", "red"])


class TestFusionAlgorithm:
    """Test FusionAlgorithm enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert FusionAlgorithm.RRF.value == "rrf"
        assert FusionAlgorithm.DBSF.value == "dbsf"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(FusionAlgorithm.RRF, str)
        assert FusionAlgorithm.RRF == "rrf"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in FusionAlgorithm]
        assert len(members) == 2
        assert "rrf" in members
        assert "dbsf" in members


class TestSearchAccuracy:
    """Test SearchAccuracy enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert SearchAccuracy.FAST.value == "fast"
        assert SearchAccuracy.BALANCED.value == "balanced"
        assert SearchAccuracy.ACCURATE.value == "accurate"
        assert SearchAccuracy.EXACT.value == "exact"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(SearchAccuracy.BALANCED, str)
        assert SearchAccuracy.BALANCED == "balanced"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in SearchAccuracy]
        assert len(members) == 4
        assert all(
            accuracy in members
            for accuracy in ["fast", "balanced", "accurate", "exact"]
        )


class TestVectorType:
    """Test VectorType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert VectorType.DENSE.value == "dense"
        assert VectorType.SPARSE.value == "sparse"
        assert VectorType.HYDE.value == "hyde"

    def test_is_string_enum(self):
        """Test that it's a string enum."""
        assert isinstance(VectorType.DENSE, str)
        assert VectorType.DENSE == "dense"

    def test_all_members(self):
        """Test all enum members are defined."""
        members = [e.value for e in VectorType]
        assert len(members) == 3
        assert all(vtype in members for vtype in ["dense", "sparse", "hyde"])


class TestEnumInheritance:
    """Test that all enums properly inherit from str and Enum."""

    def test_all_enums_are_string_enums(self):
        """Test that all enums are string enums."""
        enums_to_test = [
            Environment,
            LogLevel,
            EmbeddingProvider,
            CrawlProvider,
            ChunkingStrategy,
            SearchStrategy,
            EmbeddingModel,
            QualityTier,
            DocumentStatus,
            CollectionStatus,
            FusionAlgorithm,
            SearchAccuracy,
            VectorType,
        ]

        for enum_class in enums_to_test:
            # Check that the enum inherits from str and Enum
            assert issubclass(enum_class, str)
            assert issubclass(enum_class, Enum)

            # Check that instances are strings
            first_member = next(iter(enum_class))
            assert isinstance(first_member, str)
            assert isinstance(first_member.value, str)


class TestEnumComparisons:
    """Test enum comparisons and string operations."""

    def test_string_comparison(self):
        """Test that enums can be compared with strings."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.DEVELOPMENT == "development"
        assert Environment.DEVELOPMENT != "production"

    def test_case_sensitive(self):
        """Test that enum comparisons are case-sensitive."""
        assert LogLevel.INFO == "INFO"
        assert LogLevel.INFO != "info"
        assert LogLevel.INFO != "Info"

    def test_in_operator(self):
        """Test that enums work with the 'in' operator."""
        valid_environments = ["development", "testing", "production"]
        assert Environment.DEVELOPMENT in valid_environments

        # Also works with enum values
        all_environments = list(Environment)
        assert Environment.DEVELOPMENT in all_environments
