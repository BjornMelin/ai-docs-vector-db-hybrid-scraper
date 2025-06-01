"""Test QdrantConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import CollectionHNSWConfigs
from config.models import HNSWConfig
from config.models import QdrantConfig


class TestQdrantConfig:
    """Test QdrantConfig model validation and behavior."""

    def test_default_values(self):
        """Test QdrantConfig with default values."""
        config = QdrantConfig()

        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.prefer_grpc is False
        assert config.collection_name == "documents"

        # Performance settings
        assert config.batch_size == 100
        assert config.max_retries == 3

        # Legacy HNSW settings
        assert config.hnsw_ef_construct == 200
        assert config.hnsw_m == 16
        assert config.quantization_enabled is True

        # Collection-specific HNSW configurations
        assert isinstance(config.collection_hnsw_configs, CollectionHNSWConfigs)
        assert config.enable_hnsw_optimization is True

    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        config1 = QdrantConfig(url="http://localhost:6333")
        assert config1.url == "http://localhost:6333"

        config2 = QdrantConfig(url="https://qdrant.example.com")
        assert config2.url == "https://qdrant.example.com"

        # URL with trailing slash should be stripped
        config3 = QdrantConfig(url="http://localhost:6333/")
        assert config3.url == "http://localhost:6333"

        # Invalid URLs
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(url="qdrant.example.com")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("url",)
        assert "must start with http:// or https://" in str(errors[0]["msg"])

    def test_batch_size_constraints(self):
        """Test batch size constraints."""
        # Valid batch sizes
        config1 = QdrantConfig(batch_size=1)
        assert config1.batch_size == 1

        config2 = QdrantConfig(batch_size=1000)
        assert config2.batch_size == 1000

        # Invalid batch sizes
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(batch_size=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)
        assert "greater than 0" in str(errors[0]["msg"])

        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(batch_size=1001)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)
        assert "less than or equal to 1000" in str(errors[0]["msg"])

    def test_max_retries_constraints(self):
        """Test max retries constraints."""
        # Valid retry counts
        config1 = QdrantConfig(max_retries=0)
        assert config1.max_retries == 0

        config2 = QdrantConfig(max_retries=10)
        assert config2.max_retries == 10

        # Invalid retry counts
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(max_retries=-1)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)

        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(max_retries=11)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)

    def test_timeout_constraints(self):
        """Test timeout constraints."""
        # Valid timeout
        config = QdrantConfig(timeout=60.0)
        assert config.timeout == 60.0

        # Invalid timeout
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(timeout=0.0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_hnsw_parameters(self):
        """Test HNSW parameter constraints."""
        # Valid HNSW parameters
        config = QdrantConfig(hnsw_ef_construct=500, hnsw_m=32)
        assert config.hnsw_ef_construct == 500
        assert config.hnsw_m == 32

        # Invalid HNSW parameters
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(hnsw_ef_construct=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("hnsw_ef_construct",)

        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(hnsw_m=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("hnsw_m",)

    def test_collection_hnsw_configs(self):
        """Test collection-specific HNSW configurations."""
        config = QdrantConfig()

        # Check default collection configs
        assert isinstance(config.collection_hnsw_configs.api_reference, HNSWConfig)
        assert config.collection_hnsw_configs.api_reference.m == 20
        assert config.collection_hnsw_configs.api_reference.ef_construct == 300

        assert isinstance(config.collection_hnsw_configs.tutorials, HNSWConfig)
        assert config.collection_hnsw_configs.tutorials.m == 16

        assert isinstance(config.collection_hnsw_configs.blog_posts, HNSWConfig)
        assert config.collection_hnsw_configs.blog_posts.m == 12

        assert isinstance(config.collection_hnsw_configs.code_examples, HNSWConfig)
        assert config.collection_hnsw_configs.code_examples.m == 18

        assert isinstance(config.collection_hnsw_configs.general, HNSWConfig)
        assert config.collection_hnsw_configs.general.m == 16

    def test_custom_collection_hnsw_configs(self):
        """Test custom collection HNSW configurations."""
        custom_hnsw = CollectionHNSWConfigs(
            api_reference=HNSWConfig(m=24, ef_construct=400),
            tutorials=HNSWConfig(m=14, ef_construct=180),
        )

        config = QdrantConfig(collection_hnsw_configs=custom_hnsw)

        assert config.collection_hnsw_configs.api_reference.m == 24
        assert config.collection_hnsw_configs.api_reference.ef_construct == 400
        assert config.collection_hnsw_configs.tutorials.m == 14

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(url="http://localhost:6333", unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = QdrantConfig(
            url="https://qdrant.cloud.com",
            api_key="test-api-key",
            collection_name="my_docs",
            batch_size=50,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["url"] == "https://qdrant.cloud.com"
        assert data["api_key"] == "test-api-key"
        assert data["collection_name"] == "my_docs"
        assert data["batch_size"] == 50

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"url":"https://qdrant.cloud.com"' in json_str
        assert '"api_key":"test-api-key"' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = QdrantConfig(url="http://localhost:6333", batch_size=100)

        updated = original.model_copy(
            update={"url": "http://remote:6333", "batch_size": 200}
        )

        assert original.url == "http://localhost:6333"
        assert original.batch_size == 100
        assert updated.url == "http://remote:6333"
        assert updated.batch_size == 200

    def test_type_validation(self):
        """Test type validation for fields."""
        # Test boolean field
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(prefer_grpc={"value": True})  # Dict can't coerce to bool

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("prefer_grpc",)

        # Test float field
        with pytest.raises(ValidationError) as exc_info:
            QdrantConfig(timeout="30 seconds")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("timeout",)
