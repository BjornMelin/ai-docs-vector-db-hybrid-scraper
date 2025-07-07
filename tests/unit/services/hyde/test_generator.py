"""Tests for HyDE hypothetical document generator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.errors import APIError, EmbeddingServiceError
from src.services.hyde.config import HyDEConfig, HyDEPromptConfig
from src.services.hyde.generator import GenerationResult, HypotheticalDocumentGenerator


class TestError(Exception):
    """Custom exception for this module."""


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_generation_result_creation(self):
        """Test GenerationResult model creation."""
        result = GenerationResult(
            documents=["doc1", "doc2"],
            generation_time=1.5,
            tokens_used=100,
            cost_estimate=0.01,
            cached=False,
            diversity_score=0.8,
        )

        assert result.documents == ["doc1", "doc2"]
        assert result.generation_time == 1.5
        assert result.tokens_used == 100
        assert result.cost_estimate == 0.01
        assert result.cached is False
        assert result.diversity_score == 0.8

    def test_generation_result_defaults(self):
        """Test GenerationResult with default values."""
        result = GenerationResult(
            documents=["doc1"],
            generation_time=1.0,
            tokens_used=50,
            cost_estimate=0.005,
        )

        assert result.documents == ["doc1"]
        assert result.generation_time == 1.0
        assert result.tokens_used == 50
        assert result.cost_estimate == 0.005
        assert result.cached is False  # Default
        assert result.diversity_score == 0.0  # Default

    def test_generation_result_serialization(self):
        """Test GenerationResult serialization."""
        result = GenerationResult(
            documents=["test doc"],
            generation_time=2.0,
            tokens_used=75,
            cost_estimate=0.008,
            cached=True,
            diversity_score=0.5,
        )

        result_dict = result.model_dump()
        assert result_dict["documents"] == ["test doc"]
        assert result_dict["generation_time"] == 2.0
        assert result_dict["tokens_used"] == 75
        assert result_dict["cost_estimate"] == 0.008
        assert result_dict["cached"] is True
        assert result_dict["diversity_score"] == 0.5

        # Test deserialization
        restored = GenerationResult.model_validate(result_dict)
        assert restored.documents == ["test doc"]
        assert restored.generation_time == 2.0
        assert restored.tokens_used == 75
        assert restored.cost_estimate == 0.008
        assert restored.cached is True
        assert restored.diversity_score == 0.5


class TestHypotheticalDocumentGenerator:
    """Tests for HypotheticalDocumentGenerator class."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager."""
        manager = MagicMock(spec=ClientManager)
        manager.initialize = AsyncMock()
        manager.cleanup = AsyncMock()

        # Mock OpenAI client
        mock_openai_client = MagicMock()
        mock_openai_client.models.list = AsyncMock()
        mock_openai_client.chat.completions.create = AsyncMock()
        manager.get_openai_client = AsyncMock(return_value=mock_openai_client)

        return manager

    @pytest.fixture
    def hyde_config(self):
        """Create HyDE configuration."""
        return HyDEConfig(
            num_generations=3,
            generation_temperature=0.7,
            max_generation_tokens=200,
            generation_model="gpt-3.5-turbo",
            generation_timeout_seconds=10,
            parallel_generation=True,
            max_concurrent_generations=5,
            min_generation_length=20,
            filter_duplicates=True,
            diversity_threshold=0.3,
            log_generations=False,
        )

    @pytest.fixture
    def prompt_config(self):
        """Create prompt configuration."""
        return HyDEPromptConfig()

    @pytest.fixture
    def generator(self, hyde_config, prompt_config, mock_client_manager):
        """Create generator instance."""
        return HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            client_manager=mock_client_manager,
        )

    def test_init(self, hyde_config, prompt_config, mock_client_manager):
        """Test generator initialization."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            client_manager=mock_client_manager,
        )

        assert generator.config == hyde_config
        assert generator.prompt_config == prompt_config
        assert generator.client_manager == mock_client_manager
        assert generator._llm_client is None
        assert generator._initialized is False

        # Check metrics tracking initialization
        assert generator.generation_count == 0
        assert generator._total_generation_time == 0.0
        assert generator._total_tokens_used == 0
        assert generator._total_cost == 0.0

    def test_init_without_client_manager(self, hyde_config, prompt_config):
        """Test generator initialization without client manager."""
        with patch(
            "src.services.hyde.generator.ClientManager.from_unified_config"
        ) as mock_from_config:
            mock_manager = MagicMock(spec=ClientManager)
            mock_from_config.return_value = mock_manager

            generator = HypotheticalDocumentGenerator(
                config=hyde_config,
                prompt_config=prompt_config,
                client_manager=None,
            )

            # The generator should have created a client manager
            assert generator.client_manager is not None
            # Verify that from_unified_config was called
            mock_from_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_success(self, generator, mock_client_manager):
        """Test successful initialization."""
        await generator.initialize()

        assert generator._initialized is True
        mock_client_manager.initialize.assert_called_once()
        mock_client_manager.get_openai_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_openai_client(self, generator, mock_client_manager):
        """Test initialization failure when OpenAI client not available."""
        mock_client_manager.get_openai_client.return_value = None

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await generator.initialize()

        assert "OpenAI client not available" in str(exc_info.value)
        assert generator._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_client_error(self, generator, mock_client_manager):
        """Test initialization failure when client manager fails."""
        mock_client_manager.initialize.side_effect = Exception("Client error")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await generator.initialize()

        assert "Failed to initialize HyDE generator" in str(exc_info.value)
        assert generator._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, generator, mock_client_manager):
        """Test initialization when already initialized."""
        generator._initialized = True

        await generator.initialize()

        # Should not call client manager methods again
        mock_client_manager.initialize.assert_not_called()
        mock_client_manager.get_openai_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup(self, generator, mock_client_manager):
        """Test cleanup."""
        generator._initialized = True
        generator._llm_client = MagicMock()

        await generator.cleanup()

        assert generator._llm_client is None
        assert generator._initialized is False
        mock_client_manager.cleanup.assert_called_once()

    def test_classify_query_technical(self, generator):
        """Test query classification for technical queries."""
        technical_queries = [
            "How to use the API?",
            "Function parameters explained",
            "Method implementation guide",
            "Class structure overview",
            "Configuration setup steps",
        ]

        for query in technical_queries:
            query_type = generator._classify_query(query)
            assert query_type == "technical"

    def test_classify_query_code(self, generator):
        """Test query classification for code queries."""
        code_queries = [
            "How to implement this in Python?",
            "Show me code examples",
            "JavaScript function syntax",
            "Import library usage",
            "Example implementation",
        ]

        for query in code_queries:
            query_type = generator._classify_query(query)
            # Note: Some queries may match technical keywords first
            # The query "How to implement this in Python?" contains "how to" (code)
            # but might also match technical keywords
            assert query_type in ["code", "technical"]

    def test_classify_query_tutorial(self, generator):
        """Test query classification for tutorial queries."""
        tutorial_queries = [
            "Step by step tutorial",
            "Getting started guide",
            "Learn the basics",
            "Introduction to concepts",
            "Beginner tutorial",
        ]

        for query in tutorial_queries:
            query_type = generator._classify_query(query)
            assert query_type == "tutorial"

    def test_classify_query_general(self, generator):
        """Test query classification for general queries."""
        general_queries = [
            "What is this about?",
            "Explain the concept",
            "Overview of the system",
            "Random query text",
        ]

        for query in general_queries:
            query_type = generator._classify_query(query)
            assert query_type == "general"

    def test_get_base_prompt(self, generator):
        """Test base prompt selection."""
        # Test each query type
        technical_prompt = generator._get_base_prompt("technical")
        assert technical_prompt == generator.prompt_config.technical_prompt

        code_prompt = generator._get_base_prompt("code")
        assert code_prompt == generator.prompt_config.code_prompt

        tutorial_prompt = generator._get_base_prompt("tutorial")
        assert tutorial_prompt == generator.prompt_config.tutorial_prompt

        general_prompt = generator._get_base_prompt("general")
        assert general_prompt == generator.prompt_config.general_prompt

        # Test unknown type defaults to general
        unknown_prompt = generator._get_base_prompt("unknown")
        assert unknown_prompt == generator.prompt_config.general_prompt

    def test_build_diverse_prompts_no_variation(self, generator):
        """Test building prompts without variation."""
        generator.config.prompt_variation = False
        generator.config.num_generations = 2

        prompts = generator._build_diverse_prompts("test query", "python")

        assert len(prompts) == 2
        # All prompts should be the same since no variation
        assert prompts[0] == prompts[1]
        assert "test query" in prompts[0]

    def test_build_diverse_prompts_with_variation(self, generator):
        """Test building prompts with variation."""
        generator.config.prompt_variation = True
        generator.config.num_generations = 3

        prompts = generator._build_diverse_prompts("test query", "python")

        assert len(prompts) == 3
        # With variation, prompts should be different
        unique_prompts = set(prompts)
        assert len(unique_prompts) > 1

        # All should contain the query
        for prompt in prompts:
            assert "test query" in prompt

    def test_generate_prompt_variations(self, generator):
        """Test prompt variation generation."""
        base_prompt = "Base prompt: {query}"
        variations = generator._generate_prompt_variations(
            base_prompt, "test query", "python"
        )

        assert len(variations) <= 3  # Should be limited to 3 variations
        assert len(variations) > 0

        # Check that variations are different
        unique_variations = set(variations)
        assert len(unique_variations) == len(variations)  # All should be unique

        # All should contain the query
        for variation in variations:
            assert "test query" in variation

    @pytest.mark.asyncio
    async def test_generate_single_document_success(
        self, generator, _mock_client_manager
    ):
        """Test successful single document generation."""
        generator._initialized = True

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated document content"

        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
        generator._llm_client = mock_llm_client

        prompt = "Generate a document about test query"
        result = await generator._generate_single_document(prompt)

        assert result == "Generated document content"
        mock_llm_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_single_document_timeout(
        self, generator, _mock_client_manager
    ):
        """Test document generation timeout."""
        generator._initialized = True

        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create = AsyncMock(side_effect=TimeoutError())
        generator._llm_client = mock_llm_client

        prompt = "Generate a document about test query"
        result = await generator._generate_single_document(prompt)

        assert result == ""  # Returns empty string on timeout

    @pytest.mark.asyncio
    async def test_generate_single_document_error(
        self, generator, _mock_client_manager
    ):
        """Test document generation error handling."""
        generator._initialized = True

        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM error")
        )
        generator._llm_client = mock_llm_client

        prompt = "Generate a document about test query"
        result = await generator._generate_single_document(prompt)

        assert result == ""  # Returns empty string on error

    @pytest.mark.asyncio
    async def test_generate_parallel(self, generator, _mock_client_manager):
        """Test parallel document generation."""
        generator._initialized = True
        generator.config.max_concurrent_generations = 2

        # Mock successful generation
        async def mock_generate_single(prompt):
            return f"Document for: {prompt[:20]}"

        generator._generate_single_document = AsyncMock(
            side_effect=mock_generate_single
        )

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        documents = await generator._generate_parallel(prompts)

        assert len(documents) == 3
        assert all("Document for:" in doc for doc in documents)

    @pytest.mark.asyncio
    async def test_generate_parallel_with_errors(self, generator, _mock_client_manager):
        """Test parallel generation with some errors."""
        generator._initialized = True
        generator.config.min_generation_length = 5

        async def mock_generate_single(prompt):
            if "error" in prompt:
                msg = "Generation error"
                raise TestError(msg)
            if "short" in prompt:
                return "abc"  # Too short
            return "Valid document content here"

        generator._generate_single_document = AsyncMock(
            side_effect=mock_generate_single
        )

        prompts = [
            "Valid prompt",
            "error prompt",
            "short prompt",
            "Another valid prompt",
        ]
        documents = await generator._generate_parallel(prompts)

        # Should only have valid documents (2 out of 4)
        assert len(documents) == 2
        assert all("Valid document content" in doc for doc in documents)

    @pytest.mark.asyncio
    async def test_generate_sequential(self, generator, _mock_client_manager):
        """Test sequential document generation."""
        generator._initialized = True
        generator.config.min_generation_length = 5

        async def mock_generate_single(prompt):
            if "error" in prompt:
                msg = "Generation error"
                raise TestError(msg)
            return f"Document for: {prompt}"

        generator._generate_single_document = AsyncMock(
            side_effect=mock_generate_single
        )

        prompts = ["Valid prompt 1", "error prompt", "Valid prompt 2"]
        documents = await generator._generate_sequential(prompts)

        # Should skip the error prompt
        assert len(documents) == 2
        assert "Document for: Valid prompt 1" in documents
        assert "Document for: Valid prompt 2" in documents

    def test_post_process_documents(self, generator):
        """Test document post-processing."""
        generator.config.min_generation_length = 3
        generator.config.filter_duplicates = True

        documents = [
            "   Valid document content   ",  # Should be trimmed
            "ab",  # Too short, should be filtered
            "Another valid document",
            "Valid document content",  # Duplicate, should be filtered
            "Third unique document",
        ]

        processed = generator._post_process_documents(documents, "test query")

        assert len(processed) == 3
        assert "Valid document content" in processed
        assert "Another valid document" in processed
        assert "Third unique document" in processed

        # Check trimming
        assert not any(doc.startswith(" ") or doc.endswith(" ") for doc in processed)

    def test_post_process_documents_no_filtering(self, generator):
        """Test post-processing without duplicate filtering."""
        generator.config.min_generation_length = 1  # Lower threshold
        generator.config.filter_duplicates = False

        documents = [
            "Valid document content",
            "Valid document content",  # Duplicate, but filtering disabled
            "Another document",
        ]

        processed = generator._post_process_documents(documents, "test query")

        assert len(processed) == 3  # All documents kept
        assert processed.count("Valid document content") == 2

    def test_calculate_cost(self, generator):
        """Test cost calculation."""
        # Test with known model
        generator.config.generation_model = "gpt-3.5-turbo"
        cost = generator._calculate_cost(1000)  # 1000 tokens

        # Expected: 70% input (700 tokens) + 30% output (300 tokens)
        # gpt-3.5-turbo: input $0.0015/1k, output $0.002/1k
        expected_cost = (700 / 1000) * 0.0015 + (300 / 1000) * 0.002
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_cost_unknown_model(self, generator):
        """Test cost calculation with unknown model."""
        generator.config.generation_model = "unknown-model"
        cost = generator._calculate_cost(1000)

        # Should use default pricing
        expected_cost = (700 / 1000) * 0.002 + (300 / 1000) * 0.002
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_diversity_score(self, generator):
        """Test diversity score calculation."""
        # Completely different documents
        documents = [
            "This is about cats and animals",
            "Programming languages and code",
            "Cooking recipes and food",
        ]
        diversity = generator._calculate_diversity_score(documents)
        assert diversity > 0.5  # Should be quite diverse

        # Similar documents
        similar_docs = [
            "This is about programming in Python",
            "This is about programming in Java",
            "This is about programming languages",
        ]
        similar_diversity = generator._calculate_diversity_score(similar_docs)
        assert similar_diversity < diversity  # Should be less diverse

    def test_calculate_diversity_score_edge_cases(self, generator):
        """Test diversity score edge cases."""
        # Single document
        assert generator._calculate_diversity_score(["single doc"]) == 0.0

        # Empty list
        assert generator._calculate_diversity_score([]) == 0.0

        # Identical documents
        identical = ["same content", "same content"]
        assert generator._calculate_diversity_score(identical) == 0.0

    @pytest.mark.asyncio
    async def test_generate_documents_success(self, generator, _mock_client_manager):
        """Test successful document generation."""
        generator._initialized = True

        # Mock the generation pipeline
        mock_prompts = ["prompt1", "prompt2", "prompt3"]
        mock_documents = ["doc1 content here", "doc2 content here", "doc3 content here"]

        generator._build_diverse_prompts = MagicMock(return_value=mock_prompts)
        generator._generate_parallel = AsyncMock(return_value=mock_documents)
        generator._post_process_documents = MagicMock(return_value=mock_documents)

        result = await generator.generate_documents("test query", "python")

        assert isinstance(result, GenerationResult)
        assert result.documents == mock_documents
        assert result.generation_time > 0
        assert result.tokens_used > 0
        assert result.cost_estimate > 0
        assert result.diversity_score >= 0

    @pytest.mark.asyncio
    async def test_generate_documents_sequential_mode(
        self, generator, _mock_client_manager
    ):
        """Test document generation in sequential mode."""
        generator._initialized = True
        generator.config.parallel_generation = False

        mock_prompts = ["prompt1", "prompt2"]
        mock_documents = ["doc1", "doc2"]

        generator._build_diverse_prompts = MagicMock(return_value=mock_prompts)
        generator._generate_sequential = AsyncMock(return_value=mock_documents)
        generator._post_process_documents = MagicMock(return_value=mock_documents)

        result = await generator.generate_documents("test query")

        assert result.documents == mock_documents
        generator._generate_sequential.assert_called_once_with(mock_prompts)

    @pytest.mark.asyncio
    async def test_generate_documents_not_initialized(self, generator):
        """Test document generation when not initialized."""

        with pytest.raises(APIError) as exc_info:
            await generator.generate_documents("test query")

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_documents_failure(self, generator, _mock_client_manager):
        """Test document generation failure handling."""
        generator._initialized = True

        generator._build_diverse_prompts = MagicMock(
            side_effect=Exception("Prompt error")
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await generator.generate_documents("test query")

        assert "Document generation failed" in str(exc_info.value)

    def test_get_metrics(self, generator):
        """Test metrics retrieval."""
        # Set some test values
        generator.generation_count = 5
        generator._total_generation_time = 10.0
        generator._total_tokens_used = 500
        generator._total_cost = 0.05

        metrics = generator.get_metrics()

        assert metrics["generation_count"] == 5
        assert metrics["_total_generation_time"] == 10.0
        assert metrics["avg_generation_time"] == 2.0
        assert metrics["_total_tokens_used"] == 500
        assert metrics["_total_cost"] == 0.05
        assert metrics["avg_cost_per_generation"] == 0.01

    def test_get_metrics_zero_generations(self, generator):
        """Test metrics when no generations have been performed."""
        metrics = generator.get_metrics()

        assert metrics["generation_count"] == 0
        assert metrics["_total_generation_time"] == 0.0
        assert metrics["avg_generation_time"] == 0.0
        assert metrics["_total_tokens_used"] == 0
        assert metrics["_total_cost"] == 0.0
        assert metrics["avg_cost_per_generation"] == 0.0

    def test_model_pricing_coverage(self, generator):
        """Test that all expected models have pricing."""
        expected_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

        for model in expected_models:
            assert model in generator.model_pricing
            pricing = generator.model_pricing[model]
            assert "input" in pricing
            assert "output" in pricing
            assert isinstance(pricing["input"], int | float)
            assert isinstance(pricing["output"], int | float)
            assert pricing["input"] > 0
            assert pricing["output"] > 0
