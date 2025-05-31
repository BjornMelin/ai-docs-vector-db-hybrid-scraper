"""Comprehensive tests for HyDE hypothetical document generator."""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.infrastructure.client_manager import ClientManager
from src.services.errors import EmbeddingServiceError
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEPromptConfig
from src.services.hyde.generator import GenerationResult
from src.services.hyde.generator import HypotheticalDocumentGenerator


@pytest.fixture
def mock_config():
    """Create mock HyDE configuration."""
    return HyDEConfig(
        enable_hyde=True,
        num_generations=3,
        generation_temperature=0.7,
        max_generation_tokens=200,
        generation_model="gpt-3.5-turbo",
        generation_timeout_seconds=10,
        parallel_generation=True,
        max_concurrent_generations=5,
        use_domain_specific_prompts=True,
        prompt_variation=True,
        min_generation_length=20,
        filter_duplicates=True,
        log_generations=True,
        track_metrics=True,
    )


@pytest.fixture
def mock_prompt_config():
    """Create mock prompt configuration."""
    return HyDEPromptConfig(
        technical_prompt="Answer this technical question: {query}",
        code_prompt="Provide code for: {query}",
        tutorial_prompt="Explain step by step: {query}",
        general_prompt="Answer: {query}",
        technical_keywords=["api", "function", "method", "configuration"],
        code_keywords=["python", "javascript", "code", "implement"],
        tutorial_keywords=["tutorial", "guide", "learn", "getting started"],
    )


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    manager = AsyncMock(spec=ClientManager)
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()

    # Mock OpenAI client
    mock_openai_client = AsyncMock()
    mock_openai_client.models.list = AsyncMock(return_value=MagicMock())

    # Mock chat completion
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = "This is a generated hypothetical document about the query."
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    manager.get_openai_client = AsyncMock(return_value=mock_openai_client)
    return manager


@pytest.fixture
def hyde_generator(mock_config, mock_prompt_config, mock_client_manager):
    """Create HyDE generator for testing."""
    return HypotheticalDocumentGenerator(
        config=mock_config,
        prompt_config=mock_prompt_config,
        client_manager=mock_client_manager,
    )


class TestHyDEGeneratorInitialization:
    """Test HyDE generator initialization."""

    def test_generator_initialization(
        self, hyde_generator, mock_config, mock_prompt_config, mock_client_manager
    ):
        """Test basic generator initialization."""
        assert hyde_generator.config == mock_config
        assert hyde_generator.prompt_config == mock_prompt_config
        assert hyde_generator.client_manager == mock_client_manager
        assert hyde_generator._llm_client is None
        assert hyde_generator._initialized is False

        # Check metrics initialization
        assert hyde_generator.generation_count == 0
        assert hyde_generator.total_generation_time == 0.0
        assert hyde_generator.total_tokens_used == 0
        assert hyde_generator.total_cost == 0.0

        # Check model pricing
        assert "gpt-3.5-turbo" in hyde_generator.model_pricing
        assert "gpt-4" in hyde_generator.model_pricing

    @pytest.mark.asyncio
    async def test_initialize_success(self, hyde_generator, mock_client_manager):
        """Test successful initialization."""
        await hyde_generator.initialize()

        assert hyde_generator._initialized is True
        mock_client_manager.initialize.assert_called_once()
        mock_client_manager.get_openai_client.assert_called_once()

        # Verify OpenAI client was tested
        llm_client = await mock_client_manager.get_openai_client()
        llm_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_openai_client(
        self, hyde_generator, mock_client_manager
    ):
        """Test initialization when OpenAI client is not available."""
        mock_client_manager.get_openai_client.return_value = None

        with pytest.raises(EmbeddingServiceError, match="OpenAI client not available"):
            await hyde_generator.initialize()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(
        self, hyde_generator, mock_client_manager
    ):
        """Test initialization when connection test fails."""
        mock_openai_client = await mock_client_manager.get_openai_client()
        mock_openai_client.models.list.side_effect = Exception("Connection failed")

        with pytest.raises(
            EmbeddingServiceError, match="Failed to initialize HyDE generator"
        ):
            await hyde_generator.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, hyde_generator, mock_client_manager):
        """Test that initialization is idempotent."""
        await hyde_generator.initialize()
        await hyde_generator.initialize()  # Second call

        # Should only initialize once
        mock_client_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, hyde_generator, mock_client_manager):
        """Test generator cleanup."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = MagicMock()

        await hyde_generator.cleanup()

        mock_client_manager.cleanup.assert_called_once()
        assert hyde_generator._llm_client is None
        assert hyde_generator._initialized is False


class TestQueryClassification:
    """Test query classification for prompt selection."""

    def test_classify_technical_query(self, hyde_generator):
        """Test classification of technical queries."""
        technical_queries = [
            "How to use the API function?",
            "Configuration parameters for the method",
            "Function returns error message",
        ]

        for query in technical_queries:
            query_type = hyde_generator._classify_query(query)
            assert query_type == "technical"

    def test_classify_code_query(self, hyde_generator):
        """Test classification of code queries."""
        code_queries = [
            "How to implement this in Python?",
            "JavaScript code example needed",
            "Show me the code to import library",
        ]

        for query in code_queries:
            query_type = hyde_generator._classify_query(query)
            assert query_type == "code"

    def test_classify_tutorial_query(self, hyde_generator):
        """Test classification of tutorial queries."""
        tutorial_queries = [
            "Tutorial for getting started",
            "Learn how to use this guide",
            "Step by step introduction",
        ]

        for query in tutorial_queries:
            query_type = hyde_generator._classify_query(query)
            assert query_type == "tutorial"

    def test_classify_general_query(self, hyde_generator):
        """Test classification of general queries."""
        general_queries = [
            "What is the weather today?",
            "Explain machine learning concepts",
            "Random question about nothing specific",
        ]

        for query in general_queries:
            query_type = hyde_generator._classify_query(query)
            assert query_type == "general"


class TestPromptGeneration:
    """Test prompt generation and variations."""

    def test_get_base_prompt_technical(self, hyde_generator):
        """Test getting base prompt for technical queries."""
        prompt = hyde_generator._get_base_prompt("technical")
        assert "technical question" in prompt.lower()
        assert "{query}" in prompt

    def test_get_base_prompt_code(self, hyde_generator):
        """Test getting base prompt for code queries."""
        prompt = hyde_generator._get_base_prompt("code")
        assert "code" in prompt.lower()
        assert "{query}" in prompt

    def test_get_base_prompt_tutorial(self, hyde_generator):
        """Test getting base prompt for tutorial queries."""
        prompt = hyde_generator._get_base_prompt("tutorial")
        assert "step by step" in prompt.lower()
        assert "{query}" in prompt

    def test_get_base_prompt_general(self, hyde_generator):
        """Test getting base prompt for general queries."""
        prompt = hyde_generator._get_base_prompt("general")
        assert "answer" in prompt.lower()
        assert "{query}" in prompt

    def test_get_base_prompt_unknown_type(self, hyde_generator):
        """Test getting base prompt for unknown query type."""
        prompt = hyde_generator._get_base_prompt("unknown")
        # Should fall back to general prompt
        assert "answer" in prompt.lower()

    def test_build_diverse_prompts_basic(self, hyde_generator):
        """Test building diverse prompts."""
        hyde_generator.config.num_generations = 3
        hyde_generator.config.prompt_variation = True

        query = "How to use machine learning?"
        domain = "technical"

        prompts = hyde_generator._build_diverse_prompts(query, domain)

        assert len(prompts) == 3
        assert all(query in prompt for prompt in prompts)
        # Should have some variation in prompts
        assert len(set(prompts)) > 1  # At least some should be different

    def test_build_diverse_prompts_no_variation(self, hyde_generator):
        """Test building prompts without variation."""
        hyde_generator.config.prompt_variation = False
        hyde_generator.config.num_generations = 2

        query = "What is Python?"
        prompts = hyde_generator._build_diverse_prompts(query)

        assert len(prompts) == 2
        # Without variation, prompts might be the same
        assert all(query in prompt for prompt in prompts)

    def test_generate_prompt_variations(self, hyde_generator):
        """Test prompt variation generation."""
        base_prompt = "Answer this question: {query}"
        query = "How to use APIs?"
        domain = "technical"

        variations = hyde_generator._generate_prompt_variations(
            base_prompt, query, domain
        )

        assert len(variations) >= 1
        assert all(query in variation for variation in variations)
        assert all(domain in variation for variation in variations)
        # Should have different prefixes/styles
        assert len(set(variations)) == len(variations)


class TestDocumentGeneration:
    """Test hypothetical document generation."""

    @pytest.mark.asyncio
    async def test_generate_documents_basic(self, hyde_generator, mock_client_manager):
        """Test basic document generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        query = "How to use machine learning algorithms?"

        result = await hyde_generator.generate_documents(query)

        assert isinstance(result, GenerationResult)
        assert len(result.documents) > 0
        assert result.generation_time > 0
        assert result.tokens_used > 0
        assert result.cost_estimate >= 0
        assert result.diversity_score >= 0

        # Check metrics were updated
        assert hyde_generator.generation_count == 1
        assert hyde_generator.total_generation_time > 0

    @pytest.mark.asyncio
    async def test_generate_documents_with_domain(
        self, hyde_generator, mock_client_manager
    ):
        """Test document generation with domain context."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        query = "API documentation best practices"
        domain = "technical"
        context = {"project": "api_docs", "audience": "developers"}

        result = await hyde_generator.generate_documents(query, domain, context)

        assert isinstance(result, GenerationResult)
        assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_generate_documents_parallel(
        self, hyde_generator, mock_client_manager
    ):
        """Test parallel document generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()
        hyde_generator.config.parallel_generation = True
        hyde_generator.config.num_generations = 3

        query = "Python programming tutorial"

        start_time = time.time()
        result = await hyde_generator.generate_documents(query)
        generation_time = time.time() - start_time

        assert isinstance(result, GenerationResult)
        assert len(result.documents) > 0
        # Parallel should be reasonably fast
        assert generation_time < 5.0

    @pytest.mark.asyncio
    async def test_generate_documents_sequential(
        self, hyde_generator, mock_client_manager
    ):
        """Test sequential document generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()
        hyde_generator.config.parallel_generation = False
        hyde_generator.config.num_generations = 2

        query = "Database optimization techniques"

        result = await hyde_generator.generate_documents(query)

        assert isinstance(result, GenerationResult)
        assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_generate_documents_not_initialized(self, hyde_generator):
        """Test document generation when not initialized."""
        query = "test query"

        with pytest.raises(
            EmbeddingServiceError, match="HyDE generator not initialized"
        ):
            await hyde_generator.generate_documents(query)

    @pytest.mark.asyncio
    async def test_generate_documents_llm_failure(
        self, hyde_generator, mock_client_manager
    ):
        """Test document generation when LLM fails."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        # Mock LLM failure
        hyde_generator._llm_client.chat.completions.create.side_effect = Exception(
            "LLM error"
        )

        query = "test query"

        with pytest.raises(EmbeddingServiceError, match="Document generation failed"):
            await hyde_generator.generate_documents(query)


class TestSingleDocumentGeneration:
    """Test single document generation methods."""

    @pytest.mark.asyncio
    async def test_generate_single_document_success(
        self, hyde_generator, mock_client_manager
    ):
        """Test successful single document generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        prompt = "Generate a document about machine learning"

        document = await hyde_generator._generate_single_document(prompt)

        assert isinstance(document, str)
        assert len(document) > 0
        assert "generated hypothetical document" in document

    @pytest.mark.asyncio
    async def test_generate_single_document_timeout(
        self, hyde_generator, mock_client_manager
    ):
        """Test single document generation timeout."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()
        hyde_generator.config.generation_timeout_seconds = 0.1

        # Mock slow response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.2)
            return MagicMock()

        hyde_generator._llm_client.chat.completions.create.side_effect = slow_response

        prompt = "Generate a document"

        document = await hyde_generator._generate_single_document(prompt)

        # Should return empty string on timeout
        assert document == ""

    @pytest.mark.asyncio
    async def test_generate_single_document_exception(
        self, hyde_generator, mock_client_manager
    ):
        """Test single document generation with exception."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        # Mock exception
        hyde_generator._llm_client.chat.completions.create.side_effect = Exception(
            "API error"
        )

        prompt = "Generate a document"

        document = await hyde_generator._generate_single_document(prompt)

        # Should return empty string on exception
        assert document == ""


class TestParallelGeneration:
    """Test parallel generation with concurrency control."""

    @pytest.mark.asyncio
    async def test_generate_parallel_success(self, hyde_generator, mock_client_manager):
        """Test successful parallel generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        prompts = [
            "Generate document 1",
            "Generate document 2",
            "Generate document 3",
        ]

        documents = await hyde_generator._generate_parallel(prompts)

        assert len(documents) == len(prompts)
        assert all(isinstance(doc, str) for doc in documents)
        assert all(len(doc) > 0 for doc in documents)

    @pytest.mark.asyncio
    async def test_generate_parallel_with_failures(
        self, hyde_generator, mock_client_manager
    ):
        """Test parallel generation with some failures."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        # Mock some failures
        call_count = 0

        async def mock_generate(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("Generation failed")
            return "Generated document"

        with patch.object(
            hyde_generator, "_generate_single_document", side_effect=mock_generate
        ):
            prompts = ["prompt1", "prompt2", "prompt3"]
            documents = await hyde_generator._generate_parallel(prompts)

            # Should filter out failed generations
            assert len(documents) == 2  # Only successful ones

    @pytest.mark.asyncio
    async def test_generate_parallel_concurrency_limit(
        self, hyde_generator, mock_client_manager
    ):
        """Test parallel generation respects concurrency limits."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()
        hyde_generator.config.max_concurrent_generations = 2

        concurrent_count = 0
        max_concurrent_seen = 0

        async def mock_generate(prompt):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return "Generated document"

        with patch.object(
            hyde_generator, "_generate_single_document", side_effect=mock_generate
        ):
            prompts = [f"prompt{i}" for i in range(5)]
            await hyde_generator._generate_parallel(prompts)

            # Should not exceed concurrency limit
            assert max_concurrent_seen <= 2


class TestSequentialGeneration:
    """Test sequential generation."""

    @pytest.mark.asyncio
    async def test_generate_sequential_success(
        self, hyde_generator, mock_client_manager
    ):
        """Test successful sequential generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        prompts = ["prompt1", "prompt2", "prompt3"]

        documents = await hyde_generator._generate_sequential(prompts)

        assert len(documents) == len(prompts)
        assert all(isinstance(doc, str) for doc in documents)

    @pytest.mark.asyncio
    async def test_generate_sequential_with_failures(
        self, hyde_generator, mock_client_manager
    ):
        """Test sequential generation with some failures."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        call_count = 0

        async def mock_generate(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise Exception("Generation failed")
            return "Generated document"

        with patch.object(
            hyde_generator, "_generate_single_document", side_effect=mock_generate
        ):
            prompts = ["prompt1", "prompt2", "prompt3"]
            documents = await hyde_generator._generate_sequential(prompts)

            # Should continue after failure
            assert len(documents) == 2

    @pytest.mark.asyncio
    async def test_generate_sequential_short_documents(
        self, hyde_generator, mock_client_manager
    ):
        """Test sequential generation filters short documents."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()
        hyde_generator.config.min_generation_length = 10

        call_count = 0

        async def mock_generate(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return "short"  # Too short
            return "This is a long enough generated document for testing purposes"

        with patch.object(
            hyde_generator, "_generate_single_document", side_effect=mock_generate
        ):
            prompts = ["prompt1", "prompt2", "prompt3"]
            documents = await hyde_generator._generate_sequential(prompts)

            # Should filter out short document
            assert len(documents) == 2


class TestPostProcessing:
    """Test document post-processing."""

    def test_post_process_documents_basic(self, hyde_generator):
        """Test basic document post-processing."""
        raw_documents = [
            "  This is a good document  ",
            "Another quality document here",
            "   Short   ",  # Too short
            "This is the third good document",
        ]

        hyde_generator.config.min_generation_length = 5
        hyde_generator.config.filter_duplicates = False

        processed = hyde_generator._post_process_documents(raw_documents, "test query")

        assert len(processed) == 3  # Should filter out short one
        assert all(doc.strip() == doc for doc in processed)  # Should be trimmed

    def test_post_process_documents_filter_duplicates(self, hyde_generator):
        """Test duplicate filtering in post-processing."""
        raw_documents = [
            "This is a unique document",
            "This is another document",
            "This is a unique document",  # Duplicate
            "Yet another unique document",
        ]

        hyde_generator.config.filter_duplicates = True
        hyde_generator.config.min_generation_length = 3

        processed = hyde_generator._post_process_documents(raw_documents, "test query")

        assert len(processed) == 3  # Should remove duplicate
        assert len(set(processed)) == 3  # All unique

    def test_post_process_documents_no_filter_duplicates(self, hyde_generator):
        """Test post-processing without duplicate filtering."""
        raw_documents = [
            "This is a document",
            "This is a document",  # Same content
            "This is another document",
        ]

        hyde_generator.config.filter_duplicates = False
        hyde_generator.config.min_generation_length = 3

        processed = hyde_generator._post_process_documents(raw_documents, "test query")

        assert len(processed) == 3  # Should keep duplicates

    def test_post_process_documents_empty_input(self, hyde_generator):
        """Test post-processing with empty input."""
        processed = hyde_generator._post_process_documents([], "test query")
        assert processed == []


class TestMetricsAndCosts:
    """Test metrics calculation and cost estimation."""

    def test_calculate_cost_gpt_35_turbo(self, hyde_generator):
        """Test cost calculation for GPT-3.5 Turbo."""
        hyde_generator.config.generation_model = "gpt-3.5-turbo"

        tokens = 1000.0
        cost = hyde_generator._calculate_cost(tokens)

        assert cost > 0
        # Should be relatively cheap for GPT-3.5
        assert cost < 0.01

    def test_calculate_cost_gpt_4(self, hyde_generator):
        """Test cost calculation for GPT-4."""
        hyde_generator.config.generation_model = "gpt-4"

        tokens = 1000.0
        cost = hyde_generator._calculate_cost(tokens)

        assert cost > 0
        # Should be more expensive than GPT-3.5
        gpt35_cost = hyde_generator._calculate_cost(tokens)
        hyde_generator.config.generation_model = "gpt-3.5-turbo"
        assert cost > gpt35_cost

    def test_calculate_cost_unknown_model(self, hyde_generator):
        """Test cost calculation for unknown model."""
        hyde_generator.config.generation_model = "unknown-model"

        tokens = 1000.0
        cost = hyde_generator._calculate_cost(tokens)

        # Should use fallback pricing
        assert cost > 0

    def test_calculate_diversity_score_high_diversity(self, hyde_generator):
        """Test diversity calculation for diverse documents."""
        diverse_documents = [
            "This document discusses machine learning algorithms and neural networks",
            "Python programming involves writing clean, readable code with functions",
            "Database optimization requires understanding indexes and query performance",
        ]

        diversity = hyde_generator._calculate_diversity_score(diverse_documents)

        assert 0.0 <= diversity <= 1.0
        # Should have good diversity
        assert diversity > 0.5

    def test_calculate_diversity_score_low_diversity(self, hyde_generator):
        """Test diversity calculation for similar documents."""
        similar_documents = [
            "Machine learning algorithms are powerful",
            "Machine learning algorithms are very powerful",
            "Machine learning algorithms can be quite powerful",
        ]

        diversity = hyde_generator._calculate_diversity_score(similar_documents)

        assert 0.0 <= diversity <= 1.0
        # Should have low diversity due to similarity
        assert diversity < 0.5

    def test_calculate_diversity_score_single_document(self, hyde_generator):
        """Test diversity calculation for single document."""
        single_document = ["This is a single document"]

        diversity = hyde_generator._calculate_diversity_score(single_document)

        assert diversity == 0.0

    def test_calculate_diversity_score_empty_documents(self, hyde_generator):
        """Test diversity calculation for empty documents."""
        diversity = hyde_generator._calculate_diversity_score([])
        assert diversity == 0.0

    def test_get_metrics_basic(self, hyde_generator):
        """Test getting basic metrics."""
        # Set up some metrics
        hyde_generator.generation_count = 10
        hyde_generator.total_generation_time = 25.0
        hyde_generator.total_tokens_used = 5000
        hyde_generator.total_cost = 0.15

        metrics = hyde_generator.get_metrics()

        assert metrics["generation_count"] == 10
        assert metrics["total_generation_time"] == 25.0
        assert metrics["avg_generation_time"] == 2.5  # 25.0 / 10
        assert metrics["total_tokens_used"] == 5000
        assert metrics["total_cost"] == 0.15
        assert metrics["avg_cost_per_generation"] == 0.015  # 0.15 / 10

    def test_get_metrics_no_generations(self, hyde_generator):
        """Test getting metrics when no generations have occurred."""
        metrics = hyde_generator.get_metrics()

        assert metrics["generation_count"] == 0
        assert metrics["avg_generation_time"] == 0.0
        assert metrics["avg_cost_per_generation"] == 0.0


class TestGenerationResult:
    """Test GenerationResult model."""

    def test_generation_result_creation(self):
        """Test creating GenerationResult."""
        result = GenerationResult(
            documents=["doc1", "doc2"],
            generation_time=1.5,
            tokens_used=500,
            cost_estimate=0.02,
            diversity_score=0.75,
        )

        assert result.documents == ["doc1", "doc2"]
        assert result.generation_time == 1.5
        assert result.tokens_used == 500
        assert result.cost_estimate == 0.02
        assert result.cached is False  # Default
        assert result.diversity_score == 0.75

    def test_generation_result_with_cached(self):
        """Test GenerationResult with cached flag."""
        result = GenerationResult(
            documents=["cached doc"],
            generation_time=0.0,
            tokens_used=0,
            cost_estimate=0.0,
            cached=True,
        )

        assert result.cached is True
        assert result.diversity_score == 0.0  # Default


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_generate_documents_initialization_check(self, hyde_generator):
        """Test that generation requires initialization."""
        # Don't initialize
        query = "test query"

        with pytest.raises(
            EmbeddingServiceError, match="HyDE generator not initialized"
        ):
            await hyde_generator.generate_documents(query)

    @pytest.mark.asyncio
    async def test_generate_documents_comprehensive_error(
        self, hyde_generator, mock_client_manager
    ):
        """Test comprehensive error handling during generation."""
        hyde_generator._initialized = True
        hyde_generator._llm_client = await mock_client_manager.get_openai_client()

        # Mock complete failure in generation
        with patch.object(
            hyde_generator,
            "_build_diverse_prompts",
            side_effect=Exception("Prompt error"),
        ):
            query = "test query"

            with pytest.raises(
                EmbeddingServiceError, match="Document generation failed"
            ):
                await hyde_generator.generate_documents(query)

    @pytest.mark.asyncio
    async def test_initialization_with_client_manager_creation(self):
        """Test initialization when client manager is not provided."""
        config = HyDEConfig()
        prompt_config = HyDEPromptConfig()

        # Test that generator creates its own client manager
        with patch.object(ClientManager, "from_unified_config") as mock_from_config:
            mock_manager = AsyncMock()
            mock_from_config.return_value = mock_manager

            generator = HypotheticalDocumentGenerator(config, prompt_config)

            assert generator.client_manager == mock_manager
            mock_from_config.assert_called_once()
