#!/usr/bin/env python3
"""
Unit tests for HyDE hypothetical document generator.

Tests document generation, prompt engineering, diversity scoring, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.services.hyde.config import HyDEConfig
from src.services.hyde.config import HyDEPromptConfig
from src.services.hyde.generator import GenerationResult
from src.services.hyde.generator import HypotheticalDocumentGenerator


class TestGenerationResult:
    """Test cases for GenerationResult model."""

    def test_valid_generation_result(self):
        """Test creating a valid GenerationResult."""
        documents = ["doc1", "doc2", "doc3"]
        metadata = {"key": "value"}

        result = GenerationResult(
            documents=documents,
            metadata=metadata,
            success=True,
            error=None,
        )

        assert result.documents == documents
        assert result.metadata == metadata
        assert result.success is True
        assert result.error is None

    def test_failed_generation_result(self):
        """Test creating a failed GenerationResult."""
        error_msg = "Generation failed"

        result = GenerationResult(
            documents=[],
            metadata={},
            success=False,
            error=error_msg,
        )

        assert result.documents == []
        assert result.metadata == {}
        assert result.success is False
        assert result.error == error_msg

    def test_generation_result_defaults(self):
        """Test GenerationResult with default values."""
        documents = ["doc1"]

        result = GenerationResult(documents=documents)

        assert result.documents == documents
        assert result.metadata == {}
        assert result.success is True
        assert result.error is None


class TestHypotheticalDocumentGenerator:
    """Test cases for HypotheticalDocumentGenerator."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = AsyncMock()

        # Mock successful completion
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated hypothetical document"
        mock_response.choices = [mock_choice]

        client.chat.completions.create.return_value = mock_response

        return client

    @pytest.fixture
    def hyde_config(self):
        """Default HyDE configuration for testing."""
        return HyDEConfig()

    @pytest.fixture
    def prompt_config(self):
        """Default prompt configuration for testing."""
        return HyDEPromptConfig()

    @pytest.fixture
    def generator(self, mock_openai_client, hyde_config, prompt_config):
        """HypotheticalDocumentGenerator instance for testing."""
        return HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

    @pytest.mark.asyncio
    async def test_generate_documents_basic(self, generator, mock_openai_client):
        """Test basic document generation."""
        query = "What is machine learning?"

        result = await generator.generate_documents(query)

        assert isinstance(result, GenerationResult)
        assert result.success is True
        assert len(result.documents) > 0
        assert all(isinstance(doc, str) for doc in result.documents)
        assert mock_openai_client.chat.completions.create.called

    @pytest.mark.asyncio
    async def test_generate_documents_with_domain(self, generator, mock_openai_client):
        """Test document generation with domain specification."""
        query = "API authentication"
        domain = "api"

        result = await generator.generate_documents(query, domain=domain)

        assert result.success is True
        assert len(result.documents) > 0
        # Should have called OpenAI with domain-specific prompt
        assert mock_openai_client.chat.completions.create.called

    @pytest.mark.asyncio
    async def test_generate_documents_with_context(self, generator, mock_openai_client):
        """Test document generation with additional context."""
        query = "Database queries"
        context = {"language": "python", "framework": "django"}

        result = await generator.generate_documents(query, context=context)

        assert result.success is True
        assert len(result.documents) > 0
        assert "context" in result.metadata

    @pytest.mark.asyncio
    async def test_generate_documents_parallel(self, mock_openai_client):
        """Test parallel document generation."""
        config = HyDEConfig(parallel_generation=True, num_generations=3)
        prompt_config = HyDEPromptConfig()

        generator = HypotheticalDocumentGenerator(
            config=config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

        query = "How to use APIs?"

        result = await generator.generate_documents(query)

        assert result.success is True
        assert len(result.documents) == config.num_generations

    @pytest.mark.asyncio
    async def test_generate_documents_sequential(self, mock_openai_client):
        """Test sequential document generation."""
        config = HyDEConfig(parallel_generation=False, num_generations=2)
        prompt_config = HyDEPromptConfig()

        generator = HypotheticalDocumentGenerator(
            config=config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

        query = "Database design patterns"

        result = await generator.generate_documents(query)

        assert result.success is True
        assert len(result.documents) == config.num_generations

    @pytest.mark.asyncio
    async def test_generate_documents_with_diversity_scoring(self, generator):
        """Test document generation with diversity scoring enabled."""
        generator.config.enable_diversity_scoring = True

        # Mock different responses for diversity
        responses = [
            "First unique document about machine learning algorithms",
            "Second document covering neural networks and deep learning",
            "Third document discussing data preprocessing techniques",
        ]

        def mock_completion(*args, **kwargs):
            response = MagicMock()
            choice = MagicMock()
            choice.message.content = (
                responses.pop(0) if responses else "Default response"
            )
            response.choices = [choice]
            return response

        generator.llm_client.chat.completions.create.side_effect = mock_completion

        query = "Machine learning tutorials"
        result = await generator.generate_documents(query)

        assert result.success is True
        assert len(result.documents) > 0
        assert "diversity_scores" in result.metadata

    @pytest.mark.asyncio
    async def test_generate_documents_timeout_handling(self, mock_openai_client):
        """Test timeout handling during generation."""
        config = HyDEConfig(generation_timeout=0.001)  # Very short timeout
        prompt_config = HyDEPromptConfig()

        # Mock a slow response
        async def slow_completion(*args, **kwargs):
            await asyncio.sleep(0.1)  # Longer than timeout
            response = MagicMock()
            choice = MagicMock()
            choice.message.content = "Slow response"
            response.choices = [choice]
            return response

        mock_openai_client.chat.completions.create.side_effect = slow_completion

        generator = HypotheticalDocumentGenerator(
            config=config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

        query = "Timeout test query"
        result = await generator.generate_documents(query)

        # Should handle timeout gracefully
        assert isinstance(result, GenerationResult)
        # May succeed with partial results or fail gracefully

    @pytest.mark.asyncio
    async def test_generate_documents_api_error_handling(self, mock_openai_client):
        """Test handling of API errors during generation."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        config = HyDEConfig(max_retries=2)
        prompt_config = HyDEPromptConfig()

        generator = HypotheticalDocumentGenerator(
            config=config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

        query = "Error test query"
        result = await generator.generate_documents(query)

        # Should handle error gracefully and return failed result
        assert isinstance(result, GenerationResult)
        assert result.success is False
        assert result.error is not None
        assert "API Error" in result.error

    @pytest.mark.asyncio
    async def test_generate_documents_retry_logic(self, mock_openai_client):
        """Test retry logic for failed generations."""
        config = HyDEConfig(max_retries=3)
        prompt_config = HyDEPromptConfig()

        # Mock failing twice, then succeeding
        call_count = 0

        def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Failure {call_count}")

            response = MagicMock()
            choice = MagicMock()
            choice.message.content = "Success after retries"
            response.choices = [choice]
            return response

        mock_openai_client.chat.completions.create.side_effect = mock_completion

        generator = HypotheticalDocumentGenerator(
            config=config,
            prompt_config=prompt_config,
            llm_client=mock_openai_client,
        )

        query = "Retry test query"
        result = await generator.generate_documents(query)

        # Should eventually succeed after retries
        assert result.success is True
        assert len(result.documents) > 0
        assert call_count >= 3  # Should have retried

    def test_build_diverse_prompts(self, generator):
        """Test building diverse prompts for variety in generation."""
        query = "API documentation"
        domain = "api"
        context = {"framework": "REST"}

        prompts = generator._build_diverse_prompts(query, domain, context)

        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(prompt, str) for prompt in prompts)
        # Should have some variation in prompts
        if len(prompts) > 1:
            assert prompts[0] != prompts[1]

    def test_calculate_diversity_score(self, generator):
        """Test diversity score calculation for documents."""
        documents = [
            "This is about machine learning algorithms and neural networks",
            "Database design patterns and normalization techniques",
            "Web development with JavaScript and React frameworks",
        ]

        scores = generator._calculate_diversity_scores(documents)

        assert isinstance(scores, list)
        assert len(scores) == len(documents)
        assert all(isinstance(score, int | float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)

    def test_build_prompt_with_context(self, generator):
        """Test building prompts with context information."""
        query = "Database optimization"
        context = {"language": "python", "database": "postgresql"}

        prompt = generator._build_prompt_with_context(query, None, context)

        assert isinstance(prompt, str)
        assert query in prompt
        assert "python" in prompt.lower()
        assert "postgresql" in prompt.lower()

    def test_extract_document_from_response(self, generator):
        """Test extracting clean document content from LLM response."""
        # Test clean response
        clean_response = "This is a clean document about APIs."
        extracted = generator._extract_document_from_response(clean_response)
        assert extracted == clean_response

        # Test response with markdown
        markdown_response = "```\nCode example\n```\nThis is documentation."
        extracted = generator._extract_document_from_response(markdown_response)
        assert "documentation" in extracted

        # Test response with extra whitespace
        whitespace_response = "   \n\n  Document content  \n\n  "
        extracted = generator._extract_document_from_response(whitespace_response)
        assert extracted.strip() == "Document content"
