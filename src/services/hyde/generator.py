"""Hypothetical document generator for HyDE."""

import asyncio
import contextlib
import hashlib
import itertools
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.services.base import BaseService
from src.services.errors import EmbeddingServiceError

from .config import HyDEConfig, HyDEPromptConfig


logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Result of hypothetical document generation."""

    documents: list[str]
    generation_time: float
    tokens_used: int
    cost_estimate: float
    cached: bool = False
    diversity_score: float = 0.0


class HypotheticalDocumentGenerator(BaseService):
    """Generate hypothetical documents for HyDE using LLM."""

    def __init__(
        self,
        config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        api_key: str | None,
        *,
        max_retries: int = 3,
        timeout: float | None = None,
    ):
        """Initialize generator.

        Args:
            config: HyDE configuration
            prompt_config: Prompt configuration
            api_key: OpenAI API key used for Responses API calls.
            max_retries: Retry attempts configured on the OpenAI client.
            timeout: Optional request timeout for OpenAI requests.

        """
        super().__init__(None)
        self.config = config
        self.prompt_config = prompt_config
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._llm_client: AsyncOpenAI | None = None

        # Metrics tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.total_tokens_used = 0
        self.total_cost = 0.0

        # Model pricing (per 1K tokens)
        self.model_pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }

    async def initialize(self) -> None:
        """Initialize the generator.

        Validates that OpenAI client is available.

        Raises:
            EmbeddingServiceError: If OpenAI client unavailable
        """

        if self._initialized:
            return

        if not self._api_key:
            msg = "OpenAI API key not configured for HyDE generator"
            raise EmbeddingServiceError(msg)

        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._max_retries is not None:
            client_kwargs["max_retries"] = self._max_retries
        if self._timeout is not None:
            client_kwargs["timeout"] = self._timeout

        try:
            self._llm_client = AsyncOpenAI(**client_kwargs)
            self._initialized = True
            logger.info("HyDE document generator initialized")
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to initialize HyDE generator: {exc}"
            raise EmbeddingServiceError(msg) from exc

    async def cleanup(self) -> None:
        """Cleanup generator resources.

        Releases LLM client reference.
        Safe to call multiple times.
        """
        client = self._llm_client
        self._llm_client = None
        self._initialized = False
        if client is not None:
            close_fn = getattr(client, "close", None)
            if close_fn is None:
                return
            with contextlib.suppress(ConnectionError, RuntimeError, TimeoutError):
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
        logger.info("HyDE document generator cleaned up")

    async def generate_documents(
        self,
        query: str,
        domain: str | None = None,
        context: dict[str, object] | None = None,
    ) -> GenerationResult:
        """Generate hypothetical documents for a query.

        Args:
            query: Search query to generate documents for
            domain: Optional domain hint (e.g., "python", "javascript")
            context: Optional context dictionary for better generation

        Returns:
            GenerationResult: Contains generated documents, metrics, and metadata:
                - documents: List of generated hypothetical documents
                - generation_time: Time taken to generate in seconds
                - tokens_used: Estimated token count
                - cost_estimate: Estimated cost in USD
                - diversity_score: Diversity measure between documents (0-1)

        Raises:
            EmbeddingServiceError: If generation fails or LLM is unavailable

        """
        self._validate_initialized()

        start_time = time.time()

        try:
            # Build prompts with variations for diversity
            prompts = self._build_diverse_prompts(query, domain, context)

            if self.config.parallel_generation:
                # Generate in parallel
                generation_results = await self._generate_parallel(prompts)
            else:
                # Generate sequentially
                generation_results = await self._generate_sequential(prompts)

            raw_documents = [text for text, _ in generation_results]
            token_lookup = dict(generation_results)

            # Filter and post-process
            documents = self._post_process_documents(raw_documents, query)

            generation_time = time.time() - start_time

            # Calculate metrics
            total_tokens = 0
            for doc in documents:
                tokens = token_lookup.get(doc)
                if tokens is None:
                    tokens = int(len(doc.split()) * 1.3)
                total_tokens += tokens
            cost_estimate = self._calculate_cost(total_tokens)
            diversity_score = self._calculate_diversity_score(documents)

            # Update tracking
            self.generation_count += 1
            self.total_generation_time += generation_time
            self.total_tokens_used += total_tokens
            self.total_cost += cost_estimate

            result = GenerationResult(
                documents=documents,
                generation_time=generation_time,
                tokens_used=int(total_tokens),
                cost_estimate=cost_estimate,
                diversity_score=diversity_score,
            )

            if self.config.log_generations:
                logger.debug(
                    "Generated %d documents for query '%s' in %.2fs, diversity=%.2f",
                    len(documents),
                    query,
                    generation_time,
                    diversity_score,
                )

        except Exception as e:
            logger.exception("Failed to generate hypothetical documents: ")
            msg = f"Document generation failed: {e}"
            raise EmbeddingServiceError(msg) from e

        return result

    def _build_diverse_prompts(
        self,
        query: str,
        domain: str | None = None,
        _context: dict[str, object] | None = None,
    ) -> list[str]:
        """Build diverse prompts for the query.

        Args:
            query: Search query to build prompts for
            domain: Optional domain hint for context
            context: Optional context dictionary (currently unused)

        Returns:
            list[str]: List of diverse prompts for document generation

        """
        # Determine query type
        query_type = self._classify_query(query)

        # Get base prompt template
        base_prompt = self._get_base_prompt(query_type)

        prompts = []

        if self.config.prompt_variation and len(prompts) < self.config.num_generations:
            # Generate prompt variations
            variations = self._generate_prompt_variations(base_prompt, query, domain)
            prompts.extend(variations)

        # Fill remaining slots with base prompt
        while len(prompts) < self.config.num_generations:
            prompts.append(
                f"{base_prompt}".replace("{query}", query).replace(
                    "{domain}", domain or "technical"
                )
            )

        return prompts[: self.config.num_generations]

    def _classify_query(self, query: str) -> str:
        """Classify query type for prompt selection."""
        query_lower = query.lower()

        # Check for technical keywords
        if any(
            keyword in query_lower for keyword in self.prompt_config.technical_keywords
        ):
            return "technical"

        # Check for code keywords
        if any(keyword in query_lower for keyword in self.prompt_config.code_keywords):
            return "code"

        # Check for tutorial keywords
        if any(
            keyword in query_lower for keyword in self.prompt_config.tutorial_keywords
        ):
            return "tutorial"

        return "general"

    def _get_base_prompt(self, query_type: str) -> str:
        """Get base prompt template for query type."""
        prompt_map = {
            "technical": self.prompt_config.technical_prompt,
            "code": self.prompt_config.code_prompt,
            "tutorial": self.prompt_config.tutorial_prompt,
            "general": self.prompt_config.general_prompt,
        }

        return prompt_map.get(query_type, self.prompt_config.general_prompt)

    def _generate_prompt_variations(
        self, _base_prompt: str, query: str, domain: str | None
    ) -> list[str]:
        """Generate variations of the base prompt for diversity."""
        variations = []
        variation_templates = self.prompt_config.variation_templates

        # Create different combinations
        for i in range(min(3, self.config.num_generations)):
            # Use different prefixes, instruction styles, and context additions
            prefix_idx = i % len(variation_templates["prefixes"])
            instruction_idx = i % len(variation_templates["instruction_styles"])
            context_idx = i % len(variation_templates["context_additions"])

            prefix = variation_templates["prefixes"][prefix_idx].replace(
                "{domain}", domain or "documentation"
            )
            instruction = variation_templates["instruction_styles"][instruction_idx]
            context_addition = variation_templates["context_additions"][context_idx]

            varied_prompt = (
                f"{prefix} {instruction}\n\nQuestion: {query}\n\n{context_addition}"
            )
            variations.append(varied_prompt)

        return variations

    async def _generate_parallel(self, prompts: list[str]) -> list[tuple[str, int]]:
        """Generate documents in parallel."""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)

        async def generate_single(prompt: str) -> tuple[str, int]:
            async with semaphore:
                return await self._generate_single_document(prompt)

        # Generate all documents concurrently
        tasks = [generate_single(prompt) for prompt in prompts]
        documents = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and empty documents
        results: list[tuple[str, int]] = []
        for item in documents:
            if not isinstance(item, tuple):
                continue
            text, tokens = item
            if len(text.strip()) >= self.config.min_generation_length:
                results.append((text, tokens))
        return results

    async def _generate_sequential(self, prompts: list[str]) -> list[tuple[str, int]]:
        """Generate documents sequentially."""
        documents: list[tuple[str, int]] = []

        for prompt in prompts:
            try:
                document = await self._generate_single_document(prompt)
                text, tokens = document
                if len(text.strip()) >= self.config.min_generation_length:
                    documents.append((text, tokens))
            except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                logger.warning("Failed to generate document: %s", e)
                continue

        return documents

    async def _generate_single_document(self, prompt: str) -> tuple[str, int]:
        """Generate a single hypothetical document."""
        # Type assertion for mypy/pyright
        assert self._llm_client is not None

        try:
            response = await asyncio.wait_for(
                self._llm_client.responses.create(
                    model=self.config.generation_model,
                    temperature=self.config.generation_temperature,
                    max_output_tokens=self.config.max_generation_tokens,
                    input=prompt,
                ),
                timeout=self.config.generation_timeout_seconds,
            )

            total_tokens = 0
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_tokens_value: Any | None = None
                if isinstance(usage, dict):
                    total_tokens_value = usage.get("total_tokens")
                    if not total_tokens_value:
                        total_tokens_value = (usage.get("input_tokens") or 0) + (
                            usage.get("output_tokens") or 0
                        )
                else:
                    total_tokens_value = getattr(usage, "total_tokens", None)
                    if not total_tokens_value:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        total_tokens_value = input_tokens + output_tokens

                try:
                    total_tokens = int(total_tokens_value or 0)
                except (TypeError, ValueError):
                    total_tokens = 0

            return (response.output_text or "").strip(), total_tokens

        except TimeoutError:
            logger.warning("Document generation timed out")
            return "", 0
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning("Failed to generate document: %s", e)
            return "", 0

    def _post_process_documents(self, documents: list[str], _query: str) -> list[str]:
        """Post-process generated documents."""
        if not documents:
            return documents

        processed = []
        seen_hashes = set()

        for doc in documents:
            # Clean up the document
            cleaned_doc = doc.strip()

            # Skip if too short
            if len(cleaned_doc.split()) < self.config.min_generation_length:
                continue

            # Filter duplicates if enabled
            if self.config.filter_duplicates:
                doc_hash = hashlib.sha256(cleaned_doc.encode()).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

            processed.append(cleaned_doc)

        return processed

    def _calculate_cost(self, tokens: float) -> float:
        """Calculate estimated cost for token usage."""
        model_costs = self.model_pricing.get(
            self.config.generation_model,
            {"input": 0.002, "output": 0.002},  # Default fallback
        )

        # Rough estimate: 70% input, 30% output
        input_tokens = tokens * 0.7
        output_tokens = tokens * 0.3

        return (input_tokens / 1000) * model_costs["input"] + (
            output_tokens / 1000
        ) * model_costs["output"]

    def _calculate_diversity_score(self, documents: list[str]) -> float:
        """Calculate diversity score between generated documents."""
        if len(documents) < 2:
            return 0.0

        # Simple diversity measure based on word overlap
        total_similarity = 0.0
        comparisons = 0

        for doc1, doc2 in itertools.combinations(documents, 2):
            doc1_words = set(doc1.lower().split())
            doc2_words = set(doc2.lower().split())

            if doc1_words and doc2_words:
                overlap = len(doc1_words & doc2_words)
                union = len(doc1_words | doc2_words)
                similarity = overlap / union if union > 0 else 0
                total_similarity += similarity
                comparisons += 1

        if comparisons == 0:
            return 0.0

        avg_similarity = total_similarity / comparisons
        diversity_score = 1.0 - avg_similarity  # Higher diversity = lower similarity

        return max(0.0, min(1.0, diversity_score))

    def get_metrics(self) -> dict[str, float | int]:
        """Get generation metrics.

        Returns:
            dict[str, float | int]: Metrics including:
                - generation_count: Total number of generations
                - total_generation_time: Total time spent generating
                - avg_generation_time: Average generation time per request
                - total_tokens_used: Total estimated tokens used
                - total_cost: Total estimated cost in USD
                - avg_cost_per_generation: Average cost per generation

        """
        return {
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "avg_generation_time": (
                self.total_generation_time / self.generation_count
                if self.generation_count > 0
                else 0.0
            ),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "avg_cost_per_generation": (
                self.total_cost / self.generation_count
                if self.generation_count > 0
                else 0.0
            ),
        }
