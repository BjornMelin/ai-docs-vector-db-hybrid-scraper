"""OpenAI client provider."""

import logging

try:
    from openai import AsyncOpenAI
except ImportError:
    # Create a placeholder if openai is not available
    class AsyncOpenAI:
        pass



logger = logging.getLogger(__name__)


class OpenAIClientProvider:
    """Provider for OpenAI client with health checks and circuit breaker."""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
    ):
        self._client = openai_client
        self._healthy = True

    @property
    def client(self) -> AsyncOpenAI | None:
        """Get the OpenAI client if available and healthy."""
        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check OpenAI client health."""
        try:
            if not self._client:
                return False

            # Simple API call to check connectivity
            await self._client.models.list()
            self._healthy = True
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")  # TODO: Convert f-string to logging format
            self._healthy = False
            return False

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Get embedding for text.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            raise RuntimeError("OpenAI client is not available or unhealthy")

        response = await self.client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    async def chat_completion(
        self, messages: list, model: str = "gpt-4o-mini", **kwargs
    ) -> str:
        """Get chat completion.

        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Response text

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            raise RuntimeError("OpenAI client is not available or unhealthy")

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content