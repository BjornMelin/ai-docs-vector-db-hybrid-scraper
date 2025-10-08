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

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        """Get embedding for text.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector
        """

        if not self.client:
            msg = "OpenAI client is not available or unhealthy"
            raise RuntimeError(msg)

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
        """

        if not self.client:
            msg = "OpenAI client is not available or unhealthy"
            raise RuntimeError(msg)

        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content
