"""Browser-use provider leveraging agentic automation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from src.config.browser import BrowserUseSettings
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.runtime import execute_with_retry

from .base import BrowserProvider, ProviderContext


@dataclass(slots=True)
class _BrowserUseDeps:
    """Container bundling runtime browser-use dependencies."""

    agent_cls: Any
    browser_cls: Any
    config_cls: Any
    llm: Any


class BrowserUseProvider(BrowserProvider):
    """High-level automation using browser-use Agent."""

    kind = ProviderKind.BROWSER_USE

    def __init__(self, context: ProviderContext, settings: BrowserUseSettings) -> None:
        super().__init__(context)
        self._settings = settings
        self._deps: _BrowserUseDeps | None = None

    def _load_dependencies(self) -> _BrowserUseDeps:
        """Import browser-use and LLM providers on demand."""

        try:
            browser_use = import_module("browser_use")
            agent_cls = browser_use.Agent
            browser_cls = browser_use.Browser
            config_cls = browser_use.BrowserConfig
        except (ModuleNotFoundError, AttributeError) as exc:
            raise BrowserProviderError(
                "browser-use package is not available",
                provider=self.kind.value,
            ) from exc

        llm_provider = self._settings.llm_provider
        if llm_provider == "openai":
            module = import_module("langchain_openai")
            llm_cls = module.ChatOpenAI
            llm = llm_cls(model=self._settings.model, temperature=0.0)
        elif llm_provider == "anthropic":
            module = import_module("langchain_anthropic")
            llm_cls = module.ChatAnthropic
            llm = llm_cls(model=self._settings.model, temperature=0.0)
        elif llm_provider == "gemini":
            module = import_module("langchain_google_genai")
            llm_cls = module.ChatGoogleGenerativeAI
            llm = llm_cls(model=self._settings.model, temperature=0.0)
        else:
            raise BrowserProviderError(
                f"Unsupported LLM provider: {llm_provider}",
                provider=self.kind.value,
            )

        return _BrowserUseDeps(
            agent_cls=agent_cls,
            browser_cls=browser_cls,
            config_cls=config_cls,
            llm=llm,
        )

    async def initialize(self) -> None:
        """Ensure dependencies are importable."""

        self._deps = self._load_dependencies()

    async def close(self) -> None:
        """No persistent resources to dispose."""

        self._deps = None

    def _task_from_request(self, request: ScrapeRequest) -> str:
        if request.instructions:
            steps = "\n".join(
                f"- {instr.get('description', instr)}" for instr in request.instructions
            )
            return (
                f"Navigate to {request.url} and execute the following "
                f"instructions:\n{steps}"
            )
        if request.metadata and isinstance(
            request.metadata.get("browser_use_task"), str
        ):
            return request.metadata["browser_use_task"]
        return f"Navigate to {request.url} and extract the visible content."

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Delegate to browser-use Agent."""

        if self._deps is None:  # pragma: no cover - lifecycle guard
            raise RuntimeError("Provider not initialized")

        deps = self._deps
        browser = deps.browser_cls(
            config=deps.config_cls(headless=self._settings.headless)
        )
        task = self._task_from_request(request)
        agent = deps.agent_cls(
            task=task,
            llm=deps.llm,
            browser=browser,
        )
        try:
            timeout_seconds = (request.timeout_ms or self._settings.timeout_ms) / 1000

            async def _call() -> dict | str:
                return await asyncio.wait_for(agent.run(), timeout=timeout_seconds)

            result = await execute_with_retry(
                provider=self.kind,
                operation="agent_run",
                func=_call,
            )
        finally:
            await browser.close()

        log = result if isinstance(result, dict) else {"result": result}
        content = log.get("extracted_content") or log.get("result") or ""
        return BrowserResult(
            success=True,
            url=request.url,
            title=log.get("title", ""),
            content=content,
            html=log.get("html", ""),
            metadata=log,
            provider=self.kind,
            links=None,
            assets=None,
            elapsed_ms=None,
        )
