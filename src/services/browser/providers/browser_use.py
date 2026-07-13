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
    llm: Any


class BrowserUseProvider(BrowserProvider):
    """High-level automation using browser-use Agent."""

    kind = ProviderKind.BROWSER_USE

    def __init__(
        self,
        context: ProviderContext,
        settings: BrowserUseSettings,
        *,
        openai_api_key: str | None = None,
    ) -> None:
        """Initialize browser-use provider with settings and lazy dependency loading."""
        super().__init__(context)
        self._settings = settings
        self._openai_api_key = openai_api_key
        self._deps: _BrowserUseDeps | None = None

    def _load_dependencies(self) -> _BrowserUseDeps:
        """Import browser-use and LLM providers on demand."""
        try:
            browser_use = import_module("browser_use")
            agent_cls = browser_use.Agent
            browser_cls = browser_use.Browser
        except (ModuleNotFoundError, AttributeError) as exc:
            raise BrowserProviderError(
                "browser-use package is not available",
                provider=self.kind.value,
            ) from exc

        try:
            llm_provider = self._settings.llm_provider
            if llm_provider == "openai":
                llm_kwargs: dict[str, Any] = {
                    "model": self._settings.model,
                    "temperature": 0.0,
                }
                if self._openai_api_key:
                    llm_kwargs["api_key"] = self._openai_api_key
                llm = browser_use.ChatOpenAI(**llm_kwargs)
            elif llm_provider == "anthropic":
                llm = browser_use.ChatAnthropic(
                    model=self._settings.model,
                    temperature=0.0,
                )
            elif llm_provider == "gemini":
                llm = browser_use.ChatGoogle(
                    model=self._settings.model,
                    temperature=0.0,
                )
            else:  # pragma: no cover - validated configuration
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise BrowserProviderError(
                f"Browser-use {self._settings.llm_provider} LLM is not configured",
                provider=self.kind.value,
            ) from exc

        return _BrowserUseDeps(
            agent_cls=agent_cls,
            browser_cls=browser_cls,
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
        browser = deps.browser_cls(headless=self._settings.headless)
        task = self._task_from_request(request)
        try:
            agent = deps.agent_cls(
                task=task,
                llm=deps.llm,
                browser=browser,
            )
            timeout_seconds = (request.timeout_ms or self._settings.timeout_ms) / 1000

            async def _call() -> Any:
                return await asyncio.wait_for(agent.run(), timeout=timeout_seconds)

            result = await execute_with_retry(
                provider=self.kind,
                operation="agent_run",
                func=_call,
            )
        finally:
            await browser.kill()

        success, log = _normalize_agent_history(result)
        content = log.get("extracted_content") or log.get("result") or ""
        history = log.get("history") or []
        state = history[-1].get("state", {}) if history else {}
        return BrowserResult(
            success=success,
            url=request.url,
            title=log.get("title") or state.get("title", ""),
            content=str(content),
            html=log.get("html", ""),
            metadata=log,
            provider=self.kind,
            links=None,
            assets=None,
            elapsed_ms=None,
        )


def _normalize_agent_history(result: Any) -> tuple[bool, dict[str, Any]]:
    """Normalize browser-use's ``AgentHistoryList`` into the provider contract."""
    if isinstance(result, dict):
        return bool(result.get("success", True)), dict(result)
    if isinstance(result, str):
        return True, {"result": result}

    final_result = result.final_result()
    successful = result.is_successful()
    payload = result.model_dump(mode="json")
    payload["result"] = final_result or ""
    payload["success"] = successful is True
    return successful is True, payload
